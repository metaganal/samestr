
import itertools
import logging
import os
from os.path import basename, exists

import numpy as np
from scipy import sparse
from Bio import Seq, SeqRecord, AlignIO
from Bio.Align import MultipleSeqAlignment

from samestr.utils.utilities import load_numpy_file
from samestr.filter import consensus


LOG = logging.getLogger(__name__)

# allele subsets with >= 2 members, used for the inclusion-exclusion correction
_MULTI_SUBSETS = [s for k in range(2, 5) for s in itertools.combinations(range(4), k)]


def pairwise_counts(x, chunk=None):  # chunk unused (chunking disabled)
    """
    All-vs-all counts over the alignment positions of x (n_samples, n_pos, 4):

    shared[i, j]  = number of positions where i and j share at least one allele
    overlap[i, j] = number of positions covered in both i and j

    Computed with matrix products instead of an O(n) python loop over samples:

      overlap = C @ C.T                   C = coverage, (n, L)
      shared  = A @ A.T - correction      A = allele presence, flattened (n, 4L)

    A @ A.T counts every shared allele, so a position where two samples share
    k > 1 alleles is counted k times. The exact correction follows from
    inclusion-exclusion, 1[any shared] = sum_S (-1)^(|S|+1) prod_{a in S},
    whose |S| >= 2 terms are non-zero only at multi-allelic sites and are
    therefore computed as sparse products.

    Chunking over positions is disabled (memory assumed ample, <= 512 GB):
    the whole alignment is processed in one pass. float32 GEMMs on 0/1 data
    are exact while n_pos < 2**24; float64 is used beyond that.
    Peak memory ~ x itself + ~25 bytes * n * n_pos + ~20 bytes * n**2.
    """
    n, n_pos, _ = x.shape
    shared = np.zeros((n, n), dtype=np.float64)
    overlap = np.zeros((n, n), dtype=np.float64)
    gemm_dtype = np.float32 if n_pos < 2 ** 24 else np.float64

    # chunked variant (bounds memory to O(n * chunk)); re-enable if needed:
    # for lo in range(0, n_pos, chunk):
    #     a = x[:, lo:lo + chunk] > 0
    #     ... (body below, indented)
    a = x > 0                                           # (n, L, 4) bool
    n_alleles = a.sum(axis=2, dtype=np.int8)            # (n, L)

    cov = np.ascontiguousarray(n_alleles > 0, dtype=gemm_dtype)
    overlap += cov @ cov.T                              # SYRK
    del cov

    flat = a.reshape(n, -1).astype(gemm_dtype)          # (n, 4L)
    shared += flat @ flat.T                             # SYRK
    del flat

    if (n_alleles > 1).any():
        for s in _MULTI_SUBSETS:
            b = sparse.csr_matrix(a[:, :, list(s)].all(axis=2), dtype=np.float64)
            if b.nnz:
                sign = 1 if len(s) % 2 else -1
                shared += sign * (b @ b.T).toarray()

    # values are exact integers <= n_pos; int32 halves the output footprint
    return shared.astype(np.int32), overlap.astype(np.int32)


def _write_matrix(path, samples, mat):
    with open(path, 'w') as out:
        out.write('Sample\t' + '\t'.join(samples) + '\n')
        # convert one row at a time: avoids a full python-object copy of mat
        for sample, row in zip(samples, mat):
            out.write(sample + '\t' + '\t'.join(map(str, row.tolist())) + '\n')


def _dominant_msa(d, samples):
    """Vectorized conversion of dominant-variant frequencies to sequences."""
    present = d > 0                                         # (n, L, 4)
    n_alleles = present.sum(axis=2)
    idx = present.argmax(axis=2)                            # first present allele

    # consensus() leaves at most one allele per site; if not, pick randomly
    multi = n_alleles > 1
    if multi.any():
        r = np.random.random((int(multi.sum()), 4))
        r[~present[multi]] = -1
        idx[multi] = r.argmax(axis=1)

    idx[n_alleles == 0] = 4                                 # gap
    lut = np.frombuffer(b'ACGT-', dtype='S1')
    seqs = lut[idx]                                         # (n, L) bytes

    return MultipleSeqAlignment([
        SeqRecord.SeqRecord(id=s, description=s,
                            seq=Seq.Seq(row.tobytes().decode('ascii')))
        for s, row in zip(samples, seqs)
    ])


def compare(args):

    # if exists, skip
    output_name = os.path.join(args['output_dir'], basename(args['input_file']))
    if exists(output_name):
        LOG.info('Skipping %s. Output file exists.' % args['clade'])
        return True

    # load sample order
    with open(args['input_name'], 'r') as file:
        samples = file.read().strip().split('\n')

    # skip if fewer than args['samples_min_n'] samples
    if len(samples) < 2:
        return None

    LOG.info('Comparing %s found in %s samples.' %
             (args['clade'], len(samples)))

    # load freqs
    x = load_numpy_file(args['input_file'])

    # rename samples to samples.dom
    if args['dominant_variants'] or args['dominant_variants_added']:
        dom_samples = [s + '.dom' for s in samples]
        d = consensus(x)

    # analyze only dominant variants
    if args['dominant_variants']:
        x = d
        samples = dom_samples

    # add dominant variants separately as .dom samples
    elif args['dominant_variants_added']:
        x = np.append(x, d, axis=0)
        samples += dom_samples

    shared, overlap = pairwise_counts(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = np.nan_to_num(shared / overlap)

    _write_matrix('%s/%s.closest.txt' % (args['output_dir'], args['clade']), samples, shared)
    _write_matrix('%s/%s.overlap.txt' % (args['output_dir'], args['clade']), samples, overlap)
    _write_matrix('%s/%s.fraction.txt' % (args['output_dir'], args['clade']), samples, fraction)

    # output dominant variants as msa
    if 'dominant_variants_msa' in args and args['dominant_variants_msa']:

        if args['dominant_variants'] or args['dominant_variants_added']:
            samples = dom_samples
        else:
            d = consensus(x)

        seqs_msa = _dominant_msa(d, samples)

        # write alignment fasta
        msa_filename = os.path.join(args['output_dir'], args['clade'] + '.msa.fa')
        with open(msa_filename, 'w') as out:
            AlignIO.write(seqs_msa, out, 'fasta')
