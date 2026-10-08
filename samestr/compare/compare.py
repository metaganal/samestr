
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
from samestr.stats.alignment_stats import _site_pi_from_counts


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


def pairwise_pi(x, min_cov=4, min_allele_count=4):
    """
    All-vs-all nucleotide diversity over the positions of x (n_samples, n_pos,
    4) that both samples of a pair cover with depth >= min_cov.

    A non-dominant allele supported by fewer than min_allele_count reads is
    discarded first (its reads leave the depth too), so sequencing errors do
    not count as diversity; the dominant allele is always kept (on a tie, the
    first in ACGT order). The defaults, 4 reads per site and 4 reads per
    minor allele, follow Wasney et al. 2026 (Nat Commun,
    doi:10.1038/s41467-026-70705-8), whose pi and Fst are the ones below, and
    match samestr stats' Schloissnig pi ratios (min_allele_count=4).
    min_allele_count=1 keeps every read.

    n_sites[i, j]    = number of such positions
    pi_within[i, j]  = sum over them of sample i's within-sample pi, the
                       probability that two of its reads differ,
                       (D^2 - sum_a n_a^2) / (D (D - 1)) (as samestr stats'
                       average_nucleotide_diversity; asymmetric: row i over
                       the positions it shares with column j)
    pi_between[i, j] = sum over them of the probability that a read of i and
                       a read of j differ, 1 - sum_a f_ia f_ja (no finite-depth
                       correction: the two reads are never the same read)

    Divided by n_sites these are per-site means over one site set, so
    pairwise Fst (Hudson et al. 1992) follows as
        1 - (pi_within[i, j] + pi_within[j, i]) / 2 / pi_between[i, j]
    (see `samestr fst`).

    Matrix products, as pairwise_counts: with M the (n, L) evaluable-site
    indicator, F the allele frequencies (zero where not evaluable, flattened
    to (n, 4L)) and P the per-site within pi (zero where not evaluable),
        n_sites = M @ M.T,  pi_between = n_sites - F @ F.T,  pi_within = P @ M.T.
    float64 throughout: Fst is 1 minus a ratio of two close sums, so the sums
    need the precision. Peak memory ~ x + ~48 bytes * n * n_pos.
    """
    n = x.shape[0]
    x = np.array(x, dtype=np.float64)                    # a copy: filtered below
    if min_allele_count > 1:
        minor = np.ones(x.shape, dtype=bool)
        np.put_along_axis(minor, x.argmax(axis=2)[:, :, None], False, axis=2)
        x[minor & (x < min_allele_count)] = 0.
        del minor
    depth = x.sum(axis=2)                                # (n, L)
    ok = depth >= max(int(min_cov), 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        pw = np.where(ok, _site_pi_from_counts(x, depth), 0.)
        f = np.where(ok[:, :, None], x / depth[:, :, None], 0.)
    del x, depth

    m = ok.astype(np.float64)
    n_sites = m @ m.T
    f = f.reshape(n, -1)
    pi_between = n_sites - f @ f.T
    del f
    pi_within = pw @ m.T
    # rounding can leave -1e-16 where two samples carry identical frequencies
    np.maximum(pi_between, 0., out=pi_between)
    return pi_within, pi_between, n_sites.astype(np.int64)


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

    # nucleotide diversity within and between samples, per shared site, for
    # pairwise Fst (`samestr fst`); NaN where a pair shares no such site
    pi_within, pi_between, n_sites = pairwise_pi(
        x, args.get('pi_min_cov', 4), args.get('pi_min_allele_count', 4))
    with np.errstate(divide='ignore', invalid='ignore'):
        pi_within /= n_sites
        pi_between /= n_sites
    _write_matrix('%s/%s.pi_within.txt' % (args['output_dir'], args['clade']), samples, pi_within)
    _write_matrix('%s/%s.pi_between.txt' % (args['output_dir'], args['clade']), samples, pi_between)
    _write_matrix('%s/%s.pi_sites.txt' % (args['output_dir'], args['clade']), samples, n_sites)
    del pi_within, pi_between, n_sites

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
