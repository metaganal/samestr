
import os
import gzip
from os.path import basename, exists
import logging
import numpy as np
import pandas as pd
from scipy import stats

from samestr.utils.utilities import load_numpy_file
from samestr.utils.file_mapping import clade_path
from samestr.filter.filter_freqs import read_marker_positions

LOG = logging.getLogger(__name__)


def coverage(x):
    # calculate coverage for each sample
    # returns MxN numpy array
    return x.sum(axis=2)


def _site_summaries(x, chunk=None):  # chunk unused (chunking disabled)
    """
    One pass over x (n_samples, n_sites, 4), returning per-site coverage
    (sum), dominant-allele coverage (max) and number of observed alleles.
    Chunking over sites is disabled (memory assumed ample, <= 512 GB).
    """
    # chunked variant (bounds the (n, L, 4) bool temporary); re-enable if needed:
    # n, n_sites, _ = x.shape
    # cov = np.empty((n, n_sites), dtype=np.result_type(x.dtype, np.float64))
    # dom = np.empty_like(cov)
    # n_alleles = np.empty((n, n_sites), dtype=np.int8)
    # for lo in range(0, n_sites, chunk):
    #     c = x[:, lo:lo + chunk]
    #     cov[:, lo:lo + chunk] = c.sum(axis=2)
    #     dom[:, lo:lo + chunk] = c.max(axis=2)
    #     n_alleles[:, lo:lo + chunk] = (c > 0).sum(axis=2)
    cov = x.sum(axis=2, dtype=np.result_type(x.dtype, np.float64))
    dom = x.max(axis=2).astype(cov.dtype, copy=False)
    n_alleles = (x > 0).sum(axis=2, dtype=np.int8)
    return cov, dom, n_alleles


def _masked_row_median(values, mask):
    """
    Row-wise median of values[mask], NaN for rows without any selected entry.
    Same result as np.nanmedian on a copy with unselected entries set to NaN:
    one in-place row sort (NaN sorts last) plus a gather of the middle
    element(s), instead of nanmedian's per-row python loop.
    """
    a = np.where(mask, values, np.nan)
    a.sort(axis=1)
    counts = mask.sum(axis=1)
    rows = np.nonzero(counts)[0]
    c = counts[rows]
    median = np.full(values.shape[0], np.nan)
    median[rows] = (a[rows, (c - 1) // 2] + a[rows, c // 2]) / 2
    return median


def _row_sum_mean(values, mask, counts):
    """Row-wise mean of values over mask, NaN for empty rows. Same summation
    as np.nanmean (zero-fill, then row sum), so results are bit-identical."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(mask, values, 0).sum(axis=1) / counts


# # Calculate nucleotide diversity from allele coverage
# # Scripts from https://github.com/JosephLalli/PiMaker/tree/master/pimaker/calcpi.py
# def calculate_pi_math(read_cts):
#     """
#     Calculates 'pi math' from read counts.

#     Per `Nelson and Hughes (2015)`_, pi is the sum of the product of the read
#     counts of all possible nucleotide combinations (AC, AG, AT, CG, CT, GT).
#     This function performs this math for all sites in the read count array.

#     Args:
#         read_cts:
#             A (number of samples x n x 4) array of read counts. n is an
#             arbitrary number of nucleotide sites.
#     Returns:
#         A (number of samples x n x 7) array containing the total number of
#         reads at each site for each sample in the third axis' 0 index,
#         and then the (AC, AG, AT, CG, CT, GT) products of the number of reads
#         for each site and each sample at nucleotides at indices 1-6.
#     """
#     pi_math = np.zeros(shape=(read_cts.shape[0:2] + (7,)), dtype=np.int32)
#     pi_math[:, :, 0] = np.sum(read_cts, axis=2)
#     pi_math[:, :, 1] = read_cts[:, :, 0] * read_cts[:, :, 1]
#     pi_math[:, :, 2] = read_cts[:, :, 0] * read_cts[:, :, 2]
#     pi_math[:, :, 3] = read_cts[:, :, 0] * read_cts[:, :, 3]
#     pi_math[:, :, 4] = read_cts[:, :, 1] * read_cts[:, :, 2]
#     pi_math[:, :, 5] = read_cts[:, :, 1] * read_cts[:, :, 3]
#     pi_math[:, :, 6] = read_cts[:, :, 2] * read_cts[:, :, 3]
#     return pi_math


# def per_site_pi(pi_math):
#     """
#     Calculates pi at each site of each sample.

#     Uses the formula in `Nelson and Hughes (2015)`_.
#     pi = sum(pi_math)/((depth**2 - depth)/2) for each site and each sample.

#     Args:
#         pi_math: A (number of samples x n x 7) array of pi math. See
#         :func:`calculate_pi_math` for a description of this array.
#     Returns:
#         A (number of samples x n) array of pi values.
#     """
#     with np.errstate(divide='ignore', invalid='ignore'):
#         result = np.sum(pi_math[:, :, 1:], axis=2) / ((pi_math[:, :, 0]**2 - pi_math[:, :, 0]) / 2)
#     return np.nan_to_num(result)


# def avg_pi_per_sample(site_pi, length=None):
#     """
#     Calculates average pi per sample from an array of pi values.

#     Args:
#         site_pi:
#             A (number of samples x n) array of pi values.
#         length:
#             Optional, default None. The length of the sequence(s) for which
#             we are calculating pi. Can be array of lengths, or scalar value.
#             If not specified, avg_pi_per_sample will use the length of the
#             second axis (the nucleotide axis) as the length of the sequence.
#     Returns:
#         A (number of samples) shaped array of average pi values per sample.
#     """
#     if length is None:
#         length = site_pi.shape[1]
#     return np.nansum(site_pi, axis=1) / length


# --------------------------------------------------------------------------
# Population-genetic diversity statistics
#
# All functions take allele counts x of shape (n_samples, n_sites, 4).
#
# Two flavours are provided:
#   * within-sample (read-based): each sample is treated as a pool of reads,
#     i.e. the "sample size" at a site is its read depth. This measures the
#     intra-sample (strain-mixture) diversity of a clade in a metagenome.
#   * across-sample (population): each sample contributes its consensus
#     (dominant) allele, i.e. one haplotype per sample, as in a classic
#     alignment of n sequences. Missing data (uncovered sites) is handled per
#     site, unless stated otherwise.
#
# Diversity values are returned per site (normalised by the number of sites
# that could be evaluated), so they are comparable between clades/samples
# with different breadth of coverage.
# --------------------------------------------------------------------------


def _filter_counts(x, min_count=1):
    """Return allele counts as float64 with counts < min_count set to 0
    (e.g. to suppress singleton sequencing errors)."""
    x = np.asarray(x, dtype=np.float64)
    if min_count > 1:
        x = np.where(x >= min_count, x, 0.)
    return x


def harmonic_a1(n):
    """
    Watterson's a1 = sum_{i=1}^{n-1} 1/i, vectorised over integer array n.
    Returns 0 for n < 2.
    """
    n = np.asarray(n, dtype=np.int64)
    n_max = int(n.max()) if n.size else 0
    table = np.zeros(max(n_max, 2) + 1)
    if n_max >= 2:
        table[2:] = np.cumsum(1. / np.arange(1, n_max))
    return table[np.clip(n, 0, None)]


def harmonic_a2(n):
    """Tajima's a2 = sum_{i=1}^{n-1} 1/i^2, vectorised over integer array n."""
    n = np.asarray(n, dtype=np.int64)
    n_max = int(n.max()) if n.size else 0
    table = np.zeros(max(n_max, 2) + 1)
    if n_max >= 2:
        table[2:] = np.cumsum(1. / np.arange(1, n_max)**2)
    return table[np.clip(n, 0, None)]


def _site_pi_from_counts(counts, depth):
    """
    Unbiased per-site heterozygosity / nucleotide diversity from allele
    counts along the last axis: the probability that two alleles drawn
    without replacement differ,
        pi = (D^2 - sum_a n_a^2) / (D * (D - 1)),
    which equals sum_{a<b} n_a n_b / (D (D - 1) / 2) (Nelson & Hughes 2015).
    NaN where depth < 2.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        pi = (depth**2 - (counts**2).sum(axis=-1)) / (depth * (depth - 1))
    pi[depth < 2] = np.nan
    return pi


# --- within-sample (read-based) statistics --------------------------------

def per_site_pi(x, min_count=1):
    """
    Within-sample nucleotide diversity at each site.

    Args:
        x: (n_samples, n_sites, 4) allele counts.
        min_count: allele counts below this value are ignored.
    Returns:
        (n_samples, n_sites) float array, NaN where depth < 2.
    """
    x = _filter_counts(x, min_count)
    return _site_pi_from_counts(x, x.sum(axis=2))


def nucleotide_diversity(x, min_cov=2, min_count=1, length=None):
    """
    Average within-sample nucleotide diversity (pi) per sample.

    Args:
        x: (n_samples, n_sites, 4) allele counts.
        min_cov: only sites with at least this depth are evaluated (>= 2).
        min_count: allele counts below this value are ignored.
        length: optional scalar or (n_samples,) denominator. By default the
            number of evaluated sites per sample is used.
    Returns:
        (n_samples,) array of mean pi per site, NaN if no site was evaluated.
    """
    x = _filter_counts(x, min_count)
    depth = x.sum(axis=2)
    site_pi = _site_pi_from_counts(x, depth)
    evaluated = depth >= max(min_cov, 2)
    if length is None:
        length = evaluated.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(evaluated, site_pi, 0.).sum(axis=1) / length


def watterson_theta(x, min_cov=2, min_count=1, length=None):
    """
    Within-sample Watterson's theta per sample, per site.

    Each site is treated as a sample of n = depth reads; a site is
    segregating if more than one allele is observed (after min_count
    filtering). Because depth varies along the genome, the estimator sums
    each segregating site weighted by its own 1/a1(n):
        theta_W = sum_{segregating i} 1 / a1(depth_i) / L
    Note: theta_W is very sensitive to sequencing errors; consider
    min_count > 1 for read data.

    Args:
        x: (n_samples, n_sites, 4) allele counts.
        min_cov: only sites with at least this depth are evaluated (>= 2).
        min_count: allele counts below this value are ignored.
        length: optional scalar or (n_samples,) denominator. By default the
            number of evaluated sites per sample is used.
    Returns:
        (n_samples,) array, NaN if no site was evaluated.
    """
    x = _filter_counts(x, min_count)
    depth = x.sum(axis=2)
    evaluated = depth >= max(min_cov, 2)
    segregating = evaluated & ((x > 0).sum(axis=2) > 1)
    if length is None:
        length = evaluated.sum(axis=1)
    rows, cols = np.nonzero(segregating)
    weighted = np.bincount(rows,
                           weights=1. / harmonic_a1(depth[rows, cols]),
                           minlength=x.shape[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        return weighted / length


# --- across-sample (population) statistics ---------------------------------

def consensus_alleles(x):
    """
    Dominant allele index (0-3) per sample and site, -1 where uncovered.
    Ties are resolved towards the lower allele index.

    Returns:
        (n_samples, n_sites) int8 array.
    """
    alleles = np.argmax(x, axis=2).astype(np.int8)
    alleles[np.asarray(x).sum(axis=2) == 0] = -1
    return alleles


def population_allele_counts(x):
    """
    Number of samples carrying each consensus allele per site.

    Returns:
        (n_sites, 4) int array; row sums give the per-site sample size n.
    """
    alleles = consensus_alleles(x)
    return np.stack([(alleles == a).sum(axis=0) for a in range(4)], axis=1)


def population_stats(x, min_samples=2, complete_deletion=False):
    """
    Classical population-genetic statistics across samples, treating each
    sample's consensus allele as one haplotype.

    Args:
        x: (n_samples, n_sites, 4) allele counts.
        min_samples: only sites covered in at least this many samples are
            evaluated (>= 2).
        complete_deletion: if True, only sites covered in all samples are
            used (fixed n), as required for the textbook Tajima's D.
    Returns:
        dict with
          n_sites_evaluated: number of evaluated sites (L)
          n_segregating:     number of segregating sites (S)
          pi:                mean pairwise differences per site
          theta_w:           Watterson's theta per site,
                             sum_{segregating i} 1/a1(n_i) / L
          tajimas_d:         Tajima's D (NaN if S == 0). With missing data
                             (complete_deletion=False) n is set to the
                             rounded harmonic mean of per-site sample sizes
                             at evaluated sites, an approximation.
    """
    n_samples = x.shape[0]
    ac = population_allele_counts(x).astype(np.float64)
    n = ac.sum(axis=1)
    if complete_deletion:
        evaluated = n == n_samples
    else:
        evaluated = n >= max(min_samples, 2)
    ac, n = ac[evaluated], n[evaluated]
    L = int(evaluated.sum())

    site_pi = _site_pi_from_counts(ac, n)
    segregating = (ac > 0).sum(axis=1) > 1
    S = int(segregating.sum())
    k = float(site_pi.sum())  # average number of pairwise differences
    theta_w_sum = float((1. / harmonic_a1(n[segregating])).sum())

    if L == 0:
        return dict(n_sites_evaluated=0, n_segregating=0, pi=np.nan,
                    theta_w=np.nan, tajimas_d=np.nan)

    n_eff = int(np.rint(L / (1. / n).sum()))  # == n_samples if complete
    return dict(n_sites_evaluated=L,
                n_segregating=S,
                pi=k / L,
                theta_w=theta_w_sum / L,
                tajimas_d=tajimas_d(k, S, n_eff))


def tajimas_d(k, S, n):
    """
    Tajima's D (Tajima 1989).

    Args:
        k: average number of pairwise differences (sum of per-site pi).
        S: number of segregating sites.
        n: number of sequences (haplotypes).
    Returns:
        float, NaN if S == 0 or n < 3 (degenerate).
    """
    if S == 0 or n < 3:
        return np.nan
    a1 = float(harmonic_a1(n))
    a2 = float(harmonic_a2(n))
    b1 = (n + 1) / (3. * (n - 1))
    b2 = 2. * (n**2 + n + 3) / (9. * n * (n - 1))
    c1 = b1 - 1. / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1**2
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)
    return (k - S / a1) / np.sqrt(e1 * S + e2 * S * (S - 1))


# --------------------------------------------------------------------------
# Within-host polymorphism rate (Garud et al. 2019, PLoS Biol;
# Madi et al. 2023, eLife 12:e78530)
#
#   "The polymorphism rate of a species in a sample was computed as the
#    proportion of synonymous sites in core genes with intermediate allele
#    frequencies (0.2 <= f <= 0.8)."
#
# with sites excluded if D < 0.3 * Dbar or D > 3 * Dbar (Dbar: median depth
# at protein coding sites with nonzero coverage) and samples excluded if
# Dbar < 5. Synonymous = fourfold degenerate, nonsynonymous = onefold
# degenerate sites. Clade markers stand in for the paper's core genes
# (species-specific, single-copy coding genes, i.e. also free of genes shared
# between species which the paper blacklists).
# --------------------------------------------------------------------------

_BASES = 'ACGT'
_AMINO = ('KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF')
_CODON_AA = {a + b + c: _AMINO[16 * i + 4 * j + k]
             for i, a in enumerate(_BASES) for j, b in enumerate(_BASES)
             for k, c in enumerate(_BASES)}  # standard / bacterial (table 11)


def _codon_degeneracy():
    """(64, 3) int8 table: for codon index 16*i + 4*j + k (ACGT order) and
    codon position, the number of nucleotides (incl. the reference) that
    encode the same amino acid (4 = fourfold degenerate/synonymous, 1 =
    onefold degenerate/nonsynonymous). Stop codons get 0."""
    table = np.zeros((64, 3), dtype=np.int8)
    for codon, aa in _CODON_AA.items():
        if aa == '*':
            continue
        idx = 16 * _BASES.index(codon[0]) + 4 * _BASES.index(codon[1]) + \
            _BASES.index(codon[2])
        for pos in range(3):
            table[idx, pos] = sum(
                _CODON_AA[codon[:pos] + b + codon[pos + 1:]] == aa
                for b in _BASES)
    return table


# the six unordered allele pairs (indices into ACGT), in this order
_ALLELE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _codon_synonymous_pairs():
    """(64, 3) uint8 table of 6-bit masks: bit k is set if the two alleles
    of _ALLELE_PAIRS[k], placed at that codon position (the other two
    positions as in the reference codon), encode the same amino acid, i.e.
    a polymorphism between them is synonymous. Stop codons get 0."""
    table = np.zeros((64, 3), dtype=np.uint8)
    for codon, aa in _CODON_AA.items():
        if aa == '*':
            continue
        idx = 16 * _BASES.index(codon[0]) + 4 * _BASES.index(codon[1]) + \
            _BASES.index(codon[2])
        for pos in range(3):
            aas = [_CODON_AA[codon[:pos] + b + codon[pos + 1:]]
                   for b in _BASES]
            table[idx, pos] = sum(1 << k for k, (a, b) in
                                  enumerate(_ALLELE_PAIRS) if aas[a] == aas[b])
    return table


_DEGENERACY = _codon_degeneracy()
_SYN_PAIRS = _codon_synonymous_pairs()


def site_annotation(seq):
    """
    Per-position annotation of a protein coding sequence, read in frame 0 on
    the given strand:
      degeneracy: (L,) int8, number of nucleotides (1-4) encoding the
                  reference amino acid; 0 at codons with non-ACGT bases and
                  at stop codons.
      syn_pairs:  (L,) uint8 6-bit mask, bit k set if a polymorphism between
                  the alleles of _ALLELE_PAIRS[k] is synonymous (assuming
                  the rest of the codon is the reference).
    Returns None if seq does not look like an in-frame CDS (length not a
    multiple of 3 or an internal stop codon), so it can be skipped.
    """
    seq = str(seq).upper()
    if len(seq) < 3 or len(seq) % 3:
        return None
    lut = np.full(256, -1, dtype=np.int16)
    for i, b in enumerate(_BASES):
        lut[ord(b)] = i
    nt = lut[np.frombuffer(seq.encode(), dtype=np.uint8)].reshape(-1, 3)
    valid = (nt >= 0).all(axis=1)
    idx = np.where(valid, 16 * nt[:, 0] + 4 * nt[:, 1] + nt[:, 2], 0)
    deg = np.where(valid[:, None], _DEGENERACY[idx], 0).astype(np.int8)
    stop = valid & (deg == 0).all(axis=1)
    if stop[:-1].any():
        return None  # internal stop: not in frame 0 / not a CDS
    syn = np.where(valid[:, None], _SYN_PAIRS[idx], 0).astype(np.uint8)
    return deg.ravel(), syn.ravel()


def site_degeneracy(seq):
    """Degeneracy (1-4) per position of an in-frame CDS, see
    site_annotation. None if seq is not an in-frame CDS."""
    ann = site_annotation(seq)
    return None if ann is None else ann[0]


def _read_fasta_gz(path):
    """Minimal gzipped FASTA reader -> dict name -> sequence."""
    seqs, name, chunks = {}, None, []
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        seqs[name] = ''.join(chunks)
    return seqs


def clade_site_annotation(marker_dir, clade, n_sites, remaining_pos=None):
    """
    Degeneracy and synonymous allele pairs per alignment column of a clade
    (see site_annotation), from the SameStr database
    (`<clade>.markers.fa.gz` + `<clade>.positions.txt.gz`).

    Args:
        n_sites: number of columns of the alignment.
        remaining_pos: original positions kept in the alignment (written
            to `<clade>.pos.txt` by `samestr filter --delete-pos`).
    Returns:
        ((n_sites,) int8 degeneracy, 0 = unknown/not coding in frame,
         (n_sites,) uint8 synonymous-pair masks,
         number of markers usable as CDS, total number of markers),
        or None if the database files are missing or do not match.
    """
    base = marker_dir + '/' + clade_path(clade, filebase=True)
    pos_file, fa_file = base + '.positions.txt.gz', base + '.markers.fa.gz'
    if not (exists(pos_file) and exists(fa_file)):
        LOG.warning('%s: marker sequences/positions not found at %s.' %
                    (clade, base))
        return None
    positions = read_marker_positions(pos_file)
    seqs = _read_fasta_gz(fa_file)
    n_orig = max([end for _, end, _ in positions.values()] + [0])
    degeneracy = np.zeros(n_orig, dtype=np.int8)
    syn_pairs = np.zeros(n_orig, dtype=np.uint8)
    n_cds = 0
    for marker, (start, end, length) in positions.items():
        seq = seqs.get(marker)
        if seq is None or len(seq) != length:
            continue
        ann = site_annotation(seq)
        if ann is not None:
            degeneracy[start:end], syn_pairs[start:end] = ann
            n_cds += 1
    if remaining_pos is None:
        if n_orig != n_sites:
            LOG.warning('%s: alignment has %s columns but markers span %s '
                        'positions.' % (clade, n_sites, n_orig))
            return None
        return degeneracy, syn_pairs, n_cds, len(positions)
    remaining_pos = np.asarray(remaining_pos, dtype=np.int64)
    if remaining_pos.shape[0] != n_sites or \
            (n_sites and remaining_pos.max() >= n_orig):
        LOG.warning('%s: kept positions do not match the alignment.' % clade)
        return None
    return (degeneracy[remaining_pos], syn_pairs[remaining_pos], n_cds,
            len(positions))


def polymorphism_rate(x=None, cov=None, maf=None, median_depth=None,
                      site_mask=None, f_min=0.2, f_max=0.8,
                      min_median_depth=5, depth_low=0.3, depth_high=3.):
    """
    Within-host polymorphism rate per sample (Garud et al. 2019; Madi et al.
    2023): the fraction of evaluable sites with an intermediate allele
    frequency f_min <= f <= f_max.

    A site is evaluable in a sample if it is in site_mask (e.g. fourfold
    degenerate sites of core genes) and its depth D satisfies
        depth_low * Dbar <= D <= depth_high * Dbar,
    where Dbar is the sample's median depth over covered sites. Samples with
    Dbar < min_median_depth are excluded (NaN).

    f is the frequency of the non-dominant alleles, (D - D_dom) / D, i.e.
    the minor allele frequency at bi-allelic sites (MIDAS' alt-allele
    frequency in [0.2, 0.8] is symmetric, so equivalent there).

    Pre-calculated statistics are used when supplied, so x is not needed:
        cov:          (n_samples, n_sites) depth, x.sum(axis=2)
        maf:          (n_samples, n_sites) (cov - dom) / cov
        median_depth: (n_samples,) median depth over covered sites
    Anything missing is derived from x (n_samples, n_sites, 4).

    Args:
        site_mask: optional (n_sites,) bool array of sites to consider.
    Returns:
        dict of (n_samples,) arrays: n_polymorphic, n_sites_eval,
        polymorphism_rate (NaN for excluded samples / no evaluable sites).
    """
    if cov is None or maf is None:
        if x is None:
            raise ValueError('Either x or both cov and maf are required.')
        cov_x, dom_x, _ = _site_summaries(x)
        if cov is None:
            cov = cov_x
        if maf is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                maf = (cov_x - dom_x) / cov_x
    if median_depth is None:
        median_depth = _masked_row_median(cov, cov > 0)
    median_depth = np.asarray(median_depth, dtype=np.float64)

    with np.errstate(invalid='ignore'):
        lo = (depth_low * median_depth)[:, None]
        hi = (depth_high * median_depth)[:, None]
        evaluable = (cov > 0) & (cov >= lo) & (cov <= hi)
        if site_mask is not None:
            evaluable &= np.asarray(site_mask, dtype=bool)[None, :]
        polymorphic = evaluable & (maf >= f_min) & (maf <= f_max)

    n_eval = evaluable.sum(axis=1)
    n_poly = polymorphic.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.where(n_eval > 0, n_poly / n_eval, np.nan)
    excluded = ~(median_depth >= min_median_depth)  # also NaN medians
    rate[excluded] = np.nan
    return dict(n_polymorphic=n_poly, n_sites_eval=n_eval,
                polymorphism_rate=rate)


def nucleotide_diversity_ratio(x, degeneracy, syn_pairs, cov=None,
                               n_alleles=None, min_cov=2,
                               min_allele_count=4, min_allele_freq=0.01):
    """
    Within-sample nucleotide diversity ratios of Schloissnig et al. 2013
    (Nature 493:45): pi(N)/pi(S), the nucleotide diversity analogue of
    pN/pS, and pi(non-degenerate sites)/pi(fourfold degenerate sites),
    which depends less on the mutation spectrum (transition/transversion
    ratio).

    Per site i, pi_i is the probability that two reads drawn without
    replacement differ, sum_{a<b} 2 n_a n_b / (D (D - 1)). Each allele pair
    (a, b) is classified as synonymous or non-synonymous from the reference
    codon context, so pi_i = pi_S,i + pi_N,i. Diversity is normalised by
    the number of sites of each class (Nei & Gojobori 1986): a position of
    degeneracy d contributes (d - 1) / 3 synonymous and (4 - d) / 3
    non-synonymous sites, so
        pi(S) = sum_i pi_S,i / sum_i (d_i - 1) / 3,
        pi(N) = sum_i pi_N,i / sum_i (4 - d_i) / 3,
        pi(nondeg) = mean pi_i over d_i == 1,  pi(4fold) = over d_i == 4,
    over coding sites (degeneracy > 0) with depth >= min_cov.

    As for Schloissnig's SNP calls, a non-dominant allele only counts if it
    is supported by >= min_allele_count reads and has a frequency
    >= min_allele_freq; other non-dominant reads are discarded.

    Pre-calculated statistics are used when supplied:
        cov:       (n_samples, n_sites) depth, x.sum(axis=2)
        n_alleles: (n_samples, n_sites) number of observed alleles;
                   diversity is only computed at sites with > 1 allele
                   (pi == 0 elsewhere), so x is only read there.

    Args:
        x: (n_samples, n_sites, 4) allele counts.
        degeneracy, syn_pairs: (n_sites,) arrays from clade_site_annotation.
    Returns:
        dict of (n_samples,) arrays: pi_syn, pi_nonsyn, pi_n_pi_s,
        pi_nondeg, pi_4fold, pi_nondeg_pi_4fold (ratios NaN if the
        denominator diversity is 0 or nothing was evaluable).
    """
    if cov is None or n_alleles is None:
        cov_x, _, n_alleles_x = _site_summaries(x)
        cov = cov_x if cov is None else cov
        n_alleles = n_alleles_x if n_alleles is None else n_alleles
    n_samples = cov.shape[0]
    degeneracy = np.asarray(degeneracy)
    coding = degeneracy > 0

    with np.errstate(invalid='ignore'):
        evaluable = coding[None, :] & (cov >= max(min_cov, 2))

    # number of sites per class and sample (Nei-Gojobori site counts)
    # (masked row sums: no (n_samples, n_sites) float temporaries)
    def row_sum(weights):
        w = np.broadcast_to(weights, evaluable.shape)
        return np.sum(w, axis=1, where=evaluable)

    deg_f = degeneracy.astype(np.float64)
    syn_sites = row_sum(np.where(coding, (deg_f - 1) / 3, 0.))
    nonsyn_sites = row_sum(np.where(coding, (4 - deg_f) / 3, 0.))
    n_nondeg = row_sum((degeneracy == 1).astype(np.float64))
    n_4fold = row_sum((degeneracy == 4).astype(np.float64))

    # diversity at evaluable multi-allelic sites only
    rows, cols = np.nonzero(evaluable & (n_alleles > 1))
    counts = np.asarray(x[rows, cols], dtype=np.float64)
    depth = counts.sum(axis=1, keepdims=True)
    dominant = counts.argmax(axis=1)
    alt = np.ones_like(counts, dtype=bool)
    alt[np.arange(len(dominant)), dominant] = False
    with np.errstate(divide='ignore', invalid='ignore'):
        weak = alt & ((counts < min_allele_count) |
                      (counts / depth < min_allele_freq))
    counts[weak] = 0.
    depth = counts.sum(axis=1)

    pair_pi = np.stack([counts[:, a] * counts[:, b]
                        for a, b in _ALLELE_PAIRS], axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        pair_pi *= np.where(depth > 1, 2. / (depth * (depth - 1)), 0.)[:, None]
    is_syn = ((syn_pairs[cols][:, None] >> np.arange(6, dtype=np.uint8)) & 1
              ).astype(bool)
    pi_s_site = np.where(is_syn, pair_pi, 0.).sum(axis=1)
    pi_n_site = np.where(is_syn, 0., pair_pi).sum(axis=1)
    pi_site = pi_s_site + pi_n_site
    del counts, pair_pi, is_syn

    def per_sample(weights, keep=None):
        if keep is not None:
            return np.bincount(rows[keep], weights=weights[keep],
                               minlength=n_samples)
        return np.bincount(rows, weights=weights, minlength=n_samples)

    site_deg = degeneracy[cols]
    with np.errstate(divide='ignore', invalid='ignore'):
        pi_syn = per_sample(pi_s_site) / syn_sites
        pi_nonsyn = per_sample(pi_n_site) / nonsyn_sites
        pi_nondeg = per_sample(pi_site, site_deg == 1) / n_nondeg
        pi_4fold = per_sample(pi_site, site_deg == 4) / n_4fold
        pi_n_pi_s = np.where(pi_syn > 0, pi_nonsyn / pi_syn, np.nan)
        pi_nondeg_pi_4fold = np.where(pi_4fold > 0, pi_nondeg / pi_4fold,
                                      np.nan)
    return dict(pi_syn=pi_syn, pi_nonsyn=pi_nonsyn, pi_n_pi_s=pi_n_pi_s,
                pi_nondeg=pi_nondeg, pi_4fold=pi_4fold,
                pi_nondeg_pi_4fold=pi_nondeg_pi_4fold)


#: columns of the per-sample statistics table, in order
#: (sample_stats / `<clade>.aln_stats.txt`)
STAT_COLUMNS = (
    'Sample', 'mean_cov', 'median_cov', 'n_sites', 'n_gaps', 'n_covered',
    'n_mono', 'n_duo', 'n_tri', 'n_quat', 'n_poly', 'f_covered', 'f_mono',
    'f_duo', 'f_tri', 'f_quat', 'f_poly', 'mean_dom_cov', 'mean_f_dom_cov',
    'median_f_dom_cov', 'mean_dom_cov_polysites',
    'median_dom_cov_polysites', 'mean_f_dom_cov_polysites',
    'median_f_dom_cov_polysites', 'mean_cov_polysites', 'n_binom',
    'f_binom', 'n_binom_segata', 'f_binom_segata',
    'average_nucleotide_diversity', 'watterson_theta', 'mean_maf',
    'median_maf', 'mean_maf_polysites', 'median_maf_polysites',
    'n_poly_intermediate', 'n_sites_depth_ok', 'polymorphism_rate',
    'n_poly_intermediate_syn', 'n_sites_depth_ok_syn',
    'polymorphism_rate_syn',
    'n_poly_intermediate_nonsyn', 'n_sites_depth_ok_nonsyn',
    'polymorphism_rate_nonsyn', 'pi_syn', 'pi_nonsyn', 'pi_n_pi_s',
    'pi_nondeg', 'pi_4fold', 'pi_nondeg_pi_4fold')


def sample_stats(x, samples, site_deg=None, syn_pairs=None,
                 dominant_variants=False):
    """
    Per-sample alignment statistics of one clade (the table aln2stats
    writes), from allele counts in memory.

    Args:
        x: (n_samples, n_sites, 4) allele counts.
        samples: (n_samples,) sample names.
        site_deg, syn_pairs: optional (n_sites,) site annotation from
            clade_site_annotation; without it the synonymous/nonsynonymous
            polymorphism rates and the diversity ratios are NaN.
        dominant_variants: analyze only dominant variants.
    Returns:
        pandas DataFrame with one row per sample and columns STAT_COLUMNS.
    """
    samples = np.asarray(samples)
    if samples.shape[0] != x.shape[0]:
        raise ValueError('%s sample names for %s samples.' %
                         (samples.shape[0], x.shape[0]))

    # Every statistic below depends on x only through three per-site values:
    # coverage (sum over alleles), dominant-allele coverage (max over alleles)
    # and the number of observed alleles. consensus() keeps exactly one
    # allele with the maximal count, so for the dominant variants
    # sum == max and n_alleles == (max > 0). Hence consensus() itself, a full
    # copy of x plus a python loop over tied sites, is not needed here.
    cov, dom, n_alleles = _site_summaries(x)

    # stats: within-sample nucleotide diversity and Watterson's theta
    # (per site, over sites with depth >= 2). Needs the full allele counts.
    # With dominant variants only, every site is monomorphic and both are 0
    # (NaN without evaluable sites).
    if dominant_variants:
        n_evaluable = (dom >= 2).sum(axis=1)
        average_nucleotide_diversity = np.where(n_evaluable > 0, 0., np.nan)
        theta_w = average_nucleotide_diversity.copy()
    else:
        average_nucleotide_diversity = nucleotide_diversity(x)
        theta_w = watterson_theta(x, min_count=2)

    # stats: nucleotide diversity ratios pi(N)/pi(S) and
    # pi(non-degenerate)/pi(fourfold) (Schloissnig et al. 2013). Reads x
    # only at multi-allelic coding sites; with dominant variants only there
    # are none, so all diversities are 0 and the ratios NaN.
    pi_keys = ('pi_syn', 'pi_nonsyn', 'pi_n_pi_s', 'pi_nondeg', 'pi_4fold',
               'pi_nondeg_pi_4fold')
    if site_deg is not None:
        pi_ratios = nucleotide_diversity_ratio(
            x, site_deg, syn_pairs, cov=cov,
            n_alleles=(dom > 0).astype(np.int8) if dominant_variants
            else n_alleles)
    else:
        pi_ratios = {k: np.full(x.shape[0], np.nan) for k in pi_keys}

    if dominant_variants:
        # analyze only dominant variants
        cov = dom.copy()
        n_alleles = (dom > 0).astype(np.int8)

    n_samples, n_sites_total = cov.shape
    covered = cov > 0
    poly = n_alleles > 1

    with np.errstate(divide='ignore', invalid='ignore'):

        # stats: horizontal coverage
        n_sites = np.repeat(n_sites_total, n_samples)
        n_covered = covered.sum(axis=1)
        n_gaps = n_sites - n_covered

        # stats: vertical coverage
        mean_cov = _row_sum_mean(cov, covered, n_covered)
        median_cov = _masked_row_median(cov, covered)
        mean_cov[np.isnan(mean_cov)] = 0
        median_cov[np.isnan(median_cov)] = 0

        # stats: n of variant sites, monomorphic, .., polymorphic
        n_mono = (n_alleles == 1).sum(axis=1)
        n_duo = (n_alleles == 2).sum(axis=1)
        n_tri = (n_alleles == 3).sum(axis=1)
        n_quat = (n_alleles == 4).sum(axis=1)
        n_poly = poly.sum(axis=1)

        # polymorphic sites as per binomial cum. dist. func.
        # At non-polymorphic sites dom == cov, so cdf(cov; cov, p) == 1 and
        # never passes the test: evaluate the (costly) cdf at polymorphic
        # sites only.
        illumina_error_rate = 0.3 / 100  # Q25+
        segata_error_rate = 1 / 100  # Q20
        p_value = 0.05
        rows, cols = np.nonzero(poly)
        k, nn = dom[rows, cols], cov[rows, cols]
        n_binom = np.bincount(
            rows[stats.binom.cdf(k, nn, 1.0 - illumina_error_rate) < p_value],
            minlength=n_samples)
        f_binom = n_binom / n_covered
        n_binom_segata = np.bincount(
            rows[stats.binom.cdf(k, nn, 1.0 - segata_error_rate) < p_value],
            minlength=n_samples)
        f_binom_segata = n_binom_segata / n_covered
        del rows, cols, k, nn

        # stats: fraction of covered sites,
        # stats: fraction of covered sites with variant, monomorphic, .., polymorphic
        f_covered = n_covered / n_sites
        f_mono = n_mono / n_covered
        f_duo = n_duo / n_covered
        f_tri = n_tri / n_covered
        f_quat = n_quat / n_covered
        f_poly = n_poly / n_covered


        # stats: vertical coverage of dominant variants
        # at all sites (dom > 0 exactly where cov > 0)
        f_dom = dom / cov
        mean_dom_cov = _row_sum_mean(dom, covered, n_covered)
        mean_f_dom_cov = _row_sum_mean(f_dom, covered, n_covered)
        median_f_dom_cov = _masked_row_median(f_dom, covered)

        # at polymorphic sites
        mean_dom_cov_polysites = _row_sum_mean(dom, poly, n_poly)
        median_dom_cov_polysites = _masked_row_median(dom, poly)
        mean_f_dom_cov_polysites = _row_sum_mean(f_dom, poly, n_poly)
        median_f_dom_cov_polysites = _masked_row_median(f_dom, poly)

        # mean coverage at polymorphic sites
        mean_cov_polysites = _row_sum_mean(cov, poly, n_poly)

        # stats: minor allele frequency, i.e. the fraction of coverage not
        # supporting the dominant variant, (cov - dom) / cov
        # at all covered sites
        maf = (cov - dom) / cov
        mean_maf = _row_sum_mean(maf, covered, n_covered)
        median_maf = _masked_row_median(maf, covered)

        # at polymorphic sites
        mean_maf_polysites = _row_sum_mean(maf, poly, n_poly)
        median_maf_polysites = _masked_row_median(maf, poly)

    # stats: within-host polymorphism rate (Garud et al. 2019; Madi et al.
    # 2023) from the precomputed depth, MAF and median depth: all marker
    # sites, synonymous (fourfold degenerate) and nonsynonymous (onefold
    # degenerate) sites. NaN if the degeneracy could not be determined.
    prate = polymorphism_rate(cov=cov, maf=maf, median_depth=median_cov)
    nan = np.full(n_samples, np.nan)
    prate_4d = prate_1d = dict(n_polymorphic=nan, n_sites_eval=nan,
                               polymorphism_rate=nan)
    if site_deg is not None:
        prate_4d = polymorphism_rate(cov=cov, maf=maf, median_depth=median_cov,
                                     site_mask=site_deg == 4)
        prate_1d = polymorphism_rate(cov=cov, maf=maf, median_depth=median_cov,
                                     site_mask=site_deg == 1)
    del maf

    values = [
        samples, mean_cov, median_cov, n_sites, n_gaps, n_covered,
        n_mono, n_duo, n_tri, n_quat, n_poly, f_covered, f_mono, f_duo, f_tri,
        f_quat, f_poly, mean_dom_cov, mean_f_dom_cov, median_f_dom_cov,
        mean_dom_cov_polysites, median_dom_cov_polysites,
        mean_f_dom_cov_polysites, median_f_dom_cov_polysites,
        mean_cov_polysites, n_binom, f_binom, n_binom_segata, f_binom_segata,
        average_nucleotide_diversity, theta_w, mean_maf, median_maf,
        mean_maf_polysites, median_maf_polysites,
        prate['n_polymorphic'], prate['n_sites_eval'],
        prate['polymorphism_rate'],
        prate_4d['n_polymorphic'], prate_4d['n_sites_eval'],
        prate_4d['polymorphism_rate'],
        prate_1d['n_polymorphic'], prate_1d['n_sites_eval'],
        prate_1d['polymorphism_rate'],
        *[pi_ratios[k] for k in pi_keys]
    ]
    # one column per statistic, keeping each statistic's numeric dtype
    return pd.DataFrame(dict(zip(STAT_COLUMNS, values)),
                        columns=list(STAT_COLUMNS))


def aln2stats(args):

    # if exists, skip
    output_name = os.path.join(args['output_dir'], basename(args['input_file']))
    if exists(output_name):
        LOG.info('Skipping %s. Output file exists.' % args['clade'])
        return True

    # load sample order
    with open(args['input_name'], 'r') as file:
        samples = file.read().strip().split('\n')

    LOG.info('Gathering stats for %s found in %s samples.' %
             (args['clade'], len(samples)))

    # load freqs
    x = load_numpy_file(args['input_file'])

    # site degeneracy from the marker sequences (for synonymous/fourfold and
    # nonsynonymous/onefold polymorphism rates). `samestr filter
    # --delete-pos` stores the kept original positions next to its output.
    kept_file = os.path.join(os.path.dirname(args['input_file']),
                             args['clade'] + '.pos.txt')
    remaining_pos = np.loadtxt(kept_file, dtype=np.int64, ndmin=1) \
        if exists(kept_file) else None
    site_deg = syn_pairs = None
    annotation = clade_site_annotation(args['marker_dir'], args['clade'],
                                       x.shape[1], remaining_pos)
    if annotation is not None:
        site_deg, syn_pairs, n_cds, n_markers = annotation
        LOG.debug('%s: %s of %s markers usable as in-frame CDS.' %
                  (args['clade'], n_cds, n_markers))

    df = sample_stats(x, samples, site_deg, syn_pairs,
                      dominant_variants=args['dominant_variants'])
    del x

    # write df to file
    ofn = '%s/%s.aln_stats.txt' % (args['output_dir'], args['clade'])
    df.to_csv(ofn, sep='\t', index_label=False, index=False)
