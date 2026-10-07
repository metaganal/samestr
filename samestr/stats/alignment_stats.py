
import os
from os.path import basename, exists
import logging
import numpy as np
import pandas as pd
from scipy import stats

from samestr.utils.utilities import load_numpy_file

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

    # Every statistic below depends on x only through three per-site values:
    # coverage (sum over alleles), dominant-allele coverage (max over alleles)
    # and the number of observed alleles. consensus() keeps exactly one
    # allele with the maximal count, so for the dominant variants
    # sum == max and n_alleles == (max > 0). Hence consensus() itself, a full
    # copy of x plus a python loop over tied sites, is not needed here.
    cov, dom, n_alleles = _site_summaries(x)

    # stats: within-sample nucleotide diversity and Watterson's theta
    # (per site, over sites with depth >= 2). Needs the full allele counts,
    # so computed before x is released. With dominant variants only, every
    # site is monomorphic and both are 0 (NaN without evaluable sites).
    if args['dominant_variants']:
        n_evaluable = (dom >= 2).sum(axis=1)
        average_nucleotide_diversity = np.where(n_evaluable > 0, 0., np.nan)
        theta_w = average_nucleotide_diversity.copy()
    else:
        average_nucleotide_diversity = nucleotide_diversity(x)
        theta_w = watterson_theta(x, min_count=2)
    del x

    if args['dominant_variants']:
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
        del maf

    # convert to pandas df
    df = pd.DataFrame(data=[
        np.array(samples), mean_cov, median_cov, n_sites, n_gaps, n_covered,
        n_mono, n_duo, n_tri, n_quat, n_poly, f_covered, f_mono, f_duo, f_tri,
        f_quat, f_poly, mean_dom_cov, mean_f_dom_cov, median_f_dom_cov,
        mean_dom_cov_polysites, median_dom_cov_polysites,
        mean_f_dom_cov_polysites, median_f_dom_cov_polysites,
        mean_cov_polysites, n_binom, f_binom, n_binom_segata, f_binom_segata,
        average_nucleotide_diversity, theta_w, mean_maf, median_maf,
        mean_maf_polysites, median_maf_polysites
    ])
    df = df.T
    df.columns = [
        'Sample', 'mean_cov', 'median_cov', 'n_sites', 'n_gaps', 'n_covered',
        'n_mono', 'n_duo', 'n_tri', 'n_quat', 'n_poly', 'f_covered', 'f_mono',
        'f_duo', 'f_tri', 'f_quat', 'f_poly', 'mean_dom_cov', 'mean_f_dom_cov',
        'median_f_dom_cov', 'mean_dom_cov_polysites',
        'median_dom_cov_polysites', 'mean_f_dom_cov_polysites',
        'median_f_dom_cov_polysites', 'mean_cov_polysites', 'n_binom',
        'f_binom', 'n_binom_segata', 'f_binom_segata',
        'average_nucleotide_diversity', 'watterson_theta', 'mean_maf',
        'median_maf', 'mean_maf_polysites', 'median_maf_polysites'
    ]

    # write df to file
    ofn = '%s/%s.aln_stats.txt' % (args['output_dir'], args['clade'])
    df.to_csv(ofn, sep='\t', index_label=False, index=False)
