import logging
import os
from os.path import exists, isdir, join

import numpy as np
import pandas as pd

from samestr.summarize.read_samestr_data import read_distmat, upper_pair_indices

LOG = logging.getLogger(__name__)

# `samestr compare` outputs this reads, per clade
PI_SUFFIXES = ('.pi_within.txt', '.pi_between.txt', '.pi_sites.txt')

PAIR_COLUMNS = ['clade', 'row', 'col', 'n_sites', 'pi_within_row',
                'pi_within_col', 'pi_between', 'fst', 'pi_stats_row',
                'pi_stats_col', 'fst_stats']


def hudson_fst(pi_within_i, pi_within_j, pi_between):
    """
    Pairwise Fst (Hudson et al. 1992): 1 - mean within-sample diversity /
    between-sample diversity, NaN where pi_between is 0 or undefined.
    Arguments broadcast.
    """
    pi_w = (np.asarray(pi_within_i, dtype=np.float64) +
            np.asarray(pi_within_j, dtype=np.float64)) / 2
    pi_b = np.asarray(pi_between, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(pi_b > 0, 1 - pi_w / pi_b, np.nan)


def clade_fst(pi_within, pi_between, n_sites, pi_stats=None, min_sites=5000):
    """
    Pairwise Fst of one clade from `samestr compare` matrices.

    Args:
        pi_within: (n, n) mean within-sample pi of the row sample over the
            positions it shares with the column sample.
        pi_between: (n, n) mean between-sample pi over those positions.
        n_sites: (n, n) number of those positions.
        pi_stats: optional (n,) per-sample pi from `samestr stats`
            (average_nucleotide_diversity, over each sample's own covered
            positions), for the stats-based Fst.
        min_sites: pairs sharing fewer positions are undefined (NaN).
    Returns:
        (fst, fst_stats): (n, n) arrays. fst uses pi_within over the pair's
        shared positions, so both terms are over one site set; fst_stats
        uses pi_stats, whose site sets differ between the two samples, and
        is NaN without pi_stats. The diagonal is NaN.
    """
    pi_within = np.asarray(pi_within, dtype=np.float64)
    usable = np.asarray(n_sites) >= min_sites
    np.fill_diagonal(usable, False)

    fst = np.where(usable, hudson_fst(pi_within, pi_within.T, pi_between),
                   np.nan)
    if pi_stats is None:
        fst_stats = np.full_like(fst, np.nan)
    else:
        p = np.asarray(pi_stats, dtype=np.float64)
        fst_stats = np.where(usable,
                             hudson_fst(p[:, None], p[None, :], pi_between),
                             np.nan)
    return fst, fst_stats


def read_stats_pi(stats_dir, clade, samples):
    """
    average_nucleotide_diversity per sample from `<clade>.aln_stats.txt`, in
    the order of `samples`; None if the file is missing. Samples compared as
    dominant variants (`.dom` suffix) take the stats of their base name.
    """
    fn = join(stats_dir, '%s.aln_stats.txt' % clade)
    if not exists(fn):
        LOG.warning('%s: no %s, so fst_stats is NaN.' % (clade, fn))
        return None
    stats = pd.read_csv(fn, sep='\t', index_col='Sample')
    if 'average_nucleotide_diversity' not in stats.columns:
        LOG.warning('%s: %s has no average_nucleotide_diversity (written by '
                    'an older samestr stats), so fst_stats is NaN.' %
                    (clade, fn))
        return None
    pi = stats['average_nucleotide_diversity']
    names = [s if s in pi.index or not s.endswith('.dom') else s[:-4]
             for s in samples]
    missing = [s for s in names if s not in pi.index]
    if missing:
        LOG.warning('%s: %s sample(s) not in %s, e.g. %s; their fst_stats '
                    'is NaN.' % (clade, len(missing), fn, missing[0]))
    return pi.reindex(names).to_numpy(dtype=np.float64)


def list_clades(compare_dir, clades=None):
    """Clades with all three pi matrices in `compare_dir`."""
    found = {f[:-len(PI_SUFFIXES[0])] for f in os.listdir(compare_dir)
             if f.endswith(PI_SUFFIXES[0])}
    complete = sorted(c for c in found
                      if all(exists(join(compare_dir, c + s))
                             for s in PI_SUFFIXES))
    for c in sorted(found.difference(complete)):
        LOG.warning('Skipping %s: missing %s.' % (c, ', '.join(
            s for s in PI_SUFFIXES if not exists(join(compare_dir, c + s)))))
    if clades is not None:
        complete = [c for c in complete if c in clades]
    return complete


def _write_matrix(path, samples, mat):
    pd.DataFrame(mat, index=pd.Index(samples, name='Sample'),
                 columns=samples).to_csv(path, sep='\t', na_rep='nan')


def fst(args):
    """
    `samestr fst`: pairwise Fst per clade from `samestr compare` (pi within
    and between samples over shared positions) and, optionally, `samestr
    stats` (per-sample pi). Writes `<clade>.fst.txt`, `<clade>.fst_stats.txt`
    and one long table of all pairs, `fst_pairs.tsv`.
    """
    compare_dir, out_dir = args['compare_dir'], args['output_dir']
    stats_dir = args.get('stats_dir')
    if not isdir(compare_dir):
        raise ValueError('--compare-dir %s is not a directory.' % compare_dir)
    clades = list_clades(compare_dir, args.get('clade'))
    if not clades:
        LOG.error('No clade with %s in %s. Run `samestr compare` first (this '
                  'samestr writes them).' % (', '.join(PI_SUFFIXES),
                                             compare_dir))
        return None
    LOG.info('Calculating pairwise Fst: %s clade(s).' % len(clades))

    pairs = []
    for clade in clades:
        pw = read_distmat(compare_dir, clade, '.pi_within.txt')
        pb = read_distmat(compare_dir, clade, '.pi_between.txt')
        ns = read_distmat(compare_dir, clade, '.pi_sites.txt')
        samples = list(pw.index.astype(str))
        if not all(list(m.index.astype(str)) == samples and
                   list(m.columns.astype(str)) == samples for m in (pw, pb, ns)):
            LOG.warning('Skipping %s: the pi matrices do not share one sample '
                        'order.' % clade)
            continue
        pw, pb, ns = (m.to_numpy(dtype=np.float64) for m in (pw, pb, ns))
        pi_stats = read_stats_pi(stats_dir, clade, samples) \
            if stats_dir else None

        f, f_stats = clade_fst(pw, pb, ns, pi_stats, args['min_sites'])
        _write_matrix(join(out_dir, '%s.fst.txt' % clade), samples, f)
        _write_matrix(join(out_dir, '%s.fst_stats.txt' % clade), samples,
                      f_stats)

        i, j = upper_pair_indices(pd.Index(samples))
        lab = np.asarray(samples, dtype=object)
        ps = (np.full(len(samples), np.nan) if pi_stats is None
              else pi_stats)
        pairs.append(pd.DataFrame({
            'clade': clade, 'row': lab[i], 'col': lab[j],
            'n_sites': ns[i, j].astype(np.int64),
            'pi_within_row': pw[i, j], 'pi_within_col': pw[j, i],
            'pi_between': pb[i, j], 'fst': f[i, j],
            'pi_stats_row': ps[i], 'pi_stats_col': ps[j],
            'fst_stats': f_stats[i, j]}, columns=PAIR_COLUMNS))
        n_ok = int(np.isfinite(f[i, j]).sum())
        LOG.info('%s: %s of %s pair(s) share >= %s positions.' %
                 (clade, n_ok, len(i), args['min_sites']))

    table = pd.concat(pairs, ignore_index=True) if pairs else \
        pd.DataFrame(columns=PAIR_COLUMNS)
    table.to_csv(join(out_dir, 'fst_pairs.tsv'), sep='\t', index=False,
                 na_rep='NA')
    return True
