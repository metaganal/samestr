# Changelog

Changes in this fork of [SameStr](https://github.com/danielpodlesny/samestr),
on top of upstream `ada3ca5` (*Fix/clade input issue 20251119*, #43). Version
is still `1.2025.111`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Not committed yet. These changes make `samestr stats` callable as a library,
which [covtools](../covtools/covtools-0.1.0/CHANGELOG.md) depends on. They also
let samestr install next to numpy 2, pandas 3 and Python 3.14.

### Added

- `samestr.stats.alignment_stats.sample_stats(x, samples, site_deg=None,
  syn_pairs=None, dominant_variants=False)` computes the per-sample statistics
  table from an in-memory `(n_samples, n_sites, 4)` allele-count array.
- `STAT_COLUMNS`: the table's column names, in order.
- `samestr.stats` exports `sample_stats`, `STAT_COLUMNS` and
  `clade_site_annotation`.
- **`samestr compare` writes pairwise nucleotide diversity.** It covers the
  positions that both samples of a pair cover at depth `--pi-min-cov` or more.
  Before counting, a non-dominant allele with fewer than
  `--pi-min-allele-count` reads is discarded, so sequencing errors do not
  count as diversity. The dominant allele is always kept. Both defaults are 4,
  as in Wasney et al. 2026 (*Nat Commun*, doi:10.1038/s41467-026-70705-8):
  that paper requires at least 4 reads at a site in both samples for π and
  Fst, and at least 4 reads for a minor allele. Schloissnig et al. 2013 and
  `samestr stats`' π ratios use the same 4-read minimum. New files per clade:
  - `<clade>.pi_between.txt`: the mean probability that a read of the row
    sample and a read of the column sample differ, 1 − Σₐ f_ia·f_ja.
  - `<clade>.pi_within.txt`: the row sample's mean within-sample π over the
    positions it shares with the column sample. It uses the same unbiased
    estimator as `average_nucleotide_diversity`. The matrix is asymmetric.
  - `<clade>.pi_sites.txt`: the number of those positions.
  - All three come from `pairwise_pi`, which uses matrix products like
    `pairwise_counts`. The existing outputs are unchanged, and `summarize`
    does not read the new files.
- **`samestr fst`** computes pairwise Fst per clade (Hudson et al. 1992):
  1 − mean(π_within) / π_between.
  - Input: `--compare-dir`, plus `--stats-dir` (optional).
  - `<clade>.fst.txt` uses π_within over the pair's shared positions, so both
    terms are over one site set.
  - `<clade>.fst_stats.txt` uses `samestr stats`' per-sample
    `average_nucleotide_diversity` instead, which covers each sample's own
    positions. It is NaN without `--stats-dir`. That π applies neither
    compare's depth threshold nor its minor-allele filter, so with real
    sequencing error `fst_stats` reads lower than `fst`.
  - `fst_pairs.tsv` lists every pair with all the terms.
  - Pairs sharing fewer than `--min-sites` positions (default 5000) are NaN.
    So are pairs with π_between = 0, such as two identical clonal samples.
  - Library functions: `samestr.fst.hudson_fst` and `samestr.fst.clade_fst`.

### Changed

- `aln2stats` keeps its file handling and calls `sample_stats` for the
  numbers.
  - File handling covers the skip check, sample names, array loading, the
    `<clade>.pos.txt` lookup, the marker annotation and the write.
  - `<clade>.aln_stats.txt` is byte-identical to `dbaf6ef`. This was checked
    with and without a marker database, with `--delete-pos`, and with
    `--dominant-variants`.
- `sample_stats` builds the table one typed column at a time, so numeric
  columns keep their dtypes. The old code built a transposed object array.
- `requirements.txt` uses lower bounds instead of exact pins: `numpy>=1.24`,
  `pandas>=1.5`, `scipy>=1.10`, `biopython>=1.81` (were `==1.24.2`, `==1.5.3`,
  `==1.10.0`, `==1.81`).
- `pyproject.toml` takes its dependencies from `requirements.txt`
  (`dynamic = ["dependencies"]`). Before this, `[project]` declared none, so
  pip installed samestr without them.

### Notes

- Sequencing error left in the counts acts as within-sample diversity at
  every position, which pulls Fst between distinct strains below 1. Take two
  clonal strains that differ at 3% of positions, at depth 40 and a 0.2% error
  rate:
  - with `--pi-min-allele-count 1` (no filter), Fst ≈ 0.88;
  - with the default of 4, Fst = 1.00.
- `samestr compare` needs about 48 more bytes per sample per position while it
  computes the π matrices. It works in float64, because Fst is 1 minus a
  ratio of two close sums.
- `x` stays in memory until `sample_stats` returns. The old code freed it
  before the coverage statistics. Peak memory in `aln2stats` is therefore
  roughly one copy of `x` higher.
- The means are sums over floats, so their last bit depends on the array's
  memory layout. A C-contiguous array, such as a loaded `.npy`, gives the
  same row whether it is passed alone or inside the full cohort. A
  fancy-indexed, non-contiguous view may not.

## Fork commits (2026-10-07)

### Added

- **`samestr stats` population-genetic statistics** (`b86a1f9`). New
  `aln_stats.txt` columns:
  - `average_nucleotide_diversity`: within-sample, unbiased per-site pi
    (Nelson & Hughes 2015).
  - `watterson_theta`: per-site depth as the sample size, alleles with fewer
    than 2 reads ignored.
  - `mean_maf`, `median_maf`, `mean_maf_polysites`, `median_maf_polysites`.
  - `population_stats` and `tajimas_d` (across samples, on consensus alleles)
    are library functions only. They are not written to the table.
- **Within-host polymorphism rate** (`7cafb17`), after Garud et al. 2019 and
  Madi et al. 2023. It is the share of sites with an intermediate allele
  frequency (0.2–0.8), inside a depth window of 0.3–3× the sample's median
  depth. Samples with a median depth below 5 are excluded.
  - It is reported over all marker sites, over synonymous (fourfold
    degenerate) sites and over nonsynonymous (onefold degenerate) sites.
  - New columns: `n_poly_intermediate*`, `n_sites_depth_ok*` and
    `polymorphism_rate*`, with the suffixes `_syn` and `_nonsyn`.
  - Site degeneracy comes from the marker sequences
    (`<clade>.markers.fa.gz` + `<clade>.positions.txt.gz`, via
    `clade_site_annotation`). It follows `<clade>.pos.txt` when
    `samestr filter --delete-pos` removed columns.
- **Nucleotide diversity ratios** (`7200958`), after Schloissnig et al. 2013:
  `pi_syn`, `pi_nonsyn`, `pi_n_pi_s`, `pi_nondeg`, `pi_4fold` and
  `pi_nondeg_pi_4fold`.
  - Synonymous and nonsynonymous sites are counted as in Nei & Gojobori.
  - A non-dominant allele counts only with at least 4 reads and a frequency of
    at least 1%.
- Without a usable marker annotation, the codon-aware columns are NaN.

### Changed

- **`samestr stats`** computes every per-site quantity in one pass, from
  coverage, dominant-allele coverage and the number of observed alleles.
  - This replaces `consensus()` and a per-sample loop (`b86a1f9`).
  - Medians use a vectorised masked row median.
  - The binomial test is run only at polymorphic sites.
- **`samestr compare`** (`e6501cd`): the all-vs-all `closest`, `overlap` and
  `fraction` matrices come from matrix products (`pairwise_counts`), not a
  per-sample loop.
  - Shared-allele counts use an exact inclusion–exclusion correction, done as
    sparse products at multi-allelic sites.
  - The dominant-variant MSA is built vectorised. Ties are still broken at
    random.
- **`samestr summarize`** (`e6501cd`):
  - Taxon co-occurrence is one float64 matrix product per taxonomic level,
    read at precomputed sample pairs. This replaces outer-merging each level's
    long table.
  - Profiles are binarised without a per-cell Python call (`applymap`).
  - A sample dropped by `numeric_only` is reindexed in with zeros.

### Fixed

- **`samestr summarize`** (`e6501cd`): the strain co-occurrence sample names
  have the taxonomic-profile extension stripped, or the longest matching
  prefix of it (e.g. `.mp4` from `.mp4.txt`). Before this they did not match
  the taxon co-occurrence names, and the outer merge failed to join the rows.
- **`samestr stats`**: `clade_path` is imported from
  `samestr.utils.file_mapping` instead of `samestr.utils` (`dbaf6ef`).
