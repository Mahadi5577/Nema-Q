# NEMA-Q Pre-Registration

> **FROZEN 2026-07-17.** §1–§6 are frozen as committed in
> `dd61326` at https://github.com/Mahadi5577/Nema-Q (initial commit).
> No edits to §1–§6 after this point; amendments go in §7 with dates.
> This document is the paper's registered-analysis artifact for the
> confirmatory runs (seeds 10–19).

## 1. Hypotheses (falsifiable, with pre-declared falsifiers)

- **H1 (hyperbolicity):** Hyperbolic encoder > parameter-matched Euclidean encoder
  (NEMA-Q vs NEMA-E) on low-δ datasets; gain shrinks monotonically with δ.
  *Test:* per-dataset paired Wilcoxon (one-sided, hyp > euc) + Stouffer's Z
  pooled across datasets + Spearman(δ_mean, gain) across datasets
  (`nemaq.analysis.stats.geometry_interaction`). The pooled test is primary:
  Phase-4 Cora showed the single-dataset comparison is underpowered at n=10
  (paired d = 0.44 → ~40 seeds for 80% power alone); pooling across the
  δ-stratified suite is the registered remedy. The per-dataset test on the
  lowest-δ dataset (Disease) is confirmatory; other per-dataset tests are
  descriptive.
  *Falsifier:* pooled Stouffer p ≥ 0.05 OR no negative δ–gain correlation.

- **H2 (quantum, low-label):** NEMA-Q > NEMA-C (surrogate) at ≤5 labels/class;
  gap closes at standard label rates.
  *Test:* paired Wilcoxon per (dataset, label-rate); interaction check across rates.
  *Falsifier:* surrogate matches PQC at all label rates.

- **H3 (trainability):** PQC circuit-weight gradient variance stays within 2
  orders of magnitude of the best classical branch's gradient variance
  (relative criterion — absolute BP-literature floors do not transfer to a
  trained hybrid; classical branches are the matched healthy-gradient
  reference at the same loss scale).
  *Test:* telemetry summary `h3_pass_relative == true` on ≥ 9/10 seeds.
  *Falsifier:* sustained relative variance collapse.

- **H4 (attribution validity):** Fusion gate weights track leave-branch-out deltas.
  *Test:* Spearman ρ per branch, pooled across datasets; threshold ρ > 0.5.
  *Falsifier:* ρ ≤ 0.5.

- **H5 (frozen vs trained PQC):** A frozen-random PQC (NEMA-R) matches or
  exceeds the trained PQC (NEMA-Q) under an identical scaffold — i.e. the
  PQC's value, where present, comes from the fixed nonlinear random
  projection, not from trained quantum features. Motivated by Phase-4 Cora
  (frozen > trained in 10/10 seeds, paired mean +3.9 pts, p = 0.0002,
  d = 1.85); registered here BEFORE any Phase-5 run on other datasets.
  *Test:* paired Wilcoxon (one-sided, NEMA-R ≥ NEMA-Q) per dataset;
  Holm–Bonferroni within the H5 family.
  *Falsifier:* trained PQC significantly exceeds frozen-random on the
  majority of datasets.

## 2. Datasets and stratification

Filled from `experiments/figures/eda/delta_table.md` + `eda_stats.json`
(sampled Gromov δ, 2026-07-16):

| Dataset | δ_mean | δ_max | Edge homophily | Role |
|---|---|---|---|---|
| disease | 0.000 | 0.0 | 0.875 | positive control (low δ) |
| cora | 0.356 | 2.0 | 0.810 | benchmark |
| pubmed | 0.399 | 3.5 | 0.802 | negative control (high δ) |
| citeseer | 0.550 | 3.5 | 0.736 | negative control (high δ) |

Splits: cora/citeseer/pubmed use the Planetoid public split; disease has no
public split → ratio 30/10/60 with `split_seed: 0` (independent of model seed).

## 3. Models and controls

GCN, GAT, MLP, NEMA-Q, NEMA-C (surrogate), NEMA-R (frozen random), NEMA-E (Euclidean).
Parameter matching asserted by `tests/test_param_match.py` at tolerance 15%.

## 4. Statistical protocol

10 seeds per (model, dataset); paired Wilcoxon signed-rank; Holm–Bonferroni over
{H1-pooled, H2, H5} primary family; Cohen's d reported for all comparisons; α = 0.05.
Optional H1 power extension: the Cora geometry pair (NEMA-Q vs NEMA-E) may be
extended to 40 seeds (declared here in advance; seeds 0–39, same split protocol) —
if run, the 40-seed Cora test replaces the 10-seed one, never supplements it.
No comparisons beyond those listed here enter the hypothesis family (anything
else is exploratory and labeled as such).

## 5. Hyperparameter budget and locked recipes

Original Phase-0 template declared 50 Optuna trials per (model, dataset).
Actual pilot-phase tuning (seeds 0–9, exploratory) used smaller manual
validation-only sweeps with an equal budget shape across models on each
dataset; test accuracy was read once per locked config. The recipes below are
LOCKED for the confirmatory runs (seeds 10–19) and are identical to the
committed yaml configs.

Default recipe (`configs/base.yaml`): lr 0.01, weight_decay 5e-4,
dropout 0.5, aux_weight 0.3 (annealed), patience 100, epochs 500.

| Dataset | Model(s) | Deviations from default |
|---|---|---|
| cora | all | none (default recipe) |
| pubmed | all | none (default recipe) |
| citeseer | GCN / GAT / HGCN | none |
| citeseer | NEMA-Q, NEMA-E, NEMA-R | lr 0.005, aux_weight 2.0, gate_bias_init −4.0 |
| citeseer | trunk-only | none |
| disease | GCN | lr 0.05, wd 5e-4, dropout 0.5 |
| disease | GAT | lr 0.05, wd 5e-5, dropout 0.5 |
| disease | HGCN | lr 0.05, wd 5e-5, dropout 0.2 |
| disease | NEMA-Q, NEMA-E, NEMA-R | lr 0.05, aux_weight 0.3, gate_bias_init −2.0 |
| disease | trunk-only | none |

Rationale for the citeseer deviation is diagnosed and documented (quantum-
branch collapse under the default recipe; deep-supervision rescue). All sweep
cells and logs are preserved in the Colab notebook.

## 6. Reporting commitment

All four hypotheses are reported regardless of outcome. A null H2 is a headline
result ("component accounting localizes where the PQC does not help"), not a
buried footnote.

## 7. Amendments (dated, append-only)

- 2026-07-17: Freeze recorded. §1–§6 content is identical to commit
  `dd61326` (repository initial commit); this header/amendment note is the
  only post-freeze edit. All pilot results (seeds 0–9) predate the freeze
  and are labeled exploratory in the paper; confirmatory runs (seeds 10–19)
  must execute from a clone of the frozen repository so run manifests
  record the true git hash (pilot manifests say "no-git").
- 2026-07-17 (transcription fix, same day as freeze): the Disease tuned
  recipes in §5 and the committed disease yamls omitted `patience: 200`,
  which the executed Colab sweep and pilot test reads actually used
  (sweep cells select and evaluate all Disease models with patience 200;
  all other datasets use the default patience 100). The yamls are corrected
  to record what was run; no selection decision changed. Disease locked
  recipes therefore read: GCN lr .05/wd 5e-4/dr .5, GAT lr .05/wd 5e-5/dr .5,
  HGCN lr .05/wd 5e-5/dr .2, NEMA-Q/E/R lr .05/aux .3/gate_bias −2 —
  each with patience 200.
