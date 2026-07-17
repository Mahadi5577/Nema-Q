# NEMA-Q — Q1 Research Pipeline

**Project:** NEMA-Q: Non-Euclidean Memory-Augmented Quantum Graph Neural Network
**Investigator:** MD Nurol Amin Mahadi
**Pipeline authored:** 2026-07-16
**Target:** Q1 journal submission (see §8 for venue analysis)

---

## 1. Proposal Assessment for Q1 Readiness

### 1.1 What the proposal gets right (keep these — they are your reviewer shields)

| Strength | Why it matters at Q1 |
|---|---|
| Instrumentation-first design philosophy | Directly answers the #1 reviewer complaint about hybrid QML papers: "you compared whole models, not components." |
| Pre-declared reportable negative results | Signals scientific maturity; aligns with registered-report culture. |
| Honest QML risk framing (barren plateaus, dequantization) | Reviewers in quantum venues *will* raise Tang 2019 and McClean 2018. The proposal pre-empts both. |
| Classical bypass / graceful degradation | Gives the paper a result even if the quantum branch contributes nothing. |

### 1.2 Fatal gaps for Q1 (must fix — each one alone is a rejection reason)

**G1. Single dataset (Cora).** No Q1 venue accepts a graph-learning claim validated on one small citation graph. Required: a dataset suite **stratified by Gromov δ-hyperbolicity**, so the hyperbolic-gain hypothesis is tested where it should hold AND where it should fail (negative controls). Minimum suite:

| Dataset | Role | Expected hierarchy |
|---|---|---|
| Cora, Citeseer, Pubmed | Standard citation benchmarks | Moderate (power-law) |
| Disease, Airport (HGCN paper datasets) | High-hyperbolicity positive controls | High (tree-like, δ≈0–1) |
| Amazon Photo / Computers | Co-purchase, larger scale | Moderate |
| WebKB (Cornell/Texas/Wisconsin) or Actor | Low-hierarchy / heterophilous negative controls | Low |

**G2. No statistical protocol.** Required: ≥10 seeds per configuration, mean ± std, paired Wilcoxon signed-rank across seeds/splits, Holm–Bonferroni correction over the hypothesis family, and effect sizes (Cohen's d). "We beat baseline by 0.4%" without this is a desk reject at TNNLS.

**G3. No concrete dequantization control.** The proposal says "classical control paths of matched capacity" but names none. Required: a **classical surrogate branch** with parameter count matched to the PQC (small MLP and/or random Fourier features head), swapped in via config flag. The headline quantum claim must be *NEMA-Q vs NEMA-C (surrogate)*, not *NEMA-Q vs GCN*.

**G4. No hyperparameter fairness protocol.** Hybrid papers die when reviewers suspect the baseline was under-tuned. Required: equal tuning budget (e.g., 50 Optuna trials per model per dataset), identical search-space philosophy, all trial logs released.

**G5. No falsifiable hypotheses.** The proposal describes an architecture, not an experiment. Register these before running the evaluation matrix (§4, Phase 0):

- **H1 (hyperbolicity):** The hyperbolic encoder outperforms a parameter-matched Euclidean encoder on datasets with low δ (tree-like), with the gain monotonically shrinking as δ grows. *Falsifier:* no correlation between δ and gain.
- **H2 (quantum, low-label):** The PQC branch outperforms the parameter-matched classical surrogate in the low-label regime (≤5 labels/class), and the gap closes as labels increase. *Falsifier:* surrogate matches PQC at all label rates.
- **H3 (trainability):** Per-parameter gradient variance of the PQC stays above 10⁻⁴ throughout training at the chosen width/depth (no barren plateau in practice). *Falsifier:* variance collapse.
- **H4 (attribution validity):** Learned fusion-gate weights correlate (Spearman ρ > 0.5) with leave-branch-out performance deltas across datasets. *Falsifier:* gates uninformative.
- **Pre-declaration:** any falsified hypothesis is reported as a finding, not buried.

**G6. Simulator-only scope not defended.** State explicitly: Phase I/II are statevector-simulation studies of a *hybrid inductive bias*, not a quantum-advantage claim. Add a shot-noise ablation (finite-shot simulation at 256/1024/4096 shots) to show the effect survives realistic measurement noise. Never use the phrase "quantum advantage."

**G7. Missing related-work coverage.** Background cites hyperbolic GNNs and PQC basics but not: quantum graph neural networks (Verdon et al. 2019), equivariant QGNNs, quantum graph kernels, geometric QML (Meyer et al. 2023), hyperbolic attention networks, κ-GCN / fully hyperbolic GNNs (Chen et al. 2022), and post-2019 dequantization literature. The novelty claim ("no published architecture combines...") must survive a systematic search — run it and document it (Phase 0).

**G8. Proposal document incomplete.** Sections 3+ (methodology, work plan, timeline, deliverables, risk register) are absent from the .docx. This pipeline document supplies them; back-port into the proposal.

**G9. "Memory-Augmented" in the title is unimplemented.** Nothing in the summary describes a memory mechanism. Either (a) add an explicit memory module (e.g., a hyperbolic key-value memory over class prototypes) with its own ablation, or (b) rename. A title promising a component the paper doesn't ablate is a guaranteed reviewer flag. **Decision needed in Phase 0.**

---

## 2. Architecture Under Test

```
                    ┌──────────────────────────┐
  x, A ──┬────────► │ Hyperbolic encoder        │──┐
         │          │ (Poincaré ball, geoopt)   │  │
         │          └──────────────────────────┘  │   ┌────────────┐   ┌──────────┐
         ├────────► ┌──────────────────────────┐  ├──►│ Gated       │──►│ Classifier│──► ŷ
         │          │ Quantum branch (PQC,      │  │   │ fusion      │   └──────────┘
         │          │ PennyLane, n_q≤8, L≤4)   │──┤   │ (learned α) │
         │          └──────────────────────────┘  │   └────────────┘
         │          ┌──────────────────────────┐  │
         └────────► │ Classical bypass (GCN)    │──┘
                    └──────────────────────────┘

Swap-in controls (config flag, same fusion/head):
  quantum branch → classical surrogate (param-matched MLP / RFF)
  quantum branch → frozen-random PQC (recipe control)
  hyperbolic     → Euclidean encoder (param-matched)
```

Every branch is hook-instrumented: per-branch gradient norms/variance, fusion gate values, PQC expressibility (Sim et al. 2019 KL-vs-Haar) and Meyer–Wallach entanglement, per-node leave-branch-out attribution.

---

## 3. Experiment Matrix (the paper's evidence table)

**Models (9):** GCN, GAT, MLP, HGCN-style hyperbolic-only, PQC-only head, NEMA-Q (full), NEMA-C (surrogate swap), NEMA-R (frozen-random PQC), NEMA-E (Euclidean swap).
**Datasets (8–10):** per §1.2 G1, each with measured δ-hyperbolicity.
**Label regimes (3):** standard splits; 5 labels/class; 10 labels/class.
**Seeds:** 10.
**Total runs:** ~9 × 9 × 3 × 10 ≈ 2,400 (cheap: small models, single GPU; PQC simulation is the bottleneck — budget it, cache circuit outputs where params frozen).

Secondary studies: qubit/depth sweep vs gradient variance (H3), shot-noise ablation (G6), fusion-gate vs leave-branch-out correlation (H4), runtime/memory profile.

---

## 4. Pipeline Phases

### Phase 0 — Positioning & registration (2 weeks)
- Systematic novelty search (Scopus/arXiv: "hyperbolic quantum graph", "quantum GNN", "geometric quantum machine learning"); write related-work matrix.
- Resolve G9 (memory module: implement or rename).
- Freeze hypotheses H1–H4 + analysis plan in `experiments/PREREGISTRATION.md`, commit-hash it. This is your registered-report artifact.
- Pick target venue (§8) and pull its checklist (reproducibility, ethics, data availability).

**Gate 0:** novelty confirmed, hypotheses committed, venue chosen.

### Phase 1 — Infrastructure (2 weeks) *(scaffold already created — see §6)*
- Repro harness: global seeding, deterministic flags, env pinning, per-run manifest (config + git hash + env → JSON).
- Config-driven runner, unit tests, CI (pytest + smoke train on tiny graph).
- Dataset module incl. δ-hyperbolicity estimator; produce the dataset-statistics table (goes in paper §Datasets).

**Gate 1:** `pytest` green; smoke run trains end-to-end on Cora; δ table generated.

### Phase 2 — Baseline fidelity (2–3 weeks)
- Implement GCN/GAT/MLP; **reproduce published Cora/Citeseer/Pubmed numbers within ±0.5%** on standard splits. If you can't reproduce baselines, nothing downstream is publishable.
- Optuna tuning harness with fixed budget; log all trials.

**Gate 2:** baseline reproduction table matches literature.

### Phase 3 — Component builds (4–5 weeks)
- 3a Hyperbolic encoder (geoopt Poincaré ball; tangent-space aggregation; learnable curvature). Unit tests: exp/log inverse consistency, distortion on synthetic trees < Euclidean.
- 3b Quantum branch (PennyLane TorchLayer; angle embedding + strongly-entangling layers; n_q ∈ {4,6,8}, depth ∈ {1,2,4}). Unit tests: gradient flow, output range, param count.
- 3c Fusion + surrogate + frozen-random controls; parameter-count matching asserted in tests.
- 3d Telemetry: gradient hooks, expressibility, entanglement, attribution. Each with a test on a known circuit (e.g., idle circuit → expressibility ≈ worst case).

**Gate 3:** all component tests green; each branch trains alone on Cora without divergence.

### Phase 4 — Feasibility (Phase I deliverable of the proposal) (2 weeks)
- Full NEMA-Q trains stably on Cora, 10 seeds; telemetry captured; H3 checked at this scale.
- Write internal Phase-I report (this satisfies the original proposal's deliverables 1–3).

**Gate 4:** stable training + telemetry dashboards; go/no-go on qubit/depth setting.

### Phase 5 — Full evaluation matrix (4–6 weeks)
- Run §3 matrix via `scripts/run_matrix.py`; every run writes a manifest; no manual result copying — analysis reads run directories.
- Statistical analysis per pre-registered plan (`nemaq.analysis.stats`).

**Gate 5:** results table with corrected p-values and effect sizes; H1–H4 verdicts.

### Phase 6 — Attribution & diagnostics deep-dive (3 weeks)
- Gradient-telemetry analysis (barren-plateau evidence), expressibility-vs-performance correlation, per-node attribution case studies, fusion-gate analysis (H4).
- This section is the paper's *novelty core* — the component-level accounting no prior hybrid paper does. Spend the effort here, not on chasing +0.3% accuracy.

### Phase 7 — Writing & artifact release (4 weeks)
- Paper drafting; figures from `nemaq.analysis.plots` only (regenerable from run dirs).
- Public repo: code + configs + trial logs + run manifests + exact splits; Zenodo DOI; reproduce-everything script.
- Internal adversarial review before submission (simulate the dequantization reviewer and the statistics reviewer).

**Total: ~5.5–6.5 months to submission.**

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Quantum branch ≈ surrogate (null result) | **High** | Pre-registered as reportable; paper pivots to "rigorous component accounting shows where PQC does/doesn't help" — still Q1-viable in QML venues, arguably stronger. |
| Barren plateau at useful width | Medium | Depth/qubit sweep early (Phase 4 gate); shallow circuits + local cost. |
| Hyperbolic gain doesn't track δ | Medium | That's H1's falsifier — reportable; check per-dataset curvature learned. |
| PQC simulation too slow for matrix | Medium | n_q ≤ 8, batch circuits, lightning backend, subsample nodes fed to quantum branch (document as design choice). |
| Baseline reproduction fails | Low | Gate 2 blocks everything; use published reference implementations to debug. |
| Novelty scooped mid-project | Low–Med | Phase 0 search + arXiv alert; instrumentation angle differentiates even against similar architectures. |

---

## 6. Codebase (created alongside this document)

```
nema-q/
├── pyproject.toml               # pinned deps: torch, torch-geometric, geoopt, pennylane
├── README.md
├── configs/                     # YAML per experiment; base + overrides
│   ├── base.yaml
│   ├── cora_gcn.yaml
│   ├── cora_nemaq.yaml
│   └── ablations/nemaq_surrogate.yaml
├── src/nemaq/
│   ├── utils/        seed.py, config.py, manifest.py
│   ├── data/         loader.py, hyperbolicity.py
│   ├── models/       classical.py, hyperbolic.py, quantum.py, surrogate.py, fusion.py, nemaq.py
│   ├── telemetry/    gradients.py, expressibility.py, attribution.py
│   ├── training/     trainer.py
│   └── analysis/     stats.py
├── scripts/          train.py, run_matrix.py, compute_hyperbolicity.py
├── tests/            unit tests per module + smoke test
└── experiments/      PREREGISTRATION.md (template), runs/ (gitignored outputs)
```

Design rules enforced in code:
1. **Every run writes a manifest** (config, git hash, seed, env, metrics) — analysis consumes manifests, never hand-copied numbers.
2. **Controls are config swaps**, not code forks: `model.quantum_mode: pqc | surrogate | frozen_random`, `model.geometry: hyperbolic | euclidean`.
3. **Parameter matching is asserted**, not assumed (`tests/test_param_match.py`).
4. **Telemetry always on** — instrumentation-first means it's not optional.

---

## 7. Reproducibility Checklist (build into paper appendix)

- [ ] All seeds, splits, configs in repo
- [ ] Env pinned (`pyproject.toml` + lock)
- [ ] One-command reproduction per table/figure
- [ ] All Optuna trial logs released
- [ ] Run manifests with git hashes released
- [ ] Statistical analysis code released
- [ ] Compute budget reported (GPU-hours, simulator wall-time)
- [ ] Negative results reported per pre-registration

---

## 8. Venue Analysis (Q1)

| Venue | Fit | Notes |
|---|---|---|
| **Quantum Machine Intelligence** (Springer, Q1) | **Best fit** | Hybrid QML methodology + honest component accounting is exactly their scope; instrumentation angle strong. |
| **npj Quantum Information** (Q1) | Good if H2 positive | Higher bar for quantum significance; risky if quantum result is null. |
| **IEEE TNNLS** (Q1) | Good | Frame as neural-architecture + rigorous ablation study; quantum as one component. Statistics protocol (G2) is mandatory here. |
| **Machine Learning** (Springer, Q1) | Good fallback | Values negative/null results with rigor. |
| **Neurocomputing / Knowledge-Based Systems** (Q1) | Fallback | Faster turnaround, lower prestige within Q1. |

Recommendation: write for **Quantum Machine Intelligence**, keep TNNLS formatting-compatible as fallback.

---

## 8b. ICEQT'26 Review-Response Matrix (added 2026-07-16)

The ICEQT'26 acceptance (paper ICE7405, "NEMA-Q: A Hyperbolic Quantum-Classical
GNN with Observable-Level Explainability", scores ~13/20) provides the referee
gap-list for the Q1 extension. Every item maps to a codebase artifact:

| Referee point | Q1 response | Artifact | Status |
|---|---|---|---|
| B: no tuned GCN/GAT/hyperbolic baselines, same split, param counts | Baselines under identical splits + tuning budget; param counts in every manifest | `configs/cora_{gcn,gat,hgcn}.yaml`; `param_counts` in manifests | Built; needs matrix run |
| B: Cora-only, non-standard split | δ-stratified suite + standard public splits + low-label protocol | `data/loader.py`, `data/hyperbolicity.py` | Built; needs runs |
| B: QOA lacks faithfulness / perturbation / randomization checks | Full faithfulness harness: masking, perturbation flips, Adebayo randomization | `telemetry/qoa.py` + notebook §13 | Built; needs trained-model verdicts |
| B: high seed variance, minority-class F1 collapse | Residual-trunk fusion (std 0.10→0.02, quantified ablation); per-class F1 + collapse detector in metrics | `models/fusion.py`; `test_min_class_f1` in manifests | Done / instrumented |
| B: best-seed noise study | Noise/shot ablations over ALL seeds via `shots` config; never best-seed | `model.shots` + matrix protocol | Config ready |
| B: stats/CIs for ablation gaps | 10 seeds, paired Wilcoxon, Holm–Bonferroni, Cohen's d, pre-registered | `analysis/stats.py`, `experiments/PREREGISTRATION.md` | Built |
| B: reproduction details (model selection, hyperparams fixed?) | Run manifests (config+git hash+env), preregistration freeze, public repo | `utils/manifest.py` | Built; needs GitHub |
| A: template/citation/running-title issues | Springer LNCS template hygiene for camera-ready; venue template for Q1 | paper repo | Open |

Additional Q1-only items beyond the reviews: trunk-only diagnostic
(`ablations/nemaq_trunk_only.yaml`) to close the NEMA-Q-vs-GCN scaffold gap,
and the frozen-vs-trained PQC comparison promoted to pre-registered H5.

## 9. Immediate Next Actions

1. Run Phase 0 novelty search; fill `experiments/PREREGISTRATION.md`.
2. Decide G9 (memory module vs rename).
3. `pip install -e .[dev]` and run `pytest` — confirm scaffold on your machine.
4. Start Phase 2 baseline reproduction (highest-risk cheap task — do it before any exotic component work).
