# SESSION_CONTEXT — NEMA-Q Q1 Extension (handoff, 2026-07-17)

**2026-07-17 session progress:**
- Disease tuned recipes now IN yamls (`configs/disease_*.yaml` + `ablations/disease_nemaq_{euclidean,frozen_random}.yaml`; trunk_only stays default, matching citeseer pattern). Colab must re-pull these before confirmatory runs.
- PREREGISTRATION.md §2 filled (δ-table + homophily + splits) and §5 rewritten: locked per-dataset recipes table, honest note that Optuna plan → manual equal-shape sweeps. Freeze still gated on git push.
- Draft expanded ~5.5k words: §3 full equations (exp/log maps, Möbius layer, angle embedding, SEL circuit, gated fusion, deep-supervision loss — all verified against src), §4.1 real Table 1 from eda_stats.json, §4.2 falsifiers spelled out, §4.3 stats formulas, §8 formal QOA + harness pass criteria. §5 numbering bug fixed (5.3/5.4 were swapped). Figure map: 1 arch, 2 waterfall, 3 forest, 4 H5 paired, 5 stability, 6 geometry-δ, 7 QOA faithfulness; captions in draft as blockquotes.
- Fig 1 DONE: `paper/figures/fig1_architecture.tex` compiles clean with local MiKTeX pdflatex (works locally — only torch is broken).
- QMI sn-jnl template extracted to `paper/sn-template/sn-article-template/` (sn-jnl.cls; use `sn-mathphys-num` bst for numeric citations).
- Remaining: §9/appendices expansion to ~7k if desired; Phase 7 LaTeX conversion; integrity gate; `[TO-FILL]` items below unchanged.

**Project:** nemaq — ICEQT'26 accepted (ICE7405, acceptance email = `ICE7405.pdf`; referee report inside it) → Q1 extension for **Quantum Machine Intelligence** (Springer, sn-jnl, numeric citations).
**Track:** component accounting + explainability. NOT accuracy chasing.
**Workflow:** all training runs in Colab via `NEMA_Q_colab.ipynb` (77 cells, self-contained `%%writefile` cells mirroring `src/`). Local torch is BROKEN (WinError 193) — never run training locally; local repo is source of truth for code, Colab executes.
**Colab discipline:** after editing any `%%writefile` cell → rerun it → reload module in a SEPARATE cell (`importlib.reload`, leaf module first) → never paste reload code inside writefile cells (broke 3× this session).

## Final pilot results (seeds 0–9, test acc, mean±sd) — ALL `[PILOT]`

**Cora (public split, default recipe):** GAT .8231±.0090 · GCN .8191±.0088 · HGCN .7862±.0092 · trunk-only .7769±.0140 · frozen-random .7470±.0255 · NEMA-Q .7079±.0209 · euclidean .6868±.0408 · surrogate .6925±.0555 · softmax .6039±.1013
Paired (Wilcoxon): scaffold −4.2 (trunk≈HGCN p=.157 → scaffold cost = hyperbolic trunk, not a bug); frozen insertion −3.0 p=.002; **trained-vs-frozen −3.9, 10/10 seeds, p=.002, d=1.85**; residual-vs-softmax +10.4 p=.013, Levene p=.008.

**Citeseer (public):** GCN .7168 · GAT .7145 · HGCN .6666 · **tuned** NEMA-Q .6536±.0185 · tuned frozen .6678±.0177 · tuned euclidean .6211±.0289 · trunk-only .6457±.0180 · untuned NEMA-Q .5075±.0626 (collapse, diagnosed).
Tuned recipe (validation-only, sweeps 1–3): **lr .005, aux_weight 2.0, gate_bias_init −4.0, patience 100** — in `configs/citeseer_nemaq.yaml` + both ablation configs. Paired tuned: H5 frozen−trained +1.42, 8/10, p=.0273, d=.85; H1 hyp−euc +3.25, 9/10, p=.0039, d=1.02.
**Collapse mechanism (paper §7):** quantum-branch angle σ=.0053 (vs .0128 cora/.0159 pubmed) → near-constant q output (σ=.013) + loudest grads (1e-6 vs trunk 1e-8, geo starved 1e-11) + uniform gate (.177±.005) → destabilized shared head. Fragmentation alternative TESTED AND REJECTED (isolated nodes .500 n=12 vs connected .401). Fix = deep supervision aux=2.0 (grid edge, noted). Angle range ±0.15 of ±π on ALL datasets (near-identity regime — rationalizes H5).

**Pubmed (public, default):** GCN .7910 · GAT .7764 · HGCN .7697 · NEMA-Q .7516±.0311 · frozen .7721 (7/10, exact p TO-FILL) · euclidean .7378. H1 +1.4 p=.275.

**Disease (ratio 30/10/60, tuned):** baselines were majority-class degenerate (.7974 all seeds) until tuned — GCN .9148±.0048 (lr .05, wd 5e-4, dr .5) · GAT .9121 (lr .05, wd 5e-5, dr .5) · HGCN .8839 (lr .05, wd 5e-5, dr .2) · NEMA-Q tuned .8928±.0351 (lr .05, aux .3, gb −2) · frozen .8944 · euclidean .8906. H5 +0.16 p=.865 (equal); H1 +0.22 p=.695 (null). ~1/10 seeds relapse to majority (report w/ minF1 detector). **Disease tuned configs NOT yet written to yaml files — recipes only in this list + Colab sweep cells.**

**δ-stratification (EDA):** disease .000 / cora .356 / pubmed .399 / citeseer .550; homophily .875/.810/.802/.736.

**Hypothesis verdicts (pilot):**
- **H5 (frozen ≥ trained): holds 4/4** — the paper's headline.
- **H1(a) direction: holds pooled** (Stouffer ≈.004, recipe-heterogeneity caveat — cora/pubmed pairs still default recipe). **H1(b) δ-mechanism: FALSIFIED** — gains INCREASE with δ; GCN beats HGCN on δ=0 tree.
- H2: surrogate ≈ trained mean (p=.46), 2.7× variance (Levene .02), Cora only. H3: passes incl. collapsed runs (necessary-not-sufficient point). H4: NOT computed yet.
- **QOA faithfulness: 11/12 pass** (masking+perturbation all 4 datasets; randomization passes cora/citeseer/pubmed, FAILS disease → input-dominated attribution, framed as harness-has-teeth).

## Paper (academic-pipeline Stage 2 done through Phase 6)

- **Draft:** `paper/NEMAQ_QMI_draft.md` (~4,900 words; target ~7k after expansion). Peer-review round 1 applied (dangling H2/H3/H4 fixed, pooling caveat added, C1–C4 contributions). All numbers `[PILOT]`-tagged.
- **References:** `paper/references.md` — 56 entries, [V]=live-verified / [C]=canonical needs DOI check at integrity gate. Key positioning refs: Herbert arXiv:2309.13967 (NFL untrained circuits — H5's closest prior, engaged in §2); arXiv:2602.01828 (geometry–task alignment — supports H1(b) falsification); Heese QMI 2025 circuit-Shapley (QOA differentiation); Larocca Nat Rev Phys 2025 (BP); GInX-Eval (OOD critique acknowledged in §8).
- **Remaining paper work:** (1) expand §3/§4/§8 with equations + figure captions → ~7k; (2) TikZ architecture diagram (Fig 1 — only missing figure); (3) Phase 7 LaTeX sn-jnl conversion; (4) pipeline Stage 2.5 integrity check (verify all [C] refs + numbers vs manifests); (5) Stage 3 review etc.
- **Figures all exist** in zip `nemaq_full_export_20260716_165543.zip` → `experiments/figures/{eda,results,xai}/` (waterfall, forest×4, stability×4, paired×8, h1_geometry_delta, EDA×4+δ-scatter, XAI 5×4 datasets).

## `[TO-FILL]` in draft (blocked on user/runs)

1. ~~GitHub push~~ DONE 2026-07-17: https://github.com/Mahadi5577/Nema-Q (private), initial commit `4605c73`.
2. ~~Freeze PREREGISTRATION.md~~ DONE: §1–§6 frozen at `4605c73`; freeze header + §7 amendment recorded. Hash filled in draft §4.2 + Declarations (Zenodo DOI still TO-FILL at submission).
3. **Confirmatory runs seeds 10–19** → notebook §18 (cells 89–93, added 2026-07-17, commit `304265f`) does everything: clone-from-repo (real git hash), full matrix incl. cora surrogate/softmax + trunk_only everywhere, H4 + randomization ρ per nemaq run (h4.json), stats cell (H5/H1 one-sided Wilcoxon, Stouffer, Spearman δ-gain), export zip. User: run deps cell → §18 cells ONLY (skip %%writefile). Repo must be public or PAT-clone. Results → replace every `[PILOT]` in draft.
   Notebook audit done: 92-cell `NEMA_Q_colab_final.ipynb` (user's executed copy, gitignored) diffed vs repo — all %%writefile cells identical to src/; only stale cell 68 (unquoted `off` → YAML False, superseded by cell 69 in-notebook); tuned-config writes in sweep cells matched repo EXCEPT disease patience 200 → repo yamls fixed + prereg §7 amendment (commit `304265f`). Repo notebook = patched 97-cell version.
4. AI-disclosure statement (Stage 5 disclosure mode).
5. Flip repo private → public at submission (prereg verifiability) + Zenodo archive DOI.

## Key decisions log

- Pilot(0–9)/confirmatory(10–19) split = answer to "prereg after seeing data" problem; everything so far = exploratory pilot, honestly labeled.
- Tuning: validation-only, per-dataset, equal budget shape all models; test read once per locked config.
- Paper framing: honest nulls ARE the contribution; H5 phrased "matches or exceeds", never "quantum fails".
- Caveman output mode active (user pref).
- Stale untuned citeseer + degenerate disease runs still in `experiments/runs/` on Colab — move to `experiments/pilot/` before regenerating figures, else collect_scores mixes populations.

## Code state (local repo, all synced to notebook)

- `src/nemaq/analysis/`: `eda.py`, `figures.py`, `xai.py`, `plotstyle.py`, `stats.py` (+`geometry_interaction`).
- `qoa.py` fixed (enable_grad + deepcopy-cache clear). `fusion.py`/`nemaq.py`: `gate_bias_init` config knob. `trainer.py`: `return_model` flag. `loader.py`: Disease + ratio split.
- Configs: 4 datasets × {gcn,gat,hgcn,nemaq} + ablations {euclidean,frozen_random,trunk_only} per dataset. Citeseer nemaq+ablations tuned; disease tuned recipes NOT in yamls yet.
- Tests: `test_geometry_interaction.py` (passes locally), `test_analysis_figures.py` (Colab-only), `test_qoa.py` fixed (28 pass in Colab).
