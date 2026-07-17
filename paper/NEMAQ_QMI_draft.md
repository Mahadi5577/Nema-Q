# Component Accounting in Hybrid Quantum–Classical Graph Learning: Frozen Circuits Match Trained Ones, and Explainability Shows Why

**MD Nurol Amin** — Daffodil International University, Dhaka, Bangladesh

*Extended version of ICEQT'26 paper ICE7405 (NEMA-Q). Target: Quantum Machine Intelligence.*

> **DRAFT STATUS:** All numbers marked `[PILOT]` are from pilot seeds 0–9 and will be replaced by confirmatory runs (seeds 10–19) after preregistration freeze. `[TO-FILL]` marks values pending a run.

---

## Abstract

Hybrid quantum–classical models are typically evaluated by a single aggregate accuracy number, which cannot say *which* component earned it. We propose component accounting: an evaluation protocol that decomposes a hybrid architecture's performance into per-component contributions using parameter-matched configuration swaps, paired-seed statistics, and gradient telemetry, applied here to NEMA-Q, a hybrid graph network combining a hyperbolic encoder, a variational quantum circuit (PQC), and a classical bypass trunk under gated residual fusion. Across four node-classification datasets stratified by Gromov δ-hyperbolicity (Cora, Citeseer, Pubmed, Disease), the accounting yields three findings that a single accuracy column would hide. First, a frozen-random PQC matches or exceeds its trained counterpart on every dataset (Cora: +3.9 accuracy points, 10/10 seeds, Wilcoxon p=0.002, d=1.85 `[PILOT]`); the value the quantum branch adds, where it adds any, comes from a fixed nonlinear random projection rather than trained quantum features. Second, gated fusion fails catastrophically when a branch's compressed input loses per-node signal variance: on Citeseer the quantum branch's angle embedding collapses to near-constant output while contributing the largest gradients in the network, destabilizing training (0.508±0.063 vs. 0.717 for GCN `[PILOT]`); per-branch deep supervision restores trunk-level accuracy, and we state a measurable precondition for stable hybrid fusion. Third, the hyperbolic encoder outperforms a parameter-matched Euclidean control in pooled analysis, but the gains do not track δ-hyperbolicity — the standard tree-likeness motivation fails on this suite. All claims are validated by a Quantum Observable Attribution (QOA) explainability suite equipped with masking, perturbation, and model-randomization faithfulness checks (11 of 12 checks pass; the single failure itself localizes an input-dominated attribution regime). Code, run manifests, and preregistration are released.

**Keywords:** quantum machine learning, graph neural networks, explainability, hyperbolic geometry, variational quantum circuits, ablation methodology

---

## 1. Introduction

A recurring pattern in hybrid quantum–classical machine learning is the composite architecture: a quantum circuit embedded among classical modules, evaluated end-to-end, and reported with a single accuracy number. When such a model performs well, the quantum component absorbs the credit; when it performs poorly, the result is rarely published. Neither outcome tells us what the circuit actually contributed. Recent benchmarking critiques identify exactly this attribution gap — quantum models compared against untuned classical baselines, under undisclosed budgets, with no isolation of components — as a dominant methodological bottleneck in the field [refs: QML benchmarking cluster].

This paper treats the attribution question as the primary object of study. We take NEMA-Q — a hybrid graph neural network that fuses a hyperbolic (Poincaré-ball) encoder, a variational quantum circuit branch, and a classical GCN bypass through gated residual fusion — and subject it to *component accounting*: a protocol in which every architectural claim is paired with a control reachable by configuration swap alone, evaluated under paired seeds and identical splits, with per-branch gradient telemetry recorded throughout, and with attribution methods whose own faithfulness is tested rather than assumed.

The accounting produces results that are individually simple but jointly uncomfortable for standard narratives:

1. **A frozen-random quantum circuit is never worse than a trained one.** Under an identical scaffold, freezing the PQC's variational parameters at random values matches or exceeds training them on all four datasets, decisively so on Cora (+3.9 points, every seed, d=1.85 `[PILOT]`). The useful content of the quantum branch, where it exists, is a fixed random nonlinear projection — the quantum analogue of random-feature methods — not learned quantum structure.

2. **Hybrid fusion has a measurable failure precondition.** On Citeseer, the default recipe collapses 21 points below a plain GCN. Telemetry localizes the mechanism: the quantum branch's compressed angle embedding has the lowest input variance of any dataset, producing a branch that is nearly constant across nodes yet injects the largest gradients in the network into shared layers. We state the precondition (per-branch signal variance), show it is measurable before training, demonstrate the repair (per-branch deep supervision), and falsify an alternative explanation.

3. **The textbook motivation for hyperbolic encoders fails while the encoder itself succeeds.** Hyperbolic beats Euclidean in pooled analysis across the suite, but the gains *increase* with Gromov δ rather than decrease — the opposite of the tree-likeness story — and on the δ=0 tree dataset a plain GCN beats a dedicated hyperbolic GNN.

None of these findings is visible in an aggregate accuracy table, and the third is invisible without a δ-stratified multi-dataset design. Our contributions, in order of generality: (C1) the component-accounting protocol itself — configuration-swap controls, paired preregistered statistics, per-branch telemetry — as a reusable evidentiary standard for hybrid QML; (C2) the frozen-versus-trained control and its outcome, which we argue should become a mandatory baseline for any trained quantum module; (C3) Quantum Observable Attribution with a faithfulness harness that can fail, and once does; (C4) the fusion failure-mode analysis with its measurable preflight precondition.

**Relation to the conference version.** The ICEQT'26 paper introduced NEMA-Q and Quantum Observable Attribution on Cora only, with five seeds and no tuned baselines. This version answers every methodological objection raised in its review: tuned GCN/GAT/HGCN baselines under identical splits with parameter counts (§4); three additional datasets stratified by δ-hyperbolicity (§4.1); a faithfulness validation harness for QOA — masking, perturbation, and model-randomization checks (§8); paired statistical tests with multiple-comparison control for all ablation gaps (§4.3); and full reproduction detail via per-run manifests (§4.4). The frozen-vs-trained finding, the fusion failure-mode analysis, and the geometry-mechanism falsification are new.

## 2. Related Work

**Quantum graph learning.** Quantum graph neural networks span Hamiltonian-based constructions [Verdon 2019], equivariant circuit families [Mernyei 2022], and recent message-passing formulations analyzed in the Weisfeiler–Leman hierarchy [arXiv:2606.26873]. Most report end-to-end accuracy on small benchmarks; component-level attribution of the quantum contribution is rare. Our architecture is deliberately conservative — a small angle-embedded circuit with local Pauli-Z readout [Benedetti 2019, Schuld & Killoran 2019] — because the object of study is the accounting, not the circuit.

**Random and untrained quantum features.** Classically, random feature maps rival trained representations at a fraction of the cost [Rahimi & Recht 2007]. The quantum counterparts — quantum reservoir computing and quantum extreme learning machines — exploit untrained dynamics for information processing [Mujal 2021; Comms. Phys. 2023; npj QI 2025]. A recent no-free-lunch theorem shows that averaged over tasks, no untrained circuit outperforms any other [arXiv:2309.13967]. Our H5 result is complementary and sharper in one direction: for a *specific* trained architecture on *specific* tasks, training the circuit is empirically never better and sometimes reliably worse than freezing it — a statement about optimization in situ, not about task-averaged expressivity. Trainability obstructions offer a mechanism: barren plateaus and related landscape pathologies [McClean 2018; Cerezo 2021; Holmes 2022; Larocca 2025], compounded in our setting by an angle-embedding regime that concentrates inputs near zero rotation (§7).

**Dequantization and classical surrogates.** Trained PQC models can often be replaced by efficient classical surrogates [Schreiber 2023; Sweke 2023; Shin 2024], and some circuit families are classically simulable outright [arXiv:2408.12739]. Our surrogate control (NEMA-C) sits in this lineage: a parameter-matched classical branch substituted for the PQC. Component accounting extends the surrogate question from "can the quantum model be replaced after training?" to "which architectural slot, if any, did quantumness fill during training?"

**Hyperbolic graph learning.** Poincaré embeddings [Nickel & Kiela 2017] and hyperbolic GNNs [Chami 2019; Liu 2019] motivate negative curvature by the tree-like, scale-free structure of real graphs, standardly quantified by Gromov δ-hyperbolicity [Adcock 2013; review: arXiv:2202.13852]. Recent critical work finds that hyperbolic gains track geometry–task alignment rather than raw structural tree-likeness [arXiv:2602.01828]; our δ-stratified falsification (§6) provides independent, preregistered evidence for that revisionist view.

**Explainability for quantum models.** Quantum XAI is nascent: Shapley values over circuit gates [arXiv:2301.09138, QMI 2025], classical feature-importance adaptations [arXiv:2405.08917; Electronics 2024], representation-level frameworks [QRLaXAI, QMI 2025], and a general treatment of what explaining QML can and cannot mean [arXiv:2412.14753]. Quantum Observable Attribution differs in target: it attributes a *hybrid* model's per-node decisions to the measured observables at the quantum–classical interface — the boundary where quantum information becomes classical feature — rather than to gates or input features. Crucially, we subject QOA to the sanity-check discipline developed for classical saliency [Adebayo 2018; Hooker 2019] and adopted by the graph-XAI evaluation literature [arXiv:2208.09339; GInX-Eval arXiv:2309.16223; arXiv:2310.01820], which no prior quantum attribution work has done.

## 3. Architecture

NEMA-Q is a three-branch node classifier (Fig. 1). Given a graph $G=(V,E)$ with node features $x_i \in \mathbb{R}^F$, every branch $b$ maps $(X, E)$ to a node embedding $h_{b,i} \in \mathbb{R}^{e}$ of shared width $e=64$; ablations are configuration swaps, never code forks. We describe each branch, the fusion rule, and the training objective in the exact form implemented; every symbol below corresponds to a named tensor in the released code.

> **Figure 1.** NEMA-Q architecture. Three parallel branches — hyperbolic encoder (geo), variational quantum circuit (q), and classical GCN bypass (trunk) — map node features to width-$e$ embeddings. The bypass is the trunk of a gated residual fusion; geo and q enter through sigmoid gates initialized nearly closed ($\sigma(-2)\approx0.12$). Dashed boxes mark the configuration swaps that generate every control in the paper: geometry ∈ {hyperbolic, euclidean, off}, quantum_mode ∈ {pqc, frozen_random, surrogate, off}, fusion ∈ {residual, softmax}. Per-branch auxiliary heads (deep supervision) and per-branch gradient telemetry taps are shown in gray.

**Hyperbolic branch (geo).** An HGCN-style encoder on the Poincaré ball $\mathbb{B}^e_c = \{u \in \mathbb{R}^e : c\|u\|^2 < 1\}$ with learnable curvature $c$. Exponential and logarithmic maps at the origin,

$$\exp_0^c(v) = \tanh(\sqrt{c}\,\|v\|)\,\frac{v}{\sqrt{c}\,\|v\|}, \qquad \log_0^c(u) = \operatorname{artanh}(\sqrt{c}\,\|u\|)\,\frac{u}{\sqrt{c}\,\|u\|},$$

carry vectors between the ball and its tangent space at the origin. Raw features are first compressed Euclideanly, $t_i = \mathrm{clip}_{2.5}(W_f x_i)$, with tangent norms clipped at 2.5 before the exp-map — mapping high-dimensional sparse features (e.g. Cora's 1433-dim bags-of-words) directly onto the ball saturates points at the boundary where gradients vanish. Each of two graph-convolution layers then applies a Möbius linear transform followed by tangent-space aggregation:

$$u_i = (W \otimes_c h_i) \oplus_c \exp_0^c(\beta), \qquad h'_i = \exp_0^c\!\Big(\phi\Big(\tfrac{1}{|\mathcal{N}(i)|+1}\textstyle\sum_{j \in \mathcal{N}(i)\cup\{i\}} \log_0^c(u_j)\Big)\Big),$$

where $\otimes_c$ and $\oplus_c$ are Möbius matrix–vector product and addition, and $\phi$ is ReLU + dropout (identity in the final layer). The branch output is $\log_0^c$-mapped so fusion operates in a shared tangent space. Curvature $c$ is a learnable (positivity-constrained) parameter; its learned per-dataset value is itself reportable evidence for H1. The `euclidean` swap replaces the entire branch with a width-matched GCN encoder (control NEMA-E); `off` removes it.

**Quantum branch (q).** A classical compressor maps features to $n=4$ bounded rotation angles,

$$a_i = \pi \tanh(W_c x_i + \beta_c) \in [-\pi, \pi]^n,$$

which enter a depth-$L$ ($L=2$) strongly-entangling circuit [Schuld & Killoran 2019] via single-qubit $R_Y$ rotations:

$$|\psi_i(\theta)\rangle = U_{\text{ent}}(\theta)\,\Big(\textstyle\bigotimes_{k=1}^{n} R_Y(a_{ik})\Big)\,|0\rangle^{\otimes n},$$

with $U_{\text{ent}}$ the StronglyEntanglingLayers ansatz (per-layer general single-qubit rotations plus a ring of CNOTs; $|\theta| = 3nL$). The branch reads out the $n$ local Pauli-Z expectations and projects them to the branch width:

$$o_{ik} = \langle \psi_i(\theta) | Z_k | \psi_i(\theta) \rangle, \qquad h_{q,i} = W_o\, o_i + \beta_o.$$

The observable vector $o_i$ is the quantum–classical boundary that QOA attributes against (§8). Swaps: `frozen_random` — identical circuit with $\theta \sim \mathcal{U}[0, 2\pi)^{|\theta|}$ drawn once from a fixed seed and excluded from the optimizer (control NEMA-R); `surrogate` — parameter-matched classical MLP (NEMA-C); `off`. The circuit is deliberately small, with $n \le 8$, $L \le 4$ enforced as assertions in code, to stay inside barren-plateau safe budgets [Larocca 2025]; exact statevector simulation is used throughout, with finite-shot execution as a preregistered ablation.

**Classical bypass (trunk).** A plain two-layer GCN encoder producing $h_{\text{tr},i}$.

**Gated residual fusion.** Each branch is projected to the fused width $d=64$, $p_{b,i} = W_b h_{b,i}$, and the non-trunk branches enter as sigmoid-gated residuals on the trunk:

$$g_i = \sigma\big(W_g\,[\,p_{b_1,i}; \dots ; p_{b_B,i}\,] + \beta_g\big) \in (0,1)^{B-1}, \qquad z_i = p_{\text{tr},i} + \textstyle\sum_{b \ne \text{tr}} g_{i,b}\; p_{b,i},$$

with $W_g$ zero-initialized and $\beta_g$ initialized at $-2$ ($\sigma(-2)\approx0.12$), so training starts from "approximately a GCN" and exotic branches must earn gate mass. This is the graceful-degradation guarantee — tested, and on one dataset broken and repaired, in §7; the initialization is exposed as the config knob `gate_bias_init` that the repair tunes. Per-node gate values $g_{i,b}$ are recorded for telemetry and H4. The `softmax` fusion swap replaces the rule with all-branch competition, $g_i = \mathrm{softmax}(W_g[\cdot] + \beta_g) \in \Delta^{B-1}$, $z_i = \sum_b g_{i,b}\,p_{b,i}$ — retained as the instability ablation (§5.4). Class logits are $\hat{y}_i = W_h z_i$.

**Deep supervision.** Each branch carries an auxiliary linear classifier $A_b$ trained against the labels; the total objective on the training mask is

$$\mathcal{L} = \mathrm{CE}(\hat{y}, y) + w(t)\,\frac{1}{B}\sum_{b} \mathrm{CE}(A_b h_{b}, y), \qquad w(t) = \lambda_{\text{aux}}\Big(1 - \frac{t}{T}\Big),$$

with the auxiliary weight annealed linearly to zero over the $T$ training epochs: early anti-starvation [Lee 2015] without late-training interference. Section 7 quantifies why this term is load-bearing — $\lambda_{\text{aux}}$ is the single knob that repairs the Citeseer collapse.

Parameter matching between the PQC and surrogate branches is enforced by test at 15% tolerance; per-branch and total counts are recorded in every run manifest (Appendix A).

## 4. Experimental Protocol

### 4.1 Datasets and δ-stratification

We evaluate on Cora, Citeseer, Pubmed (Planetoid public splits) and Disease [Chami 2019] (synthetic SIR tree; stratified 30/10/60 split, no public split exists). Sampled four-point Gromov δ on the largest component stratifies the suite (Table 1): Disease δ=0.000 is the exact tree — the positive control where hyperbolic geometry must help if the tree-likeness mechanism is real — with Cora, Pubmed, and Citeseer progressively less hyperbolic. The suite simultaneously spans two potential confounders that §7 will need: feature density varies by two orders of magnitude (0.009–1.0) and largest-component fraction from 0.64 to 1.0. Full EDA (degree distributions, class balance, δ-estimation detail) in Appendix B.

**Table 1.** Dataset suite, δ-stratified. δ is sampled four-point Gromov hyperbolicity on the largest connected component; homophily is edge homophily; density is feature density (fraction of nonzero feature entries).

| Dataset | Nodes | Edges | Feat. | Classes | δ_mean | δ_max | Homophily | Density | Largest CC |
|---|---|---|---|---|---|---|---|---|---|
| Disease | 1,044 | 1,043 | 1,000 | 2 | 0.000 | 0.0 | 0.875 | 1.000 | 1.00 |
| Cora | 2,708 | 5,278 | 1,433 | 7 | 0.356 | 2.0 | 0.810 | 0.013 | 0.92 |
| Pubmed | 19,717 | 44,324 | 500 | 3 | 0.399 | 3.5 | 0.802 | 0.100 | 1.00 |
| Citeseer | 3,327 | 4,552 | 3,703 | 6 | 0.550 | 3.5 | 0.736 | 0.009 | 0.64 |

### 4.2 Hypotheses

Preregistered before the confirmatory runs (repository commit `[TO-FILL: hash]`):

- **H1 (geometry, two-part).** (a) Direction: hyperbolic > parameter-matched Euclidean (NEMA-Q vs. NEMA-E), pooled across datasets by Stouffer's Z over per-dataset one-sided Wilcoxon tests. The pooled test is primary because the conference pilot showed single-dataset comparisons are underpowered at n=10 seeds (Cora paired d=0.44 implies ~40 seeds for 80% power alone). (b) Mechanism: paired gain decreases with δ (Spearman over datasets). Part (b) is the standard literature claim; we register both so that either can fail independently. *Falsifiers:* pooled p ≥ 0.05 (a); no negative δ–gain correlation (b).
- **H2 (quantum vs. surrogate).** NEMA-Q > NEMA-C at ≤5 labels per class, with the gap closing at standard label rates; paired Wilcoxon per (dataset, label rate). *Falsifier:* surrogate matches the PQC at all label rates.
- **H3 (trainability telemetry).** PQC circuit-weight gradient variance stays within two orders of magnitude of the best classical branch's gradient variance on ≥9/10 seeds — a relative criterion, because absolute barren-plateau floors from the literature do not transfer to a trained hybrid; the classical branches are the matched healthy-gradient reference at the same loss scale. *Falsifier:* sustained relative variance collapse.
- **H4 (gate–attribution validity).** Fusion gate mass tracks leave-branch-out accuracy deltas: Spearman ρ > 0.5 per branch, pooled across datasets. *Falsifier:* ρ ≤ 0.5.
- **H5 (frozen vs. trained).** A frozen-random PQC matches or exceeds the trained PQC under an identical scaffold (one-sided Wilcoxon per dataset, Holm–Bonferroni within family). Registered after the Cora pilot that motivated it and before any run on the remaining datasets. *Falsifier:* trained significantly exceeds frozen on a majority of datasets.

### 4.3 Statistics

Ten seeds per configuration; seeds and splits are shared across models so every comparison is paired at the seed level, which removes between-seed variance from all reported differences. For a comparison between configurations A and B we form the per-seed differences $\Delta_s = \mathrm{acc}_A(s) - \mathrm{acc}_B(s)$ and report: the Wilcoxon signed-rank test on $\{\Delta_s\}$ (exact null distribution at n=10; one-sided where the hypothesis is directional) [Demšar 2006]; the seed win count; and a paired effect size $d = \bar{\Delta}/\mathrm{sd}(\Delta)$. Variance claims use Levene's test. Cross-dataset direction is pooled by Stouffer's method, $Z = \sum_j z_j / \sqrt{m}$ over the $m$ per-dataset one-sided tests [Stouffer 1949], and the primary hypothesis family {H1-pooled, H2, H5} is controlled by Holm–Bonferroni [Holm 1979]; per-dataset tests outside the registered family are descriptive. Pilot (seeds 0–9) and confirmatory (10–19) populations are disjoint; every number below is `[PILOT]` unless stated.

### 4.4 Tuning policy and reproducibility

Hyperparameters are selected on validation accuracy only, per dataset, with the same budget shape for all models on that dataset (grids in Appendix C); test accuracy is read once per locked configuration. This policy exists because the default recipe produced two artifacts that untuned comparisons would have misreported: baseline majority-class collapse on Disease (all baselines at 0.797 until the learning rate was tuned; 0.915 after) and the NEMA-Q training collapse on Citeseer (§7). Every run writes a manifest (resolved config, seed, git hash, environment, metrics, telemetry summary); all analysis code consumes manifests only.

## 5. Results I: Component Accounting

### 5.1 The waterfall

Table 2 reports the full Cora matrix `[PILOT]`. Baselines: GAT 0.823±0.009, GCN 0.819±0.009, HGCN 0.786±0.009. NEMA-Q (trained PQC, hyperbolic, residual fusion) reaches 0.708±0.021 — a 11.1-point deficit against GCN that the accounting decomposes exactly (Fig. 2):

| Stage | Accuracy | Paired step | p |
|---|---|---|---|
| GCN reference | 0.819 | — | — |
| + scaffold (trunk-only: fusion+heads, no exotic branches) | 0.777±0.014 | −4.2 | <0.001 |
| + frozen-random PQC | 0.747±0.026 | −3.0 | 0.002 |
| + training the PQC | 0.708±0.021 | −3.9 | 0.002 |

Three observations. The scaffold itself costs 4.2 points — and trunk-only is statistically indistinguishable from the dedicated hyperbolic baseline HGCN (p=0.157), so this is the price of the hyperbolic encoder, not an implementation defect. Attaching a random frozen circuit costs a further 3.0 points. *Training* that circuit costs 3.9 more — the single largest attributable step is the one that standard practice assumes is beneficial.

> **Figure 2.** Component-accounting waterfall on Cora `[PILOT]`. Each bar is a configuration swap from the previous one; annotations give the paired step size and Wilcoxon p. The decomposition sums exactly to the GCN→NEMA-Q deficit.

> **Figure 3.** Forest plots, one panel per dataset: test accuracy (mean ± 95% CI over 10 paired seeds) for all baselines, NEMA-Q, and every configuration-swap control under that dataset's locked recipe.

### 5.2 H5: frozen matches or beats trained, everywhere

| Dataset | Trained | Frozen | Paired diff | Wins | p | d |
|---|---|---|---|---|---|---|
| Cora | 0.708±0.021 | 0.747±0.026 | +3.9 | 10/10 | 0.002 | 1.85 |
| Citeseer (tuned) | 0.654±0.019 | 0.668±0.018 | +1.4 | 8/10 | 0.027 | 0.85 |
| Pubmed | 0.752±0.031 | 0.772±0.007 | +2.1 | 7/10 | n.s. | 0.72 |
| Disease (tuned) | 0.893±0.035 | 0.894±0.043 | +0.2 | 5/10 | 0.865 | — |

`[PILOT]` The direction never reverses. Where the hybrid struggles (Cora, Citeseer), freezing helps decisively; where the task is easy (Disease), training is merely useless. The registered falsifier — trained significantly better on a majority of datasets — is not approached on any dataset. (Pubmed: exact paired p `[TO-FILL: recompute at confirmatory]`; the pilot direction is 7/10 seeds.) Section 9 discusses mechanism; §7 supplies the input-regime half of it.

> **Figure 4.** H5 paired-seed plots, one panel per dataset: per-seed test accuracy of trained PQC (NEMA-Q) vs. frozen-random PQC (NEMA-R) connected by seed. Slope direction, not vertical position, carries the H5 verdict.

### 5.3 Remaining preregistered hypotheses

**H2 (surrogate).** The parameter-matched classical surrogate reaches 0.693±0.056 on Cora — statistically indistinguishable from the trained PQC in mean (p=0.46) but markedly less stable (2.7× the variance, Levene p=0.02) `[PILOT]`. The dequantization reading: nothing the trained circuit does on these tasks exceeds a matched classical branch; the full low-label-regime test of H2 runs with the confirmatory suite `[TO-FILL]`.

**H3 (trainability telemetry).** The relative criterion — PQC circuit-weight gradient variance within two orders of magnitude of the healthiest classical branch — passes on all inspected runs (`h3_pass_relative` true in manifests), including the collapsed Citeseer runs, which is itself diagnostic: H3 certifies gradients *flow*; §7 shows flowing gradients can still carry no per-node information. Trainability telemetry is necessary but not sufficient.

**H4 (gate–attribution validity).** Spearman correlation between fusion-gate mass and leave-branch-out deltas is computed per branch in the confirmatory suite `[TO-FILL: H4 table]`; the §7 finding of a uniform gate over a dead branch already bounds expectations — gates track branch *usage*, not branch *usefulness*, exactly when signal variance collapses.

### 5.4 Fusion stability

Softmax fusion is the variance story: 0.604±0.101 on Cora versus 0.708±0.021 residual — a 10.4-point mean gap (p=0.013) and a five-fold standard-deviation ratio (Levene p=0.008) `[PILOT]`. The classical surrogate exhibits the same instability under softmax competition, so this is a fusion pathology, not a quantum one. All headline configurations therefore use residual fusion, and the softmax column is retained as the ablation that justifies it.

> **Figure 5.** Seed-level stability, one panel per dataset: per-seed test accuracy distributions (strip + box) for residual vs. softmax fusion and for each control. Variance, not mean, separates the fusion modes.

## 6. Results II: Geometry — the Encoder Works, the Textbook Mechanism Does Not

Per-dataset paired gains (hyperbolic − Euclidean, identical recipes) `[PILOT]`:

| Dataset | δ | Gain | p (two-sided) | d |
|---|---|---|---|---|
| Disease | 0.000 | +0.2 | 0.695 | — |
| Cora | 0.356 | +2.1 | 0.322 | 0.44 |
| Pubmed | 0.399 | +1.4 | 0.275 | 0.40 |
| Citeseer | 0.550 | +3.3 | 0.004 | 1.02 |

**H1(a) direction:** all gains non-negative; Stouffer-pooled one-sided p≈0.004 `[PILOT]` — the hyperbolic encoder earns its place in the architecture. One caveat attaches to the pilot pooling: the Citeseer and Disease pairs ran under per-dataset tuned recipes (§4.4) while Cora and Pubmed used defaults, so the pooled statistic mixes recipe conditions; the confirmatory suite runs all four pairs under their locked recipes and supersedes this number.

> **Figure 6.** Geometry × δ. Left: paired hyperbolic−Euclidean gain (mean ± 95% CI) against dataset δ_mean; the preregistered H1(b) prediction is a negative slope, the observed trend is positive. Right: per-dataset paired-seed plots (NEMA-Q vs. NEMA-E).

**H1(b) mechanism: falsified.** The gains *rise* with δ (Spearman trend positive; the preregistered prediction was negative). The only significant per-dataset gain occurs on the *least* tree-like dataset, and on the exact tree — where the mechanism story predicts its maximum — the gain is indistinguishable from zero and a plain GCN outperforms the dedicated HGCN baseline by 3.1 points (0.915 vs. 0.884 `[PILOT]`). Whatever the hyperbolic encoder contributes on this suite, δ-hyperbolicity does not predict it; the Citeseer case suggests robustness to extreme feature sparsity as a candidate, consistent with the geometry–task-alignment reading of hyperbolic gains [arXiv:2602.01828]. We report this as a preregistered two-part outcome: direction confirmed, mechanism rejected — a combination a single-dataset study could not produce.

## 7. Results III: A Failure Mode of Gated Fusion, Diagnosed and Repaired

Under the default recipe, NEMA-Q collapses on Citeseer: 0.508±0.063 versus 0.717 for GCN, with early stopping firing within ~40 effective epochs, whole classes dropped (minimum per-class F1 reaching 0.0), and validation tracking test — a training failure, not overfitting `[PILOT]`.

**Mechanism.** Per-branch gradient telemetry from the failed runs shows a three-order-of-magnitude imbalance: hyperbolic-branch gradient variance ~10⁻¹¹ (starved), trunk ~10⁻⁸, quantum branch 10⁻⁷–10⁻⁶ (loudest in the network). The quantum branch's input is the culprit: Citeseer's sparse features (density 0.008, the suite's lowest) drive the compressed angle embedding to the lowest per-dimension variance of any dataset (σ=0.005 vs. 0.013–0.016), so the circuit outputs are nearly constant across nodes (output σ=0.013) while the branch's parameters still receive large gradients. The fusion gate admits this channel *uniformly* — gate value 0.177±0.005 across all nodes — a pathway carrying no per-node information but maximal gradient noise into the shared head. Notably, the entire angle range used is ±0.15 of the available ±π on every dataset: the embedding operates in a narrow near-linear regime everywhere, and Citeseer merely falls below the usable-variance floor. This observation independently rationalizes H5 (§9).

**Alternative tested and rejected.** Citeseer is also the most fragmented graph in the suite (largest component 64% of nodes). Fragmentation, however, does not explain the failure: in the collapsed model, isolated test nodes are classified no worse than connected ones (0.500, n=12, vs. 0.401, n=988) — a small isolated-node sample, but the effect required for fragmentation to explain a 21-point collapse would point sharply the other way.

**Repair.** The mechanism predicts the fix: strengthen per-branch supervision so branches learn discriminative features regardless of gate dynamics. A validation-only sweep confirms monotone recovery in the auxiliary-loss weight (val 0.505 → 0.661 as aux 0.3 → 2.0, with quieter gate initialization −4 and lr 0.005), restoring test accuracy to 0.654±0.019 — at trunk-only level (0.646±0.018), with no class collapse and normal convergence `[PILOT]`. The same recipe was applied to all Citeseer hybrid variants for the paired comparisons of §§5–6.

**Precondition.** Stable gated fusion requires each branch to retain per-node signal variance at its output; for angle-embedded quantum branches this is measurable *before training* from the compressed-input variance. We propose reporting this statistic as a standard preflight check for hybrid architectures.

## 8. Quantum Observable Attribution, With Its Own Audit

QOA attributes each node's predicted-class logit to the measured observables at the quantum–classical boundary. Let $o_i \in \mathbb{R}^n$ be node $i$'s Pauli-Z expectation vector (§3) and $\hat{y}_i$ its predicted class. QOA substitutes the observable tensor into the forward graph as a leaf variable and differentiates the predicted-class logit through projection, fusion, and head:

$$\mathrm{QOA}(i,k) \;=\; \frac{\partial\, \ell_{i,\hat{y}_i}}{\partial\, o_{ik}}\; \cdot\; o_{ik},$$

gradient × input over the observable vector. Two properties matter. First, the attribution target is neither the circuit's gates [arXiv:2301.09138] nor the input features, but the interface where quantum information becomes classical feature — so QOA answers the hybrid-specific question: *which physical measurements does the classical readout actually use?* Second, because $o_i$ is recomputed exactly (statevector simulation) and substituted as a leaf, the method needs no relaxation or sampling; it is exact for the model as trained.

The conference version's reviewers noted, correctly, that gradient saliency without faithfulness evidence is decoration. The extension supplies a three-check harness, run per dataset on the trained pilot models:

1. **Masking faithfulness:** zeroing the top-attributed observable per node (k=1) must reduce the predicted-class logit more than zeroing the bottom-attributed one: pass iff mean drop(top) > mean drop(bottom) over test nodes.
2. **Perturbation test:** Gaussian noise (σ=0.25) injected on the top-attributed observable must flip more predictions than on the bottom-attributed one: pass iff flip-rate(top) > flip-rate(bottom).
3. **Model-randomization sanity check** [Adebayo 2018]: re-randomizing fusion and head must collapse the attribution pattern: pass iff Spearman |ρ| between original and randomized attributions < 0.5. A method that survives randomization is reading input structure, not the model.

Across the four datasets, QOA passes 11 of 12 checks `[PILOT]`: masking and perturbation faithfulness hold everywhere (top-attributed observables produce strictly larger logit drops and higher flip rates than bottom-attributed ones on every dataset; per-dataset magnitudes in Fig. 7), and randomization passes on Cora, Citeseer, and Pubmed. The single failure is Disease, where attributions remain correlated with the randomized model's (|ρ| ≥ 0.5): on a two-class task whose quantum branch operates in the near-constant regime of §7, the attribution pattern is dominated by input structure rather than learned parameters — precisely the pathology the Adebayo check exists to expose. We therefore report QOA as validated on the three citation networks and flag Disease attributions as input-dominated rather than model-specific. A harness that can fail, and does once, is the point: attribution methods should earn trust per dataset, not by construction.

> **Figure 7.** QOA faithfulness harness, one panel per dataset: masking logit-drops (top vs. bottom observable), perturbation flip rates (top vs. bottom), and the randomization Spearman ρ against its 0.5 threshold. Disease fails the randomization check only.

We acknowledge the out-of-distribution critique of masking-style fidelity metrics [GInX-Eval]: zeroed observables are off-manifold inputs. The randomization check is immune to this critique, which is why the harness requires all three verdicts rather than any one.

The broader XAI suite (Appendix D) ports the conference version's five methods to the multi-dataset setting: integrated gradients on input features; QOA class×observable heatmaps; Poincaré-disk visualization of the geo branch (2-D PCA of the tangent output, exp-mapped at learned curvature); gradient-attributed per-node quantum contribution ratio r_Q; and exact branch-level Shapley values (2^B coalitions over branches — exact, unlike sampled KernelSHAP, and aligned with the leave-branch-out ground truth of H4).

## 9. Discussion

**Why frozen ≥ trained is the expected result, in hindsight.** Three independent lines converge. Barren-plateau theory predicts poor trainability of variational parameters even in small circuits when gradients are noisy and loss surfaces flat [Larocca 2025]. The no-free-lunch result for untrained circuits [arXiv:2309.13967] establishes that random circuits are not, on average, handicapped as feature maps. And §7's regime analysis shows the angle embedding confines the circuit to a small neighborhood of the identity, where its trainable parameters have little leverage but their gradients still inject noise into shared layers — the network learns *around* a fixed random projection faster than it can learn *through* a moving one. The practical corollary for hybrid design: if a quantum branch is included, freezing it is the stronger default, and any claim that training the circuit helps should be required to beat that default under paired statistics.

**What component accounting buys.** Every headline in this paper is a difference between two configuration swaps. The protocol's cost is linear in the number of controls; its benefit is that claims become falsifiable at the component level. We suggest reviewers of hybrid QML papers request, at minimum: a frozen-parameter control for any trained quantum module, a parameter-matched classical surrogate, and per-branch gradient telemetry.

**Limitations.** All quantum execution is exact statevector simulation; finite-shot and hardware-noise behavior are preregistered follow-ups, and nothing here speaks to hardware readiness. Results are pilot-tagged pending the confirmatory seed population; the preregistration separates the two. Four datasets support the pooled geometry analysis only weakly (Spearman over four points); the δ-trend rejection is robust in sign but not in magnitude. The auxiliary-weight repair on Citeseer selected a grid-edge value (2.0), and the tuning budget, though equal-shaped across models, was necessarily modest. Finally, NEMA-Q's absolute accuracy trails tuned classical baselines on every standard benchmark — this paper quantifies and explains that deficit rather than claiming advantage.

## 10. Conclusion

Component accounting turns a hybrid model from a monolithic claim into an auditable ledger. Applied to NEMA-Q, the ledger reads: the hyperbolic encoder pays for itself (though not for the reason the literature gives), the fusion scaffold has a measurable and repairable failure mode, and the variational quantum circuit — the component the architecture is named for — contributes most when it is not trained at all. We offer the protocol, the instrumentation, and the preregistered-null discipline as the paper's primary contribution, and the frozen-circuit finding as its cautionary headline.

---

## Declarations

**Data availability.** All datasets are public (Planetoid; Disease from the HGCN repository). Code, configuration files, run manifests, and the preregistration document are available at `[TO-FILL: repository URL + archived DOI]`.

**Author contributions (CRediT).** MD Nurol Amin: Conceptualization, Methodology, Software, Investigation, Formal analysis, Writing — original draft, Writing — review & editing.

**Conflict of interest.** The author declares no competing interests.

**Funding.** No external funding.

**AI usage disclosure.** `[TO-FILL at disclosure mode: venue-appropriate statement — code assistance and drafting support with human verification of all results and citations]`

---

## Appendix A: Parameter counts and per-seed tables `[from manifests]`
## Appendix B: EDA — degree distributions, class balance, δ estimation detail
## Appendix C: Tuning grids and validation traces (sweeps 1–3, Disease baseline grid)
## Appendix D: XAI suite figures for all datasets
## Appendix E: Preregistration document (frozen text + amendment log)
