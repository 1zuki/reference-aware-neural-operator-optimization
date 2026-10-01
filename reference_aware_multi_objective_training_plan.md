# When Numerical References Disagree: Negative Transfer and Optimization Interference in Solver-Conditioned Neural Operators

## Purpose and locked research position

This document is the detailed research plan for the DS106 Optimization and Applications project and its possible SciML continuation.

The project is **not locked around inventing a generic optimizer**. It is locked around an empirical question:

> **Does observable disagreement between matched numerical references predict when parameter sharing causes negative transfer in a solver-conditioned neural PDE surrogate?**

The conditional optimization question is:

> **If it does, can that information improve the trade-off between shared learning and reference-specific specialization?**

The hypothesized pathway is:

\[
\text{matched target disagreement}
\rightarrow
\text{gradient/update interference}
\rightarrow
\text{sharing cost or negative transfer}.
\]

Each arrow is a hypothesis. This is a mechanistic pathway in an observational matched-reference study, not a causal claim. Stronger causal conclusions would require deliberate interventions on numerical-reference discrepancy and are deferred. A negative pairwise gradient cosine is only a local optimization diagnostic. Negative transfer must be defined using held-out performance relative to a specialization control.

The project uses `ReGRO` as an internal family name and retains `ReGRO-C` as a candidate mechanism name only. The constrained projection itself is not claimed as novel before a scoped literature review and the decision gates.

---

## 1. Fit with DS106 and publication path

The course asks for an 8–15-page Elsevier-style report on either an algorithm improvement or a fair comparison of optimization algorithms. This project fits the stronger algorithm-improvement path while retaining a rigorous evaluation fallback.

The course version teaches:

- multi-objective optimization;
- constrained updates and feasible sets;
- gradient geometry and preconditioning;
- negative transfer and parameter sharing;
- experimental design, controls, and uncertainty;
- reproducible scientific-ML evaluation.

The publication ladder is evidence-gated:

1. **DS106:** matched-reference sharing diagnostic plus controlled baseline study;
2. **method or workshop paper:** only if disagreement carries information and a candidate method improves the trade-off;
3. **thesis or serious journal continuation:** second PDE system, stronger causal controls, theory or calibration analysis, and broader replication.

No Q1, high-Q2, or multi-paper outcome is promised in advance.

---

## 2. Connection to the parent tsunami benchmark

The parent `tsunami-surrogate` project studies neural PDE surrogates for tsunami-like wave propagation. Identical bathymetry-source scenarios are evaluated by three numerical references:

- Hydrostatic;
- MUSCL-HR;
- a study-defined linear weakly dispersive Boussinesq-type reference.

The reference outputs are aligned at common requested times over finite-horizon trajectories. They are related numerical targets, not interchangeable labels or physical ground truth. Their differences can combine equation-set, discretization, time-integration, boundary, and sampling effects.

The parent manuscript's main question is how reference choice changes surrogate evaluation. Its training procedure is conventional enough to serve as a controlled baseline. This project asks whether the matched reference structure changes joint optimization and the cost of sharing. The parent benchmark and previous results are prior context; the optimization analysis, controls, and experiments must be new.

### 2.1 Primary model

Use the solver-conditioned mapping:

\[
f_\theta(x,r)\rightarrow y_r,
\]

where `x` is the common scenario and `r` identifies the requested numerical reference.

The solver-anonymous model is a secondary limitation/control. Without `r`, identical inputs can correspond to multiple targets and the problem has an irreducible conditional-ambiguity component. Its poor performance cannot by itself prove optimization interference.

### 2.2 Baseline contract

Before running new experiments, record the live parent values for:

- benchmark version and data lineage;
- train, validation, and test scenario identities;
- normalization and denormalization;
- model architecture and parameter count;
- horizon-weighted loss;
- AdamW learning rate, weight decay, clipping, schedule, batch size, stopping rule, and checkpoint criterion;
- hardware and update budget.

Do not copy private parent data, checkpoints, credentials, or unrelated artifacts into this repository.

---

## 3. Formal problem setup

Let the reference set be

\[
\mathcal R=\{H,M,B\}
\]

for Hydrostatic, MUSCL-HR, and Boussinesq. For a matched batch, define a reference-specific horizon-weighted loss:

\[
L_r(\theta)=
\frac{\sum_{t=1}^{T}w_t\,\operatorname{MSE}\left(f_\theta(x,r)^{(t)},y_r^{(t)}\right)}{\sum_{t=1}^{T}w_t}.
\]

The ordinary pooled objective is

\[
L_{\mathrm{mean}}(\theta)=\frac{1}{|\mathcal R|}\sum_{r\in\mathcal R}L_r(\theta),
\]

with reference-specific gradients

\[
g_r=\nabla_\theta L_r(\theta).
\]

The pooled gradient assumes that the average direction is a good update for all references. The study tests when that assumption fails and whether the failure is harmful.

---

## 4. Three distinct quantities

### 4.1 Matched target disagreement

For the same scenario `x` and requested horizon `t`, define

\[
D_{ij}(x,t)=d\left(y_i(x,t),y_j(x,t)\right),
\]

where `d` is a declared metric such as denormalized RMSE or relative `L2`. Compare absolute disagreement with a scale-normalized version such as

\[
D^{\mathrm{rel}}_{ij}(x,t)=
\frac{\lVert y_i(x,t)-y_j(x,t)\rVert}
{\tfrac12(\lVert y_i(x,t)\rVert+\lVert y_j(x,t)\rVert)+\epsilon}.
\]

Do not use the relative form alone because it can be unstable near zero. Preserve the pairwise and local structure where possible; do not reduce everything immediately to one scalar per reference.

### 4.2 Parameter-space gradient geometry

For two references, define

\[
C_{ij}=\frac{g_i^\top g_j}{\|g_i\|\,\|g_j\|+\varepsilon}.
\]

`C_ij < 0` indicates local gradient conflict. It does not establish poor generalization.

### 4.3 Negative transfer or sharing cost

Compare the shared conditioned model with specialization controls. For local metric `e_r^m(x,t)` under model/control `m`, define the aligned local sharing costs

\[
NT_r^{\mathrm{ind}}(x,t)=e_r^{\mathrm{shared}}(x,t)-e_r^{\mathrm{independent}}(x,t),
\]

\[
NT_r^{\mathrm{head}}(x,t)=e_r^{\mathrm{shared}}(x,t)-e_r^{\mathrm{shared\ trunk+heads}}(x,t).
\]

The aggregate headline measures remain

\[
NT_r^{\mathrm{ind}}=E_r^{\mathrm{shared}}-E_r^{\mathrm{independent}},
\qquad
NT_r^{\mathrm{head}}=E_r^{\mathrm{shared}}-E_r^{\mathrm{shared\ trunk+heads}}.
\]

The independent-model gap includes capacity, representation, regularization, and optimization effects. Therefore also compare a shared trunk with reference-specific heads and report parameter, FLOP, storage, and training-compute differences. Analyze aligned quantities `D_{rj}(x,t)` and local `NT_r^k(x,t)` together. Scenario and horizon observations are repeated measurements nested within seed and solver pair, not independent solver relationships; the three-reference benchmark contains only three solver pairs.

For directional transfer, periodically form a virtual update `d_i` from reference `i` alone and evaluate

\[
T_{i\rightarrow j}=L_j(\theta+d_i)-L_j(\theta).
\]

`T_{i\rightarrow j}<0` means that the update from `i` helps `j`; `T_{i\rightarrow j}>0` means that it harms `j`. This established diagnostic captures asymmetric transfer and is not claimed as a novel method.

The primary research object is the relationship among `D`, shared-parameter `C`, directed update effects, and local or aggregate `NT`, not any one of them in isolation.

---

## 5. Central hypotheses and nulls

### H1: disagreement is associated with optimization interference

Matched `D_ij(x,t)` is associated with shared-parameter gradient/update conflict or realized finite-step loss harm after accounting for training phase, scale, and scenario difficulty.

### H2: disagreement predicts sharing cost only if it carries information

Matched disagreement predicts local or aggregate `NT` after controlling for solver identity, horizon, scenario difficulty, loss scale, shared-parameter gradient/update norms and cosines, training phase, and model capacity. A stratified shuffled-disagreement control should remove scenario correspondence while preserving marginal distributions.

### H3: not every conflict should be removed

Uniform conflict-free updates may suppress useful reference-specific behavior. If disagreement is informative, a candidate method should preserve specialization where the matched targets diverge while correcting avoidable interference where they agree.

### H4: a candidate method improves a trade-off, not just a mean

The relevant outcome includes mean error, worst-reference error, per-reference error, sharing cost, and resource cost. A lower mean that sacrifices one reference is not a clear success.

### Null hypotheses

- ordinary AdamW and equal scalarization are already sufficient;
- negative cosine is unrelated to held-out negative transfer;
- matched disagreement adds no information beyond identity and generic gradient statistics;
- any apparent gain comes from capacity, regularization, update count, or random variation.

The study is successful if it can support or reject these nulls cleanly.

---

## 6. Stage 0: freeze the scientific contract

Before method implementation:

1. inspect the live parent training and evaluation code;
2. record all data, normalization, architecture, loss, seed, and checkpoint boundaries;
3. confirm how a matched batch is formed across references;
4. confirm how per-reference gradients can be computed without changing the forward model;
5. identify genuinely shared parameters for gradient geometry. In the shared-trunk/reference-head model, compute primary cosine and interference metrics on the shared trunk only; report conditioning and reference-specific head blocks separately rather than treating their disjoint gradients as meaningful conflict or similarity;
6. reserve a fresh final-test pool, or explicitly separate prior-test evidence from new method selection;
7. define compute, memory, update, and storage accounting;
8. define the final baseline list before looking at final-test results.

This stage protects against leakage and makes the later comparison interpretable.

---

## 7. Stage 1: sharing and gradient diagnostic

Run ordinary AdamW on the conditioned model. At fixed intervals log:

- `L_H`, `L_M`, `L_B`;
- per-reference gradients and norms on genuinely shared parameters;
- `C_HM`, `C_HB`, `C_MB`;
- pairwise conflict frequency and duration;
- directed virtual update effects `T_i_to_j` and their asymmetry;
- actual AdamW candidate update and applied update;
- finite-step changes `L_r(theta+d)-L_r(theta)`;
- matched `D_ij(x,t)` and horizon-wise summaries;
- absolute and normalized disagreement;
- scenario-difficulty controls such as target energy, source amplitude, spatial variance, and baseline single-reference error;
- validation error for each reference;
- seed, batch, epoch, checkpoint, and configuration identifiers.

Use the following model controls:

1. fully shared conditioned model;
2. shared trunk plus reference-specific heads;
3. independently trained reference-specific models.

The diagnostic must answer:

1. How often do gradients conflict?
2. Which reference pair and horizon are affected?
3. Does disagreement predict conflict or realized harm?
4. Does the relationship persist across seeds and training phases?
5. Does sharing measurably hurt held-out performance?
6. How much of the gap is explained by capacity or compute?

### Gate A decision

If sharing cost is absent, report the null sharing result and stop the optimizer branch. If sharing cost is present, continue to the information test.

---

## 8. Stage 2: information test with matched and shuffled disagreement

The crucial control is a stratified shuffle. For each solver pair and horizon, permute the scenario correspondence used to form `D_ij` while preserving its marginal distribution. Compare:

\[
D_{ij}^{\mathrm{matched}}(x,t)
\]

with

\[
D_{ij}^{\mathrm{shuffled}}(x,t).
\]

Test whether matched disagreement predicts:

- per-reference sharing cost;
- finite-step loss harm;
- update conflict;
- eventual held-out error;
- benefit or harm from a candidate correction.

Use nested models such as

\[
M_0:\ NT\sim\text{solver pair}+\text{horizon}+\text{scenario difficulty}+\text{capacity},
\]

\[
M_1:\ M_0+\text{shared-parameter gradient/update geometry},
\qquad
M_2:\ M_1+D_{ij}.
\]

Test whether matched disagreement predicts local `NT_r^k(x,t)`, aggregate `NT_r^k`, directed update effects, realized finite-step harm, or update interference. Use grouped or hierarchical uncertainty by scenario, solver pair, horizon, and seed. Compare out-of-sample predictive value and report effect sizes and intervals, not only scatter-plot correlations. Do not treat thousands of scenario/horizon rows as thousands of independent solver pairs.

### Gate B decision

If matched disagreement adds predictive information beyond confounds and shuffled disagreement loses that information, continue to a candidate optimizer. If not, retain the strongest diagnostic result and do not force disagreement-aware routing.

---

## 9. Stage 3: candidate update-alignment mechanism

Only after Gates A and B pass, define the smallest candidate mechanism. The candidate family is:

> an update-level feasible-set or update-alignment method whose relaxation is informed by pairwise matched-reference geometry.

The original scalar harm rule is not assumed. A candidate must explicitly justify:

- how pairwise disagreement maps to constraints or relaxation;
- why the sign and direction of the mapping are appropriate;
- whether the signal is global, scenario-dependent, horizon-dependent, or spatial;
- whether it operates on raw gradients or the actual AdamW proposal;
- how optimizer state is handled after modifying a stateful update;
- how first-order predictions are validated against finite-step losses.

### 9.1 Candidate mathematical template

If an update-level feasible set is retained, use a generic form:

\[
\min_d\;\frac12\|d-d_0\|_{M_t}^2
\quad\text{subject to}\quad
d\in\mathcal F(D,g,\text{state}),
\]

where `d_0` is the ordinary optimizer proposal and `M_t` is its declared metric. The feasible set must be defined by the tested pairwise geometry, not by an unvalidated scalar per-reference sacrifice rule.

This mathematical template overlaps with established constrained-update families. Its potential contribution is the tested use of matched numerical-reference geometry, if that signal survives the controls.

### 9.2 Required candidate ablations

- zero or fixed relaxation;
- true matched disagreement;
- stratified shuffled disagreement;
- reversed or monotonicity-check mapping;
- global versus local/horizon signal if feasible;
- projection only versus projection plus optimizer-state alignment;
- first-order predicted harm versus realized finite-step harm.

### Gate C decision

Continue only if the candidate improves the sharing/specialization trade-off over strong baselines on fresh data across seeds and its extra cost is reported.

---

## 10. Baselines and prior-art boundary

The minimum comparison is:

1. AdamW with equal scalarization;
2. simple or random weighting;
3. PCGrad or CAGrad;
4. GUA or the closest available actual-update projection baseline;
5. the candidate disagreement-conditioned method, if justified.

Depending on compute, include ConFIG, GradVac, GEM, MGDA, GradNorm, or metric-aware multi-objective Adam in the related-work or comparison set. Do not include many methods with inconsistent budgets.

GEM-style constrained updates, gradient surgery, actual-update projection, adaptive metric handling, task-affinity or directed update-effect measures, and HARMONIC-style feasible-set methods are related prior-art families. GUA and MAdam are recent preprints as of this plan. Verify bibliographic details and venue status in the scoped literature review before formal manuscript claims.

The novelty hypothesis is not `we project gradients`. It is:

> **matched numerical-reference geometry may provide actionable information about how strongly cross-reference interference should be corrected or relaxed.**

---

## 11. Capacity, compute, and fairness controls

For every model and optimizer report:

- parameter count;
- active and total training FLOPs if available;
- optimizer update count;
- wall-clock time;
- peak memory;
- optimizer-state memory;
- number of forward/backward passes per update;
- batch construction and effective sample count.

The independent baseline has more total capacity than one shared model. Use the shared-trunk/reference-head intervention and resource-matched analyses to avoid calling every joint-versus-independent gap optimization interference.

The candidate must not receive more tuning or final-test exposure than the baselines.

---

## 12. Metrics

### Accuracy and sharing cost

For each reference report denormalized RMSE, MAE, relative `L2`, per-horizon error, and supported waveform or gauge metrics. Report absolute and normalized target disagreement, scenario-difficulty summaries, and:

\[
E_{\mathrm{mean}}=\frac13\sum_r E_r,
\qquad
E_{\mathrm{worst}}=\max_r E_r,
\]

\[
NT_r^{\mathrm{ind}}=E_r^{\mathrm{shared}}-E_r^{\mathrm{independent}},
\qquad
NT_r^{\mathrm{head}}=E_r^{\mathrm{shared}}-E_r^{\mathrm{shared\ trunk+heads}}.
\]

Also report local `NT_r^{ind}(x,t)` and `NT_r^{head}(x,t)` and the corresponding aggregate specialization and head-sharing gaps.

### Optimization behavior

- pairwise cosine distributions;
- cosine distributions computed on genuinely shared parameters, with conditioning/head blocks reported separately;
- conflict frequency and duration;
- gradient and update norms;
- directed update effects `T_i_to_j` and their asymmetry;
- finite-step per-reference loss changes;
- convergence and checkpoint curves;
- candidate projection distance or constraint residual;
- optimizer-state drift or alignment diagnostics.

### Uncertainty and robustness

- at least three seeds for the course core when feasible;
- five seeds for a serious continuation when feasible;
- paired scenario-level bootstrap or other justified intervals;
- distribution shift and resolution transfer only under a preserved parent protocol;
- fresh final-test evaluation after method selection.

---

## 13. What is deferred

The following are deliberately outside the DS106 core:

- learned bilevel routing or temporal loss schedules;
- geometry-preservation loss as the main objective;
- horizon-wise, spatial, or spectral optimizer routing;
- disagreement-aware curriculum learning;
- multiple neural-operator backbones;
- Burgers, Darcy, Navier-Stokes, or another PDE benchmark;
- broad Bayesian optimizer aggregation;
- five-seed journal-scale replication;
- theory of optimal disagreement-to-slack calibration.

These become continuation options only after the core result is stable.

---

## 14. Fallback research paths

If the gradient/disagreement gate fails, the project remains viable:

### 14.1 Diagnostic/evaluation study

Characterize sharing cost, gradient geometry, matched versus shuffled disagreement, and the failure of conflict as a predictor of transfer. This is a valid DS106 and potentially useful SciML result.

### 14.2 Bilevel temporal-loss optimization

The parent loss uses hand-designed horizon weights. A later study could learn a smooth positive weight schedule with an inner model-training problem and an outer validation objective. This is deferred because it changes the question and adds hypergradient complexity.

### 14.3 Spectral or Sobolev preconditioning

A later study could use FNO mode structure or derivative/frequency losses to design a structured preconditioner. Adding an FFT term alone is not a sufficient novelty claim.

### 14.4 Evaluation-only optimizer comparison

If implementation time is short, compare two to five optimizers under a strict fairness protocol. This is a valid course fallback with a lower publication ceiling.

---

## 15. Report structure

The 8–15-page report can use:

1. Abstract and research question.
2. Introduction and relation to the parent benchmark.
3. Related work and prior-art boundary.
4. Multi-reference setup and measured quantities `D`, `C`, directed update effects, and `NT`.
5. Sharing and gradient diagnostic.
6. Candidate method, only if Gates A and B pass.
7. Experimental protocol, capacity controls, and leakage controls.
8. Results and uncertainty.
9. Ablations, limitations, and fallback interpretation.
10. Conclusion and evidence-based continuation decision.

The report should be written around the observed phenomenon. It must remain honest if the candidate optimizer is unnecessary or ineffective.

---

## 16. Pseudocode-level workflow

```text
Input:
    matched batch of scenarios
    labels y_H, y_M, y_B
    solver-conditioned model f_theta(x, r)

Stage 1: ordinary AdamW diagnostic
    1. Compute predictions and reference-specific losses L_H, L_M, L_B.
    2. Compute per-reference gradients g_H, g_M, g_B.
    3. Compute target disagreement D_ij(x, t).
    4. Log cosines, norms, candidate/applied updates, and finite-step loss changes.
    5. Compare shared, shared-trunk/head, and independent controls.

Stage 2: information test
    6. Construct stratified shuffled D_ij while preserving pair/horizon marginals.
    7. Test matched versus shuffled predictors of sharing cost and update harm.

Stage 3: conditional method test
    8. If Gates A and B pass, define the smallest candidate feasible set or update alignment.
    9. Compare AdamW, simple weighting, PCGrad/CAGrad, GUA, and the candidate.
   10. Include fixed, shuffled, reversed, and state-alignment ablations.
   11. Select on validation data only.
   12. Evaluate once on fresh final-test data and report paired uncertainty and resource cost.
```

---

## 17. Success criteria

A strong result can be any of the following, provided it is reproducible and carefully controlled:

1. matched disagreement predicts sharing cost or realized update harm beyond generic diagnostics;
2. the relationship disappears under stratified shuffling;
3. a candidate update method improves mean and/or worst-reference behavior without suppressing valid specialization;
4. a null result shows that gradient conflict or disagreement is not a useful predictor;
5. a negative method result identifies why generic conflict correction is inappropriate in this setting.

The project does not require a large percentage improvement. It requires a defensible answer to the central question.

---

## 18. Final execution order

1. Freeze the parent manuscript and reported results as prior work.
2. Inspect the live parent training code and scientific contract.
3. Build the matched AdamW diagnostic and the three sharing controls.
4. Run Gates A and B using training/validation data only.
5. Freeze the baseline list and final-test protocol.
6. Implement the smallest candidate update-alignment mechanism only if justified.
7. Run matched baselines, ablations, seed replications, and resource accounting.
8. Evaluate once on fresh final-test data, or clearly separate prior-test evidence if fresh data are impossible.
9. Write the DS106 report around the measured phenomenon.
10. Decide whether the evidence supports a workshop, journal, or thesis continuation.

---

## 19. Source and provenance notes

- [`sources/a.md`](sources/a.md) and [`sources/b.md`](sources/b.md) are temporary-chat critiques preserved as planning material.
- [`sources/shared_chatgpt_response.md`](sources/shared_chatgpt_response.md) archives the later shared discussion.
- [`sources/shared_chatgpt_response_earlier.md`](sources/shared_chatgpt_response_earlier.md) archives the earlier shared discussion.
- The parent benchmark is `/home/izu/Projects/tsunami-surrogate`.
- Shared-chat literature and venue claims are historical until verified against primary sources.
