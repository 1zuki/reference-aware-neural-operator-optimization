# Project Decision and Roadmap

## 1. Locked decision

This project will investigate whether **observable disagreement between matched numerical references predicts when parameter sharing causes negative transfer in a solver-conditioned neural PDE surrogate**.

The conditional optimization question is whether that signal can improve the trade-off between shared learning and reference-specific specialization.

The research program is evidence-gated. The project does not assume that gradient conflict is harmful, that target disagreement implies legitimate conflict, or that a new optimizer is needed. A candidate optimizer is implemented only after the diagnostic evidence supports it.

The DS106 deliverable is the first bounded study. A workshop, journal, or thesis continuation is a later decision based on reproducible results, not a promise made in advance.

## 2. Project identity

- **Repository name:** `reference-aware-neural-operator-optimization`
- **Repository description:** Research code and experiments for disagreement-aware multi-objective optimization of solver-conditioned neural PDE surrogates trained on matched numerical references.
- **Working paper title:** **When Numerical References Disagree: Negative Transfer and Optimization Interference in Solver-Conditioned Neural Operators**
- **Internal method family:** `ReGRO` (Reference-Geometry-Aware Gradient Routing)
- **Candidate mechanism:** `ReGRO-C`, a disagreement-conditioned relaxed update-alignment method, only if the decision gates justify it.

The paper title is phenomenon-first. The projection or QP is a candidate mechanism, not the presumed novelty.

## 3. Central model and mechanistic hypothesis

The primary setting is a solver-conditioned model:

```text
f_theta(x, r) -> y_r
```

where `x` is a matched scenario and `r` identifies Hydrostatic, MUSCL-HR, or Boussinesq. Let `L_r` be the reference-specific loss and `g_r = grad_theta L_r`.

For matched scenario and horizon, define target disagreement:

```text
D_ij(x, t) = distance(y_i(x, t), y_j(x, t))
```

The study tests the hypothesized pathway:

```text
matched target disagreement
        -> gradient/update interference
        -> sharing cost or negative transfer
```

The first arrow concerns optimization geometry. The second must be established with held-out performance. This is a mechanistic hypothesis in an observational matched-reference study, not a causal claim; deliberate interventions on reference discrepancy are deferred. Negative pairwise cosine is a diagnostic, not a definition of negative transfer.

## 4. Why this setting is appropriate

The parent `tsunami-surrogate` project supplies:

- identical scenarios evaluated by multiple numerical references;
- common requested times and finite-horizon trajectories;
- reference-specific and cross-reference diagnostics;
- an existing solver-conditioned neural-operator setup;
- a conventional AdamW procedure suitable as a controlled baseline.

The parent manuscript asks how numerical-reference choice changes surrogate evaluation. This project asks whether the same reference structure changes joint optimization and the cost of sharing. The benchmark, data-generation framework, and prior results remain prior context; the new optimization analysis and experiments must be new.

## 5. Decision gates

### Gate A: problem

Does parameter sharing hurt held-out performance consistently enough to matter?

Required comparisons:

1. fully shared solver-conditioned model;
2. shared trunk with reference-specific heads;
3. independently trained reference-specific models.

The analysis must report parameter count, FLOPs or training compute, optimizer updates, and storage so that a joint-versus-independent gap is not overinterpreted as pure gradient interference.

If Gate A fails, report the null sharing result and stop the optimizer story.

### Gate B: signal

Does matched disagreement explain when sharing hurts beyond solver identity, horizon, scenario difficulty, loss magnitude, shared-parameter gradient geometry, training phase, and model capacity?

Required controls:

- stratified shuffled disagreement within solver pair and horizon;
- matched-versus-shuffled comparison with preserved marginal distributions;
- capacity and sharing interventions;
- regression or grouped analysis that reports uncertainty and effect sizes rather than only correlations.

If Gate B fails, disagreement-aware optimization is not justified even if sharing is costly.

### Gate C: actionability

Can using the signal improve the sharing/specialization trade-off over strong simple and modern baselines on fresh data, across seeds, with measured overhead and no unexplained sacrifice of one reference?

If yes, the result can support a method paper or thesis continuation. If no, retain the diagnostic result and report the null or negative method result honestly.

## 6. Experimental order

### Stage 0: freeze the contract

- record parent benchmark version, data lineage, splits, normalization, architecture, horizon loss, checkpoint rule, and AdamW settings;
- identify a fresh final-test pool or explicitly mark the existing test set as prior evidence;
- confirm how matched batches and separate per-reference gradients are obtained;
- identify genuinely shared parameters for gradient geometry; analyze conditioning and reference-specific heads separately rather than treating disjoint blocks as artificial orthogonality;
- define resource accounting before method comparisons.

### Stage 1: measure sharing cost and gradients

Run ordinary AdamW on the conditioned model and log:

- per-reference train and validation losses;
- per-reference gradients, norms, and pairwise cosine similarities;
- directed virtual update effects `T_i_to_j = L_j(theta + d_i) - L_j(theta)`;
- actual AdamW candidate updates and applied updates;
- matched target disagreement `D_ij(x,t)`;
- absolute and normalized disagreement together with scenario-difficulty controls such as target energy, source amplitude, spatial variance, and baseline single-reference error;
- finite-step loss changes after applying a candidate update;
- horizon-wise and, only if affordable, spatial or spectral summaries;
- seed, batch, epoch, checkpoint, and configuration identifiers.

Use the independent and shared-trunk controls to define negative transfer operationally. For local metric `e_r^m(x,t)` under model/control `m`, define:

```text
NT_r^ind(x,t)  = e_r^shared(x,t) - e_r^independent(x,t)
NT_r^head(x,t) = e_r^shared(x,t) - e_r^shared-trunk+heads(x,t)
```

The aggregate headline measures are `NT_r^ind = E_r(shared) - E_r(independent)` and `NT_r^head = E_r(shared) - E_r(shared-trunk+heads)`. Analyze aligned quantities `D_rj(x,t)` and local `NT_r^k(x,t)` together. Scenario and horizon observations are repeated measurements nested within seed and solver pair, not independent solver relationships; there are only three reference pairs.

### Stage 2: test whether disagreement carries information

Compare true matched disagreement with a stratified shuffled version. Preserve solver-pair and horizon distributions while destroying scenario correspondence. Use nested models such as:

```text
M0: sharing cost ~ solver pair + horizon + scenario difficulty + capacity
M1: M0 + shared-parameter gradient/update geometry
M2: M1 + matched disagreement D_ij
```

Test whether `D_ij` predicts local `NT_r^k(x,t)`, aggregate `NT_r^k`, directed update effects, realized finite-step harm, or update interference. Compare out-of-sample predictive value and grouped or hierarchical uncertainty; do not treat thousands of scenario/horizon rows as thousands of independent solver pairs.

The required null hypothesis is that ordinary scalarization and AdamW may already be sufficient.

### Stage 3: conditional optimizer study

Only if Gates A and B pass, implement a minimal candidate mechanism. It should be described as an update-level feasible-set or update-alignment method whose relaxation is informed by pairwise matched-reference geometry.

The initial method must keep the following choices explicit:

- pairwise versus global disagreement;
- global versus horizon-dependent signal;
- gradient-space versus post-Adam update-space operation;
- first-order predicted harm versus realized finite-step loss change;
- optimizer-state treatment when a proposed update is modified.

The old scalar rule `epsilon_r = kappa q_r ||g_r|| ||d_0||` is not locked. A unilateral per-reference slack does not follow directly from pairwise disagreement and must be justified or replaced after the diagnostic.

## 7. Baselines

The core baseline set is:

1. AdamW with equal scalarization;
2. simple or random loss weighting;
3. PCGrad or CAGrad as classical multi-objective controls;
4. GUA or the closest available actual-update projection baseline;
5. candidate disagreement-conditioned relaxed update alignment, only after the gates pass.

ConFIG, GradVac, GEM, MGDA, GradNorm, and metric-aware multi-objective Adam are related methods to consider in the literature review or as compute permits. The exact final list must be frozen before final-test evaluation.

GUA and MAdam are recent preprints as of this plan. They are still essential prior art for novelty and comparison, but should be labeled as preprints in the manuscript. The paper should say `we did not identify prior work in our scoped search` until a reproducible literature review is complete.

Every comparison must match data, batch construction, model architecture, validation/checkpoint rules, optimizer updates, seed protocol, and either compute or wall-clock budget. Report extra memory and runtime for methods that require per-reference gradients or projections.

## 8. Candidate method boundary

The project no longer treats ReGRO-C's original QP as a selected contribution. GEM-style non-harm constraints, actual-update projection, task-affinity or directed update-effect measures, and HARMONIC-style feasible-set methods are established related families; the recent GUA formulation makes the overlap especially close.

The potentially distinctive hypothesis is narrower:

> matched, scenario-dependent numerical-reference geometry may contain information about how strongly an update should correct or relax cross-reference interference.

The candidate must therefore be compared with:

- zero or fixed relaxation;
- true matched disagreement;
- stratified shuffled disagreement;
- reversed or monotonicity-check policies;
- projection-only versus projection plus optimizer-state alignment.

Any gain must be accompanied by evidence that the signal matters, not merely that extra regularization or a different update scale helped.

## 9. Evaluation and evidence requirements

### Accuracy

- per-reference denormalized RMSE, MAE, and relative `L2`;
- mean and worst-reference error;
- per-horizon error curves;
- supported trajectory, gauge, peak, or arrival-time diagnostics;
- distribution-shift and resolution-transfer results only if they remain valid under the parent contract.

### Optimization behavior

- pairwise gradient cosines and conflict frequency;
- cosines computed on genuinely shared parameters, with conditioning/head blocks reported separately;
- gradient and update norms;
- directed update effects `T_i_to_j` and their asymmetry;
- local and aggregate specialization/head-sharing gaps;
- actual optimizer proposal versus applied update;
- realized finite-step loss changes;
- convergence speed and checkpoint behavior;
- projection distance or feasible-set residuals;
- wall-clock time, memory, parameter count, and optimizer-state cost.

### Scientific validity

- exact data and checkpoint lineage;
- fixed train/validation/test roles;
- seed list and configuration hashes;
- fresh final-test status;
- paired scenario-level intervals or equivalent uncertainty estimates;
- clear separation of numerical-reference discrepancy from physical truth and physical validation.

A lower mean error alone is not sufficient. A method is supported only if the optimization effect is interpretable, survives controls and ablations, and does not hide a material cost in another reference or resource budget.

## 10. What is in the DS106 core

- one parent tsunami benchmark;
- one fixed solver-conditioned neural-operator architecture;
- Gates A and B as the mandatory scientific core;
- a small strong baseline set;
- a candidate optimizer only if justified by the gates;
- at least three seeds for the core comparison when compute allows;
- an 8–15-page Elsevier-style report organized around the measured phenomenon.

The report can include a conditional method section. It must also be publishable as a diagnostic/evaluation study if the method gate fails.

## 11. Deferred extensions

The following are intentionally outside the locked DS106 core:

- learned bilevel routing or temporal loss schedules;
- geometry-preservation loss as the central objective;
- temporal, spatial, or spectral routing inside the optimizer;
- disagreement-aware curriculum learning;
- simultaneous comparison of multiple neural-operator backbones;
- Burgers, Darcy, Navier-Stokes, or another PDE benchmark;
- five-seed journal-scale replication;
- a theory of optimal disagreement-to-slack calibration;
- a promise of two or three papers before results exist.

They remain continuation options if the core evidence is strong.

## 12. Fallback paths

- **Gate A fails:** a rigorous diagnostic/evaluation paper on sharing cost and the relationship between reference disagreement and optimization geometry.
- **Gate A passes, Gate B fails:** a SciML study showing that sharing is costly but matched disagreement adds no useful information beyond ordinary diagnostics.
- **Gates A and B pass, Gate C fails:** an evidence-based analysis of why the candidate mechanism does not improve the trade-off.
- **All gates pass:** a method paper or thesis continuation with a second PDE system, stronger theory, and broader replication as later work.

## 13. Execution order

1. Freeze the parent manuscript and reported results as prior work.
2. Inspect the live parent training code and scientific contract.
3. Build the matched AdamW diagnostic and sharing-control protocol.
4. Run Gates A and B on training/validation data only.
5. Freeze the baseline list and method-selection protocol.
6. Implement the smallest candidate update-alignment mechanism if justified.
7. Run matched baseline and ablation experiments.
8. Evaluate once on the fresh final-test pool, or clearly separate prior-test evidence if a fresh pool is impossible.
9. Replicate the core result across seeds and report statistical and resource uncertainty.
10. Decide whether the final artifact is a DS106 paper, workshop paper, or journal/thesis continuation.

## 14. Provenance and source files

- [`sources/a.md`](sources/a.md) and [`sources/b.md`](sources/b.md) are temporary-chat critiques preserved as planning inputs.
- [`sources/shared_chatgpt_response.md`](sources/shared_chatgpt_response.md) archives the later shared discussion.
- [`sources/shared_chatgpt_response_earlier.md`](sources/shared_chatgpt_response_earlier.md) archives the earlier shared discussion.
- [`reference_aware_multi_objective_training_plan.md`](reference_aware_multi_objective_training_plan.md) contains the detailed mathematical exploration and extension ideas.
- The parent benchmark is `/home/izu/Projects/tsunami-surrogate`; this repository does not copy its private data, checkpoints, or credentials.
- Literature and venue claims from shared chats are historical until verified against primary sources.
