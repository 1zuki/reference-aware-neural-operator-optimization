# Reference-Aware Neural Operator Optimization

Research planning and implementation repository for studying negative transfer and optimization interference in solver-conditioned neural PDE surrogates trained on matched numerical references.

## Locked direction

The project is locked around this question:

> Does observable disagreement between matched numerical references predict when sharing a neural surrogate causes negative transfer?

The conditional optimization question is:

> If it does, can that information improve the trade-off between shared learning and reference-specific specialization?

The central mechanistic hypothesis is:

```text
matched target disagreement
        -> optimization/update interference
        -> sharing cost or negative transfer
```

Every arrow is an empirical question. Negative cosine similarity is a gradient diagnostic; it is not, by itself, evidence of negative transfer.

This is a hypothesized pathway in an observational matched-reference study, not a causal claim. Stronger causal conclusions would require deliberate interventions on numerical-reference discrepancy and are deferred.

## Project identity

- **Repository name:** `reference-aware-neural-operator-optimization`
- **Description:** Research code and experiments for disagreement-aware multi-objective optimization of solver-conditioned neural PDE surrogates trained on matched numerical references.
- **Working paper title:** **When Numerical References Disagree: Negative Transfer and Optimization Interference in Solver-Conditioned Neural Operators**
- **Internal method family:** `ReGRO` (Reference-Geometry-Aware Gradient Routing)
- **Candidate mechanism:** `ReGRO-C`, retained as an internal name for a disagreement-conditioned relaxed update-alignment method if the diagnostic gates justify it.

The title is deliberately phenomenon-first. The project does not claim that the projection mechanism is a new general-purpose optimizer.

## Why this benchmark

The parent project, [`tsunami-surrogate`](../tsunami-surrogate), provides a useful scientific-ML setting:

- identical scenarios evaluated by Hydrostatic, MUSCL-HR, and Boussinesq numerical references;
- matched trajectories sampled at common requested times;
- a solver-conditioned surrogate mapping `f_theta(x, r) -> y_r`;
- reference-specific, cross-reference, horizon, and robustness diagnostics;
- a conventional AdamW training procedure that can serve as a controlled baseline.

The parent manuscript studies how numerical-reference choice changes surrogate evaluation. This project studies the separate question of whether matched reference choice changes the optimization problem and the cost of parameter sharing. The parent work is prior context; its private data, checkpoints, and credentials are not copied here.

## What we will do

1. Freeze the parent benchmark contract: data lineage, splits, normalization, architecture, horizon loss, checkpoint rules, and baseline optimizer.
2. Measure matched target disagreement, per-reference gradients, actual optimizer updates, and held-out performance under ordinary AdamW.
3. Compare a fully shared conditioned model, a shared-trunk/reference-head model, and independently trained reference models to measure sharing cost while exposing capacity and compute confounds.
4. Test whether matched disagreement predicts local and aggregate negative transfer after controlling for solver identity, horizon, scenario difficulty, loss scale, gradient statistics, training phase, and model capacity.
5. Use a stratified shuffled-disagreement control that preserves solver-pair and horizon distributions while destroying scenario correspondence.
6. Only if the evidence supports the hypothesis, evaluate a disagreement-conditioned relaxed update-alignment mechanism against AdamW, simple weighting, classical gradient surgery, and the closest actual-update baseline.
7. Use fresh final-test data, multiple seeds, paired intervals, and resource accounting before making a method or publication claim.

## Decision gates

### Gate A: problem

Does parameter sharing hurt held-out performance consistently enough to matter? Independent/specialized models and a shared-trunk/reference-head control are mandatory. If this gate fails, the optimizer story stops.

### Gate B: signal

Does matched disagreement explain when sharing hurts beyond identity, horizon, scenario difficulty, loss magnitude, shared-parameter gradient geometry, training phase, and capacity? Matched versus stratified shuffled disagreement is required. If this gate fails, disagreement-aware optimization is not justified even if sharing is costly.

### Gate C: actionability

Can using the signal improve the sharing/specialization trade-off over strong simple and modern baselines on fresh data, across seeds, with acceptable cost and no unexplained sacrifice of a reference? If yes, the work can support a method paper or thesis continuation.

## Candidate mechanism and prior-art boundary

The candidate mechanism is an update-level feasible-set method whose relaxation is informed by pairwise matched-reference geometry. The exact feasible set is intentionally conditional on Gates A and B.

The following are baseline mechanisms or prior-art families, not standalone novelty claims: GEM-style constrained updates, PCGrad, CAGrad, GradVac, ConFIG, GUA, metric-aware multi-objective Adam, and task-affinity or directed update-effect methods. HARMONIC and the exact bibliographic status of recent methods should be checked in the scoped literature review before manuscript claims are written. The paper must say `we did not identify prior work in our scoped search` until that review is reproducible.

## Scope

### DS106 core

- one parent tsunami benchmark;
- one fixed solver-conditioned neural-operator architecture;
- diagnostic and sharing controls;
- a small, fair baseline set;
- candidate optimizer only if the gates pass;
- at least three seeds for the core comparison when compute allows;
- validation-only method selection and a fresh or explicitly separated final test.

### Deferred extensions

- learned bilevel routing or loss schedules;
- geometry-preservation loss as a central objective;
- temporal or spatially localized optimizer routing;
- disagreement-aware curriculum learning;
- multiple neural-operator backbones;
- Burgers, Darcy, Navier-Stokes, or another PDE benchmark;
- broad optimizer sweeps;
- theory of optimal disagreement-to-slack calibration.

These are continuation options, not promises of multiple papers.

## Repository contents

- [`PROJECT_DECISION_AND_ROADMAP.md`](PROJECT_DECISION_AND_ROADMAP.md): canonical locked decision, gates, controls, baselines, and execution order.
- [`reference_aware_multi_objective_training_plan.md`](reference_aware_multi_objective_training_plan.md): detailed mathematical and experimental plan.
- [`sources/a.md`](sources/a.md): first temporary-chat critique, preserved as source material.
- [`sources/b.md`](sources/b.md): second temporary-chat critique, preserved as source material.
- [`sources/shared_chatgpt_response.md`](sources/shared_chatgpt_response.md): archive of the later shared discussion.
- [`sources/shared_chatgpt_response_earlier.md`](sources/shared_chatgpt_response_earlier.md): archive of the earlier shared discussion.

The temporary-chat files and shared archives are historical planning inputs. Their literature, venue, and novelty claims require independent verification before appearing in a manuscript.

## Current status

The research direction is locked at the planning level. No optimizer result is claimed, no parent benchmark artifact is copied, and no commit, remote, dataset, checkpoint, or secret is created by this planning repository.
