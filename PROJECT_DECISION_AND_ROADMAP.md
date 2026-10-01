# Project Decision and Roadmap

## 1. Decision in one paragraph

The project will study whether matched numerical references create harmful optimization interference when a solver-conditioned neural PDE surrogate is trained on them jointly. The selected method is **ReGRO-C: Disagreement-Calibrated Harm-Constrained AdamW for Multi-Reference Neural PDE Surrogates**. The method modifies the optimizer step using reference-specific gradients and a disagreement-calibrated harm budget. The DS106 submission is the first, bounded version of the study. A journal continuation is possible only if the diagnostic finds a real and repeatable problem and the method survives matched comparisons, ablations, and fresh evaluation.

The project is not claiming a Q1 result in advance. Q1 or high-Q2 publication is an ambition and a later decision based on evidence.

## 2. Recommended project identity

- **Repository name:** `reference-aware-neural-operator-optimization`
- **Repository description:** Research code and experiments for disagreement-aware multi-objective optimization of solver-conditioned neural PDE surrogates trained on matched numerical references.
- **Initial paper title:** **Disagreement-Calibrated Harm-Constrained Optimization for Multi-Reference Neural PDE Surrogates**
- **Alternative title:** **When Numerical References Disagree: Harm-Constrained Optimization for Multi-Reference Neural PDE Surrogates**
- **Method short name:** `ReGRO-C`
- **Broader method family from the shared discussion:** `ReGRO` (Reference-Geometry-Aware Gradient Routing)

The title names the optimization contribution without claiming that one numerical solver is physical ground truth.

## 3. What the shared discussions contributed

The two archived conversations are exploratory design material. Together they established the following useful ideas:

1. The existing tsunami-surrogate benchmark is a valid application for an optimization project because it has matched scenarios, multiple numerical references, finite-horizon trajectories, and a conventional training pipeline.
2. Hydrostatic, MUSCL-HR, and Boussinesq should be treated as related but distinct reference-specific objectives rather than as interchangeable labels.
3. A solver-conditioned model is the primary setting:

   ```text
   f_theta(x, r) -> y_r
   ```

   where `x` is the shared scenario and `r` identifies the requested reference.
4. The important distinction is between parameter-space gradient conflict and target-space reference disagreement. A conflict between two gradients is not automatically harmful if the corresponding numerical references genuinely differ.
5. The first binding decision must be empirical: measure the gradients before inventing a more complicated optimizer.
6. The work should compare against established optimizers and multi-objective methods under matched data, update, seed, and compute budgets.
7. The existing benchmark is prior work and an application setting. The new contribution must be the optimization formulation, diagnostics, experiments, and analysis.

The shared response also contains literature suggestions and venue-level language. Those are preserved in `sources/`, but they are historical suggestions, not independently verified claims in this repository.

## 4. Selected method: ReGRO-C

For reference `r`, let

```text
g_r = grad_theta L_r(theta)
```

where `L_r` is the reference-specific horizon-weighted loss on a matched batch. Ordinary AdamW produces a candidate parameter step `d_0` after its moment and preconditioning calculation. ReGRO-C then solves a small constrained projection problem:

```text
minimize_d       0.5 * ||d - d_0||^2_{P_t^{-1}}

subject to       g_r^T d <= epsilon_r       for every reference r
                 ||d||_{P_t^{-1}} <= Delta_t
```

Here `P_t` is the AdamW diagonal preconditioner. The constraints limit reference-specific loss harm while keeping the update close to the ordinary AdamW step and inside a trust-region-like step budget.

The harm budget is calibrated from reference disagreement:

```text
epsilon_r = kappa * q_r * ||g_r|| * ||d_0||
```

`q_r` summarizes how much reference-specific freedom is justified by the paired target geometry. References that are close receive stricter protection; references that genuinely disagree retain more room for a solver-specific update. The exact normalized discrepancy and calibration constants are implementation decisions to be fixed before the main comparison and held fixed across seeds.

The constraints are solved jointly. This makes the update order-independent and gives the method a clearer optimization interpretation than applying a pairwise projection heuristic to gradients in an arbitrary sequence.

### Why this is the selected direction

The earlier shared proposal, discrepancy-weighted PCGrad, is a useful diagnostic and ablation, but by itself it can be criticized as a heuristic reweighting of gradient surgery. ReGRO-C makes the novelty more defensible:

- it operates on the actual AdamW candidate step rather than only on raw gradients;
- it states explicit per-reference harm constraints;
- it calibrates allowable harm using paired numerical-reference disagreement;
- it solves the constraints jointly, without pair-order dependence;
- it exposes measurable constraint violations, projection distance, and compute overhead.

This is still a hypothesis. The diagnostic may show that a simpler baseline is sufficient or that conflict is too rare to justify the method.

## 5. DS106 core study

### 5.1 Research question

> When one solver-conditioned neural PDE surrogate is trained against multiple matched numerical references, does reference disagreement create structured negative transfer, and can a disagreement-calibrated harm-constrained AdamW step reduce that harm without suppressing legitimate reference-specific behavior?

### 5.2 Required model and data setting

- Use the solver-conditioned neural operator as the primary model.
- Keep the architecture and horizon-weighted loss fixed while comparing optimizers.
- Use matched scenarios so the three reference targets correspond to the same inputs.
- Keep Hydrostatic, MUSCL-HR, and Boussinesq identities explicit.
- Preserve the existing train/validation/test boundaries and record any changes.
- Do not place the parent paper's private data, checkpoints, or credentials in this repository.

### 5.3 Staged experiment plan

#### Stage 0: Freeze the contract

- Record the parent benchmark version, preprocessing, split, normalization, architecture, loss, and current AdamW settings.
- Decide which data are genuinely fresh for method selection and final evaluation.
- Confirm how separate per-reference gradients can be obtained without changing the forward model.

#### Stage 1: Gradient diagnostic (binding gate)

- Run ordinary AdamW with per-reference losses and gradients logged.
- Measure pairwise gradient cosine similarity, gradient norms, conflict frequency, target discrepancy, and their relationship over training.
- Break the analysis down by forecast horizon and, if affordable, by spatial region or frequency band.

Decision rule:

- If conflict is frequent, structured, and associated with negative transfer, implement ReGRO-C.
- If conflict is rare or unrelated to reference disagreement, do not force a gradient-conflict paper. Use the strongest supported fallback: a carefully controlled evaluation study or another mechanism motivated by the observed pathology.

#### Stage 2: Minimal optimizer comparison

Compare, under matched batches and update budgets:

1. ordinary AdamW;
2. a generic multi-objective baseline such as PCGrad or CAGrad;
3. ReGRO-C;
4. ReGRO-Fixed, which uses a fixed harm/disagreement rule as an ablation;
5. optionally, one additional established baseline such as GradNorm or Nash-MTL if compute permits.

Use at least three seeds for the core comparison. Do not add dynamic curricula or learned routing until this stage is understood.

#### Stage 3: Robustness and ablations

- Remove disagreement calibration while retaining the constrained step.
- Replace the joint constraints with an order-dependent pairwise correction to test the claimed advantage.
- Vary the harm scale `kappa` and trust-region scale `Delta_t`.
- Compare equal reference weighting with a worst-reference risk measure.
- Test whether gains persist under distribution shift and resolution transfer if those parent benchmark protocols remain valid.

#### Stage 4: Publication-strength extension

- Use a fresh final-test pool that was not used for optimizer selection.
- Add more seeds and paired statistical intervals.
- Report runtime, memory, solver calls, projection distance, and constraint violations.
- Replicate on a second PDE benchmark only if the core result is stable and the DS106 scope is already complete.

## 6. Evaluation and evidence requirements

Report all three levels of evidence separately:

### Accuracy

- per-reference RMSE;
- per-reference relative `L2` error;
- mean reference error;
- worst-reference error;
- trajectory and virtual-gauge diagnostics where supported by the parent benchmark.

### Optimization behavior

- pairwise gradient cosine distributions;
- conflict frequency and duration;
- gradient norms and update norms;
- loss curves and convergence speed;
- ReGRO-C constraint violations;
- projection distance from the AdamW candidate step;
- wall-clock time and memory overhead.

### Scientific validity

- exact data and checkpoint lineage;
- fixed train/validation/test roles;
- seed list and configuration hashes;
- fresh final-test status;
- numerical-reference discrepancy versus surrogate error;
- explicit separation of numerical verification from physical validation.

The method is successful only if any accuracy gain is accompanied by an interpretable optimization effect and survives the relevant ablations. A lower average loss alone is not enough.

## 7. Retained ideas from the shared proposal

These ideas remain in scope as diagnostics, ablations, or later extensions:

- paired numerical-reference geometry;
- solver-conditioned training;
- harmful versus reference-supported gradient conflict;
- ReGRO-Fixed as a controlled ablation;
- reference-geometry distortion metrics;
- independent-model negative-transfer comparisons;
- AdamW, PCGrad, CAGrad, and possibly GradNorm/Nash-MTL;
- fresh final-test evaluation;
- temporal disagreement analysis;
- spectral disagreement analysis if the first result supports it.

## 8. Deliberately deferred ideas

The following are left out of the DS106 core to protect causal interpretation, implementation time, and reproducibility:

- making the geometry-preservation loss the central objective;
- bilevel learned routing or a learned disagreement gate;
- temporal or frequency-dependent routing inside the optimizer;
- disagreement-aware curriculum learning;
- simultaneous comparison of multiple neural-operator backbones;
- Burgers, Darcy, Navier-Stokes, or other extra PDE benchmarks;
- five-seed full journal replication;
- broad Bayesian gradient-aggregation comparisons;
- a claim that solver disagreement is physical truth;
- a claim that ReGRO-C improves all references or all datasets before results exist.

These are not discarded. They are extension candidates if the diagnostic and core method provide a credible foundation.

## 9. Planned report structure

The 8-15 page Elsevier-style report can use:

1. Introduction and motivation from matched numerical references.
2. Related optimization methods and multi-reference surrogate setting.
3. Problem formulation and gradient-conflict diagnostic.
4. ReGRO-C constrained-step method.
5. Experimental protocol and leakage controls.
6. Results: accuracy, conflict, constraints, convergence, and cost.
7. Ablations and limitations.
8. Conclusion and the evidence-based publication decision.

The report should be written around the observed optimization pathology. It should not present the algorithm as successful before the diagnostic and matched experiments support that claim.

## 10. Immediate next actions

1. Inspect the `tsunami-surrogate` training code and identify the exact per-reference loss/gradient boundary.
2. Run the ordinary AdamW diagnostic before implementing ReGRO-C.
3. Record the diagnostic output in a small experiment ledger with seed, split, configuration, and checkpoint identifiers.
4. Implement the smallest order-independent constrained-step prototype only if the diagnostic gate passes.
5. Compare AdamW, one generic multi-objective baseline, and ReGRO-C under matched budgets.
6. Decide whether the deliverable is a DS106 paper only, a workshop-style result, or a candidate journal continuation.

## 11. Source and provenance notes

- `reference_aware_multi_objective_training_plan.md` is the comprehensive exploratory plan.
- `sources/shared_chatgpt_response.md` archives the later shared discussion, including the ReGRO refinement.
- `sources/shared_chatgpt_response_earlier.md` archives the earlier shared discussion.
- The parent benchmark is in `/home/izu/Projects/tsunami-surrogate`; this repository does not copy it or add a remote automatically.
- Shared-page literature references and venue claims are not treated as verified evidence until checked against primary sources.
