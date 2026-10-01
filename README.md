# Reference-Aware Neural Operator Optimization

Research code and experiments for disagreement-aware multi-objective optimization of solver-conditioned neural PDE surrogates trained on matched numerical references.

## Project status

This repository is the planning and implementation home for a DS106 Optimization and Applications final project, with a possible journal continuation. The current contribution is a research hypothesis and experimental roadmap; no optimizer result is claimed yet.

## Proposed paper

**Disagreement-Calibrated Harm-Constrained Optimization for Multi-Reference Neural PDE Surrogates**

Short method name: **ReGRO-C**.

The project asks whether numerical-reference disagreement creates structured negative transfer when one solver-conditioned neural operator is trained jointly on matched Hydrostatic, MUSCL-HR, and Boussinesq targets. ReGRO-C modifies the AdamW candidate step with jointly solved per-reference harm constraints whose budgets are calibrated from paired target disagreement.

## Why this benchmark

The parent project, [`tsunami-surrogate`](../tsunami-surrogate), provides a scientific-ML setting with:

- identical scenarios evaluated by multiple numerical references;
- solver-conditioned surrogate mappings;
- finite-horizon wave trajectories;
- reference-specific and cross-reference evaluation;
- an existing AdamW training procedure that can serve as a controlled baseline.

The parent manuscript studies how reference choice changes surrogate evaluation. This project studies the separate question of whether reference choice changes the optimization problem itself. The parent work remains prior context; its data, checkpoints, and private artifacts are not copied into this repository.

## First decision gate

Before implementing a new optimizer, run an ordinary AdamW gradient diagnostic. Log per-reference gradients, pairwise cosine similarity, gradient conflict frequency, target discrepancy, negative-transfer indicators, and their evolution over the forecast horizon.

- If the diagnostic shows structured conflict related to target disagreement, implement and test ReGRO-C.
- If it does not, pivot to the strongest supported evaluation or optimization question instead of forcing the original story.

## Planned study

1. Freeze the parent benchmark contract: data splits, normalization, architecture, horizon loss, and baseline settings.
2. Run the gradient diagnostic.
3. Compare AdamW, a generic multi-objective baseline such as PCGrad or CAGrad, and ReGRO-C under matched batches, update budgets, and at least three seeds.
4. Evaluate per-reference accuracy, mean and worst-reference error, gradient conflict, convergence, constraint violations, wall-clock time, and memory overhead.
5. Add ablations and a fresh final-test evaluation if the core result is stable.
6. Decide from the evidence whether the outcome is a DS106 report, a workshop-style result, or a larger journal continuation.

## Scope boundary

The DS106 core does not include learned bilevel routing, a central geometry-preservation loss, frequency-dependent optimizer routing, a disagreement-aware curriculum, multiple neural-operator backbones, extra PDE benchmarks, or a five-seed journal replication. Those remain documented extension candidates.

No Q1 or high-Q2 outcome is guaranteed. Any publication claim must follow from reproducible results, primary-source literature checks, and a fresh evaluation protocol.

## Repository contents

- [`reference_aware_multi_objective_training_plan.md`](reference_aware_multi_objective_training_plan.md): comprehensive exploratory plan and fallback paths.
- [`PROJECT_DECISION_AND_ROADMAP.md`](PROJECT_DECISION_AND_ROADMAP.md): selected method, staged decision gates, planned experiments, and deferred extensions.
- [`sources/shared_chatgpt_response.md`](sources/shared_chatgpt_response.md): archive of the later shared discussion.
- [`sources/shared_chatgpt_response_earlier.md`](sources/shared_chatgpt_response_earlier.md): archive of the earlier shared discussion.

The shared archives preserve visible user/assistant turns and attachment metadata. The original PDF attachments are not copied here. Literature and venue claims in those conversations are preserved as historical planning material and require independent verification before use in a paper.

## Suggested repository metadata

- **Name:** `reference-aware-neural-operator-optimization`
- **Description:** Research code and experiments for disagreement-aware multi-objective optimization of solver-conditioned neural PDE surrogates trained on matched numerical references.

The local repository is intentionally initialized without a remote, commit, push, dataset, checkpoint, or secret.
