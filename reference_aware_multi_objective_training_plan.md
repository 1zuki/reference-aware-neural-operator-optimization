# Reference-Aware Multi-Objective Training for Neural PDE Surrogates

## Purpose of this document

This document turns the complete project idea into a research-style plan that can be discussed with the professor and later developed into the DS106 final project. It preserves the connection to the existing tsunami-surrogate manuscript, explains the optimization problem, gives concrete mathematical formulations, proposes an initial diagnostic before any new optimizer is implemented, and lays out baselines, experiments, hypotheses, risks, and scope.

The central proposed direction is:

> **Reference-Aware Multi-Objective Training of Neural PDE Surrogates under Numerical Solver Disagreement**

The project asks whether disagreement between numerical reference solvers creates measurable optimization interference when one neural PDE surrogate is trained across multiple references, and whether an optimizer can use the known reference disagreement to handle that interference more intelligently.

The project is intentionally framed as an optimization study, not merely as an “optimizer shootout.” A plain comparison such as AdamW versus SGD versus RMSProp would fit a course project, but would have a lower research ceiling. The stronger version first identifies a training pathology, then proposes and evaluates a method designed around that pathology.

---

## 1. Fit with the DS106 final project

The DS106 course is **Optimization and Applications**. The syllabus covers gradient methods, steepest descent, Newton-type methods, constrained and nonlinear optimization, and other optimization algorithms. The final seminar/report is a group project on a topic related to optimization and applications, with the report encouraged to follow the format of a scientific paper.

The professor's requested format and scope are:

- Elsevier two-column format.
- Approximately 8–15 pages.
- Either:
  - improving an algorithm, which is harder and could potentially become a publishable paper; or
  - evaluating 2–5 algorithms, which is safer and better suited to simply passing the course.

This project targets the first option while retaining a fallback path to the second option. It studies an actual optimization problem arising in an existing scientific machine-learning benchmark rather than using an artificial toy problem.

The project is difficult enough to teach substantial material:

- multi-objective optimization;
- gradient geometry and cosine similarity;
- gradient conflict and negative transfer;
- gradient projection or gradient surgery;
- adaptive objective weighting;
- worst-case and risk-sensitive objectives;
- numerical conditioning and preconditioning;
- long-horizon sequence loss design;
- reproducible scientific experiments;
- uncertainty about whether the proposed method helps, hurts, or is unnecessary.

A potential Q1 or high-Q2 publication should not be promised in advance. The project has a credible research path, but publication depends on whether the initial diagnostic finds a real and systematic problem, whether the method is genuinely distinct from existing multi-objective methods, and whether the evaluation is broad and statistically convincing.

---

## 2. Connection to the existing submission

The project reuses the benchmark and scientific setting from an existing manuscript currently in submission:

> **Reference-Aware Evaluation of Neural PDE Surrogates: A Controlled Multi-Reference Benchmark for Tsunami-Like Waves**

The current manuscript studies neural PDE surrogates for tsunami-like wave propagation. Identical bathymetry–source scenarios are advanced by three numerical references:

- **Hydrostatic**;
- **MUSCL-HR**;
- a study-defined linear weakly dispersive **Boussinesq-type** reference.

The solver outputs are aligned at 50 common requested times from 8.4 to 420 benchmark-time units. The references are not treated as interchangeable ground truth. Hydrostatic and MUSCL-HR approximate depth-averaged shallow-water equations with different discretizations, while the Boussinesq-type comparison introduces a different weakly dispersive equation family. Their pairwise differences combine equation-set, discretization, time-integration, boundary, and output-sampling effects.

The manuscript's main message is that **numerical-reference choice changes the apparent evaluation result**. It separates:

- same-reference surrogate error;
- raw solver-to-solver discrepancy;
- cross-reference surrogate discrepancy.

A surrogate being closer to another numerical solver in one metric does not establish physical superiority without observational truth or a stronger physical reference.

The current manuscript deliberately uses established architectures and a controlled, mostly conventional training procedure. Its contribution is the reference-aware evaluation framework, not a new neural architecture or optimizer. That leaves a clean continuation:

- **Paper 1:** numerical reference choice affects evaluation.
- **This project/Paper 2:** numerical reference choice may affect optimization itself.

The new contribution must be clearly separated from the submitted manuscript. The existing dataset/framework can be cited as prior work and reused as the benchmark, but the new optimization algorithm, gradient analysis, experimental protocol, and results must be new.

### 2.1 Verified benchmark facts useful for the motivation

The current manuscript reports, on 2,500 held-out scenarios, global solver-gap RMSEs of approximately:

- Hydrostatic–MUSCL-HR: **0.00171**;
- MUSCL-HR–Boussinesq: **0.00305**;
- Hydrostatic–Boussinesq: **0.00374**.

In the pooled-reference comparison:

- a solver-anonymous FNO obtains mean RMSEs of approximately **0.00258, 0.00274, and 0.00357** against the three references;
- supplying solver identity reduces target-specific errors to approximately **0.00212, 0.00252, and 0.00289**.

This is important motivation: the same scenario input can correspond to three different target trajectories, and solver identity helps the model represent reference-specific mappings.

The benchmark includes substantial structure that can support a serious optimization study:

- 50-frame finite-horizon trajectories;
- matched scenarios across the three references;
- 10,000 training, 1,000 validation, and 2,500 test scenarios per reference in the current setup;
- distribution-shift and resolution-transfer tests;
- architecture comparisons involving FNO and other neural models;
- virtual-gauge and waveform diagnostics;
- seven-member ensemble analysis;
- numerical verification and inter-code comparison evidence.

The current pooled experiment balances the three references, compares a solver-anonymous FNO with a solver-conditioned FNO, and uses the same architecture and optimization objective for the variants.

### 2.2 Current optimization procedure to be treated as the baseline

The manuscript currently trains with:

- **AdamW**;
- initial learning rate \(5\times 10^{-4}\);
- weight decay \(10^{-6}\);
- gradient clipping at maximum norm 1.0;
- cosine learning-rate annealing to \(5\times 10^{-5}\);
- batch size 64 for most direct models, with smaller batches for memory-heavy models;
- early stopping patience 20 epochs and a maximum of 256 epochs;
- validation relative \(L_2\) for checkpoint selection.

The loss is a fixed horizon-weighted MSE:

\[
\mathcal L
=\frac{\sum_{t=1}^{T} w_t\,\mathrm{MSE}_t}{\sum_{t=1}^{T} w_t},
\]

with

\[
w_t=w_{\min}+(w_{\max}-w_{\min})
\left(\frac{t-1}{T-1}\right)^p,
\]

and the current values

\[
w_{\min}=0.8,\qquad w_{\max}=2.0,\qquad p=1.7.
\]

This fixed weighting puts more emphasis on later forecast frames. The new project can retain this loss as the controlled baseline while changing only the multi-reference optimization procedure.

---

## 3. The central research question

The current pooled training effectively treats the three references as one combined objective. Let

\[
H=\text{Hydrostatic},\qquad
M=\text{MUSCL-HR},\qquad
B=\text{Boussinesq}.
\]

For a solver-conditioned model,

\[
f_\theta(x,r)\rightarrow y_r,
\]

where \(x\) is the common scenario input and \(r\) is the requested reference identity. There are three reference-specific losses:

\[
L_H(\theta),\qquad L_M(\theta),\qquad L_B(\theta).
\]

Ordinary pooled training can be written as

\[
L_{\mathrm{avg}}(\theta)
=\frac{1}{3}\left(L_H+L_M+L_B\right),
\]

with gradient

\[
g=\nabla_\theta L_{\mathrm{avg}}
=\frac{1}{3}(g_H+g_M+g_B),
\qquad
 g_r=\nabla_\theta L_r.
\]

This is reasonable, but it makes an implicit assumption:

> The average of the three reference-specific gradients is a good descent direction for all three references.

That assumption may fail. A step that reduces one reference's loss can increase another's loss. This is the multi-objective optimization problem hidden inside the existing benchmark.

The core research question is:

> **When a neural PDE surrogate is trained against multiple numerical references, does reference disagreement produce systematic gradient conflict, and can a reference-aware optimizer reduce harmful negative transfer without erasing legitimate reference-specific behavior?**

The method should not blindly force every gradient to agree. Some conflicts may be caused by optimization interference, while others may reflect real differences between the numerical solvers.

---

## 4. Why the solver-conditioned model should be the primary setting

The solver-anonymous model receives \(x\) but not solver identity. For an identical scenario, the target may be any of

\[
y_H,\qquad y_M,\qquad y_B.
\]

Thus the anonymous problem is effectively

\[
x\rightarrow\{y_H,y_M,y_B\},
\]

rather than a well-defined single mapping \(x\rightarrow y\). Under MSE, the theoretical optimum tends toward a conditional-average or compromise prediction, roughly related to

\[
f^*(x)=\mathbb E[Y\mid X=x].
\]

No optimizer can completely remove this ambiguity because the requested reference is missing from the input.

The primary new study should therefore use the solver-conditioned model:

\[
(x,H)\rightarrow y_H,
\]

\[
(x,M)\rightarrow y_M,
\]

\[
(x,B)\rightarrow y_B.
\]

Now the mapping is well-defined, and the remaining problem is parameter interference between reference-specific objectives. That is the part an optimization method can genuinely address.

The anonymous pooled model can remain as a secondary analysis or limitation experiment, but it should not be the main basis for claiming that optimization solved the problem.

---

## 5. Gradient conflict and how to measure it

For each reference, compute a gradient

\[
g_H=\nabla_\theta L_H,
\qquad
 g_M=\nabla_\theta L_M,
\qquad
 g_B=\nabla_\theta L_B.
\]

For two references \(i,j\), measure gradient alignment using cosine similarity:

\[
C_{ij}
=\frac{g_i^\top g_j}{\|g_i\|\,\|g_j\|+\varepsilon}.
\]

Interpretation:

\[
C_{ij}\approx 1
\quad\Rightarrow\quad
\text{strong agreement},
\]

\[
C_{ij}\approx 0
\quad\Rightarrow\quad
\text{mostly independent directions},
\]

\[
C_{ij}<0
\quad\Rightarrow\quad
\text{gradient conflict}.
\]

The gradient similarity matrix is

\[
C=
\begin{bmatrix}
1&C_{HM}&C_{HB}\\
C_{HM}&1&C_{MB}\\
C_{HB}&C_{MB}&1
\end{bmatrix}.
\]

A conflict occurs when

\[
g_i^\top g_j<0.
\]

For example, if

\[
g_M=\begin{bmatrix}1\\1\end{bmatrix},
\qquad
 g_B=\begin{bmatrix}-1\\-0.5\end{bmatrix},
\]

then

\[
g_M^\top g_B=-1.5<0.
\]

An update that reduces one objective can therefore increase the other objective.

The first diagnostic should record, over training:

\[
C_{HM}(e),\qquad C_{HB}(e),\qquad C_{MB}(e),
\]

where \(e\) is the epoch or training step. It should also record the individual losses, gradient norms, update norms, and the fraction of batches with negative pairwise cosine similarity.

---

## 6. Reference disagreement as an additional signal

The unusual advantage of this problem is that the disagreement between the objectives is observable in the data. For the same scenario and requested time, compare the numerical outputs directly:

\[
D_{ij}=d(y_i,y_j),
\]

where \(d\) may be denormalized RMSE, relative \(L_2\), or another clearly stated benchmark metric.

The reference-discrepancy matrix is

\[
D=
\begin{bmatrix}
0&D_{HM}&D_{HB}\\
D_{HM}&0&D_{MB}\\
D_{HB}&D_{MB}&0
\end{bmatrix}.
\]

The two signals mean different things:

- \(D_{ij}\): how differently the numerical solvers describe the same scenario;
- \(C_{ij}\): how differently the corresponding objectives try to update the neural network.

The proposed study connects them:

\[
\text{numerical PDE discrepancy}
\rightarrow
\text{optimization geometry}
\rightarrow
\text{surrogate performance}.
\]

A scatter plot of \(D_{ij}\) versus \(C_{ij}\), or \(D_{ij}\) versus \(1-C_{ij}\), can test whether larger solver disagreement predicts stronger gradient conflict.

### 6.1 Two important cases

#### Case A: small solver discrepancy but conflicting gradients

\[
D_{HM}\approx 0,
\qquad
 C_{HM}<0.
\]

The solvers predict nearly the same trajectory, but their gradients fight. This may indicate optimization noise, parameter coupling, conditioning, normalization effects, or different feature use rather than meaningful solver physics. It is a plausible case for strong conflict correction.

#### Case B: large solver discrepancy and conflicting gradients

\[
D_{HB}\gg 0,
\qquad
 C_{HB}<0.
\]

This may represent legitimate reference-specific behavior. Hydrostatic/MUSCL-HR are shallow-water references, while Boussinesq introduces weak dispersion. Blindly forcing the gradients to align may destroy useful reference-specific learning.

This motivates the central distinction:

> **Some gradient conflict is optimization interference; some is legitimate numerical or physical-reference disagreement. The method should treat them differently.**

---

## 7. Temporal and spatial reference disagreement

The output is a trajectory, approximately

\[
Y\in\mathbb R^{50\times64\times64}.
\]

A single scalar \(D_{ij}\) hides when the references diverge. Define a horizon-specific discrepancy:

\[
D_{ij}^{(t)}
=\operatorname{RMSE}\left(y_i^{(t)},y_j^{(t)}\right).
\]

Potential behavior:

- Hydrostatic and MUSCL-HR may agree more closely at early times;
- they may diverge later because small numerical differences accumulate;
- Boussinesq may show a different divergence pattern as dispersive effects matter more over time.

A horizon-dependent agreement factor can therefore be defined as

\[
A_{ij}^{(t)}
=\exp\left(-\frac{D_{ij}^{(t)}}{\tau}\right),
\]

where \(\tau\) is a scale parameter estimated from the training data or selected on validation data.

The same idea can be localized spatially, although this is a later extension. A disagreement field could be defined by

\[
D_{ij}(t,x,y)=|y_i(t,x,y)-y_j(t,x,y)|
\]

or by a normalized local difference. This would allow the optimizer or loss to distinguish regions where references agree from regions where their numerical behavior diverges.

The safest project scope is initially global or horizon-wise disagreement. Full spatially localized gradient handling is interesting but increases implementation and memory complexity substantially.

---

## 8. Proposed method: Reference-Discrepancy-Aware Gradient Optimization

A provisional name is:

> **Reference-Discrepancy-Aware Gradient Optimization**

The name can be changed after the method is mathematically settled and related work has been checked.

### 8.1 Batch-level objectives

For a matched batch containing the same scenarios under all three references, compute

\[
L_H,\qquad L_M,\qquad L_B.
\]

The losses may retain the existing horizon-weighted MSE, or be decomposed by horizon if the temporal extension is used:

\[
L_r=\frac{\sum_t w_t L_{r,t}}{\sum_t w_t}.
\]

Compute separate gradients:

\[
g_r=\nabla_\theta L_r.
\]

This requires retaining or recomputing per-reference gradients. A memory-efficient implementation may compute them sequentially, flatten the gradients, and discard the computation graph after each backward pass.

### 8.2 Agreement strength

Convert the measured target discrepancy into an agreement strength:

\[
A_{ij}=e^{-D_{ij}/\tau}.
\]

Therefore:

- similar solver outputs produce \(A_{ij}\approx1\);
- strongly different solver outputs produce \(A_{ij}\approx0\).

For the temporal version:

\[
A_{ij}^{(t)}=e^{-D_{ij}^{(t)}/\tau}.
\]

The agreement scale must be normalized carefully. Raw RMSE values depend on units, normalization, and output scale. Possible choices include standardizing \(D_{ij}\) by a training-set median, percentile, or robust scale before applying the exponential.

### 8.3 Conflict correction

A PCGrad-like projection for reference \(i\) against reference \(j\) would remove the component of \(g_i\) that conflicts with \(g_j\). A discrepancy-aware version is

\[
g_i'
=
 g_i
-
 A_{ij}
\frac{g_i^\top g_j}{\|g_j\|^2+\varepsilon}
 g_j,
\]

applied only when

\[
g_i^\top g_j<0.
\]

Interpretation:

- if \(D_{ij}\) is small, then \(A_{ij}\) is large and the conflict is corrected strongly;
- if \(D_{ij}\) is large, then \(A_{ij}\) is small and the method preserves more reference-specific behavior;
- if gradients already agree, no correction is needed.

For three references, corrections can be applied pairwise. The order should be specified and tested, or a symmetric correction should be designed to avoid order dependence.

A possible symmetric formulation is to construct a corrected gradient per reference from all pairwise conflicts, then clip or normalize the correction to prevent one pair from dominating:

\[
\widetilde g_i
=
 g_i
-
\sum_{j\neq i}
\mathbf 1[g_i^\top g_j<0]
 A_{ij}
\frac{g_i^\top g_j}{\|g_j\|^2+\varepsilon}
 g_j.
\]

This formula is a starting point, not a final claim. Its stability, symmetry, and interaction with AdamW must be tested.

### 8.4 Combining the corrected gradients

The simplest final direction is the equal-weight average:

\[
g^*=\frac13(\widetilde g_H+\widetilde g_M+\widetilde g_B).
\]

Then apply AdamW using \(g^*\), or treat \(g^*\) as the gradient passed to an AdamW-style moment update.

A more general combination is

\[
g^*=\sum_{r\in\{H,M,B\}}\alpha_r\widetilde g_r,
\qquad
\alpha_r\ge0,
\qquad
\sum_r\alpha_r=1.
\]

The first version should keep \(\alpha_r=1/3\) so that the contribution of gradient correction can be isolated. Dynamic weighting should be introduced as a separate ablation or second method variant.

### 8.5 Dynamic worst-reference weighting

To emphasize the reference currently performing worst, define normalized losses \(\widetilde L_r\) and

\[
\alpha_r
=
\frac{\exp(\beta\widetilde L_r)}
{\sum_j\exp(\beta\widetilde L_j)}.
\]

Here \(\beta\) controls how strongly the method focuses on the worst reference. This moves the training objective conceptually closer to

\[
\min_\theta\max\{L_H,L_M,L_B\}
\]

rather than minimizing only the mean.

The method should not use raw losses with incompatible scales. Losses should be normalized by reference-specific baselines, running averages, or validation-calibrated scales. Otherwise one reference may receive a larger weight simply because its target has a larger numerical scale.

An alternative robust objective is a CVaR-like or soft-max loss over reference losses. For only three references, soft-max weighting is likely simpler and easier to explain.

### 8.6 AdamW interaction

AdamW itself transforms gradients through adaptive first- and second-moment estimates. There are at least two possible insertion points:

1. **Gradient-space correction before AdamW:** compute and correct \(g_H,g_M,g_B\), combine them into \(g^*\), then feed \(g^*\) to the ordinary AdamW moment update.
2. **Per-reference preconditioning before combination:** apply an Adam-like preconditioner to each \(g_r\), then perform reference-aware combination.

The first option is safer for the initial project because it keeps the optimizer baseline recognizable and isolates the effect of the multi-objective correction. The paper should explicitly state which gradient space is used and whether conflict is measured before or after AdamW preconditioning.

### 8.7 Optional horizon-aware correction

A richer variant computes gradients per forecast horizon:

\[
g_{r,t}=\nabla_\theta L_{r,t},
\]

then uses

\[
A_{ij}^{(t)}=e^{-D_{ij}^{(t)}/\tau}
\]

for each time \(t\). The corrected horizon gradients are combined using the existing horizon weights \(w_t\).

This is scientifically attractive because the solver disagreement may increase over time, but it is computationally expensive and may be too large for the first implementation. It is best treated as a planned extension if the global or reference-level method shows promise.

---

## 9. A second possible mechanism: disagreement-aware curriculum

Gradient surgery is not the only possible method. A simpler and potentially more stable alternative uses solver disagreement to control training difficulty.

Define a disagreement score for scenario \(x\) and horizon \(t\):

\[
u(x,t)=\operatorname{Var}\{y_H(x,t),y_M(x,t),y_B(x,t)\}
\]

or an averaged pairwise discrepancy.

Low \(u\) means that the references agree. High \(u\) means that the scenario or horizon is solver-dependent.

Define a curriculum weight

\[
w(x,t,e)=
\exp\left(-\frac{u(x,t)}{\tau_e}\right),
\]

where \(e\) is the epoch and \(\tau_e\) increases over training.

Early training:

\[
\tau_e\text{ small}
\]

so the model focuses on reference-consistent dynamics.

Later training:

\[
\tau_e\uparrow
\]

so the model progressively sees more solver-disagreement regions.

The conceptual learning sequence is

\[
\text{common dynamics}
\rightarrow
\text{hard solver-specific dynamics}.
\]

This curriculum is easier to implement than full per-reference gradient surgery and can be used as:

- a standalone method;
- an ablation;
- a component combined with discrepancy-aware gradient correction.

A useful ablation matrix is:

1. ordinary pooled training;
2. generic gradient surgery;
3. disagreement-aware curriculum;
4. disagreement-aware gradient correction;
5. curriculum plus gradient correction.

---

## 10. Hypotheses

### H1 — Numerical-reference disagreement creates measurable optimization conflict

\[
D_{ij}
\quad\text{is associated with}\quad
1-C_{ij}.
\]

In plain language: the more the numerical references disagree, the more their gradients tend to conflict.

This must be tested before building the method. If gradients are aligned almost all the time, gradient surgery is not the right research story.

### H2 — Not all gradient conflicts should be removed

A method that treats every negative cosine similarity equally may damage reference-specific learning. Discrepancy-aware correction should preserve more of the conflict when the underlying numerical references genuinely disagree.

Expected relationship:

\[
\text{reference-aware correction}
>
\text{uniform conflict correction}
\]

for preserving target-specific accuracy while reducing harmful interference.

### H3 — Reference-aware optimization reduces negative transfer

The method should improve the mean objective and, more importantly, reduce the weakest-reference error:

\[
\frac13\sum_r L_r
\]

and/or

\[
\max_r L_r.
\]

The worst-reference metric is particularly important because a pooled model can appear good on average while failing badly on one reference.

### H4 — Solver identity and optimization interact

The method should be more effective for the solver-conditioned model than for the anonymous model, because the conditioned model represents three well-defined mappings while the anonymous model is inherently ambiguous.

### H5 — Conflict is structured by forecast horizon

If solver differences accumulate over time, gradient conflict may be weak at early horizons and stronger at later horizons. The proposed method should therefore be analyzed by horizon, not only by one global loss.

---

## 11. The first experiment: do not invent the method yet

The first experiment should be a cheap diagnostic using ordinary AdamW training on the existing solver-conditioned FNO.

At intervals, such as every \(K\) batches or every epoch, log:

- \(L_H,L_M,L_B\);
- gradient norms \(\|g_H\|,\|g_M\|,\|g_B\|\);
- pairwise cosine similarities:
  - \(C_{HM}\);
  - \(C_{HB}\);
  - \(C_{MB}\);
- the fraction of pairwise conflicts;
- reference discrepancies \(D_{HM},D_{HB},D_{MB}\);
- horizon-wise discrepancies \(D_{ij}^{(t)}\) if feasible;
- parameter update norm;
- validation error for each reference;
- eventual per-reference test error.

The diagnostic should answer:

1. How often are gradients conflicting?
2. Which reference pair conflicts most?
3. Does conflict increase later in training?
4. Does conflict differ by forecast horizon?
5. Does solver discrepancy predict gradient conflict?
6. Do high-conflict batches correspond to larger eventual prediction errors?
7. Are conflicts persistent across random seeds?
8. Does the worst-reference loss receive too little gradient contribution under equal averaging?

A useful figure is a scatter plot with

- x-axis: \(D_{ij}\);
- y-axis: \(C_{ij}\) or \(1-C_{ij}\);
- one point per batch, scenario group, or horizon region;
- separate colors for \(HM\), \(HB\), and \(MB\).

Another useful figure plots \(C_{HM}(e),C_{HB}(e),C_{MB}(e)\) across training epochs.

### Diagnostic decision rule

If all three gradients are aligned almost all the time, for example with negligible negative-cosine frequency and no relationship between discrepancy and conflict, do not force a gradient-surgery paper. Move to a different optimization contribution, such as:

- learned horizon weighting;
- reference-aware worst-case weighting;
- spectral or Sobolev preconditioning;
- a carefully designed evaluation of optimization algorithms.

If systematic conflict appears, especially when structured by reference pair, horizon, scenario difficulty, or training phase, then there is evidence of a real optimization problem rather than merely a new optimizer comparison.

---

## 12. Baselines and comparisons

At minimum, compare:

1. **AdamW pooled baseline** using the existing fixed horizon-weighted MSE and equal reference sampling.
2. **SGD or momentum SGD** as a basic optimization reference if computationally affordable.
3. **A generic multi-objective baseline**, such as PCGrad, with its hyperparameters documented.
4. **A loss-weighting baseline**, such as equal weighting versus dynamic soft-max/worst-reference weighting.
5. **The proposed reference-discrepancy-aware method.**

Possible additional baselines:

- GradNorm or another gradient-balancing method;
- MGDA or a Pareto-style multi-objective update;
- a simple projection method without discrepancy awareness;
- curriculum alone;
- an AdamW run with the same update budget and three seeds.

The baseline list should be kept manageable for the 8–15-page course paper. It is better to run a smaller set rigorously than a large set with inconsistent budgets.

Every method should use matched:

- training data;
- batch construction;
- number of optimizer updates;
- maximum epochs or wall-clock budget;
- model architecture;
- validation and checkpoint rules;
- random-seed protocol.

A secondary experiment can compare the solver-anonymous and solver-conditioned models, but the main algorithmic claim should focus on the conditioned setting.

---

## 13. Evaluation metrics

Report per-reference results rather than only one pooled average.

### 13.1 Accuracy

For each reference \(r\in\{H,M,B\}\), report:

- denormalized RMSE;
- MAE;
- relative \(L_2\);
- per-frame or per-horizon error curves;
- optionally gauge waveform error, arrival-time error, and peak error.

### 13.2 Multi-reference aggregate metrics

Report:

\[
L_{\mathrm{mean}}=\frac13(L_H+L_M+L_B),
\]

\[
L_{\mathrm{worst}}=\max(L_H,L_M,L_B),
\]

and, if useful, the standard deviation across references.

The worst-reference metric should receive special attention. A result like the following would be compelling:

| Method | Hydrostatic | MUSCL-HR | Boussinesq | Mean | Worst |
|---|---:|---:|---:|---:|---:|
| AdamW | 0.20 | 0.22 | 0.31 | 0.243 | 0.31 |
| PCGrad | 0.21 | 0.21 | 0.27 | 0.230 | 0.27 |
| Ours | **0.20** | **0.20** | **0.24** | **0.213** | **0.24** |

The exact numbers above are illustrative only, not expected results.

### 13.3 Optimization behavior

Measure:

- convergence curves;
- number of updates to reach a target validation error;
- gradient conflict frequency;
- average and distribution of pairwise cosine similarities;
- gradient norms by reference;
- worst-reference loss over training;
- update norm and clipping frequency;
- wall-clock time;
- peak memory;
- sensitivity to random seed.

### 13.4 Generalization and robustness

For a stronger research version, evaluate:

- distribution shift;
- resolution transfer;
- unseen scenario strengths or bathymetries;
- long-horizon error;
- performance under a fresh final-test pool.

The existing 2,500-scenario test pool has already been inspected extensively for the original paper. For a strict new study, generate or reserve a fresh untouched test pool for algorithm design and final evaluation. Otherwise, clearly state that the existing test results are prior benchmark evidence and that new method selection uses only training/validation data.

---

## 14. Experimental matrix

A reasonable staged matrix is:

### Stage A — Diagnostic

- model: solver-conditioned FNO;
- optimizer: ordinary AdamW;
- data: current matched three-reference training set;
- seeds: at least 3 if affordable;
- output: losses, gradients, cosine similarities, solver discrepancies, horizon structure.

### Stage B — Minimal method test

- AdamW baseline;
- generic PCGrad or equivalent baseline;
- discrepancy-aware gradient correction;
- equal objective weights;
- 3 seeds;
- same update budget.

### Stage C — Weighting and curriculum ablations

- equal weights versus dynamic worst-reference weights;
- gradient correction with and without discrepancy factor \(A_{ij}\);
- curriculum alone;
- correction alone;
- curriculum plus correction.

### Stage D — Stronger validation

- fresh final-test pool;
- distribution shift;
- resolution transfer;
- horizon-wise error;
- wall-clock and memory;
- sensitivity to \(\tau\), \(\beta\), gradient normalization, and correction order.

### Stage E — Optional second PDE benchmark

For a journal-level claim, add one or two standard PDE benchmarks such as Burgers, Darcy flow, or Navier–Stokes. This tests whether the method is a general optimization method rather than a tsunami-specific trick.

For the DS106 course, the tsunami benchmark alone is likely sufficient if the analysis is rigorous and the report clearly explains the optimization formulation.

---

## 15. Ablation plan

The contribution must be decomposed so that any improvement can be attributed correctly.

Recommended ablations:

1. AdamW with equal reference weighting.
2. AdamW with dynamic worst-reference weighting only.
3. Generic PCGrad with equal weighting.
4. Discrepancy-aware correction with equal weighting.
5. Discrepancy-aware correction with dynamic weighting.
6. Disagreement-aware curriculum only.
7. Curriculum plus discrepancy-aware correction.
8. Global discrepancy versus horizon-wise discrepancy.
9. Solver-conditioned versus solver-anonymous input.
10. With and without gradient clipping.
11. Different \(\tau\) values or robust discrepancy normalizations.
12. Different random seeds.

The method should be tested for both average improvement and worst-reference improvement. A method that lowers the average only by sacrificing the weakest reference is not a clear success.

---

## 16. Statistical and reproducibility requirements

A publishable-quality study needs more than one attractive training curve.

Use:

- at least 3 independent training seeds for the course version;
- 5 seeds if the results are intended for a serious paper and compute allows it;
- paired evaluation on identical held-out scenario identities;
- confidence intervals or bootstrap intervals for key metrics;
- consistent checkpoint and early-stopping rules;
- complete reporting of hyperparameters and compute budget;
- fixed validation data for method selection;
- an untouched final test pool where possible.

The original benchmark uses independent scenario pools and has a strong basis for paired evaluation. The new paper should preserve this strength and avoid choosing the method after repeatedly inspecting the final test set.

Record:

- code version or commit;
- configuration files;
- random seeds;
- training duration;
- hardware;
- model parameter count;
- peak GPU memory;
- number of optimizer steps;
- any failed or restarted runs.

---

## 17. Important methodological caveats

### 17.1 Reusing the current benchmark is valid, but the contribution must be new

Using the current submission's dataset and benchmark as the experimental platform is valid. It is especially appropriate because the new project asks a continuation question that the original work intentionally left open.

The report should state explicitly:

- the benchmark and data-generation framework come from the prior submission;
- the original results are treated as prior work and baseline evidence;
- the new work introduces a new optimization formulation and new experiments;
- the original manuscript/results are not rewritten as if they were produced by the new method.

### 17.2 The existing test set may no longer be pristine

The original work has already inspected the 2,500-scenario results and subgroup failures. For strict research practice, hold out a fresh final-test pool for the new algorithm. This is cheap relative to the credibility it provides.

If a fresh pool is impossible, use only training and validation data for design decisions, clearly label the old test results as prior benchmark evidence, and report the limitation.

### 17.3 Gradient conflict may be absent

The project must be falsifiable. If the reference-specific gradients are almost always aligned, then gradient surgery is not justified. The study should report that negative result and pivot toward a different optimization question rather than manufacturing conflict.

### 17.4 Reference disagreement is not automatically physical truth

The solver discrepancy measures numerical-reference disagreement. It does not tell us which solver is physically correct. The method should not claim to preserve “true physics” merely because it preserves gradients associated with a large \(D_{ij}\).

A safer statement is:

> The method distinguishes optimization interference from reference-specific target disagreement under the benchmark's numerical definitions.

### 17.5 Anonymous pooling has an irreducible ambiguity

A solver-anonymous model cannot perfectly represent three different target trajectories for the same input. Poor anonymous performance may reflect missing conditioning information rather than a bad optimizer. Therefore, claims about negative transfer should be based primarily on the conditioned model.

### 17.6 Existing multi-objective methods already exist

PCGrad, GradNorm, MGDA, Pareto methods, and related algorithms are prior art. The novelty cannot be “we project conflicting gradients.” The potentially distinctive part is using measurable numerical-reference discrepancy to decide how strongly conflict should be corrected, and testing whether that distinction matters for neural PDE surrogate training.

The related-work review must check whether a similar reference-aware or physics-aware multi-objective method already exists.

### 17.7 More complexity is not automatically better

The global reference-level method should be implemented first. Horizon-wise and spatially localized versions are extensions. A method that is simple, stable, and interpretable is preferable to an elaborate method that cannot be fairly benchmarked.

---

## 18. Potential alternative directions if the main diagnostic fails

If gradient conflict is not systematic, the broader optimization project can still continue with one of these directions.

### 18.1 Bilevel learning of the temporal loss

The current temporal weights are hand-designed:

\[
w_{\min}=0.8,\qquad w_{\max}=2.0,\qquad p=1.7.
\]

Make their parameterization learnable. The inner problem trains model parameters:

\[
\theta^*(\phi)
=\arg\min_\theta
\sum_{t=1}^{50}w_t(\phi)L_t(\theta),
\]

and an outer validation problem learns the loss parameters:

\[
\min_\phi V(\theta^*(\phi)).
\]

This teaches bilevel optimization, hypergradients, truncated unrolling, implicit differentiation, smooth positive weight constraints, and regularization. A stronger version could learn reference-specific or horizon-specific schedules.

### 18.2 Spectral or mode-wise preconditioning for FNO

AdamW is essentially a coordinate-wise adaptive method. FNO parameters have a structured Fourier-mode organization. A mode-wise preconditioner could use gradient covariance or Kronecker-style approximations:

\[
\Delta W_k=-\eta P_kG_kQ_k.
\]

This would teach preconditioning, curvature, complex gradients, spectral structure, and memory/stability trade-offs. It is high risk and should be tested on more than tsunami if making a general optimizer claim.

### 18.3 Adaptive spectral/Sobolev wavefield training

MSE can underemphasize phase, derivatives, and frequency content. A richer loss could combine

\[
L=\lambda_1L_{\mathrm{MSE}}
+\lambda_2L_{\nabla}
+\lambda_3L_{\mathrm{spectral}}
+\lambda_4L_{\mathrm{phase}}.
\]

The stronger contribution is to adapt \(\lambda_i\) according to gradient magnitudes or conflict, possibly by forecast horizon. Adding an FFT term alone is not enough for a strong novelty claim because Sobolev and spectral losses already have precedent.

### 18.4 Evaluation-only algorithm study

If time is short, compare 2–5 optimizers fairly on the same benchmark. This is a valid DS106 project, but it has a lower research ceiling than diagnosing and solving reference-specific interference.

---

## 19. Suggested paper structure for the 8–15-page Elsevier report

A compact two-column report could use this structure:

1. **Abstract**
   - problem;
   - current limitation of pooled multi-reference training;
   - proposed reference-aware method;
   - main findings.
2. **Introduction**
   - neural PDE surrogate motivation;
   - numerical references are not neutral labels;
   - gap: reference disagreement has not been connected to optimization geometry in this benchmark;
   - contributions.
3. **Related Work**
   - multi-objective optimization;
   - gradient conflict methods;
   - neural operators and tsunami surrogates;
   - numerical-reference-aware evaluation.
4. **Prior Benchmark and Problem Formulation**
   - three solvers;
   - matched scenarios and 50 frames;
   - solver-conditioned mapping;
   - baseline AdamW and fixed horizon-weighted MSE.
5. **Diagnostic Analysis**
   - gradient cosine similarity;
   - reference discrepancy;
   - relationship between them;
   - horizon behavior.
6. **Proposed Method**
   - discrepancy normalization;
   - agreement factor;
   - gradient correction;
   - dynamic weighting if included;
   - algorithm pseudocode.
7. **Experimental Setup**
   - data splits;
   - seeds;
   - baselines;
   - metrics;
   - compute and fairness controls.
8. **Results**
   - main accuracy table;
   - worst-reference performance;
   - convergence;
   - conflict statistics;
   - ablations;
   - robustness tests.
9. **Limitations and Discussion**
   - finite-horizon synthetic benchmark;
   - numerical discrepancy is not physical truth;
   - anonymous mapping ambiguity;
   - compute and generalization limits.
10. **Conclusion**
    - answer whether reference disagreement creates harmful optimization conflict;
    - summarize whether the method improves average and worst-reference behavior.

The report should not present an algorithm as successful before showing the diagnostic evidence that motivates it.

---

## 20. Pseudocode-level workflow

```text
Input:
    matched batch of scenarios
    labels y_H, y_M, y_B
    model f_theta(x, r)
    discrepancy scale tau

1. Compute predictions for each reference.
2. Compute reference-specific horizon-weighted losses L_H, L_M, L_B.
3. Compute or retrieve target discrepancies D_HM, D_HB, D_MB.
4. Backpropagate each L_r separately to obtain g_H, g_M, g_B.
5. Compute pairwise gradient cosines C_HM, C_HB, C_MB.
6. Log losses, gradient norms, discrepancies, and cosines.
7. For every conflicting pair (i, j):
       A_ij = exp(-normalized(D_ij) / tau)
       remove A_ij times the conflicting projection from g_i
8. Combine corrected gradients with equal weights initially.
9. Optionally compute dynamic worst-reference weights alpha_r.
10. Form g_star = sum_r alpha_r * corrected_g_r.
11. Apply AdamW moment and parameter update using g_star.
12. Evaluate each reference separately on validation data.
13. Select checkpoints only from validation metrics.
14. Evaluate once on a fresh final-test pool if available.
```

The implementation must define whether gradients are flattened over all parameters, how shared parameters and reference-conditioning parameters are handled, how pairwise corrections are ordered, and how corrected gradients are passed through AdamW.

---

## 21. What would count as a successful result?

A strong result does not require a dramatic improvement on every metric. The following would already be valuable:

1. The diagnostic shows that solver disagreement is associated with structured gradient conflict.
2. The proposed method reduces harmful conflict without collapsing reference-specific behavior.
3. The worst-reference error decreases consistently across seeds.
4. Mean error is maintained or improved.
5. The method is competitive with generic PCGrad or other baselines.
6. The improvement survives ablations and a fresh final test set.
7. The method's extra cost is measured and justified.

Even if the method improves only modestly, the analysis may still be a useful contribution if it clearly demonstrates how numerical-reference disagreement interacts with optimization geometry.

A particularly clear narrative would be:

\[
\text{single-reference ERM}
\rightarrow
\text{pooled ERM}
\rightarrow
\text{solver-conditioned ERM}
\rightarrow
\boxed{\text{reference-aware multi-objective optimization}}.
\]

---

## 22. Short Vietnamese message to ask the professor

### Written version

> Thầy cho em hỏi hướng này có phù hợp với dạng **cải tiến thuật toán tối ưu** của đồ án không ạ?
>
> Hiện em có một bài đang submission về **neural PDE surrogate cho mô phỏng sóng tsunami**. Trong bài đó, cùng một kịch bản được sinh nhãn bởi ba numerical solver khác nhau: Hydrostatic, MUSCL-HR và Boussinesq. Bài hiện tại chủ yếu nghiên cứu ảnh hưởng của **reference solver đến kết quả đánh giá**, còn phần training vẫn dùng AdamW và một loss cố định.
>
> Với đồ án môn này, em muốn mở rộng theo hướng **multi-objective optimization**: coi mỗi solver là một objective riêng, rồi phân tích xem gradient của các objective có xung đột khi train chung hay không. Sau đó em muốn đề xuất một phương pháp **reference-aware optimization**, tức là dựa vào mức độ khác biệt giữa các solver để quyết định khi nào nên giảm gradient conflict và khi nào nên giữ lại vì đó có thể là khác biệt numerical/physical thật sự.
>
> Em dự định so sánh với AdamW và một số multi-objective baseline, đánh giá theo accuracy trên từng solver, worst-case solver, convergence và gradient conflict. Dataset/benchmark sẽ được reuse từ bài đang submission, nhưng thuật toán tối ưu và phần thực nghiệm mới sẽ là contribution mới.
>
> Thầy thấy scope như vậy có phù hợp và đủ mức cho hướng “cải tiến thuật toán” không ạ?

### Very short spoken version

> Em đang có một paper submission về neural surrogate cho tsunami với ba numerical solver làm reference. Paper hiện tại dùng AdamW và loss cố định, chủ yếu nghiên cứu ảnh hưởng của solver đến evaluation. Em muốn làm đồ án bằng cách coi ba solver là ba optimization objectives, đo gradient conflict giữa chúng, rồi đề xuất một reference-aware gradient method dựa trên mức độ solver disagreement. Sau đó em benchmark với AdamW và các multi-objective methods, đánh giá cả accuracy từng solver và worst-reference performance. Như vậy có được xem là hướng cải tiến thuật toán đủ mạnh cho đồ án không ạ?

---

## 23. Final recommended order of work

1. Freeze the existing manuscript and its reported results as prior work.
2. Inspect the current training code and confirm how to obtain separate per-reference gradients.
3. Run the cheap AdamW diagnostic before changing the optimizer.
4. Determine whether conflict is frequent, structured, and related to target disagreement.
5. If the diagnostic supports the idea, implement the simplest global discrepancy-aware correction.
6. Compare against AdamW and one generic multi-objective baseline under matched budgets.
7. Add dynamic worst-reference weighting only after the correction is understood.
8. Add curriculum or horizon-wise disagreement only if the base method is stable.
9. Use a fresh final-test pool or clearly separate prior-test evidence from new method selection.
10. Run seed replication, ablations, and robustness tests.
11. Write the report around the observed optimization problem, not around an assumed success.
12. Decide after the evidence whether the result is best presented as a DS106 course paper, a workshop-style paper, or the basis for a larger journal submission.

The first real decision is therefore empirical:

> **Do the three reference-specific gradients actually conflict in a structured way?**

Everything else should follow from that answer.
