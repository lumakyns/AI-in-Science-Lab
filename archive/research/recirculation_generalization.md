# Beyond autoencoders: supervised recirculation through an output-to-input bridge

Research date: 2026-10-21. This is a research memo, not a claim of a new published algorithm or a change to the running MNIST experiment. Literature statements, derivations, numerical checks, and conjectures are distinguished below. The reproducible checks are in `scripts/verify_recirculation_bridge.py`; generated results belong under `checkpoints/recirculation/generalization_checks/` and are ignored artifacts.

## The answer in plain language

A supervised network can recirculate errors if an additional map translates output errors into input perturbations. It does **not** have to reconstruct the original input. For a linear feedforward network with matrix F, a particularly useful choice is a right inverse R satisfying FR = I on the output-error directions of interest. More generally, a positive symmetric part of FR gives useful directional alignment. R = F^T is another sufficient choice for nonnegative alignment, although computing it directly defeats the original motivation for avoiding weight transport.

Baldi's identity connection is one setting of the general family, not the uniquely correct feedback connection. A nonidentity bridge is already discussed in his appendix, and supervised forward-error methods already exist. A defensible new direction must improve what that bridge preserves, how it is obtained, or how it adapts to rank loss—not merely add it.

The important quantity is **usable error rank**: which independent error directions reach each hidden layer, with adequate gain and useful orientation. Raw rank, effective rank, cosine similarity, and dynamical stability answer different questions. More rank is not universally better; a low-dimensional task can need only a small error channel. Conversely, high cosine can hide discarded modes.

The proposed research direction is **rank-certified, probe-calibrated recirculation**: use shared forward perturbations to calibrate a damped right-inverse bridge on the current task-error subspace; measure what it loses; refresh or expand that subspace when a certificate fails. An optional matched-probe local correction can recover BP signals exactly in suitable linear architectures. The ingredients have substantial prior art; the potentially distinct contribution is their one-loop construction, finite-probe identities, and explicit rank/conditioning audit. Novelty remains a hypothesis requiring a deeper comparison and empirical results.

## 1. Existing work we must not rediscover

| Source | What is already there | Consequence for this project |
|---|---|---|
| [Baldi & Sadowski, 2018, Appendix B4](https://peterjsadowski.github.io/assets/pdf/baldi2018recirculation.pdf#page=36) | Nonidentity output-to-input channel C3 = C and hidden error BC(I − O); interpreted as adaptive random BP. | R ≠ I by itself is not new. The appendix still frames its derivation around autoencoder targets. |
| [Nøkland, 2016, IFA/DFA](https://arxiv.org/pdf/1609.01596) | Indirect feedback alignment sends output error to the first hidden layer, then forward through subsequent layers; DFA broadcasts separately. | Forward delivery of a supervised teaching signal predates the 2018 paper. |
| [Kohan, Rietman & Siegelmann, 2018, Error Forward-Propagation](https://arxiv.org/pdf/1808.03357) | An output-to-early-layer loop in recurrent supervised classifiers. | Non-autoencoder loop architecture is established; its recurrent/clamped dynamics differ from our simple two-pass rule. |
| [Dellaferrera & Kreiman, 2022, PEPITA](https://proceedings.mlr.press/v162/dellaferrera22a/dellaferrera22a.pdf) | A second forward pass perturbs inputs using projected output error and trains from activity differences. | Closest simple algorithmic baseline. Sign conventions and original versus perturbed presynaptic activities must be matched. |
| [Srinivasan et al., ICLR 2024](https://arxiv.org/pdf/2302.05440) | An adaptive-feedback interpretation of forward-only learning and extensions including weight mirroring. | Learning the PEPITA bridge with mirroring is also prior art. |
| [Tavakoli, Sadowski & Baldi, Tourbillon](https://arxiv.org/pdf/2107.06424) | Stacked circular autoencoders plus a supervised output layer, with optional nonsymmetric error propagation. | A direct Baldi follow-up, but its internal modules remain autoencoders. |
| [Meulemans et al., 2020, target-propagation theory](https://proceedings.neurips.cc/paper_files/paper/2020/file/e7a425c6ece20cbc9056f98699b53c6f-Paper.pdf) and [Ernoult et al., 2022](https://proceedings.mlr.press/v162/ernoult22a/ernoult22a.pdf) | Pseudoinverse/Gauss–Newton connections and schemes for learning BP-like feedback targets. | “Learn an inverse” is not a novel thesis; products of layer pseudoinverses need not equal the whole-map pseudoinverse. |
| [Meulemans et al., 2021, Deep Feedback Control](https://proceedings.neurips.cc/paper_files/paper/2021/file/25048eb6a33209cb5a815bff0cf6887c-Paper.pdf) | Feedback subspace and stability conditions, dynamical inversion, and conditional GN/minimum-norm results. | Strong prior art for control-based feedback guarantees. Its feedback acts at multiple hidden layers, unlike our single input port. |
| [Akrout et al., 2019, weight mirrors](https://arxiv.org/pdf/1904.05391) | Noise correlations can train feedback weights toward transposed forward weights. | Noise-derived adjoints are known, not a novel claim here. |
| [Garg & Vempala, 2022](https://proceedings.mlr.press/v151/garg22a.html) | FA can differ provably from GD on low-rank factorization, depending on target rank and representation width. | A theoretical rank warning; its rank parameter is not simply the rank of our input bridge. |
| [Hanut & Kadmon, low-dimensional feedback](https://arxiv.org/pdf/2502.20580) | Learned low-rank error pathways; task dimension matters, with an error-covariance/Oja variant. | Adaptive low-rank feedback and following dominant residual modes are established ideas. |
| [Boeshertz, Pascanu & Clopath, 2026 preprint](https://arxiv.org/pdf/2606.11123) | Effective-rank collapse in FA; normalization and Muon improve update geometry in their experiments. | The user's rank hypothesis has direct contemporary support, but in FA architectures rather than this particular loop. |
| [Caillon et al., FOTON](https://arxiv.org/abs/2512.20668) | Exact BP equivalence in a linear orthogonal setting; a forward-only method beyond that setting. | Orthogonalizing the path to force BP equivalence is already covered. |
| [As-Saquib et al., FTP, revised Oct. 2026](https://arxiv.org/pdf/2506.11030) | First-hidden-layer targets plus forward target propagation, including comparison to IFA. | Another close one-port supervised baseline. |
| [Safaai, Reddy & Sabatini, 2026 preprint](https://arxiv.org/pdf/2607.18574) | Activity/error inverse-moment conditioning of DFA. | Local whitening/preconditioning alone is not a novel contribution. |
| [Caillon et al., GrAPE, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/2b1c608556ce011d3a3381fe0c527136-Paper-Conference.pdf) | Forward Jacobian estimates and feedback alignment objectives. | Forward-probe feedback calibration also has close modern precedents. |

This is a targeted primary-source review, not an exhaustive priority search. Preprint claims above are reported as claims, not independently reproduced benchmark results.

## 2. Define the generalization carefully

Use **column vectors** throughout this memo; the current code uses row batches and stores PyTorch weights accordingly. Let

- x ∈ R^n be an input and t ∈ R^m a supervised target;
- h_l = A_l h_(l−1), h_0 = x, and y = F x, with F = A_L ... A_1;
- P_l = A_l ... A_1 map input to hidden layer l;
- Q_l = A_L ... A_(l+1) map that hidden layer to output, so F = Q_l P_l;
- e = t − y for squared loss 1/2 ||y − t||²;
- R ∈ R^(n×m) be an output-error-to-input feedback bridge.

For a general output loss, define e = −∂loss/∂y, in explicitly chosen output coordinates. For cross-entropy with logits, use t − softmax(logits) in logit coordinates. Do not mix a probability-space residual with a logit-space Jacobian.

Perform the original pass, cache its activities, then probe x^- = x − ε R e. Define

s_l = [h_l(x) − h_l(x^-)] / ε = P_l R e.

The local hidden update is ΔA_l = η s_l h_(l−1)^T; the final layer uses ΔA_L = η e h_(L−1)^T. All signals use frozen pre-update weights. These are independent-layer SGD updates for now, not tied parameters.

**Baldi is recovered exactly:** set m = n, t = x, R = I, ε = 1. Then x^- = x − (x − y) = y, and s_l is the original activity minus the activity after recirculating the reconstruction. This also fixes the sign convention.

A nonlinear bridge can use r(x,e), with r(x,0)=0 and derivative ∂r/∂e = R_x locally. A direct nonlinear map g(output) should enter through g(t) − g(y), not through g(y) alone: differencing preserves a zero teaching signal when the output is already correct. It need not reconstruct x. A fixed bias-free linear R is the smallest first experiment; a deep recurrence network is not required by dimensions alone. The loop is a training-time teaching channel: retain the original x and evaluate predictions on the ordinary first forward pass. Do not permanently replace the dataset input by its perturbed version.

## 3. What the bridge has to accomplish

### 3.1 Exact BP at all hidden layers is usually too much to ask

The true BP teaching signal is b_l = Q_l^T e. Exact equality requires

P_l R = Q_l^T  for every hidden layer l.

Stack these equations: P_stack R = B_stack. A solution exists iff

(I − P_stack P_stack^†) B_stack = 0.

On an error subspace with orthonormal basis U ∈ R^(m×r), replace B_stack with B_stack U and solve only for R U. Generically, the equations are inconsistent even if each individual prefix has full row rank. Consecutive layers impose extra compatibility constraints. For example, exact matching at l and l+1 implies

A_(l+1) A_(l+1)^T Q_(l+1)^T = Q_(l+1)^T.

This is an isometry condition on the relevant backward subspace, not a generic property of independent weights. Tying each encoder–decoder pair is insufficient to impose all these conditions in a deep chain.

### 3.2 A weaker, useful condition is positive alignment

For linear networks, a simple identity holds at **every** hidden layer:

b_l^T s_l = e^T Q_l P_l R e = e^T F R e.

Thus the alignment numerator is the same scalar across all hidden layers. The cosines can differ because their denominator norms differ. In our existing four-layer linear diagnostic, all three cosines sharing a sign is therefore not three independent pieces of evidence.

A sufficient uniform condition on an error subspace S = range(U) is

U^T sym(FR) U ≽ μ I_r,  μ > 0.

It gives b_l^T s_l ≥ μ ||e||² for every nonzero e ∈ S. Full rank alone does not imply this; replacing a good R by −R preserves rank and reverses the alignment.

For independent linear weights and a **single example**, the true gradient is G_l = −b_l h_(l−1)^T, and the approximate gradient is Gtilde_l = −s_l h_(l−1)^T. Therefore

<G_l, Gtilde_l> = ||h_(l−1)||² e^T F R e.

With an exact final-layer gradient, the combined direction is downhill whenever its total gradient inner product is positive. A sufficiently small SGD step then lowers that example's loss. This is not a global convergence theorem. For a minibatch, cross-example terms appear in the aggregate parameter-gradient inner product; positive per-example signal alignment alone does not settle that case.

**Two useful constructions:**

1. R = F^T gives FR = FF^T ≽ 0. Alignment is strictly positive whenever F^T e ≠ 0.
2. If F has full row rank, R = F^† gives FR = I_m and b_l^T s_l = ||e||². It is the minimum-input-norm right inverse. Other right inverses add a map N with FN = 0.

A right inverse means that an input perturbation R e produces output perturbation e. It does **not** mean R(Fx) = x: a classifier can have many inputs per label and no global input-reconstructing inverse.

For rank-deficient F, F F^† is a projector, not the identity. A right inverse on all errors cannot exist. Task-subspace inversion or damping is needed, and some BP-relevant directions can remain inaccessible to input-only perturbations.

### 3.3 Why tied/nonlinear updates need separate care

When weights are tied, shared-parameter gradients sum encoder and decoder contributions. Their inner products contain cross terms absent from the individual-layer calculation. Positivity of all b_l^T s_l does not guarantee positivity after sharing. This is directly relevant to the earlier exploding tied-recirculation runs.

For nonlinear networks, let P_l = ∂h_l/∂x, Q_l = ∂y/∂h_l, and J = Q_l P_l. Small input probes give s_l ≈ P_l R_x e, so the same hidden-activation identity holds to first order. But converting activation targets into weight updates introduces local Jacobians. In general, b^T s > 0 does not imply b^T K_l s > 0, where K_l = (∂h_l/∂θ_l)(∂h_l/∂θ_l)^T. Activation derivatives cannot simply be dropped. ReLU gate changes, saturation, finite ε, and stale feedback add further approximation error.

## 4. Rank: the useful property and its limits

At frozen linear weights, collect errors in E ∈ R^(m×N). BP produces B_l = Q_l^T E; recirculation produces S_l = P_l R E. Consequently,

rank(S_l) ≤ min(rank(P_l), rank(R), rank(E)).

There is **no extra one-port input-prefix restriction** in rank(B_l) ≤ min(rank(Q_l), rank(E)). A narrow input or early bottleneck can block feedback directions even when later hidden layers are wide. This is a fixed-network, fixed-linearization statement—not a proof that the network can never adapt its representation to solve the task.

Several distinctions matter:

- An individual error vector is not a rich matrix-rank statistic. With batch size 1, each dense outer-product weight update has rank at most 1 under both BP and recirculation. Diagnose a matrix of errors or flattened gradients across many examples at frozen weights, or a clearly labeled temporal window.
- Algebraic rank counts nonzero modes. Effective rank measures how evenly energy is distributed. Report both the singular-value entropy rank and an energy-based participation ratio, without mixing their definitions.
- Error covariance support matters more than ambient output width. For C-class softmax, the logit residual sums to zero, so its span is contained in a C−1 dimensional hyperplane. This does not bound nonlinear hidden-error matrices across examples by C−1: sample-dependent Jacobians and activation gates can rotate those vectors.
- Rank alone does not establish that the preserved subspace is the task-relevant one. Nor does it control tiny singular values. A full-rank channel with condition number 10^8 is almost blind in one direction numerically.
- High rank is not always the objective. Removing nuisance residual modes can be useful; removing a rare but necessary class direction can be disastrous. Track per-class and held-out performance, not only leading residual variance.

For a specified task basis U, preserving all its coordinates through a fixed map needs rank(P_l R U)=r. A weaker BP-relative requirement is

ker(P_l R U) ⊆ ker(Q_l^T U).

Then feedback does not silently discard an error direction that BP would use. Equality of ranks alone does not establish this kernel inclusion or directional alignment. The right-inverse condition FR=I implies P_l R has full column rank m at every hidden layer, since Q_l(P_l R)=I; this is a particularly clean rank-preservation guarantee when feasible.

**A concrete trap:** B = diag(10,1,1) and S = diag(10,0,0) have cosine 0.99015, but S has rank 1 rather than rank 3. Two error modes vanished. This is why the current cosine/sign-agreement logs are useful but incomplete.

A fixed deficient bridge cannot generally repair itself merely by extra cycles. With frozen F,R and residual update e_(k+1)=(I−αFR)e_k, any e in ker(R) remains invisible to R at every cycle. In fact R(I−αFR)^k=(I−αRF)^k R, so stacking those observations does not increase the bridge's error rank. Different learned bridges or other state dynamics can change this conclusion; arbitrary recurrent architectures are not covered by this simple result.

## 5. Proposed idea: rank-certified, probe-calibrated recirculation

### 5.1 Start with a small, measured right inverse

Maintain a task-error basis U ∈ R^(m×r), initially using all output coordinates or the known softmax sum-zero subspace. A lower-rank adaptive U can be considered later, but should include residual exploration so rare modes are not permanently excluded.

Use K input probe directions in an orthonormal matrix V ∈ R^(n×K), with K ≥ r. Freeze the forward weights. Measure central forward differences at the current x:

Y[:,j] = [f(x+ε v_j) − f(x−ε v_j)]/(2ε) ≈ J V[:,j].

Let Z = U^T Y and define the feedback map

R_λ = V Z^T (Z Z^T + λ I_r)^−1 U^T.

This is one output-to-input bridge. It needs a small r×r solve, not a whole backward pass or a separate output-to-each-layer network. In a linear network with λ=0 and Z full row rank,

U^T F R_0 U = I_r.

For e=U c, it follows that e^T F R_0 e = ||c||². Thus raw hidden updates have positive alignment in the independent-weight, single-example linear setting—even when their cosines are far below 1.

When U spans the entire output space, this is a genuine right inverse FR_0=I. For a reduced U, only the projected loop is inverted: FR_0 e can have components outside range(U). Do not claim exact correction of the complete output vector in that case.

With λ>0, the projected loop is

U^T F R_λ U = Z Z^T (Z Z^T + λ I)^−1.

Its eigenvalues are σ_i(Z)^2/(σ_i(Z)^2+λ): positive on retained modes, at most 1, and explicitly attenuated on weak modes. For orthonormal V,U,

||R_λ||_2 ≤ 1/(2 sqrt(λ)).

Damping bounds bridge amplification but cannot restore missing rank. Rank and conditioning are separately certified by the spectrum of Z. The certificate must be refreshed as the forward weights or nonlinear operating point change.

**Why this is a candidate rather than a solved innovation:** pseudoinverses, derivative-free subspace solves, PEPITA, weight mirroring, and learned low-rank feedback already exist. The precise proposed package is one-port, task-subspace, forward-probe calibration with a measured weakest-mode guarantee and adaptive probe/basis budget. A finite-probe proof alone is not evidence of a speed or accuracy advantage.

### 5.2 Optional matched-probe correction: when exact BP is possible

There is a useful second construction. For arbitrary shared probes X ∈ R^(n×K), collect Y=F X and hidden responses D_l=P_l X. Set

Rhat = X Y^T/K,    C_l = D_l D_l^T/K.

Then, exactly in a linear network,

P_l Rhat e = C_l Q_l^T e = C_l b_l.

Therefore

C_l^† P_l Rhat e = projection_range(D_l) b_l.

If D_l has full row rank, this is exactly the BP signal. Crucially, the empirical Rhat need not be close to F^T: the same probe covariance appears in both expressions and cancels. Independent probe sets would not have that exact cancellation. Damping gives a controlled spectral filter, but loses exact BP equality.

This offers a distinction between **rank sufficient for descent** and **rank sufficient for exact BP reconstruction**. K near the task rank can suffice for the right-inverse alignment construction. Exact local BP from untargeted matched probes can require K at least the reachable hidden dimension, which may be much larger. A rank-r signal matrix can still have the wrong r-dimensional subspace.

Input-only probes cannot span a hidden layer beyond range(P_l). If exact BP there is desired, define the missing task map

M_l = (I − P_l P_l^†) Q_l^T U.

A minimal residual direct-feedback correction to this projected estimator has rank rank(M_l); truncated SVD gives the best lower-rank approximation in Frobenius norm. This minimum is specific to repairing the chosen reachable projection. It is not a lower bound on every possible feedback architecture or on all learned nonlinear decoders. It suggests adding feedback only where the one-port rank/geometry audit fails, rather than defaulting to a dense broadcast at every layer.

### 5.3 What must be safeguarded

- Use independent forward weights first; tied-gradient aggregation is a separate question.
- Control both ε (probe linearity) and η (parameter update size). A stable activation loop does not imply stable parameter learning.
- Distinguish one contrastive probe from an endlessly recurrent settling procedure. Repeated residual correction has e_(k+1)≈(I−αJR)e_k; stability requires this operator to contract on the intended subspace. For R=J^T, an appropriate α is bounded by 2/||J||² on reachable modes. There is no strict contraction on unresolved null modes.
- Constrain or damp the bridge, and consider an actual-loss acceptance/backtracking check on the sampled objective. It can reject harmful directions without computing BP; it does not certify population-loss progress or prove convergence.
- Measure extra probe passes, matrix solves, memory, and refresh rate. Eliminating a backward pass is not automatically faster; frequent wide probe batches may be much slower.
- Exact identities use frozen weights and a consistent operating point. EMA covariance, reuse across examples, probe noise, finite precision, and nonlinear gates need separate experiments.

## 6. Numerical checks completed

![Frozen matrix checks: cosine can conceal rank loss, and sufficient probe span recovers BP](../checkpoints/recirculation/generalization_checks/findings.png)

The checks use a non-autoencoder linear chain 8→6→5→3 and separate deliberately deficient examples. They neither download data nor launch W&B or modify the current sweep.

| Check | Result |
|---|---|
| Adjoint bridge R=F^T | Raw hidden cosines 0.8764 and 0.5626, both with full task signal rank 3. |
| Same bridge with its sign reversed | Same rank, negative cosines. Full rank alone is insufficient. |
| Local prefix-Gram correction | Relative BP-signal errors below 1e−14 with exact Grams. |
| A common raw bridge fitted to all hidden BP maps | Best relative residual 0.5399: exact equality is generically infeasible. |
| Forward-probe right inverse, K=4, output width 3 | ||FR−I|| relative error 1.84e−15; full signal rank 3 at both hidden layers, cosines only 0.3327 and 0.2473. |
| Rank-2 task basis with K=2 probes | Projected-loop relative error 9.74e−16; feedback rank 2. |
| Cosine versus rank counterexample | Cosine 0.99015 despite rank falling 3→1. |
| Matched probes, K=7 | Rhat differs from F^T by 191% relatively; corrected hidden signals still match BP to below 7e−13. |
| K=1…6 matched-probe sweep | Projection identity holds for every K. At K=3 the signal already has rank 3 yet differs from BP by 76%; at K=6 BP is recovered to numerical precision. |
| Wide hidden layer 2→5→3 | Reachable projected BP map has rank 2; the missing residual map has rank 3. A rank-2 bypass leaves ≈17% relative error, a rank-3 repair is exact. A simple rank-count difference is not the correct repair budget. |
| Eight cycles with one fixed rank-1 bridge | Stacked temporal signal rank remains 1. |

Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 /opt/conda/envs/biai_env/bin/python \
  scripts/verify_recirculation_bridge.py \
  --output checkpoints/recirculation/generalization_checks/algebra.json
```

All checks also passed across 20 additional random seeds. These are float64 matrix identities/counterexamples, not MNIST accuracy results, biological implementation demonstrations, or a global convergence proof.

## 7. A concrete next experiment

Begin with supervised linear teacher–student regression, before MNIST classification. Vary output/task rank independently of input and hidden widths; include both narrowing and expanding prefixes. Use independent parameters, batch size 1, SGD, and a stable common learning-rate search. Evaluate BP, fixed random R, oracle F^T, oracle F^†, probe-calibrated R_λ, and matched-probe correction. Include equal-pass/compute-budget comparisons and a final-layer-only control.

Sweep K/r, λ, bridge refresh interval, probe ε, and true task rank. Use at least five seeds for a substantive claim. Keep a full-output basis as the primary control; compare adaptive error bases only afterward. Report time/forward-call budget to a fixed held-out error, not just epochs.

The discriminating hypotheses are:

1. Right-inverse calibration preserves usable error modes and yields positive **actual hidden-parameter** gradient alignment in the linear independent case without requiring cosine=1.
2. Matched versus independently estimated probe covariances differ substantially at finite K, as the exact cancellation predicts.
3. Numerical conditioning/weakest-mode coverage predicts failure beyond aggregate cosine or norm ratios.
4. Adaptive rank can reduce feedback cost only if omitted task modes are audited; blindly following leading variance may sacrifice rare classes or directions.
5. Expanding-prefix deficits motivate selective residual feedback, whereas increasing K alone cannot exceed input reachability.

Then use supervised MNIST, for example 784→20→784→20→10, with labels and a 784×10 feedback bridge. Start with linear logits and squared one-hot loss to isolate the theorem; subsequently test softmax cross-entropy with the correct logit error and a sum-zero error basis. ReLU/nonlinear networks and tied weights should be separate follow-ups, because the proven linear parameter-descent argument does not transfer automatically.

A new implementation under `src/` must follow AGENTS.md's full experiment layout and own its dataset implementation. This memo deliberately leaves the current reconstruction model, configuration, running processes, and historical artifacts untouched.

## 8. Diagnostics that make the rank claim testable

At a frozen checkpoint, collect many examples or independent output-error probes. Keep the three objects separate: output error matrix, propagated hidden error matrix, and flattened parameter-gradient matrix. Report:

- singular spectra, numerical rank at explicit relative thresholds, singular-value entropy rank, and energy participation ratio;
- BP-relevant signal energy lost by projection, and the kernel/subspace test on U;
- minimum singular value of U^T J R U or measured Z, condition number, and the minimum eigenvalue of sym(U^T J R U);
- layer signal cosine/norm/sign metrics plus the cosine, norm ratio, and inner product of the **full shared-parameter update** if parameters are tied;
- directional descent per unit step and actual loss change after an accepted step;
- class-balanced held-out accuracy/loss and coverage of residual directions outside the current U;
- forward passes, feedback storage, solve cost, wall time, and failures.

Do not interpret a positive cosine as exact gradients, a rank match as correct subspaces, a loop contraction as training convergence, or a finite linear identity as an established novel algorithm. The credible research claim would be that a measured, adaptive one-port feedback budget preserves the task's useful error geometry and improves a controlled compute/performance tradeoff over PEPITA/IFA and relevant learned-feedback baselines.
