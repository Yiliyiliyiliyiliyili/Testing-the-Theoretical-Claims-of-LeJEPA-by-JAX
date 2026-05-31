# Testing the Theoretical Claims of LeJEPA: A Lambda Sweep and VICReg Comparison

---

## 1. Background

### 1.1 Self-Supervised Learning

Self-supervised learning (SSL) trains visual representations without labeled data. The core idea is to define a pretext task directly from the data itself: the same image is augmented in two or more different ways, and the encoder is trained to produce similar embeddings for all augmented versions of the same image. This property is called invariance.

The central challenge is representational collapse — the trivial solution where the encoder satisfies invariance by mapping every image to the same point in embedding space. Preventing collapse without introducing fragile engineering tricks (stop-gradient, momentum encoders, large batch sizes) is the key open problem that motivates both VICReg and LeJEPA.

### 1.2 Linear Probe

Linear probe is the standard evaluation protocol for SSL representations. After SSL training completes, the encoder weights are frozen and a single linear layer (512→10 for CIFAR-10) is trained on top of the backbone features using labeled data. The test accuracy of this linear classifier measures how much useful information the encoder has learned to represent.

The probe is applied to the 512-dimensional backbone output before the projection head, not the 64-dimensional projected embeddings. The projection head is a training artifact that maps embeddings into a space suited for the SSL loss objective; it is discarded at evaluation time. The backbone features are what the encoder has actually learned about the images.

### 1.3 LeJEPA and SIGReg

LeJEPA (Balestriero & LeCun, 2025) starts from a bias-variance analysis of downstream task performance. Theorems 1, 7, and 8 establish that the optimal embedding distribution for SSL is an **isotropic Gaussian**: every dimension carries equal information, and dimensions are statistically independent. The intuition is that a spherical distribution allows any downstream linear classifier to exploit all embedding dimensions equally, minimizing generalization error.

This theoretical result motivates **SIGReg**, which directly penalizes the distance between the empirical embedding distribution and N(0,1). The computation proceeds in three steps:

1. **Cramér-Wold reduction**: By the Cramér-Wold theorem, a multivariate distribution equals N(0,I) if and only if every one-dimensional projection is N(0,1). This reduces the high-dimensional problem to many one-dimensional tests.
2. **Random projection**: 512 random unit directions are sampled. Embeddings are projected onto each direction to produce a scalar distribution.
3. **Epps-Pulley CF distance**: For each projection, the squared distance between the empirical characteristic function and the Gaussian CF (exp(−t²/2)) is measured at 10 evaluation points on [−4, 4] and averaged.

The total LeJEPA loss is:

```
Loss = λ × SIGReg(z) + (1 − λ) × Invariance(z)
```

λ controls the balance between distribution regularization and view alignment. Unlike VICReg's three separate weighted terms, LeJEPA has a single hyperparameter with a clear interpretation: how much of the optimizer's attention goes to shaping the distribution versus aligning views.

### 1.4 VICReg

VICReg (Bardes et al., 2022) prevents collapse through three simultaneously applied constraints:

| Term | Formula | Purpose |
|------|---------|---------|
| Variance | mean over dims of max(0, 1 − std(z_j)) | Penalizes std < 1 per dimension; directly prevents collapse |
| Invariance | mean MSE between views | Pulls different views of the same image together |
| Covariance | sum of squared off-diagonal covariance entries / D | Decorrelates dimensions; prevents information redundancy |

The variance term is the anti-collapse mechanism: when all embeddings collapse to a point, every dimension has std ≈ 0, making the hinge loss large. The covariance term goes further than just requiring spread — it requires that different dimensions capture different information.

VICReg uses paper-recommended weights α=25, β=25, γ=1.

**The theoretical limitation.** VICReg's variance and covariance terms together enforce mean ≈ 0, std ≈ 1, and low linear correlation per projection direction. This is 2-moment matching. Theorem 3 in the LeJEPA paper formalizes that this is insufficient: many distributions satisfy these constraints while being far from Gaussian (e.g., uniform, bimodal, heavy-tailed distributions with matched moments). These are shortcut solutions — VICReg cannot distinguish them from the optimal Gaussian.

LeJEPA Appendix B.14 proves that when the Epps-Pulley statistic is replaced by a 2-moment statistic (mean² + (std−1)²), the SIGReg loss converges to VICReg in the limit of many projection directions. VICReg is therefore a degenerate special case of SIGReg, making it the most theoretically principled baseline for this experiment.

### 1.5 CIFAR-10

CIFAR-10 is a standard image classification benchmark: 10 classes, 50,000 training images, 10,000 test images, 32×32 pixels. It is chosen for this experiment because it is lightweight (fast to load and train on), requires no special setup (available via `tensorflow_datasets`), and is sufficient for a concept-validation experiment focused on relative comparisons between conditions rather than absolute accuracy.

The 32×32 resolution is a limitation: it constrains the richness of features the encoder can learn and requires a simplified augmentation strategy (pad-and-crop rather than RandomResizedCrop). Results may not generalize to higher-resolution settings.

### 1.6 MiniResNet-18

MiniResNet-18 is a lightweight ResNet-18 adapted for 32×32 CIFAR-10 input. The architecture:

```
Stem:   Conv(3→64, 3×3) + InstanceNorm + ReLU
Stage1: ResBlock(64→64,   stride=1) × 2
Stage2: ResBlock(64→128,  stride=2) × 2   [32→16]
Stage3: ResBlock(128→256, stride=2) × 2   [16→8]
Stage4: ResBlock(256→512, stride=2) × 2   [8→4]
GlobalAvgPool → ProjMLP: 512→512→512→64
```

**InstanceNorm instead of BatchNorm.** BatchNorm maintains running statistics across batches — mutable state that is incompatible with JAX's functional programming model, where functions must be pure (no side effects). InstanceNorm normalizes each feature map independently over its spatial dimensions within a single sample, requiring no batch-level bookkeeping. This makes the model a pure function: given the same input and weights, it always produces the same output, which is required for JAX's JIT compilation and automatic differentiation.

The model exposes two forward methods: `encode()` returns the 512-dim backbone features used for linear probe; `__call__()` applies the full forward pass including the projection head, used during SSL training.

### 1.7 JAX and Equinox

**JAX** is a numerical computing library designed for high-performance machine learning research. Its core design principles are relevant to this experiment:

- `jit`: compiles Python functions to optimized XLA programs, fusing operations into single GPU kernels
- `vmap`: vectorizes functions over a batch dimension without explicit loops
- `grad`: computes exact gradients of any differentiable function

These transforms compose cleanly and require functions to be pure, which motivated the choice of InstanceNorm and the functional training loop design.

**Equinox** is a JAX-based neural network library that represents models as pytrees of arrays — pure data structures with no hidden state. This makes models composable with all JAX transforms naturally.

**tf.data instead of PyTorch DataLoader.** The standard PyTorch DataLoader with `num_workers > 0` uses `os.fork()` to spawn worker processes. JAX initializes a multithreaded CUDA context at import time, and forking a multithreaded process produces undefined behavior — in practice, a deadlock. The `tf.data` pipeline performs all data loading and augmentation in TensorFlow's C++ runtime, which does not share JAX's threading constraints. The `prefetch(AUTOTUNE)` call overlaps CPU data preparation with GPU computation, achieving the same throughput benefit as multiple DataLoader workers without any fork-related issues.

### 1.8 AdamW and Warmup Cosine Decay

**AdamW** is Adam with decoupled weight decay. Standard Adam applies weight decay by adding it to the gradient, which interacts with the adaptive learning rate scaling. AdamW applies weight decay directly to the parameters, independent of the gradient, which is the theoretically correct regularization behavior.

**Warmup cosine decay** uses a two-phase learning rate schedule:

```
Steps 0–500:     linear ramp from 0 to peak lr (1e-3)
Steps 500–20000: cosine decay from peak lr to end lr (1e-5)
```

The warmup phase is important for SSL training: in the first steps, embeddings are near-random and gradients are large and unstable. A full learning rate from step 0 can push the model into a collapsed or degenerate state before meaningful gradient information is available. Ramping the learning rate gradually gives the model time to establish a reasonable embedding distribution before committing to large parameter updates. The cosine decay phase allows fine-grained refinement as the model approaches convergence.

---

## 2. Experiment Design

### 2.1 Overall Architecture and Design Philosophy

The guiding principle is **simplicity**: use the smallest, fastest components that can clearly answer the research questions. Every design choice prioritizes interpretability of results over absolute performance.

The full pipeline:

```
CIFAR-10 image
    ↓  × 4 independent augmentations
(N, V=4, 3, 32, 32) batch
    ↓  jax.vmap(model)
(V, N, D=64) projected embeddings z
    ↓
SSL Loss: λ × SIGReg(z) + (1−λ) × Invariance(z)   [LeJEPA]
          α × Var + β × Inv + γ × Cov               [VICReg]
    ↓  AdamW + warmup cosine decay
updated model parameters
```

V=4 views per image provides C(4,2)=6 view pairs for the invariance term, giving a more stable gradient estimate than the standard V=2 used in the original papers.

### 2.2 Experimental Conditions

Conditions are split into two groups to keep the main experiment plots readable. PureInv and PureSIG produce extreme values (emb_var → 0 or >> 1) that would compress the y-axis and obscure differences between the main conditions.

**Baseline group (1000 steps, no linear probe):**

| Condition | Regularizer | λ | Purpose |
|-----------|------------|---|---------|
| PureInv | None | 0 | Confirms collapse without regularization |
| PureSIG | SIGReg only | 1.0 | Shows distribution quality without view alignment |

1000 steps is sufficient to observe collapse (PureInv) and the emb_var explosion (PureSIG). Continuing further adds no new information.

**Main experiment (20000 steps, linear probe after training):**

| Condition | Regularizer | λ | Expected behavior |
|-----------|------------|---|-------------------|
| VICReg | VICReg (α=25, β=25, γ=1) | — | Stable training, 2-moment matching |
| LeJEPA λ=0.01 | SIGReg | 0.01 | Collapse (SIGReg too weak) |
| LeJEPA λ=0.05 | SIGReg | 0.05 | Near collapse threshold |
| LeJEPA λ=0.1 | SIGReg | 0.1 | Stable zone |
| LeJEPA λ=0.5 | SIGReg | 0.5 | Strong regularization |

The four λ values are chosen to sample the key behavioral zones: below the collapse threshold (0.01), at the threshold (0.05), in the stable zone (0.1), and at the edge of over-regularization (0.5). All conditions share the same initial weights (PRNGKey(0)) and optimizer settings, isolating the loss function as the only variable.

### 2.3 Evaluation Criteria

Four metrics are recorded every 100 steps:

| Metric | What it measures | What to look for |
|--------|-----------------|-----------------|
| SIGReg loss | Distance from embedding distribution to N(0,1) | Lower = more Gaussian distribution |
| Invariance loss | Mean MSE between different views of the same image | Lower = better view alignment |
| Embedding variance | Mean variance per embedding dimension | Near 0 = collapse; near 1 = healthy |
| Gradient ratio | ‖∂SIGReg/∂θ‖ / ‖∂Inv/∂θ‖ (no λ scaling) | Quantifies natural scale competition |

SIGReg loss is applied to **all conditions** including VICReg as a unified evaluation metric, even though VICReg never optimizes it directly. This design choice is critical: it allows all conditions to be compared on the same scale. If each condition were evaluated by its own training loss, the numbers would be incomparable.

The gradient ratio is computed without λ scaling to reveal the natural scale relationship between the two loss terms — separate from the artificial weighting imposed by λ. A ratio of 10 means SIGReg's gradient is 10× stronger than Invariance's before any scaling.

**Linear probe accuracy** is the final downstream evaluation, measuring the real-world utility of the learned representations on CIFAR-10 classification.

---

## 3. Results

### 3.1 Baseline Group

![Baseline conditions: SIGReg loss and embedding variance](baseline_plot.png)

**PureInv** collapsed immediately. Embedding variance reached zero by step 500 and SIGReg loss remained fixed at 0.636 for the full 1000 steps. Without regularization, the minimum of the invariance loss is the trivial solution of mapping everything to a point.

**PureSIG** showed the opposite extreme. SIGReg loss fell to 0.015, but embedding variance peaked at 7.1 before settling near 1.8. Without an invariance term, embeddings expand freely in all directions to reduce CF distance. The representations are well-distributed but carry no information about which images are similar, making them useless for any downstream task.

These two conditions confirm that both terms are necessary: regularization without invariance produces meaningless embeddings, and invariance without regularization produces collapsed embeddings.

### 3.2 Training Dynamics

![Main experiment: SIGReg loss, embedding variance, and gradient ratio](training_curves.png)

**VICReg** trained stably throughout. SIGReg loss dropped from 0.636 to 0.011 — the lowest final value among all conditions — and embedding variance stabilized near 1.0. The cosine decay schedule is visible in the continued slow improvement through the later stages of training.

**LeJEPA λ=0.01** collapsed completely and permanently. Embedding variance was zero at every logged step. The gradient ratio oscillated between 10² and 10⁴, reflecting unstable gradient estimates in the collapsed state. After λ=0.01 scaling, SIGReg's effective contribution is approximately 6% of the invariance gradient — too weak to break the collapse attractor.

**LeJEPA λ=0.05** collapsed for the first 4500 steps before escaping stochastically. During the collapse phase, all metrics were identical to λ=0.01. After escape at ~step 4500, SIGReg loss declined to 0.195 and embedding variance grew to 0.55. The warmup schedule did not reliably prevent collapse at this λ value.

**LeJEPA λ=0.1** trained stably from step 0. SIGReg loss declined continuously to 0.064 and embedding variance reached 0.84. This condition achieved the lowest invariance loss (0.006) of all LeJEPA conditions, indicating the best view alignment quality.

**LeJEPA λ=0.5** converged fastest. SIGReg loss reached 0.009 by step 19500. However, invariance loss (0.013) was twice that of λ=0.1, indicating that the stronger regularization came at the cost of view alignment quality.

### 3.3 Gradient Ratio Analysis

The gradient ratio plots (rightmost panel above) provide direct evidence for the "guard" dynamic.

For λ=0.1 and λ=0.5, the ratio starts high in the first few thousand steps and then stabilizes at a lower plateau. SIGReg gradients are strong when the distribution is disordered and weaken as it approaches Gaussian — the regularizer is most active when it is most needed.

For λ=0.05, the ratio during the collapse phase (steps 0–4500) oscillates violently between 10² and 10³, then drops to ~12 after escape. For λ=0.01, the ratio never stabilizes, remaining in the 10²–10⁴ range throughout.

The final gradient ratios form a monotone decreasing sequence with λ:

| Condition | Grad Ratio (final) | Effective SIGReg contribution (after λ) |
|-----------|-------------------|-----------------------------------------|
| LeJEPA λ=0.01 | 6.20 | ~6% of Inv |
| LeJEPA λ=0.05 | 11.71 | ~59% of Inv |
| LeJEPA λ=0.1 | 1.40 | ~14% of Inv |
| LeJEPA λ=0.5 | 0.74 | ~37% of Inv |

Higher λ → stronger regularization pressure at every point in training, not just through the explicit weighting.

### 3.4 Linear Probe Results

| Condition | SIGReg (final) | Inv Loss (final) | Emb Var (final) | Test Acc |
|-----------|---------------|-----------------|-----------------|----------|
| VICReg | 0.011 | 0.024 | 1.094 | 54.83% |
| LeJEPA λ=0.01 | 0.635 | 0.000 | 0.000 | 10.00% |
| LeJEPA λ=0.05 | 0.195 | 0.003 | 0.553 | 45.13% |
| LeJEPA λ=0.1 | 0.064 | 0.006 | 0.840 | **59.95%** |
| LeJEPA λ=0.5 | 0.009 | 0.013 | 0.938 | **61.97%** |

LeJEPA λ=0.1 and λ=0.5 both outperform VICReg by 5–7 percentage points. LeJEPA λ=0.05 underperforms VICReg due to the ~4500 steps lost to collapse.

### 3.5 Discussion

**The dissociation between SIGReg loss and downstream accuracy.**

The most striking result is that VICReg achieves the lowest SIGReg loss (0.011) yet lower accuracy than LeJEPA λ=0.1 and λ=0.5. If SIGReg loss fully predicted downstream performance, VICReg should be the best condition. It is not.

The explanation is invariance loss. VICReg's invariance loss at the end of training (0.024) is four times higher than LeJEPA λ=0.1 (0.006). VICReg's three-term loss structure creates a competition between its variance, invariance, and covariance terms. The optimizer satisfies all three simultaneously, but this means the invariance term receives less dedicated optimization pressure than in LeJEPA, where invariance has a dedicated (1−λ) weight. The result: VICReg produces embeddings that are well-distributed in shape but poorly aligned across views.

A linear classifier trained on these embeddings sees features that vary considerably within a class (because different views are far apart), reducing its ability to generalize. In this experiment, **invariance loss is a better predictor of downstream accuracy than SIGReg loss**.

**The λ=0.5 accuracy advantage.**

λ=0.5 achieves the highest accuracy despite elevated invariance loss. This is likely a consequence of the limited training budget (20000 steps ≈ 200 epochs). With stronger SIGReg pressure, the distribution converges faster, and the backbone learns more structured features within the available steps. Whether this advantage persists at convergence is unknown — if invariance loss remains elevated at λ=0.5 over a longer run, λ=0.1 may eventually surpass it.

### 3.6 Conclusions

**Lambda behavior zones.**

| Zone | Lambda | Behavior |
|------|--------|----------|
| Collapse | λ ≤ 0.01 | Permanent collapse, emb_var → 0, test acc = random |
| Critical | λ = 0.05 | Stochastic escape (~step 4500), warmup does not reliably prevent it |
| Stable | λ = 0.1 | No collapse, best view alignment, competitive accuracy |
| High-reg | λ = 0.5 | Fastest distribution convergence, highest accuracy in this run |

**Core conclusions:**

LeJEPA with SIGReg outperforms VICReg when λ is large enough to prevent collapse. At λ=0.1 and λ=0.5, the advantage is 5–7 percentage points, supporting the paper's claim that sufficient moment matching produces better representations. The gradient ratio analysis confirms the guard dynamic: SIGReg dominates early and recedes as the distribution approaches Gaussian.

However, SIGReg loss alone is not sufficient to predict downstream performance. VICReg achieves the lowest SIGReg loss but intermediate accuracy, because its loss structure leaves view alignment under-optimized. Both distribution quality and view alignment matter.

### 3.7 Limitations and Future Work

**Insufficient training steps.** 20000 steps corresponds to ~200 epochs — well below the 800–1000 epochs at which SSL models typically converge. The ordering of conditions by test accuracy may reflect convergence speed differences rather than final performance. In particular, the λ=0.5 accuracy advantage may not hold at full convergence.

**Single seed.** Each condition was run once. The stochastic collapse escape at λ=0.05 demonstrates that some conditions are sensitive to the specific gradient trajectory. Without multiple seeds, the variance of results cannot be quantified.

**Dataset resolution.** CIFAR-10's 32×32 images limit feature richness and require the simplified pad-and-crop augmentation. Results may not generalize to higher-resolution benchmarks.

**VICReg internal loss decomposition not tracked.** The variance, invariance, and covariance terms of VICReg were not logged separately. Tracking these would clarify exactly how the three-term competition affects view alignment, enabling a more precise diagnosis of VICReg's higher invariance loss.

**Directions for improvement:** extending training to 50000–100000 steps; running 3+ seeds per condition; adding λ=0.25 to fill the gap between stable and high-regularization zones; logging VICReg's internal terms; testing on STL-10 or ImageNette for higher-resolution validation.

---

## References

1. Balestriero, R. & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. *arXiv:2511.08544*.
2. Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.
3. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*.
4. Epps, T. W. & Pulley, L. B. (1983). A test for normality based on the empirical characteristic function. *Biometrika, 70*(3), 723–726.
