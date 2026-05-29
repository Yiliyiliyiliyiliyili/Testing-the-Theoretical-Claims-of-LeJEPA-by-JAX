# Testing the Theoretical Claims of LeJEPA

Empirical analysis of SIGReg regularization on CIFAR-10, using JAX and Equinox.
This project runs a lambda sweep on LeJEPA and compares against VICReg to test
whether sufficient moment matching leads to better downstream representations.

---

## What This Is

LeJEPA (Balestriero & LeCun, 2025) proves that the optimal SSL embedding
distribution is an isotropic Gaussian, and proposes SIGReg — a regularizer
based on the Epps-Pulley characteristic function distance — to directly enforce
this. The paper also shows that VICReg is a degenerate special case of SIGReg,
converging to 2-moment matching in the limit of many projection directions.

This experiment tests two questions:

1. Does LeJEPA with SIGReg produce better downstream representations than VICReg?
2. What range of λ leads to stable training, and where are the collapse boundaries?

---

## Experiment Design

### Why VICReg as Baseline

The paper establishes a direct theoretical bridge between SIGReg and VICReg,
making VICReg the most principled comparison point. Any performance difference
is interpretable in terms of the paper's claims about moment matching sufficiency,
rather than arbitrary design differences between unrelated methods.

### Conditions

**Baseline group (1000 steps):**

| Condition | Regularizer | λ |
|-----------|------------|---|
| PureInv | None | 0 |
| PureSIG | SIGReg only | 1.0 |

**Main experiment (20000 steps + linear probe):**

| Condition | Regularizer | λ |
|-----------|------------|---|
| VICReg | Original VICReg (α=25, β=25, γ=1) | — |
| LeJEPA λ=0.01 | SIGReg | 0.01 |
| LeJEPA λ=0.05 | SIGReg | 0.05 |
| LeJEPA λ=0.1 | SIGReg | 0.1 |
| LeJEPA λ=0.5 | SIGReg | 0.5 |

All conditions share the same architecture (MiniResNet-18, InstanceNorm),
optimizer (AdamW + warmup cosine decay), and initial weights (PRNGKey(0)).

### Implementation

- **Framework**: JAX + Equinox
- **Dataset**: CIFAR-10 via `tensorflow_datasets`
- **Data pipeline**: `tf.data` with prefetch (avoids JAX/fork deadlock)
- **Augmentation**: SimCLR-style (pad-and-crop, flip, color jitter, grayscale)
- **Evaluation**: Frozen 512-dim backbone features, linear probe trained 100 epochs

---

## Results

### Training Curves

![Main experiment: SIGReg loss, embedding variance, and gradient ratio](training_curves.png)

### Baseline Conditions

![Baseline conditions: SIGReg loss and embedding variance](baseline_plot.png)

### Linear Probe (CIFAR-10 Test Accuracy)

| Condition | SIGReg (final) | Inv Loss (final) | Emb Var (final) | Test Acc |
|-----------|---------------|-----------------|-----------------|----------|
| VICReg | 0.011 | 0.024 | 1.094 | 54.83% |
| LeJEPA λ=0.01 | 0.635 | 0.000 | 0.000 | 10.00% |
| LeJEPA λ=0.05 | 0.195 | 0.003 | 0.553 | 45.13% |
| LeJEPA λ=0.1 | 0.064 | 0.006 | 0.840 | **59.95%** |
| LeJEPA λ=0.5 | 0.009 | 0.013 | 0.938 | **61.97%** |

---

## Key Findings

**Lambda behavior zones.** Four distinct regimes emerged:

| Zone | Lambda | Behavior |
|------|--------|----------|
| Collapse | λ ≤ 0.01 | Permanent collapse, test acc = random |
| Critical | λ = 0.05 | Stochastic escape at ~4500 steps |
| Stable | λ = 0.1 | No collapse, best view alignment |
| High-reg | λ = 0.5 | Fastest distribution convergence |

**LeJEPA outperforms VICReg when λ is large enough to prevent collapse.**
LeJEPA λ=0.1 and λ=0.5 exceed VICReg by 5–7 percentage points in test accuracy.

**SIGReg loss alone does not predict downstream performance.**
VICReg achieves the lowest final SIGReg loss (0.011) yet lower accuracy than
LeJEPA λ=0.1 and λ=0.5. The key differentiator is invariance loss: VICReg's
view alignment (0.024) is four times worse than LeJEPA λ=0.1 (0.006), leading
to semantic inconsistency that hurts the linear classifier.

**Gradient ratio confirms the guard dynamic.**
For stable LeJEPA conditions, the ratio starts high and stabilizes at a lower
plateau — SIGReg is strong when the distribution is disordered and recedes as
it approaches Gaussian, consistent with the paper's description.

For a full analysis, see [report.md](report.md).

---

## How to Run

### Run the experiment

Open `experiment.ipynb` in Google Colab with a GPU runtime (A100 recommended).
Execute cells 1–17 in order. Training takes approximately 4–5 hours on A100.

### View results without re-running

Download `results.pkl` and `show_results.py` to the same folder, then:

    pip install numpy pandas matplotlib
    python show_results.py

To specify a different path to the pkl file:

    python show_results.py --path /path/to/results.pkl

---

## References

1. Balestriero, R. & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. *arXiv:2511.08544*.
2. Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.
3. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*.
4. Epps, T. W. & Pulley, L. B. (1983). A test for normality based on the empirical characteristic function. *Biometrika, 70*(3), 723–726.
