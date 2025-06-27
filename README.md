# Min‑Max Optimization in GANs (Team Project 8)

## Overview

Generative Adversarial Networks (GANs) formulate training as a **min–max game** between a *generator* and a *discriminator*. Classical Gradient Descent–Ascent (GDA) struggles with oscillations and slow convergence, especially in high‑dimensional settings. We implement and evaluate the **convergent, dimension‑independent algorithm** proposed by *Keswani et al., 2023* as a drop‑in replacement for the GDA loop.

## Objectives

1. **Re‑implement baseline GDA** for MNIST in PyTorch.
2. **Implement Keswani’s optimizer** with momentum stabilisation.
3. **Quantitatively compare** both approaches on convergence speed, stability, Fréchet Inception Distance (FID) and Inception Score (IS).


## Repository Layout

```
├── docs/            # Report, slides, papers
├── codes/
│   ├── GDA.py    # GDA
│   ├── Keswani_algorithm.py   # Keswani optimizers
└── README.md        # This file
```

## Results

| Iterations | Optimizer | FID ↓      | IS ↑     |
| ---------: | --------- | ---------- | -------- |
|        400 | Keswani   | **219.22** | **3.96** |
|            | GDA       | 248.06     | 3.23     |
|        800 | Keswani   | **97.59**  | **4.49** |
|            | GDA       | 198.92     | 3.80     |
|      1 200 | Keswani   | **77.74**  | **5.04** |
|            | GDA       | 137.10     | 3.66     |
|      1 600 | Keswani   | **72.13**  | **5.52** |
|            | GDA       | 119.62     | 4.76     |
|      2 000 | Keswani   | **63.27**  | **5.17** |
|            | GDA       | 111.76     | 5.15     |

> **Key insight:** Keswani’s method consistently narrows the FID gap and reaches higher IS earlier, indicating faster, more stable learning.

## Challenges & Fixes

* **Memory limits on Colab:** reduced batch size from 128 → 64 and added checkpointing.
* **Metric bugs:** corrected FID and IS implementations in `evaluate.py`.
* **Unstable gradients:** tuned momentum and learning‑rate schedule.

## Future Work

* Train on higher‑resolution datasets (CIFAR‑10, CelebA‑HQ).
* Integrate with advanced architectures (DCGAN, StyleGAN‑v2).
* Explore adversarial regularisation and manifold constraints.

## Contributors

* **Atharva Kulkarni** (2023101072)
* Ansh Acharya (2023101069)
* Monosij Roy (2023111016)

## References

1. V. Keswani, O. Mangoubi, S. Sachdeva, N. K. Vishnoi, “A Convergent and Dimension‑Independent Min‑Max Optimization Algorithm,” 2023.
2. Project report & slides in `docs/` folder.
