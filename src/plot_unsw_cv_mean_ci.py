import numpy as np
import matplotlib.pyplot as plt

single = np.array([0.5951, 0.5964, 0.6022, 0.6001, 0.5853])
two    = np.array([0.6547, 0.6559, 0.6411, 0.6347, 0.6611])

def mean_ci95(x):
    m = x.mean()
    s = x.std(ddof=1)
    n = len(x)
    ci = 1.96 * s / np.sqrt(n)   # 95% CI (normal approx; fine for visualization)
    return m, ci

m1, ci1 = mean_ci95(single)
m2, ci2 = mean_ci95(two)

plt.figure(figsize=(6, 4.2))

x = np.array([0, 1])
means = [m1, m2]
cis   = [ci1, ci2]

# errorbars (mean ± 95% CI)
plt.errorbar(x, means, yerr=cis, fmt='o', capsize=6, elinewidth=2)

# overlay fold points with jitter
rng = np.random.default_rng(42)
jit1 = rng.normal(0, 0.03, size=len(single))
jit2 = rng.normal(0, 0.03, size=len(two))
plt.scatter(np.zeros_like(single)+jit1, single, s=35)
plt.scatter(np.ones_like(two)+jit2, two, s=35)

plt.xticks([0, 1], ["Single-stage", "Two-stage"])
plt.ylabel("Macro-F1")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/unsw_cv_mean_ci.png", dpi=300)
plt.show()