import numpy as np
import matplotlib.pyplot as plt

single = np.array([0.5951, 0.5964, 0.6022, 0.6001, 0.5853])
two    = np.array([0.6547, 0.6559, 0.6411, 0.6347, 0.6611])

plt.figure(figsize=(6, 4.2))

for i in range(len(single)):
    plt.plot([0, 1], [single[i], two[i]], marker='o', linewidth=2)

plt.xticks([0, 1], ["Single-stage", "Two-stage"])
plt.ylabel("Macro-F1")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/unsw_cv_paired.png", dpi=300)
plt.show()