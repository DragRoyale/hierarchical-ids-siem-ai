import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---- Your fold results ----
single_stage = [0.5951, 0.5964, 0.6022, 0.6001, 0.5853]
two_stage = [0.6547, 0.6559, 0.6411, 0.6347, 0.6611]

# Create dataframe
df = pd.DataFrame({
    "Macro-F1": single_stage + two_stage,
    "Model": ["Single-stage"] * 5 + ["Two-stage"] * 5
})

# Set clean academic style
sns.set(style="whitegrid")

plt.figure(figsize=(6, 5))

# Boxplot
sns.boxplot(x="Model", y="Macro-F1", data=df)

# Overlay individual fold points
sns.stripplot(x="Model", y="Macro-F1", data=df,
              color="black", size=6, jitter=True)

plt.ylabel("Macro-F1")
plt.title("UNSW-NB15 5-Fold Cross-Validation")

plt.tight_layout()
plt.savefig("results/unsw_cv_boxplot.png", dpi=300)
plt.show()