import json
import numpy as np
from scipy import stats

# Your CatBoost folds on UNSW (from your run)
cat = np.array([0.5347, 0.5438, 0.5536, 0.5445, 0.5408], dtype=float)

with open("results/unsw_baselines_cv_withCats.json", "r", encoding="utf-8") as f:
    d = json.load(f)

hgb = np.array(d["hgb"], dtype=float)
xgb = np.array(d["xgb"], dtype=float)

def paired(name_a, a, name_b, b):
    t, p = stats.ttest_rel(a, b)
    print(f"\n{name_a} vs {name_b}")
    print(f"{name_a} mean={a.mean():.4f} | {name_b} mean={b.mean():.4f}")
    print(f"mean diff ({name_a}-{name_b})={float((a-b).mean()):.4f}")
    print(f"t={t:.4f}  p={p:.6f}")

if __name__ == "__main__":
    paired("CatBoost", cat, "HGB", hgb)
    paired("CatBoost", cat, "XGBoost", xgb)
    paired("XGBoost", xgb, "HGB", hgb)