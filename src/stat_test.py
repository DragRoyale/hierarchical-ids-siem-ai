import numpy as np
from scipy import stats

# Your fold results
hgb_scores = np.array([0.5429, 0.4410, 0.4579, 0.3933, 0.4297])
cat_scores = np.array([0.6050, 0.5289, 0.5308, 0.5337, 0.5381])

# Paired t-test
t_stat, p_value = stats.ttest_rel(cat_scores, hgb_scores)

# Mean difference
mean_diff = np.mean(cat_scores - hgb_scores)

print("HGB mean:", np.mean(hgb_scores))
print("CatBoost mean:", np.mean(cat_scores))
print("Mean improvement:", mean_diff)
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("\nResult: Improvement is statistically significant (p < 0.05)")
else:
    print("\nResult: Improvement is NOT statistically significant")