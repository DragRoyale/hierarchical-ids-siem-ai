import numpy as np
from scipy import stats

# Single-stage XGBoost folds (from your baseline run)
single = np.array([0.5951, 0.5964, 0.6022, 0.6001, 0.5853], dtype=float)

# Two-stage XGBoost folds (from your new run)
two_stage = np.array([0.6453, 0.6559, 0.6411, 0.6347, 0.6611], dtype=float)

t, p = stats.ttest_rel(two_stage, single)
diff = (two_stage - single).mean()

print("Two-stage mean:", two_stage.mean())
print("Single-stage mean:", single.mean())
print("Mean improvement:", diff)
print("t-statistic:", t)
print("p-value:", p)