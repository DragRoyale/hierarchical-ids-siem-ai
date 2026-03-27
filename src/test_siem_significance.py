import numpy as np
from scipy.stats import ttest_rel

two_stage = np.array([0.8277, 0.9096, 0.9689, 0.8521, 0.9334])
single_stage = np.array([0.9161, 0.8130, 0.8327, 0.7335, 0.8004])

diff = two_stage - single_stage

t_stat, p_val = ttest_rel(two_stage, single_stage)

print("Two-stage mean:", two_stage.mean())
print("Single-stage mean:", single_stage.mean())
print("Mean improvement:", diff.mean())
print("t-statistic:", t_stat)
print("p-value:", p_val)