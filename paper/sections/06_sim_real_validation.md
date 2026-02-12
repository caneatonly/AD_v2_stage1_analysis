# 6. Simulation-to-Experiment Validation

## 6.1 Evaluation Setup

This section reports in-sample and out-of-sample simulation-to-experiment validation using the same initial-condition and parameter loading rules defined in the identification protocol. For each run, `theta_0` is initialized from the first sample point of the corresponding segment, and simulated trajectories are interpolated onto experimental timestamps before metric calculation.

## 6.2 Time-Series Comparison

Representative runs are reported in fixed order: in-sample group first, out-of-sample group second. Both `theta(t)` and `q(t)` are shown whenever angular-rate sensing is available, with synchronized time axes and consistent sign conventions.

[Fig. 10. Multi-run in-sample and out-of-sample sim-real trajectories (`theta`, `q`). Insert here.]
[Fig. 11. Time-domain error traces with zero-reference and RMSE bands. Insert here.]

## 6.3 Metrics and Interpretation

The mandatory absolute metrics are reported for each split: `TAAE_theta`, `TASE_theta`, `RMSE_theta`, `MAE_theta`, `MaxAbs_theta`, `dt90`, overshoot error, steady-window mean and standard deviation, `RMSE_q`, and `MAE_q`. GapScore-based ranking is reported as a secondary aggregate indicator and interpreted jointly with absolute metrics.

[Table 1. Validation metrics by condition and split. Insert here.]
