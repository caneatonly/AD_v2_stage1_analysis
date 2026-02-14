# 5. Experimental identification and model validation

## 5.1 Test procedure and data processing

This section documents preprocessing, segmentation, condition-level split constraints, and anti-leakage rules under `id_protocol.yaml` and `experiment_manifest.csv`.

## 5.2 Identification method and diagnostics

The workflow includes robust fitting, multi-start optimization, and optional bootstrap confidence analysis over selected segments. Residual checks, convergence behavior, and bootstrap confidence intervals are reported to evaluate parameter identifiability and stability.

[Fig. 9. Identification diagnostics: residuals, multi-start convergence, and bootstrap confidence intervals. Insert here.]

## 5.3 Simulation–experiment comparison

This subsection reports in-sample and out-of-sample simulation-to-experiment validation using the same initial-condition and parameter loading rules defined in the identification protocol. For each run, `theta_0` is initialized from the first sample point of the corresponding segment, and simulated trajectories are interpolated onto experimental timestamps before metric calculation.

Representative runs are reported in fixed order: in-sample group first, out-of-sample group second. Both `theta(t)` and `q(t)` are shown whenever angular-rate sensing is available, with synchronized time axes and consistent sign conventions.

[Fig. 10. Multi-run in-sample and out-of-sample sim-real trajectories (`theta`, `q`). Insert here.]
[Fig. 11. Time-domain error traces with zero-reference and RMSE bands. Insert here.]

## 5.4 Error analysis

The mandatory absolute metrics are reported for each split: `TAAE_theta`, `TASE_theta`, `RMSE_theta`, `MAE_theta`, `MaxAbs_theta`, `dt90`, overshoot error, steady-window mean and standard deviation, `RMSE_q`, and `MAE_q`. GapScore-based ranking is reported as a secondary aggregate indicator and interpreted jointly with absolute metrics.

[Table 1. Validation metrics by condition and split. Insert here.]
[Fig. 12. Condition-normalized metric heatmap and GapScore ranking. Insert here.]
