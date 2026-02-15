# 4. Model Validation Against Free-Decay Experiments

## 4.1 Validation Dataset and Split Strategy

Validation is performed using the same protocolized data interface as parameter determination, while preserving condition-level anti-leakage boundaries. Segment selection follows predefined split tags and optional cross-validation folds, and validation sets are separated from calibration usage at condition level.

The split policy evaluates extrapolation robustness rather than pointwise interpolation. Leave-one-theta-level-out settings are applied where configured.

[Fig. 13. Validation dataset coverage and split topology across operating conditions. Insert here.]

## 4.2 In-Sample Performance

In-sample comparison serves as verification of calibration behavior under the fixed parameter set determined in Section 3. For each segment, \(\theta_0\) is initialized from the first experimental sample, and simulated trajectories are sampled at experimental timestamps for metric consistency.

In-sample trajectories are shown in Fig. 14 and interpreted jointly with out-of-sample behavior.

## 4.3 Out-of-Sample Performance

Out-of-sample comparison is the primary validation target. Representative trajectories are presented with synchronized time axes for \(\theta(t)\) and \(q(t)\), when angular-rate measurements are available.

[Fig. 14. Multi-run in-sample and out-of-sample sim-real trajectories (`theta`, `q`). Insert here.]
[Fig. 15. Time-domain trajectory-error traces with zero-reference and uncertainty bands. Insert here.]

## 4.4 Error Metrics and GapScore Analysis

Validation reports absolute metrics for each split and condition: \(TAAE_\theta\), \(TASE_\theta\), \(RMSE_\theta\), \(MAE_\theta\), \(MaxAbs_\theta\), \(dt90\), overshoot error, steady-window mean and standard deviation, \(RMSE_q\), and \(MAE_q\). GapScore-based ranking complements these metrics through condition-normalized aggregation.

[Table 1. Validation metrics by condition and split. Insert here.]
[Fig. 16. Condition-wise validation metrics and GapScore ranking map. Insert here.]

## 4.5 Discrepancy Analysis

Discrepancy analysis distinguishes phase mismatch, amplitude mismatch, and terminal-state bias, and relates deviation patterns to model assumptions and operating-condition boundaries. The resulting error patterns define the confidence range used in the design-oriented analysis of Section 5.

[Fig. 17. Discrepancy-mode decomposition for representative validation cases. Insert here.]
