# 6. Model Validation Against Free-Decay Experiments

## 6.1 Validation Objective and Evidence Logic

The purpose of this section is to test whether the closed model package obtained in Sections~4--5 can reproduce free-decay transition behavior without retuning, and to characterize where the model is reliable for deployment-oriented reasoning. The validation target is therefore not limited to visual trajectory overlap. Instead, the section evaluates time-domain agreement, characteristic transition timing, terminal-state behavior, and discrepancy modes under condition-level split control.

A key point is that the validation protocol is designed to measure generalization across operating conditions rather than interpolation within a single waveform family. For this reason, data partitions are defined at the condition level, and optional leave-one-\(\theta_0\)-level-out settings are supported by the manifest and pipeline configuration. This structure is intended to prevent leakage between calibration and validation when segments share similar excitation patterns.

The current repository stage contains a baseline set of free-decay segments, and the validation structure presented here is written to scale directly to the planned expanded condition matrix. The same reporting logic is used in both cases so that new data extend the evidence base without requiring a change in evaluation criteria.

[Fig. 13. Validation dataset coverage and split topology across operating conditions. Insert here.]

## 6.2 Simulation-to-Experiment Comparison Protocol

Each validation segment is simulated using the full 3-DOF model developed in Section~4 with the calibrated parameter package from Section~5 fixed. In the current evaluation script, the initial pitch angle and pitch rate are taken from the first sample of the experimental segment, while the translational velocities are initialized as \(u_0=w_0=0\) for the free-decay replay. Simulated trajectories are then interpolated to the experimental timestamps before metric computation, which ensures that all reported discrepancies are measured on a common time grid and are not artifacts of different solver output intervals.

This section reports both in-sample and out-of-sample results, but they play different roles. In-sample results verify that the identified parameter set is consistent with the calibration data under the full-model simulation interface, whereas out-of-sample results provide the primary evidence for predictive usefulness. Throughout the section, in-sample evidence is shown first and out-of-sample evidence second to preserve a transparent progression from calibration consistency to generalization performance.

## 6.3 In-Sample Behavior and Calibration Consistency Check

In-sample validation is used as a consistency check rather than a final performance claim. The central question is whether the parameter set identified from the free-decay pipeline produces physically plausible reconstructed trajectories under the full simulation model and fixed hydrodynamic mappings. Agreement in this stage supports the internal consistency of the modeling and identification stack, including sign conventions, hydrodynamic closure, and parameter transfer from the identification routine to the full simulator.

Trajectory comparisons should be interpreted jointly in terms of shape, phase, and terminal behavior. In particular, apparent visual agreement in \(\theta(t)\) can mask systematic errors in \(q(t)\), and conversely, short-lived phase drift can appear larger than its practical impact if the amplitude and terminal-state behavior remain well captured. For this reason, the in-sample figures are presented together with error traces and metric summaries rather than as stand-alone overlays.

## 6.4 Out-of-Sample Validation and Condition-Wise Performance

Out-of-sample validation is the primary test of whether the model captures the transition mechanics in a way that is useful beyond the calibration subset. Representative trajectories for \(\theta(t)\) and \(q(t)\) are reported with synchronized time axes and consistent panel ordering so that phase drift, overshoot mismatch, and terminal-state deviation can be compared across conditions. When angular-rate measurements are unavailable or unreliable for a segment, the metric aggregation is adjusted rather than forcing incomplete comparisons.

[Fig. 14. Multi-run in-sample and out-of-sample sim-real trajectories (`theta`, `q`). Insert here.]
[Fig. 15. Time-domain trajectory-error traces with zero-reference and uncertainty bands. Insert here.]

The evaluation pipeline computes absolute and signed error metrics including \(RMSE_{\theta}\), \(MAE_{\theta}\), \(MaxAbs_{\theta}\), \(TAAE_{\theta}\), \(TASE_{\theta}\), settling-time mismatch (reported as \(dt90\) and fallback \(dt80\)), overshoot error, and steady-window statistics. When \(q\) is available, \(RMSE_q\) and \(MAE_q\) are also reported. These metrics are designed to separate trajectory-shape error, timing error, and terminal-state error so that model limitations can be localized rather than collapsed into a single scalar score.

[Table 2. Validation metrics by condition and split. Insert here.]

## 6.5 GapScore Aggregation and Discrepancy-Mode Decomposition

To complement the absolute metrics, the evaluation pipeline computes a condition-normalized composite score (GapScore) that combines angle error, timing error, overshoot mismatch, and pitch-rate error when available. Let \(\Delta\theta_{span}=|90^\circ-\theta_0|\) denote the condition-specific transition amplitude. The normalized terms are constructed from the reported metrics so that comparisons across conditions do not become dominated by large-amplitude cases. In the current implementation, the composite score is evaluated as

$$
GapScore = 0.35N_{RMSE_\theta} + 0.20N_{MAE_\theta} + 0.20N_{RMSE_q} + 0.15N_t + 0.10N_{over},
$$

with the \(q\)-error term omitted and the remaining weights renormalized when \(q\) measurements are unavailable. The resulting composite score is then used as a ranking aid, not as a substitute for the underlying metrics.

[Fig. 16. Condition-wise validation metrics and GapScore ranking map. Insert here.]

A second layer of analysis decomposes discrepancies into phase mismatch, amplitude mismatch, and terminal-state bias. This decomposition is important for engineering interpretation because different discrepancy modes imply different corrective actions. For example, dominant phase mismatch may suggest deficiencies in effective inertia or damping balance, whereas terminal-state bias may indicate hydrostatic or cable-restoring mismatch. The discrepancy analysis therefore provides the bridge between validation metrics and the uncertainty-aware design guidance developed in Section~7.

[Fig. 17. Discrepancy-mode decomposition for representative validation cases. Insert here.]

## 6.6 Validity Boundary for Design Use

The main outcome of Section~6 is not simply a statement of model accuracy, but a bounded validity region for engineering use. If the model reproduces trajectory shape, transition timing, and terminal behavior within acceptable limits over the evaluated condition set, it can be used in Section~7 as a design-analysis surrogate under clearly stated confidence boundaries. Where systematic discrepancy modes persist, those patterns are carried forward explicitly as uncertainty qualifiers rather than ignored.

This evidence-to-guidance transfer is central to the paper's contribution: the model is used for design reasoning only after its failure modes and applicability boundaries have been characterized on experimental data.
