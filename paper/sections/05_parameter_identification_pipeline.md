# 5. Parameter Identification and Reproducibility Diagnostics

## 5.1 Identification Objective and Evidence Contracts

After the model structure and hydrodynamic closure are fixed (Section~4), dynamic parameters are determined from the free-decay evidence governed in Section~3. The identification objective is to obtain a parameter set that is (i) physically interpretable, (ii) stable under the protocolized preprocessing/segmentation pipeline, and (iii) transferable without retuning to out-of-sample validation and design analysis.

All identification runs are filtered through a condition-aware split contract. Calibration segments are selected via a manifest-driven split topology so that training and validation/test evidence are separated at the operating-condition level, preventing leakage between highly similar waveform families.

## 5.2 Two-Stage Identification Structure

The identification pipeline uses a two-stage structure designed for practical identifiability.

First, rotational damping coefficients are estimated using a cycle-energy method with non-negative least squares. This stage provides a physically informed initialization for the subsequent optimization and reduces sensitivity to poor starting points.

Second, an ODE-based residual minimization stage identifies the remaining parameters under explicit bounds. In the current implementation, the optimization is carried out on a pitch-dominant reduced model for \(\theta\) and \(q\), while the full 3-DOF model defined in Section~4 is reserved for simulation-based validation and design analysis. This separation improves practical identifiability while preserving a consistent downstream validation model.

## 5.3 Optimization Formulation and Parameter Set

For a set of calibration segments, the Stage-2 optimization minimizes a stacked residual vector over simulated and measured pitch angle and pitch rate histories subject to parameter bounds. The calibrated parameter set includes the pitch-channel permeability parameter \(\mu_{\theta}\), linear rotational damping \(d_q\), a quadratic damping coefficient (implemented as `d_qq` in the fitter and corresponding to \(d_{q,abs}\) in the manuscript notation), and the cable stiffness term \(K_{cable}\) when cable effects are enabled.

## 5.4 Reproducibility Artifacts, Diagnostics, and Uncertainty Hooks

To support reproducibility and diagnosis, each identification run records protocol snapshots, the git commit hash, and the Python environment snapshot alongside fitted parameters and selected segment lists. Multi-start optimization and bootstrap resampling are supported by the protocol, even when disabled in a baseline run. In the current protocol snapshot included with this repository stage, `multi_start_n=1` and `n_boot=0`, so the baseline execution behaves as a single-start fit without bootstrap-based uncertainty expansion.

The identification evidence package is summarized by the workflow diagram and diagnostics placeholders below.

[Fig. 11. Parameter-identification workflow from segmented free-decay data to calibrated model parameters. Insert here.]
[Fig. 12. Identification diagnostics and uncertainty summaries (residuals, multi-start convergence, bootstrap intervals). Insert here.]

## 5.5 Closed Model Package for Validation and Design Analysis

At the end of Section~5, the study produces a closed model package consisting of fixed platform and hydrostatic parameters, CFD-derived static hydrodynamic mappings, and experimentally calibrated dynamic parameters. This package is transferred to validation (Section~6) without retuning and is subsequently used in design analysis (Section~7). The no-retuning rule between calibration and validation is essential to the claim that the framework provides engineering evidence rather than post hoc curve fitting.
