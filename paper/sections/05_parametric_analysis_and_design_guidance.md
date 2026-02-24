# 5. Parametric Analysis and Design Guidance

## 5.1 Purpose and Analysis Philosophy

Section 5 uses the validated model package to support deployment-oriented engineering decisions under bounded confidence. The goal is not to claim a globally optimal control or configuration law, but to translate the experimentally assessed model into practical guidance for the current platform architecture. All recommendations in this section are therefore conditional on the validity boundary established in Section 4 and on the configuration assumptions defined in Section 2.

The analysis is organized around three questions that matter for deployment practice. First, for which initial conditions is transition completion likely or unlikely under the current configuration? Second, how do actuation timing decisions trade off transition speed, overshoot, and safety-relevant behavior? Third, how sensitive are these conclusions to plausible uncertainty in hydrostatic and damping-related parameters? The subsection structure below follows this progression from feasibility, to performance trade-off, to robustness.

## 5.2 Initial-Condition Feasibility Envelope

The initial-condition envelope analysis maps model-predicted transition outcomes over the deployment-relevant space of initial pitch angle and pitch-rate states. Rather than reporting only individual trajectories, the analysis summarizes regions associated with successful transition completion, delayed convergence, stall-like behavior, and undesirable dynamic responses. This representation is useful because deployment planning decisions are typically made in terms of admissible release conditions, not isolated time histories.

The envelope should be interpreted as a model-based operational map under the calibrated parameter set and the validation-derived confidence boundary. Regions near decision boundaries are expected to be more sensitive to parameter uncertainty and should be treated conservatively in engineering use.

[Fig. 18. Operational success-envelope map in the initial-condition space. Insert here.]

## 5.3 Actuation Timing and Strategy Trade-Offs

The current platform configuration allows strategy comparisons based on actuation timing, including shutdown-angle sweeps and baseline always-on behavior. The purpose of this analysis is to expose trade-offs among transition duration, overshoot, terminal rate, and safety-relevant trajectory characteristics rather than to optimize a single scalar metric in isolation. In practice, different deployment priorities may require different operating points even under the same hardware configuration.

A useful presentation is to define a scenario-specific objective surface \(J(u_0,\theta_{off})\) and to interpret it together with explicit feasibility constraints. The objective surface should not be treated as universally transferable across missions; instead, it serves as a transparent decision aid that makes the trade-offs and assumptions visible. The preferred operating region is therefore identified as a feasible valley under the chosen weighting and safety constraints, not as a context-free optimum.

[Fig. 19. Control-strategy objective surface `J(u_0, theta_off)` and feasible valley region. Insert here.]

## 5.4 Sensitivity to Hydrostatic and Damping Uncertainty

Because the platform behavior depends on the balance among hydrostatic restoring effects, hydrodynamic moment closure, and damping, parametric guidance must be stress-tested against plausible uncertainty. This subsection evaluates sensitivity to variations in the buoyancy-gravity geometry (represented through the effective \(BG\)-related restoring characteristics) and damping-related parameters. The objective is to identify whether the feasibility envelope and recommended operating region remain structurally stable under perturbations consistent with Section 4 discrepancy patterns and the available identification evidence.

Sensitivity results should be interpreted in terms of envelope shift, boundary deformation, and change in ranking among candidate actuation strategies. Large topology changes in the envelope indicate that recommendations are fragile and require either additional experiments or tighter parameter control in hardware. Modest shifts suggest that the operating guidance is robust enough for practical decision support within the stated scope.

[Fig. 20. Sensitivity envelopes under `BG` shift and damping uncertainty. Insert here.]

## 5.5 Integrated Operating Guidance for the Current Platform

The final step combines feasibility, strategy trade-off, and sensitivity results into a deployment-oriented operating guidance chart for the current platform. The recommended region is defined as the subset of conditions that satisfy feasibility criteria, avoid high-risk boundary zones, and maintain acceptable performance under the examined uncertainty range. In other words, the recommended operating region is a robustness-filtered subset of the nominally feasible region.

This integrated presentation is important because individual analyses can be misleading when read in isolation. A condition that appears attractive in the nominal objective surface may lie near a sensitivity-induced boundary shift, while a conservative region with slightly worse nominal performance may be preferable in practice. The resulting guidance chart therefore emphasizes decision transparency and robustness rather than nominal performance alone.

[Fig. 21. Integrated operating-guidance chart combining feasibility, robustness, and strategy preference. Insert here.]

## 5.6 Scope of the Guidance and Transferability Limits

The design guidance in this section is intended for the present mission-oriented platform configuration and the transition phase studied in this manuscript. It should not be generalized to other geometries, payload layouts, or actuation architectures without re-identification and re-validation. The value of the framework lies in the procedure: once a new configuration is instrumented and validated under the same evidence logic, the same analysis structure can be reused to generate configuration-specific guidance.
