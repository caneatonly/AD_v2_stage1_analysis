# 1. Introduction

## 1.1 Mission Context and Engineering Motivation

Underwater corner reflectors are widely used as passive high-contrast markers for marine detection and localization tasks. In practical deployment, however, system reliability is governed not only by the reflector itself but also by the transition behavior of the carrier platform immediately after release. For the mission considered in this study, the platform is released in a compact, near-horizontal configuration and must autonomously evolve toward a near-vertical working posture before the payload can be regarded as operationally effective. The horizontal-launch to vertical-stabilization transition is therefore a mission-defining engineering requirement rather than a secondary kinematic detail.

This problem setting differs from the more common objective of general-purpose underwater vehicle maneuvering or trajectory tracking. Here, the platform is intentionally designed around a single critical mission phase, and the success criterion is tied to robust posture transition under constrained geometry, mass distribution, and hydrostatic layout. Consequently, platform design, dynamic modeling, parameter identification, and experimental validation must be treated as a coupled workflow rather than as independent tasks.

## 1.2 Related Work and Remaining Gap

Existing studies in underwater dynamics and marine mechatronic systems provide mature tools for rigid-body modeling, hydrodynamic coefficient estimation, CFD-assisted mechanism analysis, and experiment-based parameter identification `[R2-R6]`. These methods are essential foundations for the present work. Nevertheless, direct transfer of generic models and parameters is often unreliable when a system exhibits strong mission-specific coupling among structure, buoyancy arrangement, internal fluid motion, and deployment protocol, particularly in large-attitude transition regimes.

Two gaps are therefore important for the present application. First, many studies prioritize either platform hardware realization or model identification, while the deployment mission requires both to be co-designed and reported within a single engineering evidence chain. Second, static CFD information is frequently used without a clear governance rule for what it replaces in the reduced-order dynamics, which can lead to term duplication or ambiguous physical interpretation. A mission-oriented framework is needed in which the platform role, model structure, coefficient closure logic, and validation protocol are jointly defined and reproducible.

## 1.3 Objectives and Contributions of This Study

The objective of this paper is to establish a reproducible platform-model-validation workflow for the transition phase of a mission-oriented self-suspending underwater deployment platform. The platform is designed to achieve passive posture recovery toward the working orientation, while the dynamic model is formulated to preserve physical interpretability and remain compatible with experimentally identifiable parameters.

To this end, we develop an anisotropic permeability-corrected 3-DOF model in surge, heave, and pitch, in which \(\mu_x\), \(\mu_z\), and \(\mu_{\theta}\) describe directional inner-water coupling strength. Static CFD-derived hydrodynamic mappings are used as model-closure evidence for \(C_X(\alpha)\), \(C_Z(\alpha)\), and \(C_m(\alpha)\), whereas rotational damping is retained as an experiment-identified component. Parameters are then determined from protocolized free-decay data using condition-aware split control, followed by simulation-to-experiment validation and sensitivity-driven design analysis.

The principal contribution of the paper is not a standalone model or a standalone prototype, but a closed engineering evidence chain that links mission requirements, platform configuration, interpretable dynamics, parameter determination, and bounded design guidance under the current hardware configuration.

[Fig. 1. Mission-oriented platform architecture and deployment workflow. Insert here.]

## 1.4 Scope, Claim Boundary, and Paper Organization

This paper is deliberately scoped to the horizontal-launch to vertical-stabilization phase. Closed-loop depth control, long-duration mission behavior, and broader mission execution logistics are outside the present study and are discussed only as future extensions. In addition, all conclusions are explicitly bounded to the current geometry, mass distribution, buoyancy layout, and actuation strategy; the results should not be interpreted as universal laws for all underwater deployment platforms.

The remainder of the paper is organized as follows. Section 2 defines the platform configuration, coordinate conventions, and experimental data protocol. Section 3 presents the dynamic model, anisotropic permeability correction, CFD-based hydrodynamic closure, and parameter determination procedure. Section 4 evaluates the calibrated model against free-decay experiments under condition-level split control and analyzes discrepancy modes. Section 5 develops parametric analyses and deployment-oriented design guidance based on the validated model package. Section 6 summarizes the main findings, engineering implications, limitations, and future work.
