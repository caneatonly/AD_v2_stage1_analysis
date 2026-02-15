# 1. Introduction

## 1.1 Engineering Background and Motivation

Underwater corner reflectors serve as passive markers in marine detection and localization tasks. Deployment reliability depends on both the reflector and the carrier platform dynamics during posture transition. In the present deployment scenario, the platform is released near horizontal and is required to stabilize to a near-vertical working posture. The launch-to-stabilization transition is therefore treated as the primary engineering problem.

## 1.2 Related Work and Gap Identification

Existing underwater dynamics studies employ established approaches for hydrodynamic modeling and parameter identification, but most are developed for general-purpose vehicles or for method-focused demonstrations. For mission-specific deployment platforms, this separation introduces uncertainty because structural layout, internal-water coupling, and operating protocol are strongly coupled. Parameter transferability also degrades when large-attitude transition is intrinsic to operation. A suitable framework therefore requires integrated platform design, physically grounded dynamics, and experiment-based validation `[R2-R6]`.

## 1.3 Objectives and Contributions

This study addresses platform design and dynamic characterization in a single workflow. A mission-oriented self-suspending platform is developed with explicit constraints on structure, mass-buoyancy distribution, and deployment sequence. For the transition phase, an anisotropic permeability-corrected 3-DOF model is formulated. The parameters `mu_x`, `mu_z`, and `mu_theta` represent directional internal-water coupling in surge, heave, and pitch.

Figure 1 summarizes the platform architecture and deployment workflow used in this study.

Static CFD is used to obtain hydrodynamic coefficient mappings required for model closure. Rotational damping is retained as an identified term. Model parameters are determined from free-decay data, followed by out-of-sample simulation-to-experiment validation and parametric design analysis.

The contributions are threefold: (1) design, implementation, and experimental characterization of a mission-oriented self-suspending deployment platform; (2) an anisotropic permeability-corrected 3-DOF model with CFD-supported hydrodynamic closure; and (3) an evidence chain from parameter determination to out-of-sample validation and design-oriented parametric analysis.

## 1.4 Scope and Paper Organization

Conclusions are bounded to the current geometry, mass distribution, and actuation strategy. Section 2 describes the platform configuration and deployment scenario. Section 3 presents model development and parameter determination. Section 4 reports validation against free-decay experiments. Section 5 presents parametric analysis and design guidance. Section 6 summarizes conclusions, limitations, and future work.

[Fig. 1. Mission-oriented platform architecture and deployment workflow. Insert here.]
