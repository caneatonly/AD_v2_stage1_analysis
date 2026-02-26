# 1. Introduction

## 1.1 Mission Context and Engineering Motivation

Underwater corner reflectors are widely used as passive high-contrast markers for marine detection and localization tasks. In practical deployment, system reliability is governed not only by the reflector packaging but also by the post-release behavior of the carrier platform. In the scenario considered in this study, the platform is released in a compact, near-horizontal configuration with nonzero initial velocity and must autonomously transition to a near-vertical working posture. This horizontal-launch to vertical-stabilization phase is therefore a mission-defining engineering requirement.

This setting differs from general-purpose underwater vehicle maneuvering or long-horizon trajectory tracking. Here, the platform is intentionally designed around one critical phase with a clear success criterion (reaching and maintaining the working posture with bounded overshoot and residual rate), under constraints imposed by packaging, buoyancy layout, and deployment workflow. As a result, platform realization, dynamic modeling, parameter identification, and experimental validation must be treated as a coupled engineering workflow.

## 1.2 Related Work and the Mission-Specific Gap

Existing studies on underwater corner reflectors primarily investigate acoustic scattering characteristics and structural design variables to enhance detectability and target strength, including air-cavity concepts, array configurations, elastic-loss effects, and metasurface-based modulation (e.g., \\citep{Luo2020AirCavityCornerReflector,Chu2022EightGridAirCavityCornerReflector,Xiao2025CornerReflectorLinearArrays,Luo2025ElasticLossCornerReflector,Du2023MetasurfaceCornerReflector}). These works establish important reflector-level knowledge, but they typically do not address how a miniature carrier platform is engineered and validated to place a corner reflector under a mission-constrained deployment workflow.

Separately, underwater vehicles and payload-deployment problems have been studied in terms of separation dynamics, hydrodynamic influence factors, and attitude stabilization during release (e.g., \\citep{Wang2024AUVInitialSeparation,Deng2024PayloadReleaseAttitudeStabilization,Zhang2025RetrievalCaptureMechanism}). However, the integration of a miniature corner-reflector deployment platform with a transition-phase dynamics model, experiment-based identification, and no-retuning validation evidence chain remains, to our knowledge, only sparsely documented in the open literature.

## 1.3 Objectives and Contributions of This Study

The objective of this paper is to establish a reproducible platform--model--evidence workflow for the mission-critical transition phase of underwater corner-reflector deployment. We design and fabricate a mission-oriented self-suspending underwater platform, and we characterize the release-to-stabilization behavior through a physically interpretable model and a protocolized experimental identification and validation program.

To analyze and guide the transition from a near-horizontal, finite-velocity release to vertical stabilization, we develop an anisotropic permeability-corrected planar three-degree-of-freedom model in surge, heave, and pitch. Directional inner-water coupling is represented by three bounded parameters (\(\mu_x\), \(\mu_z\), \(\mu_{\theta}\)), static hydrodynamic loads are closed using CFD-derived coefficient mappings as controlled replacements of static force and moment terms, and rotational damping is retained as an experimentally identified component. The workflow is executed with deterministic preprocessing and segmentation, condition-level split governance to prevent leakage, and a no-retuning rule between calibration and validation.

The main contributions are therefore: (i) a realized mission-oriented self-suspending platform and deployment workflow tailored to corner-reflector deployment, (ii) an interpretable reduced-order transition model with permeability-corrected inertial coupling and governed CFD-to-model closure logic, and (iii) a closed evidence chain from free-decay identification to out-of-sample validation and sensitivity-based operating guidance.

[Fig. 1. Mission-oriented platform architecture and deployment workflow. Insert here.]

## 1.4 Scope, Claim Boundary, and Paper Organization

This paper is deliberately scoped to the horizontal-launch to vertical-stabilization phase. Closed-loop depth control, long-duration mission behavior, and broader mission logistics are outside the present study and are discussed only as future work. All conclusions are explicitly bounded to the current platform geometry, mass distribution, buoyancy layout, and actuation strategy; the results should not be interpreted as universal laws for all underwater deployment platforms.

The remainder of the paper is organized as follows. Section 2 reports the platform architecture, key design choices, and deployment scenario. Section 3 describes the experimental setup and data governance used to obtain protocolized free-decay evidence. Section 4 develops the reduced-order transition dynamics and the governed CFD-based hydrodynamic closure. Section 5 presents the parameter identification workflow and uncertainty-oriented diagnostics. Section 6 validates the closed model package against free-decay experiments under condition-level split control and analyzes discrepancy modes. Section 7 develops parametric analyses and bounded operating guidance for the current platform configuration. Section 8 summarizes the main findings, limitations, and future work.
