# 2. Platform Configuration and Deployment Scenario

## 2.1 Mission-Driven Platform Configuration

The platform studied in this work is developed for underwater corner-reflector deployment, where the reflector acts as a passive marker and the carrier platform must rapidly achieve an operational posture after release. The core engineering challenge is not merely transporting the payload but ensuring a reliable post-release attitude transition under practical deployment constraints. Accordingly, the platform is designed as a self-suspending system in which structural layout, mass distribution, and buoyancy placement are tuned for passive posture recovery during the transition phase.

The hardware concept combines a rigid-body frame, a prescribed buoyancy-center offset in the body frame, inner-water coupling effects induced by the platform structure, and an optional cable-related restoring contribution in the current implementation. This combination is central to the system behavior and motivates the modeling strategy adopted in Section 3. Mission-sensitive operational details are intentionally abstracted, but the physical assumptions that affect the transition dynamics are stated explicitly to preserve reproducibility.

For the nominal configuration used throughout this paper, the rigid-body dry mass is \(m_{dry}=2.55\,\mathrm{kg}\), the wet mass is \(m_{wet}=2.76\,\mathrm{kg}\), and the inner-water equivalent mass is \(m_{water,inner}=0.21\,\mathrm{kg}\). The rigid-body pitch inertia is \(I_{yy}=0.05741\,\mathrm{kg\,m^2}\), and the inner-water equivalent pitch inertia is \(I_{water,inner}=0.01119\,\mathrm{kg\,m^2}\). Buoyancy is represented through an equivalent mass \(B_{mass}=2.55\,\mathrm{kg}\) with nominal body-frame offset \(x_b=0.02535\,\mathrm{m}\) and \(z_b=0\). These parameters define the baseline physical configuration for model closure, identification, and evaluation.

## 2.2 Deployment Phase Definition and Coordinate Conventions

The present analysis targets a single mission-critical phase: horizontal launch followed by passive transition toward vertical stabilization. Under the convention adopted in this study, the stabilized working posture lies near \(\theta=90^\circ\). The analysis therefore focuses on the transient evolution from the release state to the stabilized posture and excludes downstream control tasks such as depth regulation or long-horizon trajectory management.

A fixed body-axis convention is used throughout the repository and manuscript to avoid sign inconsistencies across CFD post-processing, parameter identification, and time-domain simulation. The body axes are defined as \(x_b\) forward and \(z_b\) downward. Positive pitch is nose-up, and pitch rate is \(q=\dot{\theta}\) about \(+y_b\). The state interfaces are \(\nu=[u,w,q]^T\) and \(\eta=[\theta]\), and the angle of attack for hydrodynamic lookup is defined by \(\alpha=\mathrm{atan2}(w,u)\). These conventions are frozen in the implementation and should be treated as part of the model contract.

[Fig. 2. Body-axis definition, sign convention, and state interface (`u, w, q, theta`). Insert here.]

## 2.3 Experimental Program and Data Governance

Free-decay experiments provide the primary identification and validation data for this paper. Each run follows a protocolized sequence consisting of initial stabilization, recording start, a single manual excitation, free decay, and final stabilization. One raw file is required to contain only one excitation-decay event. This one-event-per-file rule is essential for deterministic preprocessing and segmentation and directly supports downstream condition-level split control.

Raw files are stored under the pipeline data contract and must satisfy a fixed column schema with monotonic timestamps. The current baseline repository contains three free-decay segments, while the planned expansion targets a 12-condition matrix spanning \(\theta_0\) and \(q_0\) levels with repeated trials. The key design choice for data governance is that splits are defined by operating-condition blocks rather than random sample points. This avoids leakage between calibration and validation when waveform segments from the same condition would otherwise share strong temporal and dynamic similarity.

[Fig. 3. Experimental condition matrix, repeat policy, and anti-leakage split strategy. Insert here.]

## 2.4 System Boundary and Assumptions for Subsequent Sections

Three assumptions delimit the claims made in the remainder of the paper. First, mission context is retained at the level required to motivate the platform and transition-phase objective, but mission-sensitive details are abstracted. Second, the conclusions are intentionally restricted to the current platform geometry, mass-buoyancy configuration, and actuation strategy. Third, CFD is used as hydrodynamic closure and mechanism-support evidence, not as a standalone novelty claim independent of the platform-model-validation workflow.

With these assumptions, Section 2 establishes the physical and experimental boundary conditions needed for the reduced-order model, the parameter determination protocol, and the validation logic developed in the following sections.
