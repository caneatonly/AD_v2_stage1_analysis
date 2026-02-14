# 2. Platform description and deployment scenario

## 2.1 Mission context

The platform in this study is developed for underwater corner-reflector deployment. In this context, the corner reflector acts as a passive high-contrast marker to support marine detection and localization-related operations. The engineering challenge is not only payload placement, but also reliable posture transition of the carrier after release. The platform must be deployable in a compact launch state and then autonomously reach a stable in-water working posture with repeatable dynamics.

Unlike a general-purpose underwater vehicle study, this work is mission-oriented from the beginning: the platform is designed around a specific deployment requirement, and the transition dynamics are investigated because they directly affect deployment success and operational readiness. To satisfy confidentiality constraints of mission details, payload-specific tactical information is abstracted, while all dynamic and identification assumptions relevant to reproducibility are retained.

## 2.2 Platform architecture

The proposed system is a self-suspending underwater platform with coupled structural and hydrostatic design. The architecture is defined by: (i) rigid-body mass distribution, (ii) buoyancy-center placement relative to the body frame, (iii) internal-water coupling represented later by permeability-corrected terms, and (iv) optional cable-related restoring effects used in the present hardware configuration.

For the nominal setup used throughout this manuscript, key physical parameters are frozen in `sim_flip/configs/params_nominal.yaml`: dry mass `m_dry = 2.55 kg`, wet mass `m_wet = 2.76 kg`, inner-water equivalent mass `m_water_inner = 0.21 kg`, pitch inertia `Iyy = 0.05741 kg m^2`, and inner-water inertia `I_water_inner = 0.01119 kg m^2`. Buoyancy is implemented as equivalent mass with `B_mass = 2.55 kg` and nominal offset `x_b = 0.02535 m`. These values define the baseline configuration for all reported simulation-experiment comparisons.

[Fig. 2. Body-axis definition, sign convention, and state interface (`u, w, q, theta`). Insert here.]

## 2.3 Phase of interest and state definition

This paper focuses on one mission-critical phase: horizontal launch to vertical stabilization. The stabilized posture is defined near `theta = 90 deg` under the project convention. The analysis excludes full closed-loop depth-keeping and long-horizon mission execution; those are treated as future work.

The state interface used across data processing, modeling, and simulation is fixed as `nu = [u, w, q]^T` and `eta = [theta]`. Sign and axis conventions are frozen in `sim_flip/src/conventions.py`: body `x_b` forward, `z_b` downward, nose-up pitch positive, and `q = dtheta/dt` about `+y_b`. Angle of attack is computed as `alpha = atan2(w, u)`. Freezing these definitions prevents silent inconsistency between CFD tables, identification scripts, and dynamic simulation.

## 2.4 Experimental program

The operational sequence used for the current dataset is: initial vertical rest, data recording start, single manual excitation, free decay, and final rest. One raw run contains exactly one excitation-decay event. This rule is enforced in the data protocol to support deterministic segmentation and condition-level split control.

The use of protocolized run definitions, deterministic segmentation, and condition-level split control is consistent with best practices in recent CFD-identification and experimental-validation studies `[R3-R6]`.

At the time of this draft, the baseline evidence includes three free-decay segments, with planned expansion to a 12-condition matrix over `theta_0` and `q_0` levels (minimum two repeats per condition). Data split is performed by condition blocks rather than random sample points, and leave-one-theta-level-out validation is adopted to test extrapolation behavior. This design links platform operation constraints with identification and out-of-sample evaluation requirements.

[Fig. 3. Experimental condition matrix, repeat policy, and anti-leakage split strategy. Insert here.]

## 2.5 Scope and assumptions

Three assumptions delimit the claims in this paper. First, mission description is abstracted, but the deployment role of the platform is explicit. Second, conclusions are restricted to the present geometry, mass distribution, and actuation strategy. Third, CFD is used as a mechanism and coefficient evidence layer for model-term mapping, not as a standalone novelty claim.

Under these assumptions, the contribution of this section is to define the system boundary and engineering context for the dynamic model and identification framework introduced in the following sections.

Reference map for placeholders `[R3-R6]` is maintained in `paper/figures/FIGURE_REQUIREMENTS.md` (Section 5).
