# 2. Platform Configuration and Deployment Scenario

## 2.1 Mission-Oriented Platform Design

The platform is developed for underwater corner-reflector deployment, where the reflector serves as a passive high-contrast marker for marine detection and localization tasks. The key challenge is reliable posture transition after release, rather than payload placement alone. The platform is required to be deployable in a compact launch state and to stabilize autonomously to the working posture.

Unlike general-purpose underwater-vehicle studies, the present work is driven by a specific deployment requirement. Transition dynamics are analyzed because they directly affect deployment success and operational readiness. Mission details are abstracted for confidentiality, while dynamic assumptions and identification protocol are kept explicit for reproducibility.

The proposed system is a self-suspending underwater platform with coupled structural and hydrostatic design. The architecture is defined by (i) rigid-body mass distribution, (ii) buoyancy-center placement in the body frame, (iii) internal-water coupling represented by permeability-corrected terms, and (iv) optional cable-related restoring effects in the current hardware configuration.

For the nominal setup, the key parameters are dry mass \(m_{dry}=2.55\,\mathrm{kg}\), wet mass \(m_{wet}=2.76\,\mathrm{kg}\), inner-water equivalent mass \(m_{water,inner}=0.21\,\mathrm{kg}\), pitch inertia \(I_{yy}=0.05741\,\mathrm{kg\,m^2}\), and inner-water inertia \(I_{water,inner}=0.01119\,\mathrm{kg\,m^2}\). Buoyancy is represented by equivalent mass \(B_{mass}=2.55\,\mathrm{kg}\) with nominal offset \(x_b=0.02535\,\mathrm{m}\). These values define the baseline configuration used in all simulation-experiment comparisons.

## 2.2 Coordinate System and State Definitions

The analysis focuses on one mission-critical phase: horizontal launch to vertical stabilization. The stabilized posture is defined near \(\theta=90^{\circ}\) under the adopted convention. Closed-loop depth-keeping and long-horizon mission execution are outside the present scope.

The state vectors are defined as \(\nu=[u,w,q]^T\) and \(\eta=[\theta]\). The body-axis convention is \(x_b\) forward, \(z_b\) downward, positive pitch nose-up, and \(q=d\theta/dt\) about \(+y_b\). Angle of attack is defined by \(\alpha=\mathrm{atan2}(w,u)\). These fixed definitions enforce consistency among coefficient mapping, parameter determination, and dynamic simulation.

[Fig. 2. Body-axis definition, sign convention, and state interface (`u, w, q, theta`). Insert here.]

## 2.3 Experimental Conditions and Data Protocol

The operational sequence in each run is initial vertical rest, recording start, single manual excitation, free decay, and final rest. Each raw run contains one excitation-decay event. This protocol enables deterministic segmentation and condition-level split control.

Protocolized run definitions, deterministic segmentation, and condition-level split control follow established experimental-validation practice `[R3-R6]`.

The current baseline dataset contains three free-decay segments, with planned expansion to a 12-condition matrix over \(\theta_0\) and \(q_0\) levels (minimum two repeats per condition). Data are split by condition blocks rather than random samples. Leave-one-theta-level-out validation is adopted to evaluate extrapolation behavior.

[Fig. 3. Experimental condition matrix, repeat policy, and anti-leakage split strategy. Insert here.]

## 2.4 Scope and Assumptions

Three assumptions delimit the claims in this paper. First, mission details are abstracted while the deployment role remains explicit. Second, conclusions are restricted to the current geometry, mass distribution, and actuation strategy. Third, CFD is used for hydrodynamic coefficient acquisition and mechanism support, rather than as a standalone novelty claim.

Under these assumptions, this section defines the system boundary and engineering context for model development, parameter determination, and validation.
