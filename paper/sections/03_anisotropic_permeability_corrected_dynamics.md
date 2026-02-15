# 3. Dynamic Model Development and Parameter Determination

## 3.1 Governing Equations

This section presents a 3-DOF governing model for the horizontal-launch to vertical-stabilization phase, with state interfaces \(\nu=[u,w,q]^T\) and \(\eta=[\theta]\). The body-axis convention follows Section 2 (Fig. 2): \(x_b\) forward, \(z_b\) downward, positive pitch nose-up, and \(q=d\theta/dt\) about \(+y_b\).

The simulation state is

$$
\mathbf{x} = [u,\; w,\; q,\; \theta]^T,
$$

where \(u\) and \(w\) are body-frame translational velocities, \(q\) is pitch rate, and \(\theta\) is pitch angle. The kinematic closure is

$$
\dot{\theta} = q.
$$

For the present phase of interest, the governing dynamics are written in a compact planar Fossen-style form with effective inertias (defined in Section 3.2):

$$
\begin{aligned}
m_x\,\dot{u} - m_z\,wq &= X_{cfd}(\alpha) - (W - B)\,\sin\theta + T, \\
m_z\,\dot{w} + m_x\,uq &= Z_{cfd}(\alpha) + (W - B)\,\cos\theta, \\
I_y\,\dot{q} &= M_{cfd}(\alpha) + M_{damp}(q) + M_{bg}(\theta) + M_{cable}(\theta,q) + M_{thruster}, \\
\dot{\theta} &= q.
\end{aligned}
$$

Here, \(W=m_{dry}g\) is the weight magnitude and \(B=B_{mass}g\) is the buoyancy magnitude. The terms \(T\) and \(M_{thruster}\) are optional actuation inputs; they are set to zero in free-decay parameter determination.

Hydrodynamic forces and moment are represented by static mappings through dynamic-pressure scaling:

$$
V = \sqrt{u^2 + w^2}, \qquad Q = \tfrac{1}{2}\rho V^2,
$$

with a numerical safeguard \(Q=0\) when \(V<V_{eps}\). The model uses

$$
X_{cfd} = Q\,A_{ref}\,C_X(\alpha), \qquad
Z_{cfd} = Q\,A_{ref}\,C_Z(\alpha), \qquad
M_{cfd} = Q\,A_{ref}\,L_{ref}\,C_m(\alpha).
$$

The attack angle is computed by

$$
\alpha_{raw} = \mathrm{atan2}(w, u),
$$

and mapped/clamped to the lookup range at runtime.

## 3.2 Anisotropic Permeability Correction

The model introduces an anisotropic permeability correction to the added-mass and added-inertia terms. Two contributions are considered: (i) outer-fluid added mass and (ii) inner-water coupling. Three directional coupling parameters

$$
\mu_x,\;\mu_z,\;\mu_{\theta} \in [0,1]
$$

represent the fraction of inner-water inertia coupled to rigid-body motion in surge, heave, and pitch, respectively. In this interpretation, \(\mu\approx 0\) corresponds to high internal permeability (weak coupling), and \(\mu\approx 1\) corresponds to near-rigid coupling.

Using the project sign convention (Fossen-style negative added-mass derivatives), total added terms are

$$
\begin{aligned}
X_{\dot{u},total} &= X_{\dot{u},outer} - \mu_x\,m_{water,inner}, \\
Z_{\dot{w},total} &= Z_{\dot{w},outer} - \mu_z\,m_{water,inner}, \\
M_{\dot{q},total} &= M_{\dot{q},outer} - \mu_{\theta}\,I_{water,inner}.
\end{aligned}
$$

The corresponding effective inertias in the ODE denominators are

$$
m_x = m_{dry} - X_{\dot{u},total}, \qquad
m_z = m_{dry} - Z_{\dot{w},total}, \qquad
I_y = I_{yy} - M_{\dot{q},total}.
$$

The effective inertias are computed from the nominal platform parameter set. Translational denominators use \(m_{dry}\), consistent with the adopted model interface.

[Fig. 4. Model-term map from physical mechanisms to equation blocks and code interfaces. Insert here.]

The force and moment decomposition in the body-fixed frame is summarized in Fig. 5.

[Fig. 5. Free-body force and moment decomposition for the transition-phase model. Insert here.]

[Fig. 6. Conceptual interpretation of anisotropic internal-water coupling parameters (`mu_x`, `mu_z`, `mu_theta`). Insert here.]

## 3.3 Hydrodynamic Coefficient Acquisition from CFD

CFD is used to close the dynamic model by supplying static mappings \(C_X(\alpha)\), \(C_Z(\alpha)\), \(C_m(\alpha)\), and the pressure-center trend \(X_{cp}(\alpha)\) for mechanism interpretation. CFD is positioned as supporting evidence for model closure.

Numerical verification is provided through mesh-independence and residual-convergence summaries to support coefficient reliability.

[Fig. 7. CFD verification results: mesh-independence and residual convergence. Insert here.]

The coefficient mappings are reported across the full attack-angle range required for transition simulation, with local enlargement in high-angle regions where response sensitivity is strongest.

[Fig. 8. Hydrodynamic coefficient mappings versus attack angle (`C_X`, `C_Z`, `C_m`). Insert here.]

Pressure-center migration is then analyzed to interpret pitch-moment behavior and static stability-region transition.

[Fig. 9. Pressure-center migration and static-stability-region transition versus attack angle. Insert here.]

Representative flow and pressure fields at selected angles are finally presented to support the observed coefficient trends and pressure-center movement.

[Fig. 10. Representative pressure and velocity field patterns at selected attack angles. Insert here.]

No additional explicit Munk-moment term is introduced when \(C_m(\alpha)\) already captures the corresponding static effect.

## 3.4 Parameter Identification Procedure

After model structure and coefficient mappings are fixed, parameters are determined using a protocolized free-decay identification procedure. The workflow includes preprocessing, deterministic segmentation, condition-level split control, energy-based damping estimation, and ODE residual minimization with multi-start initialization. Bootstrap analysis is included when uncertainty quantification is required.

[Fig. 11. Parameter-identification workflow from segmented free-decay data to calibrated model parameters. Insert here.]

The identification stage reports residual diagnostics, convergence behavior across starts, and confidence summaries to assess parameter stability and identifiability.

[Fig. 12. Identification diagnostics and uncertainty summaries (residuals, multi-start convergence, bootstrap intervals). Insert here.]

## 3.5 Identified Parameter Set and Model Closure

Section 3 yields a closed model package containing (i) fixed structural and hydrostatic parameters, (ii) CFD-supported static hydrodynamic mappings, and (iii) identified dynamic parameters for transition-phase simulation. This package is used without re-tuning in Section 4 for validation against free-decay experiments.
