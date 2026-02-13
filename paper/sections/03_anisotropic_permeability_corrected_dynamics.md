# 3. Anisotropic Permeability-Corrected Dynamics

## 3.1 Governing Equations and State Definition

This section introduces the 3-DOF governing model for the horizontal-launch to vertical-stabilization phase, with frozen interfaces `nu = [u, w, q]^T` and `eta = [theta]`. The body-axis and sign conventions follow Section 2 and `sim_flip/src/conventions.py` (Fig. 2): `x_b` forward, `z_b` downward, nose-up pitch positive, and `q = dtheta/dt` about `+y_b`.

The simulation state is

$$
\mathbf{x} = [u,\; w,\; q,\; \theta]^T,
$$

where `u` and `w` are body-frame translational velocities, `q` is pitch rate, and `theta` is the pitch angle. The kinematic closure is

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

Here, `W = m_dry g` is the weight magnitude and `B = B_mass g` is the buoyancy magnitude (implemented as equivalent mass in the parameter tree). The terms `T` and `M_thruster` are optional actuation inputs; in the free-decay identification setting they are set to zero.

Hydrodynamic forces and moment are represented by static-CFD tables through a dynamic-pressure scaling:

$$
V = \sqrt{u^2 + w^2}, \qquad Q = \tfrac{1}{2}\rho V^2,
$$

with a numerical safeguard `Q=0` when `V < V_eps` for robustness. The CFD-driven force and moment are

$$
X_{cfd} = Q\,A_{ref}\,C_X(\alpha), \qquad
Z_{cfd} = Q\,A_{ref}\,C_Z(\alpha), \qquad
M_{cfd} = Q\,A_{ref}\,L_{ref}\,C_m(\alpha),
$$

where `A_ref` and `L_ref` are fixed reference area and length (`sim_flip/configs/params_nominal.yaml`). The attack angle is computed by the frozen convention

$$
\alpha_{raw} = \mathrm{atan2}(w, u),
$$

and mapped/clamped to the CFD lookup range at runtime (details in Section 4; implementation in `sim_flip/src/cfd_table.py` and `sim_flip/src/kinematics.py`). Following the equation-governance rule, no additional explicit Munk-moment term is introduced when `C_m(\alpha)` already encodes the corresponding static effect.

## 3.2 Permeability-Corrected Added-Mass Terms

The main modeling novelty in this section is an anisotropic permeability correction to the added-mass / added-inertia totals. We conceptually separate (i) an outer-fluid added-mass set and (ii) an inner-water coupling contribution. Three directional coupling parameters

$$
\mu_x,\;\mu_z,\;\mu_{\theta} \in [0,1]
$$

represent the fraction of inner-water inertia that effectively couples to the rigid-body motion in surge (`x`), heave (`z`), and pitch (`\theta`) channels, respectively. This formulation is intentionally compact and physically interpretable: `\mu \approx 0` indicates weak coupling (high internal permeability / relative flow), while `\mu \approx 1` indicates strong coupling (near-lumped inner water).

Using the project sign convention (typical Fossen form expects negative added-mass derivatives), the total added-mass / added-inertia terms are defined as:

$$
\begin{aligned}
X_{\dot{u},total} &= X_{\dot{u},outer} - \mu_x\,m_{water,inner}, \\
Z_{\dot{w},total} &= Z_{\dot{w},outer} - \mu_z\,m_{water,inner}, \\
M_{\dot{q},total} &= M_{\dot{q},outer} - \mu_{\theta}\,I_{water,inner}.
\end{aligned}
$$

The corresponding effective inertias used in the ODE denominators are

$$
m_x = m_{dry} - X_{\dot{u},total}, \qquad
m_z = m_{dry} - Z_{\dot{w},total}, \qquad
I_y = I_{yy} - M_{\dot{q},total}.
$$

All terms above are frozen by the parameter contract and are computed in `sim_flip/src/added_mass.py` from `sim_flip/configs/params_nominal.yaml` (`rigid_body`, `added_mass_outer`, `permeability`). Consistent with the implementation, the translational denominators use `m_dry` (not `m_wet`).

## 3.3 Restoring, Damping, and Actuation Terms

The model retains Fossen-style rigid-body coupling through the Coriolis-like cross terms `\pm m\,\cdot\,(\cdot)\,q` in the surge/heave equations. Restoring, damping, and optional configuration-dependent terms are defined as follows.

Hydrostatic restoring moment is modeled by a buoyancy-center offset `(x_b, z_b)` in the body frame:

$$
M_{bg}(\theta) = B\,(z_b\,\sin\theta + x_b\,\cos\theta),
$$

where `B = B_mass g`. This term is a concise representation of the pitch moment induced by buoyancy about the chosen body reference point and is consistent with the platform parameterization in Section 2.

Rotational damping is retained as an identification-oriented term (not obtained from static CFD tables):

$$
M_{damp}(q) = -\left(d_q + d_{q,abs}|q|\right)q.
$$

This mixed linear-quadratic form provides a pragmatic balance between interpretability, numerical stability, and fit quality for free-decay data, while keeping the CFD contribution restricted to static force/moment mappings.

If a cable restoring mechanism is enabled in the hardware configuration, its contribution is modeled as a linear torsional spring-damper around an equilibrium angle `theta_eq`:

$$
M_{cable}(\theta,q) = -K_{cable}(\theta - \theta_{eq}) - C_{cable,q}\,q.
$$

When cable effects are not present, the corresponding parameters are set to disable this term (`enabled: false`), and `M_{cable}` is identically zero. Optional actuation terms `T` and `M_{thruster}` are provided as external inputs in the simulation interface but are set to zero for the free-decay identification experiments reported in this paper.

## 3.4 Model Interfaces and Backward Compatibility

The manuscript equations above map one-to-one to the repository implementation:

- State ordering: `y = [u, w, q, theta]` is used consistently in the ODE right-hand side and events (`sim_flip/src/dynamics.py`).
- Kinematics and CFD inputs: `V`, `Q`, and `alpha` handling are computed in `sim_flip/src/kinematics.py`; coefficient interpolation uses `sim_flip/src/cfd_table.py` and the default dataset `sim_flip/data/cfd_table_clean.csv`.
- Permeability correction: added-mass totals and effective inertias are computed in `sim_flip/src/added_mass.py` from the YAML parameter tree.
- Parameter interfaces: `rho/g/A_ref/L_ref`, `m_dry/Iyy/m_water_inner/I_water_inner`, `X_udot_outer/Z_wdot_outer/M_qdot_outer`, `mu_x/mu_z/mu_theta`, `B_mass/x_b/z_b`, `d_q/d_q_abs`, and optional cable terms are single-sourced in `sim_flip/configs/params_nominal.yaml`.

This explicit mapping is intended to support reproducibility: a reader can re-generate simulation trajectories and force/moment diagnostics by running the simulation interface with the frozen parameter file and the published CFD lookup table, without requiring hidden hand-tuned terms.

[Fig. 4. Model-term map from physical mechanisms to equation blocks and code interfaces. Insert here.]
