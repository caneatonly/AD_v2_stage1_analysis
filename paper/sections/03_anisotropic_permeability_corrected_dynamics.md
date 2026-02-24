# 3. Dynamic Model Development and Parameter Determination

## 3.1 Modeling Scope and Architecture of the Reduced-Order Description

This section develops the dynamic model used to describe the horizontal-launch to vertical-stabilization phase introduced in Section 2. The objective is not to reproduce every fluid-structure interaction detail directly, but to construct a reduced-order model that remains physically interpretable, computationally efficient, and compatible with the available free-decay identification data. To achieve this, the model combines three elements: a planar 3-DOF rigid-body/hydrostatic core, CFD-derived static hydrodynamic mappings for force and moment closure, and experimentally identified dynamic terms that cannot be inferred reliably from static CFD alone.

A key design principle is explicit equation governance. In particular, CFD-derived static coefficients are used to replace the corresponding static hydrodynamic force and moment mappings, while rotational damping remains an identification-oriented term and is not double counted through ad hoc additions. This governance rule is important for maintaining both interpretability and consistency between the manuscript equations and the implementation in the simulation pipeline.

## 3.2 Governing Equations for the Transition Phase

Using the state interfaces \(\nu=[u,w,q]^T\) and \(\eta=[\theta]\), the simulation state is written as

$$
\mathbf{x}=[u,\,w,\,q,\,\theta]^T,
$$

with kinematic closure

$$
\dot{\theta}=q.
$$

For the transition phase of interest, the governing dynamics are expressed in a planar Fossen-style form with effective inertias \(m_x\), \(m_z\), and \(I_y\):

$$
\begin{aligned}
m_x\dot{u}-m_zwq &= X_{cfd}(\alpha) - (W-B)\sin\theta + T, \\
m_z\dot{w}+m_xuq &= Z_{cfd}(\alpha) + (W-B)\cos\theta, \\
I_y\dot{q} &= M_{cfd}(\alpha) + M_{damp}(q) + M_{bg}(\theta) + M_{cable}(\theta,q) + M_{thruster}, \\
\dot{\theta} &= q.
\end{aligned}
$$

Here \(W=m_{dry}g\) and \(B=B_{mass}g\). The terms \(T\) and \(M_{thruster}\) denote optional actuation inputs. They are set to zero during free-decay identification and validation unless a specific actuation strategy is being studied in Section 5.

The buoyancy restoring moment is expressed as

$$
M_{bg}(\theta)=B\left(z_b\sin\theta + x_b\cos\theta\right),
$$

and the rotational damping term is modeled as

$$
M_{damp}(q)=-(d_q+d_{q,abs}|q|)q.
$$

When cable effects are enabled in the hardware configuration, the cable contribution is modeled as

$$
M_{cable}(\theta,q)=-K_{cable}(\theta-\theta_{eq})-C_{cable,q}q.
$$

These expressions match the runtime formulation implemented in the simulation module used for validation and parametric analysis.

## 3.3 Hydrodynamic Closure and Angle-of-Attack Mapping

Static hydrodynamic loads are closed through CFD-derived coefficient mappings as functions of the attack angle \(\alpha\). The body-frame speed and dynamic pressure are defined as

$$
V=\sqrt{u^2+w^2}, \qquad Q=\tfrac{1}{2}\rho V^2,
$$

with a numerical safeguard for low-speed states to prevent unstable coefficient evaluation. The model then uses

$$
X_{cfd}=QA_{ref}C_X(\alpha), \qquad
Z_{cfd}=QA_{ref}C_Z(\alpha), \qquad
M_{cfd}=QA_{ref}L_{ref}C_m(\alpha).
$$

The attack angle is computed from the frozen sign convention,

$$
\alpha=\mathrm{atan2}(w,u),
$$

and is mapped to the available CFD lookup range at runtime. The coefficient convention is consistent across the manuscript and implementation: \(C_X\) acts along \(+x_b\), \(C_Z\) acts along \(+z_b\), and \(C_m\) acts about \(+y_b\), with positive \(C_m\) corresponding to nose-up moment.

The CFD role in this paper is intentionally limited to static hydrodynamic closure and mechanism evidence. In particular, when the static pitch-moment mapping \(C_m(\alpha)\) already reflects Munk-like static effects, no additional explicit Munk-moment term is added to the reduced-order pitch equation.

## 3.4 Anisotropic Permeability-Corrected Added Mass and Effective Inertias

A central modeling feature of this study is the anisotropic permeability correction used to represent directional coupling between rigid-body motion and inner-water inertia. Instead of treating inner-water effects as either fully decoupled or fully rigidly attached, the model introduces three channel-specific coupling parameters,

$$
\mu_x,\,\mu_z,\,\mu_{\theta} \in [0,1],
$$

which describe the fraction of inner-water inertia effectively coupled to surge, heave, and pitch motion, respectively. Values near zero indicate weak coupling (high effective internal permeability), whereas values near one indicate near-rigid coupling.

Using the project sign convention for added-mass derivatives, the total added terms are written as

$$
\begin{aligned}
X_{\dot{u},total} &= X_{\dot{u},outer} - \mu_x m_{water,inner}, \\
Z_{\dot{w},total} &= Z_{\dot{w},outer} - \mu_z m_{water,inner}, \\
M_{\dot{q},total} &= M_{\dot{q},outer} - \mu_{\theta} I_{water,inner}.
\end{aligned}
$$

The effective inertias used in the ODE denominators are then

$$
m_x=m_{dry}-X_{\dot{u},total}, \qquad m_z=m_{dry}-Z_{\dot{w},total}, \qquad I_y=I_{yy}-M_{\dot{q},total}.
$$

An important modeling choice, implemented consistently in the codebase, is that the translational denominators use \(m_{dry}\) rather than \(m_{wet}\). This convention is retained throughout the manuscript to avoid hidden discrepancies between equations, configuration files, and numerical simulations.

[Fig. 4. Model-term map from physical mechanisms to equation blocks and code interfaces. Insert here.]
[Fig. 5. Free-body force and moment decomposition for the transition-phase model. Insert here.]
[Fig. 6. Conceptual interpretation of anisotropic internal-water coupling parameters (`mu_x`, `mu_z`, `mu_theta`). Insert here.]

## 3.5 CFD-Supported Coefficient Acquisition and Mechanism Interpretation

The hydrodynamic coefficient mappings used in Section 3.3 are obtained from CFD and treated as model-evidence inputs. Because the reduced-order simulation depends strongly on these mappings over a wide attitude range, the CFD reporting in this paper is organized to support both credibility and interpretability. The first requirement is numerical credibility through mesh-independence and residual-convergence summaries. The second is mechanism interpretability through coefficient trends, pressure-center migration, and representative field visualizations at selected attack angles.

The resulting CFD evidence package is therefore structured in four layers: numerical verification, coefficient mappings, pressure-center migration, and representative flow/pressure fields. This ordering is deliberate. It allows the reader to assess numerical trustworthiness before interpreting the physics encoded in the coefficient curves and moment trends used by the reduced-order model.

[Fig. 7. CFD verification results: mesh-independence and residual convergence. Insert here.]
[Fig. 8. Hydrodynamic coefficient mappings versus attack angle (`C_X`, `C_Z`, `C_m`). Insert here.]
[Fig. 9. Pressure-center migration and static-stability-region transition versus attack angle. Insert here.]
[Fig. 10. Representative pressure and velocity field patterns at selected attack angles. Insert here.]

## 3.6 Parameter Determination Workflow and Optimization Structure

After the model structure and hydrodynamic closure are fixed, dynamic parameters are determined from free-decay experiments using a protocolized pipeline. The pipeline is designed to preserve reproducibility and to make data handling decisions explicit. It includes raw signal preprocessing, deterministic segmentation, condition-level split filtering via the experiment manifest, frequency diagnostics, and a two-stage parameter identification process.

The first identification stage estimates rotational damping coefficients using a cycle-energy method with non-negative least squares. This stage provides a physically informed initialization for the second stage and reduces sensitivity to poor starting points. The second identification stage performs ODE-based residual minimization with bounded parameters. In the current implementation, the optimization is carried out on a pitch-dominant reduced model for \(\theta\) and \(q\), while the full 3-DOF model defined above is reserved for simulation-based validation and design analysis. This separation improves practical identifiability while preserving a consistent downstream validation model.

For a set of calibration segments, the Step-4 optimization minimizes a stacked residual vector over simulated and measured pitch angle and pitch rate histories, subject to parameter bounds on \(\mu_{\theta}\), \(d_q\), the quadratic damping coefficient (implemented as `d_qq` in the Step-4 fitter and corresponding to \(d_{q,abs}\) in the manuscript notation), and \(K_{cable}\). Multi-start optimization and bootstrap resampling are supported by the pipeline configuration, even when disabled in a baseline run. In the current protocol snapshot included with this repository stage, `multi_start_n=1` and `n_boot=0`, so the baseline execution behaves as a single-start fit without bootstrap uncertainty expansion. The pipeline also records reproducibility artifacts (protocol snapshot, git commit hash, and Python environment snapshot) alongside fitted parameters.

[Fig. 11. Parameter-identification workflow from segmented free-decay data to calibrated model parameters. Insert here.]
[Fig. 12. Identification diagnostics and uncertainty summaries (residuals, multi-start convergence, bootstrap intervals). Insert here.]

## 3.7 Model Closure for Validation and Design Analysis

At the end of Section 3, the study produces a closed model package consisting of fixed platform and hydrostatic parameters, CFD-derived static hydrodynamic mappings, and experimentally calibrated dynamic parameters. This package is transferred to Section 4 without retuning for simulation-to-experiment validation and is subsequently used in Section 5 for parametric analysis and deployment-oriented design guidance. The no-retuning rule between calibration and validation is essential to the claim that the framework provides engineering evidence rather than post hoc curve fitting.
