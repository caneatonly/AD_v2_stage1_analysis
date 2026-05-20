# 3. Transition-stage hybrid dynamic model

Following the platform realization and mission sequence described in Section 2, this section formulates a hybrid reduced-order dynamic model for the transition stage of the open-mouth underwater deployment platform. The model is constructed for the fairing-attached and reflector-uninflated configuration, in which the inflatable flexible corner reflector remains stored inside the detachable fairing. Its purpose is not to reproduce a full six-degree-of-freedom marine-vehicle formulation, but to provide a physically interpretable longitudinal model for the reorientation from near-horizontal release to a near-vertical working attitude.

During this stage, the platform motion is governed by rigid-platform inertia, gravity--buoyancy restoring, internal-water participation in the open-mouth fairing, large-angle hydrodynamic loading, and pitch-rate-dependent rotational dissipation. Fairing separation, reflector inflation, and subsequent depth-maintaining operation occur after this transition and are outside the scope of the present model. The resulting formulation provides the common model definition used later for parameter closure, validation, model-block ablations, and release-condition analysis.

## 3.1 Transition-stage problem definition

The transition stage starts when the deployment platform leaves the carrier or launching constraint with a near-horizontal attitude and a finite release velocity. It ends when the platform reaches a near-vertical quasi-steady working attitude suitable for the subsequent deployment actions. Thus, the transition stage is a prerequisite for reliable fairing separation and reflector inflation, although neither action is modeled in this section.

Let \(t_0\) denote the release instant. The initial release state is written as

$$
t=t_0,\qquad \theta(t_0)=\theta_0,\qquad
\nu(t_0)=\nu_0=\begin{bmatrix}u_0&w_0&q_0\end{bmatrix}^{T},
$$

where \(\theta\) is the planar pitch attitude, \(u\) and \(w\) are the body-axis translational velocities in the longitudinal plane, and \(q\) is the pitch rate. The initial release state \((\theta_0,u_0,w_0,q_0)\) is retained explicitly because these variables define the state-space coordinates for release-condition analysis.

A successful transition is defined by entry into, and retention within, a near-vertical low-rate attitude window. Let \(\theta_v=90^{\circ}\) denote the target working attitude under the pitch convention defined in Section 3.2. Within a prescribed evaluation horizon \(T_{\mathrm{eval}}\), the transition is considered successful if there exists a time \(t_s\) such that \(t_s+\tau_{\mathrm{hold}}\leq t_0+T_{\mathrm{eval}}\) and, for the hold duration \(\tau_{\mathrm{hold}}\),

$$
|\operatorname{wrap}_{180}(\theta(t)-\theta_v)|\leq \Delta_{\theta},
\qquad
|q(t)|\leq \Delta_q,
\qquad
\forall t\in [t_s,t_s+\tau_{\mathrm{hold}}].
$$

Here \(\Delta_{\theta}\), \(\Delta_q\), \(\tau_{\mathrm{hold}}\), and \(T_{\mathrm{eval}}\) are prescribed by mission tolerance, measurement resolution/noise, and the validation protocol; they are not fitted dynamic parameters. The operator \(\operatorname{wrap}_{180}(\cdot)\) returns the signed shortest angular difference in degrees. Angular thresholds are reported in degrees for readability, whereas trigonometric evaluations in the dynamic model use consistent angular units. In the simulation model, \(\theta\) is treated as an unwrapped planar pitch angle unless otherwise stated. The hold requirement prevents a transient pass through the vertical attitude from being classified as a stable working condition. Cases that do not satisfy the attitude, rate, and hold requirements within \(T_{\mathrm{eval}}\) are classified as unsuccessful for validation and feasible release-envelope analysis.

## 3.2 Coordinates, states, and reduced-order assumptions

The notation follows the standard marine-vehicle convention, with the six-degree-of-freedom position and velocity vectors denoted by

$$
\eta_6=[x,y,z,\phi,\theta,\psi]^T,
\qquad
\nu_6=[u,v,w,p,q,r]^T.
$$

The present model, however, is not developed as a full six-degree-of-freedom Fossen model. The transition behavior targeted in this study is represented in the longitudinal plane, where the dominant variables are surge velocity, heave velocity, pitch rate, and planar pitch attitude. The reduced velocity state is therefore

$$
\nu=[u,w,q]^T,
$$

with resultant speed, angle of attack, and dynamic pressure defined as

$$
V=\sqrt{u^2+w^2},
\qquad
\alpha=\operatorname{atan2}(w,u),
\qquad
Q=\frac{1}{2}\rho V^2 .
$$

This three-degree-of-freedom reduction is a compromise between a single-pitch model and an under-constrained full six-degree-of-freedom model. A single-pitch model cannot describe the evolution of release velocity, angle of attack, and translational hydrodynamic loads, which are central to the large-angle transition of the open-mouth structure. Conversely, a full model would introduce sway, roll, yaw, and additional cross-coupling coefficients that are not reliably identifiable from the available transition-stage evidence and are not the primary mechanisms represented in the present formulation. The longitudinal reduction is treated as a model assumption to be checked against three-axis IMU records, with roll and yaw amplitudes or energy reported separately from pitch in the validation evidence.

All translational velocities and hydrodynamic loads in the reduced model are expressed in the body-fixed longitudinal frame. The body-frame origin is located at the platform center of gravity (CG). The body \(x\)-axis is aligned with the platform axis, and the body \(z\)-axis is positive downward in the longitudinal plane, as shown schematically in Fig. [TBD]. Positive pitch angle and pitch rate rotate the positive body \(x\)-axis toward the positive body \(z\)-axis. Positive force components act along the positive body \(x\) and \(z\) axes, and positive moment acts in the positive pitch direction. The angle of attack \(\alpha=\operatorname{atan2}(w,u)\) follows the same body-axis convention, and the CFD coefficients \(C_X\), \(C_Z\), and \(C_m\) are tabulated with these force and moment signs. The center of buoyancy (CB) location relative to the CG is denoted by \((x_b,z_b)\). The reference schematic should show the inertial downward direction, the body axes, the positive \(\theta/q\) convention, and the CG-to-CB offset used in the restoring moment. Because \(\theta\) is used here only as a planar pitch attitude, the neighborhood of \(90^{\circ}\) is not an Euler-angle singularity of a full six-degree-of-freedom transformation.

At very low speed, \(Q\rightarrow 0\) and the CFD hydrodynamic load vanishes. For numerical evaluation, a small speed threshold \(V_{\varepsilon}\) may be used only to avoid an undefined \(\operatorname{atan2}(0,0)\): when \(V<V_{\varepsilon}\), \(Q\) is set to zero so that \(f_{\mathrm{CFD}}=0\), while \(\alpha\) is assigned a bounded interpolation value with no effect on the load. This convention is not a fitted physical parameter.

## 3.3 Hybrid physical partition

The model is built on the marine-vehicle dynamics framework, but it is not a full constant-coefficient hydrodynamic model. Instead, the transition-stage dynamics are partitioned into four physically distinct blocks. The analytical skeleton retains the rigid-platform inertia, selected body-frame inertial coupling, gravity--buoyancy restoring, and axial input channel. Permeability-corrected added inertia represents partial internal-water participation in the open-mouth fairing. CFD-derived AoA coefficient maps close the large-angle translational loads and the corresponding attitude-dependent pitching moment. Free-decay-informed rotational damping represents the remaining pitch-rate-dependent energy loss.

### 3.3.1 Analytical skeleton

The analytical part of the model keeps only the structures directly constrained by the chosen coordinates and by measured or geometric quantities: platform mass \(m\), dry pitch inertia \(I_{yy}\), surge--heave body-frame inertial coupling, gravity--buoyancy restoring, and an axial thrust input \(T\). Here \(m\) denotes the rigid-platform mass excluding the modeled internal-water participation. The same mass convention is used in the weight \(W=mg\). Internal-water participation in the open-mouth fairing is not folded into \(m\); it is introduced separately through the permeability-corrected added-inertia terms below.

Only the surge--heave body-frame inertial coupling required by the longitudinal translational dynamics is retained analytically. The translation-induced pitching moment is not represented by a separate analytical \(uw\)-type term in this skeleton, because that physical role is assigned to the CFD-derived pitching-moment map \(C_m(\alpha)\) under the non-overlapping closure partition described in Section 3.5. The thrust term \(T=T(t)\) is retained as a prescribed input along the body \(x\)-axis. The passive-transition case is obtained by setting this input to zero, but this is a case definition rather than a change in model structure. Off-axis thrust moments are neglected because the thruster axis is assumed to coincide with the body axis.

### 3.3.2 Permeability-corrected added inertia

The open-mouth fairing makes the internal water neither completely detached from the platform motion nor fully entrapped as a rigid internal mass. To represent this bounded internal-water participation without introducing an unconstrained fluid--structure interaction model, three directional participation factors are introduced:

$$
\boldsymbol{\mu}=[\mu_x,\mu_z,\mu_\theta]^T,
\qquad
0\leq \mu_x,\mu_z,\mu_\theta\leq 1.
$$

The transition-stage effective added-inertia derivatives are defined as

$$
X_{\dot u}^{t}=X_{\dot u}^{out}-\mu_x m_w,
$$

$$
Z_{\dot w}^{t}=Z_{\dot w}^{out}-\mu_z m_w,
$$

$$
M_{\dot q}^{t}=M_{\dot q}^{out}-\mu_\theta I_w,
$$

where \(X_{\dot u}^{out}\), \(Z_{\dot w}^{out}\), and \(M_{\dot q}^{out}\) are outer-body added-inertia estimates. The quantities \(m_w\) and \(I_w\) correspond to the representative mass and pitch inertia of the internal-water region associated with the open-mouth fairing. Their numerical closure is a Section 4 matter. The superscript \(t\) denotes the transition-stage effective value.

The limits \(\mu=0\) and \(\mu=1\) represent, respectively, no additional internal-water co-acceleration and complete internal-water participation in the corresponding direction. The open-mouth platform is expected to lie between these limits, with possible anisotropy among surge, heave, and pitch rotation. The factors \(\mu_x\), \(\mu_z\), and \(\mu_\theta\) should therefore be interpreted as bounded internal-water participation factors, not as Darcy permeability coefficients or arbitrary tuning gains.

The effective inertia matrix used by the reduced model is

$$
M_{\mathrm{eff}}
=
\operatorname{diag}
\left(
 m-X_{\dot u}^{t},
 m-Z_{\dot w}^{t},
 I_{yy}-M_{\dot q}^{t}
\right).
$$

The derivatives \(X_{\dot u}^{t}\), \(Z_{\dot w}^{t}\), and \(M_{\dot q}^{t}\) follow the standard marine-vehicle added-inertia sign convention. Thus, negative added-inertia derivatives increase the corresponding diagonal entries of \(M_{\mathrm{eff}}\).

### 3.3.3 CFD-derived AoA coefficient maps

The transition involves large changes in angle of attack, so constant small-perturbation hydrodynamic derivatives are not sufficient to represent the open-mouth loads over the full maneuver. The translational hydrodynamic loads and the attitude-dependent pitching moment are therefore closed by CFD-derived AoA coefficient maps:

$$
f_{\mathrm{CFD}}(\alpha,V)
=
QA
\begin{bmatrix}
C_X(\alpha)\\
C_Z(\alpha)\\
L C_m(\alpha)
\end{bmatrix},
$$

where \(A\) and \(L\) are the reference area and length, and \(C_X(\alpha)\), \(C_Z(\alpha)\), and \(C_m(\alpha)\) are the axial-force, normal-force, and pitching-moment coefficient maps. The coefficients are expressed in the reduced-model body axes. The coefficient \(C_m(\alpha)\) is defined about the CG/body-frame origin used in the reduced dynamic equations; CFD moments reported about any other point must be transferred to this reference point before constructing the coefficient map. It is used as the net quasi-static pitching-moment closure associated with translation at a prescribed angle of attack and is not a pitch-rate damping coefficient.

This representation replaces simple surge and heave quadratic-damping descriptions with a dynamic-pressure-scaled quasi-static hydrodynamic closure over the relevant AoA range. The CFD-derived AoA coefficient maps represent hydrodynamic loads scaled by \(Q\), not hydrostatic forces. Because the static CFD sweeps correspond to prescribed AoA conditions with no pitch rotation, they do not identify pure rotational damping. Pitch-rate-dependent dissipation is therefore assigned to a separate rotational damping term.

### 3.3.4 Free-decay rotational damping

The remaining pitch-rate-dependent rotational dissipation is modeled as

$$
d_m(q)=\left(d_q+d_{q|q|}|q|\right)q,
$$

where \(d_q\) and \(d_{q|q|}\) are the linear and quadratic rotational damping coefficients. In the governing equations, this damping contribution is placed on the left-hand side, so it appears as a resisting moment in the scalar pitch equation. Section 3 only defines the model structure and the physical role of this term. The free-decay test configuration, preprocessing, identification protocol, and uncertainty treatment are described in Section 4. Any compensation term associated with the free-decay experimental setup is not part of the untethered release model.

## 3.4 Governing equations

Combining the preceding components gives the transition-stage hybrid reduced-order model

$$
M_{\mathrm{eff}}\dot{\nu}
+
c_{\mathrm{hyb}}(\nu)
+
g(\theta)
+
d(q)
=
f_{\mathrm{CFD}}(\alpha,V)
+
\tau_T,
$$

where all force and moment components are expressed in the body-fixed longitudinal frame, except that the restoring vector depends on the attitude \(\theta\). The retained body-frame inertial coupling is

$$
c_{\mathrm{hyb}}(\nu)
=
\begin{bmatrix}
(m-Z_{\dot w}^{t})wq\\
-(m-X_{\dot u}^{t})uq\\
0
\end{bmatrix}.
$$

The zero third component is deliberate: the analytical pitch-row \(uw\)-type coupling is not included because the translation-induced pitching moment is assigned to the CFD moment closure \(QALC_m(\alpha)\). The restoring, damping, hydrodynamic, and thrust vectors are defined below. Here \(W=mg\) and \(B=\rho g\nabla\) denote the weight and buoyancy forces, respectively, both expressed in newtons. The mass \(m\) in \(W\) is the rigid-platform mass excluding the modeled internal-water participation. If buoyancy is reported as a buoyancy-equivalent mass in a parameter table, it is converted to force before being used in \(g(\theta)\).

$$
g(\theta)
=
\begin{bmatrix}
(W-B)\sin\theta\\
-(W-B)\cos\theta\\
-B(z_b\sin\theta+x_b\cos\theta)
\end{bmatrix},
$$

With the CG as the body-frame origin, the moment component follows from the moment of the buoyancy force acting at \((x_b,z_b)\) under the positive pitch convention defined in Fig. [TBD]; the weight force acts through the origin and therefore contributes no pitch moment. The same schematic should show the inertial vertical direction and the sign of \((x_b,z_b)\) to make the restoring convention explicit.

$$
d(q)
=
\begin{bmatrix}
0\\
0\\
(d_q+d_{q|q|}|q|)q
\end{bmatrix},
$$

$$
f_{\mathrm{CFD}}(\alpha,V)
=
QA
\begin{bmatrix}
C_X(\alpha)\\
C_Z(\alpha)\\
L C_m(\alpha)
\end{bmatrix},
\qquad
\tau_T=
\begin{bmatrix}
T\\0\\0
\end{bmatrix}.
$$

Here \(T\) may be prescribed as a time history. The corresponding scalar equations are

$$
(m-X_{\dot u}^{t})\dot u
=
-(m-Z_{\dot w}^{t})wq
+QAC_X(\alpha)
-(W-B)\sin\theta
+T,
$$

$$
(m-Z_{\dot w}^{t})\dot w
=
(m-X_{\dot u}^{t})uq
+QAC_Z(\alpha)
+(W-B)\cos\theta,
$$

$$
(I_{yy}-M_{\dot q}^{t})\dot q
=
QALC_m(\alpha)
-\left(d_q+d_{q|q|}|q|\right)q
+B(z_b\sin\theta+x_b\cos\theta).
$$

The planar pitch kinematics are

$$
\dot{\theta}=q.
$$

These equations form the reduced-order state model for the transition stage. The analytical coupling terms retain the surge--heave interaction generated by expressing translation in the body frame. The restoring vector represents the measured weight--buoyancy balance and CG--CB geometry. The CFD-derived AoA coefficient maps provide the quasi-static large-angle hydrodynamic closure, and the damping term accounts for pitch-rate-dependent energy loss.

## 3.5 Model-scope remarks and anti-double-counting treatment

The proposed formulation is hybrid by design. It should not be interpreted as a conventional constant-coefficient diagonal-damping model, nor as a purely empirical curve fit. Each model block has a distinct physical role and source: rigid-platform inertia, measured weight--buoyancy balance, and CG--CB geometry define the analytical skeleton; permeability-corrected added inertia represents open-mouth internal-water participation; CFD-derived AoA coefficient maps close quasi-static large-angle hydrodynamic loads; and free-decay data close the pitch rotational damping. This partition also supports later model-block ablations, because each physical block can be removed or replaced in a controlled manner.

A key modeling choice is the treatment of the analytical Munk-type pitching term associated with translational added inertia. In a constant-coefficient model, a pitch moment proportional to

$$
\left(X_{\dot u}^{t}-Z_{\dot w}^{t}\right)uw
$$

may appear in the pitch equation. In the present hybrid partition, \(QALC_m(\alpha)\) is used as the net quasi-static translation-induced pitching-moment closure about the CG/body-frame origin. Therefore, an additional analytical \((X_{\dot u}^{t}-Z_{\dot w}^{t})uw\) pitch moment is not superposed in this reduced model unless a separate non-overlapping decomposition is introduced. This treatment avoids assigning overlapping quasi-static pitch-moment content to two closures, without claiming that static AoA CFD identifies all acceleration-dependent added-inertia effects.

The model scope is restricted to the fairing-attached, reflector-uninflated transition stage. It does not describe fairing separation, reflector inflation, post-deployment depth control, environmental waves or currents, or full fluid--structure interaction of the inflated reflector. These exclusions keep the formulation focused on the dominant passive reorientation mechanism while retaining the axial input channel needed for later passive-baseline or assisted-transition modeling choices.
