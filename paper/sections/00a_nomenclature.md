# Nomenclature

<!-- Placement: After frontmatter (Abstract/Keywords/Highlights), before Introduction.
     This follows the standard Ocean Engineering convention (90% of OE papers).
     Symbols are organized by category; Greek symbols follow Latin symbols.
     Source of truth for values: sim_flip/configs/params_nominal.yaml
     Source of truth for conventions: sim_flip/src/conventions.py -->

## Latin Symbols

| Symbol | Description | Unit |
|--------|-------------|------|
| $A_{ref}$ | Reference area for hydrodynamic coefficients | m² |
| $B$ | Buoyancy force magnitude, $B = B_{mass} g$ | N |
| $B_{mass}$ | Buoyancy equivalent mass | kg |
| $BG$ | Distance between center of buoyancy and center of gravity | m |
| $C_m$ | Pitching moment coefficient about $+y_b$ ($C_m > 0$ = nose-up) | — |
| $C_X$ | Force coefficient along $+x_b$ | — |
| $C_Z$ | Force coefficient along $+z_b$ | — |
| $C_{cable,q}$ | Cable damping coefficient | N·m·s/rad |
| $d_q$ | Linear rotational damping coefficient | N·m·s/rad |
| $d_{q,abs}$ | Quadratic rotational damping coefficient | N·m·s²/rad² |
| $g$ | Gravitational acceleration | m/s² |
| $I_y$ | Effective pitch inertia (ODE denominator) | kg·m² |
| $I_{yy}$ | Rigid-body pitch moment of inertia | kg·m² |
| $I_{water,inner}$ | Inner-water equivalent pitch inertia | kg·m² |
| $K_{cable}$ | Cable torsional spring stiffness | N·m/rad |
| $L_{ref}$ | Reference length for moment coefficients | m |
| $m_{dry}$ | Dry mass of rigid body | kg |
| $m_{water,inner}$ | Inner-water equivalent mass | kg |
| $m_{wet}$ | Wet mass (dry mass + inner water) | kg |
| $m_x$ | Effective surge inertia (ODE denominator) | kg |
| $m_z$ | Effective heave inertia (ODE denominator) | kg |
| $M_{bg}$ | Buoyancy restoring moment | N·m |
| $M_{cable}$ | Cable restoring moment | N·m |
| $M_{cfd}$ | CFD-derived pitching moment | N·m |
| $M_{damp}$ | Rotational damping moment | N·m |
| $q$ | Pitch rate, $q = \dot{\theta}$ | rad/s |
| $Q$ | Dynamic pressure, $Q = \frac{1}{2}\rho V^2$ | Pa |
| $T$ | Thrust force along $x_b$ | N |
| $u$ | Surge velocity along $+x_b$ | m/s |
| $V$ | Total velocity magnitude, $V = \sqrt{u^2 + w^2}$ | m/s |
| $w$ | Heave velocity along $+z_b$ | m/s |
| $W$ | Weight magnitude, $W = m_{dry} g$ | N |
| $x_b$ | Buoyancy center offset, forward | m |
| $X_{cfd}$ | CFD-derived surge force | N |
| $X_{\dot{u},total}$ | Total surge added-mass derivative | kg |
| $z_b$ | Buoyancy center offset, downward | m |
| $Z_{cfd}$ | CFD-derived heave force | N |
| $Z_{\dot{w},total}$ | Total heave added-mass derivative | kg |

## Greek Symbols

| Symbol | Description | Unit |
|--------|-------------|------|
| $\alpha$ | Angle of attack, $\alpha = \mathrm{atan2}(w, u)$ | deg |
| $\theta$ | Pitch angle (nose-up positive) | rad |
| $\theta_0$ | Initial pitch angle at segment start | deg |
| $\theta_{eq}$ | Cable equilibrium angle | rad |
| $\mu_x$ | Surge permeability coupling parameter | — |
| $\mu_z$ | Heave permeability coupling parameter | — |
| $\mu_\theta$ | Pitch permeability coupling parameter | — |
| $\rho$ | Water density | kg/m³ |

## Subscripts and Superscripts

| Notation | Description |
|----------|-------------|
| $(\cdot)_b$ | Body-frame quantity |
| $(\cdot)_{inner}$ | Inner-water contribution |
| $(\cdot)_{outer}$ | Outer-fluid (external) contribution |
| $(\cdot)_{total}$ | Combined outer + permeability-corrected inner |

## Abbreviations

| Abbreviation | Description |
|--------------|-------------|
| AoA | Angle of attack |
| CFD | Computational fluid dynamics |
| CI | Confidence interval |
| DOF | Degree of freedom |
| MAE | Mean absolute error |
| RMSE | Root mean square error |
| TAAE | Time-averaged absolute error |
| TASE | Time-averaged signed error |
