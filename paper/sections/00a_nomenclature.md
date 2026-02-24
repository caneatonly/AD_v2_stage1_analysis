# Nomenclature

<!-- Placement in final manuscript: after front matter and before the Introduction. -->

## Latin Symbols

| Symbol | Description | Unit |
|--------|-------------|------|
| \(A_{ref}\) | Reference area used in hydrodynamic coefficient scaling | m² |
| \(B\) | Buoyancy force magnitude, \(B=B_{mass}g\) | N |
| \(B_{mass}\) | Buoyancy equivalent mass | kg |
| \(BG\) | Distance between center of buoyancy and center of gravity | m |
| \(C_m\) | Pitch-moment coefficient about \(+y_b\) (positive nose-up) | -- |
| \(C_X\) | Force coefficient along \(+x_b\) | -- |
| \(C_Z\) | Force coefficient along \(+z_b\) | -- |
| \(C_{cable,q}\) | Cable damping coefficient in pitch | N·m·s/rad |
| \(d_q\) | Linear rotational damping coefficient | N·m·s/rad |
| \(d_{q,abs}\) | Quadratic rotational damping coefficient | N·m·s²/rad² |
| \(g\) | Gravitational acceleration | m/s² |
| \(I_y\) | Effective pitch inertia used in the ODE denominator | kg·m² |
| \(I_{yy}\) | Rigid-body pitch moment of inertia | kg·m² |
| \(I_{water,inner}\) | Inner-water equivalent pitch inertia | kg·m² |
| \(K_{cable}\) | Cable torsional stiffness | N·m/rad |
| \(L_{ref}\) | Reference length used in moment coefficient scaling | m |
| \(m_{dry}\) | Dry mass of the rigid platform body | kg |
| \(m_{water,inner}\) | Inner-water equivalent mass | kg |
| \(m_{wet}\) | Wet mass (dry mass plus inner-water equivalent mass) | kg |
| \(m_x\) | Effective surge inertia used in the ODE denominator | kg |
| \(m_z\) | Effective heave inertia used in the ODE denominator | kg |
| \(M_{bg}\) | Buoyancy restoring moment | N·m |
| \(M_{cable}\) | Cable-induced restoring/damping moment | N·m |
| \(M_{cfd}\) | CFD-derived hydrodynamic pitching moment | N·m |
| \(M_{damp}\) | Rotational damping moment | N·m |
| \(M_{thruster}\) | Thruster-induced pitching moment | N·m |
| \(q\) | Pitch rate, \(q=\dot{\theta}\) | rad/s |
| \(Q\) | Dynamic pressure, \(Q=\tfrac{1}{2}\rho V^2\) | Pa |
| \(T\) | Thrust force along \(+x_b\) | N |
| \(u\) | Surge velocity along \(+x_b\) | m/s |
| \(V\) | Body-frame speed magnitude, \(V=\sqrt{u^2+w^2}\) | m/s |
| \(w\) | Heave velocity along \(+z_b\) | m/s |
| \(W\) | Weight magnitude, \(W=m_{dry}g\) | N |
| \(x_b\) | Forward offset of buoyancy center in the body frame | m |
| \(X_{cfd}\) | CFD-derived hydrodynamic force along \(+x_b\) | N |
| \(X_{\dot{u},total}\) | Total surge added-mass derivative | kg |
| \(X_{cp}\) | Pressure-center location used for mechanism interpretation | m |
| \(z_b\) | Downward offset of buoyancy center in the body frame | m |
| \(Z_{cfd}\) | CFD-derived hydrodynamic force along \(+z_b\) | N |
| \(Z_{\dot{w},total}\) | Total heave added-mass derivative | kg |

## Greek Symbols

| Symbol | Description | Unit |
|--------|-------------|------|
| \(\alpha\) | Angle of attack, \(\alpha=\mathrm{atan2}(w,u)\) | deg |
| \(\eta\) | Configuration-state vector, \(\eta=[\theta]\) | -- |
| \(\theta\) | Pitch angle (positive nose-up) | rad |
| \(\theta_0\) | Initial pitch angle of a free-decay segment | deg |
| \(\theta_{eq}\) | Cable equilibrium angle | rad |
| \(\mu_x\) | Surge-channel permeability coupling parameter | -- |
| \(\mu_z\) | Heave-channel permeability coupling parameter | -- |
| \(\mu_{\theta}\) | Pitch-channel permeability coupling parameter | -- |
| \(\nu\) | Velocity-state vector, \(\nu=[u,w,q]^T\) | -- |
| \(\rho\) | Water density | kg/m³ |

## Subscripts and Superscripts

| Notation | Description |
|----------|-------------|
| \((\cdot)_b\) | Quantity expressed in the body frame |
| \((\cdot)_{inner}\) | Inner-water contribution |
| \((\cdot)_{outer}\) | Outer-fluid contribution |
| \((\cdot)_{total}\) | Combined outer-fluid and permeability-corrected inner contribution |
| \((\cdot)_0\) | Initial value at segment start |

## Abbreviations

| Abbreviation | Description |
|--------------|-------------|
| AoA | Angle of attack |
| CFD | Computational fluid dynamics |
| CI | Confidence interval |
| CV | Cross-validation |
| DOF | Degree of freedom |
| MAE | Mean absolute error |
| NNLS | Non-negative least squares |
| RMSE | Root mean square error |
| TAAE | Time-averaged absolute error |
| TASE | Time-averaged signed error |
