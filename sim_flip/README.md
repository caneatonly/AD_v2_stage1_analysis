# sim_flip — 3-DOF forward simulation (horizontal → vertical flip)

Scope: implement and validate the **longitudinal-plane 3-DOF** model (Surge/Heave/Pitch) using **CFD static coefficient lookup** + identified damping + (optional) cable stiffness. Baseline first: **no thruster**.

## Global conventions (frozen)

### Frames and state
- Body frame: $x_b$ forward, $z_b$ **down** (right-handed with $y_b$ starboard).
- State: $u$ along $+x_b$ (m/s), $w$ along $+z_b$ (m/s), $q=\dot\theta$ about $+y_b$ (rad/s), $\theta$ pitch angle (rad).
- Pitch sign: **nose-up = positive** $\theta$.

### Angle of attack and units
- Speed: $V = \sqrt{u^2+w^2}$.
- Angle of attack (degrees) for lookup:
  $$\alpha_{deg} = \mathrm{atan2}(w, u)\cdot\frac{180}{\pi}$$
  with singularity protection when $V<V_\varepsilon$.
- CFD table uses **degrees** (`alpha_deg`).

### CFD coefficient meanings (body-axis)
- $C_x$: force coefficient along $+x_b$.
- $C_z$: force coefficient along $+z_b$ (down). (So **upward** hydrodynamic force corresponds to $C_z<0$.)
- $C_m$: pitching moment coefficient about $+y_b$.
- **Convention:** $C_m>0$ is **nose-up** moment.

### Critical modeling rule (no double-counting)
If we use CFD table moment $C_m(\alpha)$ in
$$M_{CFD}=QAL\,C_m(\alpha), \quad Q=\tfrac12\rho V^2,$$
then **do not** add a separate theoretical Munk moment term.

## Data contracts

### CFD table
`data/cfd_table_clean.csv` columns:
- `alpha_deg` (monotone increasing)
- `Cx`, `Cz`, `Cm`

### Nominal parameters
`configs/params_nominal.yaml` is the single source of truth for constants and identified parameters.

## Next steps (first milestone)
- Implement the 3-DOF ODE RHS and direction sanity tests.
- Run baseline flip scenarios S0/S1/S2 and output metrics ($t_{80}$, $\theta_{max}$, $q_{max}$).

## Quick run (baseline, no thruster)
Install dependencies from `sim_flip/requirements.txt`, then run:

```bash
python sim_flip/scripts/run_flip_baseline.py
```

Outputs:
- `sim_flip/outputs/baseline_timeseries.csv`
- `sim_flip/outputs/baseline_metrics.json`
