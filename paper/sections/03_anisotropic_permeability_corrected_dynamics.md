# 3. Anisotropic Permeability-Corrected Dynamics

## 3.1 Governing Equations and State Definition

This section introduces the 3-DOF governing model for the launch-to-stabilization phase, with frozen interfaces `nu = [u, w, q]^T` and `eta = [theta]`.

## 3.2 Permeability-Corrected Added-Mass Terms

Directional coupling parameters `mu_x`, `mu_z`, and `mu_theta` are used to modify the effective added-mass and added-inertia terms according to the parameter contract.

## 3.3 Restoring, Damping, and Actuation Terms

Fossen-style rigid-body inertia, Coriolis coupling, restoring effects, and rotational damping are retained according to the equation governance in the integrated plan.

## 3.4 Model Interfaces and Backward Compatibility

The implementation mapping to `sim_flip/src/dynamics.py` and config interfaces is documented here, including assumptions required for reproducibility.

[Fig. 4. Model-term map from physical mechanisms to equation blocks and code interfaces. Insert here.]
