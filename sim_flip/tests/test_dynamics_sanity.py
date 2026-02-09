import unittest

from sim_flip.src.added_mass import compute_from_param_tree
from sim_flip.src.cfd_table import load_default_cfd_interpolator
from sim_flip.src.dynamics import build_rhs, evaluate_diagnostics


class TestDynamicsSanity(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {
            "constants": {"rho": 1000.0, "g": 9.81, "A_ref": 0.0056745, "L_ref": 0.625},
            "rigid_body": {"m_dry": 2.55, "m_water_inner": 0.21, "m_wet": 2.76, "Iyy": 0.05741, "I_water_inner": 0.01119},
            "added_mass_outer": {"X_udot_outer": -0.0806, "Z_wdot_outer": -3.54, "M_qdot_outer": -0.115},
            "buoyancy_restore": {"B_mass": 2.76, "x_b": 0.0, "z_b": 0.0},
            "permeability": {"mu_x": 0.0, "mu_z": 0.0, "mu_theta": 0.0},
            "damping": {"d_q": 0.0, "d_q_abs": 0.0},
            "cable": {"enabled": False, "K_cable": 0.0, "theta_eq_deg": 0.0},
            "numerics": {"V_eps": 0.01, "alpha_hold_when_V_small": False},
        }
        self.cfd = load_default_cfd_interpolator()

    def test_forward_drag_decelerates(self) -> None:
        diag = evaluate_diagnostics(
            t=0.0,
            y=[1.0, 0.0, 0.0, 0.0],
            params=self.params,
            cfd=self.cfd,
        )
        self.assertLess(diag.X_cfd, 0.0)

    def test_vertical_motion_cfd_opposes(self) -> None:
        diag = evaluate_diagnostics(
            t=0.0,
            y=[0.0, 1.0, 0.0, 0.0],
            params=self.params,
            cfd=self.cfd,
        )
        self.assertLess(diag.Z_cfd, 0.0)

    def test_zero_speed_has_zero_dynamic_pressure(self) -> None:
        diag = evaluate_diagnostics(
            t=0.0,
            y=[0.0, 0.0, 0.0, 0.0],
            params=self.params,
            cfd=self.cfd,
        )
        self.assertEqual(diag.Q, 0.0)
        self.assertEqual(diag.X_cfd, 0.0)
        self.assertEqual(diag.Z_cfd, 0.0)
        self.assertEqual(diag.M_cfd, 0.0)

    def test_inertial_coupling_uses_eff_masses(self) -> None:
        params = {
            **self.params,
            "constants": {**self.params["constants"], "A_ref": 0.0},  # remove CFD force coupling
            "buoyancy_restore": {"B_mass": self.params["rigid_body"]["m_dry"], "x_b": 0.0, "z_b": 0.0},
            "permeability": {"mu_x": 1.0, "mu_z": 0.0, "mu_theta": 0.0},
        }
        rhs = build_rhs(params=params, cfd=self.cfd)
        # pick nonzero u,w,q so coupling terms are active
        u, w, q, theta = 0.8, -0.5, 0.7, 0.0
        u_dot, w_dot, _, _ = rhs(0.0, [u, w, q, theta])

        _, eff = compute_from_param_tree(params)
        exp_u_dot = -(eff.m_z * w * q) / eff.m_x
        exp_w_dot = +(eff.m_x * u * q) / eff.m_z
        self.assertAlmostEqual(u_dot, exp_u_dot, places=12)
        self.assertAlmostEqual(w_dot, exp_w_dot, places=12)

    def test_cable_damping_term_affects_q_dot(self) -> None:
        params = {
            **self.params,
            "constants": {**self.params["constants"], "A_ref": 0.0},
            "buoyancy_restore": {"B_mass": self.params["rigid_body"]["m_dry"], "x_b": 0.0, "z_b": 0.0},
            "cable": {
                "enabled": True,
                "K_cable": 0.0,
                "C_cable_q": 2.5,
                "theta_eq_deg": 0.0,
            },
        }
        rhs = build_rhs(params=params, cfd=self.cfd)
        u, w, q, theta = 0.0, 0.0, 0.6, 0.0
        _, _, q_dot, _ = rhs(0.0, [u, w, q, theta])
        _, eff = compute_from_param_tree(params)
        expected = -(2.5 * q) / eff.I_y
        self.assertAlmostEqual(q_dot, expected, places=12)


if __name__ == "__main__":
    unittest.main()
