import unittest

from sim_flip.src.cfd_table import load_default_cfd_interpolator
from sim_flip.src.dynamics import evaluate_diagnostics


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


if __name__ == "__main__":
    unittest.main()
