import unittest

from sim_flip.src.added_mass import compute_added_mass_totals, compute_effective_inertia


class TestAddedMass(unittest.TestCase):
    def setUp(self) -> None:
        # Representative nominal values from configs/params_nominal.yaml
        self.X_udot_outer = -0.0806
        self.Z_wdot_outer = -3.54
        self.M_qdot_outer = -0.115
        self.m_water_inner = 0.21
        self.I_water_inner = 0.01119
        self.m_dry = 2.55
        self.Iyy = 0.05741

    def test_mu_bounds_positive_effective_inertia(self) -> None:
        # mu = 0
        t0 = compute_added_mass_totals(
            X_udot_outer=self.X_udot_outer,
            Z_wdot_outer=self.Z_wdot_outer,
            M_qdot_outer=self.M_qdot_outer,
            m_water_inner=self.m_water_inner,
            I_water_inner=self.I_water_inner,
            mu_x=0.0,
            mu_z=0.0,
            mu_theta=0.0,
        )
        e0 = compute_effective_inertia(m_dry=self.m_dry, Iyy=self.Iyy, totals=t0)
        self.assertGreater(e0.m_x, 0.0)
        self.assertGreater(e0.m_z, 0.0)
        self.assertGreater(e0.I_y, 0.0)

        # mu = 1
        t1 = compute_added_mass_totals(
            X_udot_outer=self.X_udot_outer,
            Z_wdot_outer=self.Z_wdot_outer,
            M_qdot_outer=self.M_qdot_outer,
            m_water_inner=self.m_water_inner,
            I_water_inner=self.I_water_inner,
            mu_x=1.0,
            mu_z=1.0,
            mu_theta=1.0,
        )
        e1 = compute_effective_inertia(m_dry=self.m_dry, Iyy=self.Iyy, totals=t1)
        self.assertGreater(e1.m_x, 0.0)
        self.assertGreater(e1.m_z, 0.0)
        self.assertGreater(e1.I_y, 0.0)

    def test_monotonicity_with_mu(self) -> None:
        # Increasing mu should make totals more negative (subtract positive quantity)
        # and therefore increase effective inertias m_x, m_z, I_y.
        t0 = compute_added_mass_totals(
            X_udot_outer=self.X_udot_outer,
            Z_wdot_outer=self.Z_wdot_outer,
            M_qdot_outer=self.M_qdot_outer,
            m_water_inner=self.m_water_inner,
            I_water_inner=self.I_water_inner,
            mu_x=0.0,
            mu_z=0.0,
            mu_theta=0.0,
        )
        e0 = compute_effective_inertia(m_dry=self.m_dry, Iyy=self.Iyy, totals=t0)

        t1 = compute_added_mass_totals(
            X_udot_outer=self.X_udot_outer,
            Z_wdot_outer=self.Z_wdot_outer,
            M_qdot_outer=self.M_qdot_outer,
            m_water_inner=self.m_water_inner,
            I_water_inner=self.I_water_inner,
            mu_x=1.0,
            mu_z=1.0,
            mu_theta=1.0,
        )
        e1 = compute_effective_inertia(m_dry=self.m_dry, Iyy=self.Iyy, totals=t1)

        self.assertLess(t1.X_udot_total, t0.X_udot_total)
        self.assertLess(t1.Z_wdot_total, t0.Z_wdot_total)
        self.assertLess(t1.M_qdot_total, t0.M_qdot_total)

        self.assertGreater(e1.m_x, e0.m_x)
        self.assertGreater(e1.m_z, e0.m_z)
        self.assertGreater(e1.I_y, e0.I_y)

    def test_mu_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_added_mass_totals(
                X_udot_outer=self.X_udot_outer,
                Z_wdot_outer=self.Z_wdot_outer,
                M_qdot_outer=self.M_qdot_outer,
                m_water_inner=self.m_water_inner,
                I_water_inner=self.I_water_inner,
                mu_x=-0.1,
                mu_z=0.0,
                mu_theta=0.0,
            )

        with self.assertRaises(ValueError):
            compute_added_mass_totals(
                X_udot_outer=self.X_udot_outer,
                Z_wdot_outer=self.Z_wdot_outer,
                M_qdot_outer=self.M_qdot_outer,
                m_water_inner=self.m_water_inner,
                I_water_inner=self.I_water_inner,
                mu_x=0.0,
                mu_z=1.1,
                mu_theta=0.0,
            )

        with self.assertRaises(ValueError):
            compute_added_mass_totals(
                X_udot_outer=self.X_udot_outer,
                Z_wdot_outer=self.Z_wdot_outer,
                M_qdot_outer=self.M_qdot_outer,
                m_water_inner=self.m_water_inner,
                I_water_inner=self.I_water_inner,
                mu_x=0.0,
                mu_z=0.0,
                mu_theta=2.0,
            )


if __name__ == "__main__":
    unittest.main()
