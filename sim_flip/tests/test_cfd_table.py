import unittest

from sim_flip.src.cfd_table import load_cfd_table, CfdInterpolator, default_cfd_table_path


class TestCfdTable(unittest.TestCase):
    def test_grid_points_recover_exact(self):
        table = load_cfd_table(default_cfd_table_path())
        itp_pchip = CfdInterpolator.from_table(table, method="pchip")
        itp_lin = CfdInterpolator.from_table(table, method="linear")

        for a, cx, cz, cm in zip(table.alpha_deg, table.Cx, table.Cz, table.Cm):
            got_p = itp_pchip.coeffs(a)
            got_l = itp_lin.coeffs(a)
            self.assertAlmostEqual(got_p[0], cx, places=12)
            self.assertAlmostEqual(got_p[1], cz, places=12)
            self.assertAlmostEqual(got_p[2], cm, places=12)
            self.assertAlmostEqual(got_l[0], cx, places=12)
            self.assertAlmostEqual(got_l[1], cz, places=12)
            self.assertAlmostEqual(got_l[2], cm, places=12)

    def test_clamp_no_extrapolation(self):
        table = load_cfd_table(default_cfd_table_path())
        itp = CfdInterpolator.from_table(table, method="pchip")

        c_lo = itp.coeffs(table.alpha_deg[0] - 123.0)
        c_hi = itp.coeffs(table.alpha_deg[-1] + 123.0)
        self.assertEqual(c_lo, itp.coeffs(table.alpha_deg[0]))
        self.assertEqual(c_hi, itp.coeffs(table.alpha_deg[-1]))

    def test_pchip_no_overshoot_on_monotone_intervals(self):
        """If y is monotone on an interval [i,i+1], PCHIP should stay within endpoint bounds."""

        table = load_cfd_table(default_cfd_table_path())
        itp = CfdInterpolator.from_table(table, method="pchip")

        def check_series(values, eval_fn):
            eps = 1e-12
            for i in range(len(table.alpha_deg) - 1):
                a0 = table.alpha_deg[i]
                a1 = table.alpha_deg[i + 1]
                y0 = values[i]
                y1 = values[i + 1]
                dy = y1 - y0
                if dy == 0.0:
                    continue
                # Only enforce on strictly monotone intervals.
                lo = min(y0, y1) - eps
                hi = max(y0, y1) + eps
                for frac in (0.25, 0.5, 0.75):
                    aq = a0 + frac * (a1 - a0)
                    yq = eval_fn(aq)
                    self.assertGreaterEqual(yq, lo)
                    self.assertLessEqual(yq, hi)

        check_series(table.Cx, lambda a: itp.coeffs(a)[0])
        check_series(table.Cz, lambda a: itp.coeffs(a)[1])
        check_series(table.Cm, lambda a: itp.coeffs(a)[2])

    def test_linear_midpoint_matches(self):
        table = load_cfd_table(default_cfd_table_path())
        itp = CfdInterpolator.from_table(table, method="linear")

        for i in range(len(table.alpha_deg) - 1):
            a0 = table.alpha_deg[i]
            a1 = table.alpha_deg[i + 1]
            amid = 0.5 * (a0 + a1)

            for series_idx, values in enumerate((table.Cx, table.Cz, table.Cm)):
                y0 = values[i]
                y1 = values[i + 1]
                y_expect = 0.5 * (y0 + y1)
                y_got = itp.coeffs(amid)[series_idx]
                self.assertAlmostEqual(y_got, y_expect, places=12)

    def test_rule_b_alpha_over_90(self):
        """alpha=110 should fold to 70 with your specified sign rules."""

        table = load_cfd_table(default_cfd_table_path())
        itp = CfdInterpolator.from_table(table, method="pchip")

        # Reference at 70 deg (exists in table).
        cx70, cz70, cm70 = itp.coeffs(70.0)
        # Rule: alpha>90 => Cx flips, Cz unchanged, Cm flips; fold 110->70.
        cx, cz, cm = itp.coeffs_extended(110.0)
        self.assertAlmostEqual(cx, -cx70, places=12)
        self.assertAlmostEqual(cz, cz70, places=12)
        self.assertAlmostEqual(cm, -cm70, places=12)

    def test_rule_b_negative_alpha(self):
        """alpha=-20 uses abs(alpha)=20 for lookup, with Cz and Cm sign flips."""

        table = load_cfd_table(default_cfd_table_path())
        itp = CfdInterpolator.from_table(table, method="pchip")

        cx20, cz20, cm20 = itp.coeffs(20.0)
        cx, cz, cm = itp.coeffs_extended(-20.0)
        # Rule: alpha<0 => Cz flips, Cm flips, Cx unchanged.
        self.assertAlmostEqual(cx, cx20, places=12)
        self.assertAlmostEqual(cz, -cz20, places=12)
        self.assertAlmostEqual(cm, -cm20, places=12)

    def test_rule_b_negative_and_over_90(self):
        """alpha=-110 => abs->110 then fold->70; combined sign rules apply."""

        table = load_cfd_table(default_cfd_table_path())
        itp = CfdInterpolator.from_table(table, method="pchip")

        cx70, cz70, cm70 = itp.coeffs(70.0)
        cx, cz, cm = itp.coeffs_extended(-110.0)
        # alpha<0: Cz-, Cm-; alpha>90: Cx-, Cm- (so Cm becomes + overall).
        self.assertAlmostEqual(cx, -cx70, places=12)
        self.assertAlmostEqual(cz, -cz70, places=12)
        self.assertAlmostEqual(cm, cm70, places=12)


if __name__ == "__main__":
    unittest.main()
