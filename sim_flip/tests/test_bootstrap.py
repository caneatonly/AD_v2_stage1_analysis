import unittest

import numpy as np
import pandas as pd

from sim_flip.analysis.bootstrap import bootstrap_fit_segments


class TestBootstrap(unittest.TestCase):
    def test_bootstrap_outputs_ci_corr_and_samples(self) -> None:
        t = np.arange(0.0, 5.0, 0.01)
        theta = np.sin(2 * np.pi * 0.5 * t)
        q = np.gradient(theta, t)
        qd = np.gradient(q, t)
        seg = pd.DataFrame(
            {
                "theta_rad": theta,
                "q_rad_s": q,
                "q_dot_rad_s2": qd,
            }
        )

        def fit_fn(samples: list[pd.DataFrame]) -> dict[str, float]:
            x = samples[0]
            return {
                "mu_theta": float(np.mean(np.abs(x["theta_rad"]))),
                "d_q": float(np.mean(np.abs(x["q_rad_s"]))),
                "d_qq": float(np.mean(np.abs(x["q_dot_rad_s2"]))),
                "K_cable": float(np.std(x["theta_rad"])),
            }

        out = bootstrap_fit_segments(
            segment_dfs=[seg],
            fit_fn=fit_fn,
            n_boot=20,
            block_s=0.2,
            dt_s=0.01,
            seed=0,
        )

        self.assertEqual(out.n_total, 20)
        self.assertGreater(out.n_success, 0)
        self.assertIn("mu_theta", out.params_mean)
        self.assertIn("d_q", out.params_ci95)
        self.assertIn("mu_theta", out.params_corr)
        self.assertGreater(len(out.samples), 0)


if __name__ == "__main__":
    unittest.main()

