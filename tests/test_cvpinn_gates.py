"""Lightweight smoke and regression tests for reference & CV-RBA-PINN modules."""
import math
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import torch

import cvpinn
import heston_grid_vec as hgv
import heston_ref as hr


class TestCVPINNGates(unittest.TestCase):
    def setUp(self):
        self.p1 = cvpinn.Heston(*hr.P_FELLER_OK)
        self.p2 = cvpinn.Heston(*hr.P_FELLER_BAD)

    def test_gate2_expected_and_spot(self):
        for p in (self.p1, self.p2):
            self.assertLessEqual(cvpinn.gate2(p, cvpinn.Domain(v_min=0.0), "expected", n=512), 1e-6)
            self.assertLessEqual(cvpinn.gate2(p, cvpinn.Domain(v_min=1e-3), "spot", n=512), 1e-6)

    def test_gate3_xi_zero(self):
        for p in (self.p1, self.p2):
            self.assertEqual(cvpinn.gate3(p, cvpinn.Domain(v_min=0.0), n=512), 0.0)

    def test_greeks_bs_analytic(self):
        p = self.p2
        x = torch.tensor([-0.2, 0.0, 0.15], dtype=torch.float64)
        v = torch.tensor([0.005, 0.04, 0.16], dtype=torch.float64)
        tau = torch.tensor([0.02, 0.25, 1.5], dtype=torch.float64)
        fn = lambda a, b, c: cvpinn.bs_call(a, cvpinn.total_variance(b, c, p, "expected"))
        c, cx, gam, cxv = cvpinn.greeks(fn, x, v, tau)
        for i in range(3):
            xi, vi, ti = float(x[i]), float(v[i]), float(tau[i])
            w = hgv.w_exp_stable(vi, ti, p.kappa, p.theta)
            b_tau = (1.0 - math.exp(-p.kappa * ti)) / p.kappa
            self.assertAlmostEqual(float(c[i]), hr.bs(xi, w), places=12)
            self.assertAlmostEqual(float(gam[i]), hr.bs_gam(xi, w), places=11)
            self.assertAlmostEqual(float(cxv[i]), b_tau * hr.bs_xw(xi, w), places=11)

    def test_lbfgs_safe_rollback_on_nan(self):
        net = cvpinn.Mlp(3)
        with torch.no_grad():
            for param in net.parameters():
                param.zero_()
        # Confirm _run_lbfgs_safe preserves finite state and runs cleanly on convex loss
        target = torch.tensor([1.0], dtype=torch.float64)
        x = torch.zeros(4, dtype=torch.float64)
        v = torch.full((4,), 0.04, dtype=torch.float64)
        tau = torch.full((4,), 0.1, dtype=torch.float64)
        dom = cvpinn.Domain()
        loss_fn = lambda: ((net(cvpinn.base_features(x, v, tau, dom)) - target) ** 2).mean()
        info = cvpinn._run_lbfgs_safe(net, loss_fn, 10)
        self.assertFalse(info["lbfgs_rolled_back"])
        self.assertLess(info["lbfgs_post_loss"], 1e-12)


if __name__ == "__main__":
    unittest.main()
