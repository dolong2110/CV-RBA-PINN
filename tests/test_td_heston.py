"""Regression and mathematical gate tests for Tier-1 Time-Dependent Heston modules."""
import math
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import torch

import cvpinn
import cvpinn_td as tdmod
import heston_ref as hr
import run_td_experiments as rtd


torch.set_default_dtype(torch.float64)


class TestTDHeston(unittest.TestCase):
    def setUp(self):
        self.td = tdmod.default_td_heston()
        self.dom_exp = cvpinn.Domain(v_min=0.0, v_max=0.5)
        self.dom_spot = cvpinn.Domain(v_min=1e-3, v_max=0.5)

    def test_chebyshev_ode_matches_rk4(self):
        check_taus = [1e-4, 0.02, 0.25, 1.0, 2.0]
        rk = rtd.rk4_ab_reference(self.td, check_taus, n_steps=4000)
        a_cheb, b_cheb = tdmod.alpha_beta_np(check_taus, self.td)
        for i, tau in enumerate(check_taus):
            a_rk, b_rk = rk[tau]
            self.assertLess(abs(float(a_cheb[i]) - a_rk) / a_rk, 1e-11)
            self.assertLess(abs(float(b_cheb[i]) - b_rk) / b_rk, 1e-12)

    def test_gate2_td_and_v0_strip(self):
        self.assertLessEqual(tdmod.gate2_td(self.td, self.dom_exp, "expected", n=512, seed=0), 1e-6)
        self.assertLessEqual(tdmod.gate2_td(self.td, self.dom_spot, "spot", n=512, seed=0), 1e-6)
        x0 = torch.linspace(-2.0, 2.0, 64, dtype=torch.float64)
        v0 = torch.zeros_like(x0)
        t0 = torch.linspace(1e-4, 2.0, 64, dtype=torch.float64)
        lhs = tdmod.operator_td(
            lambda a, b, c: cvpinn.bs_call(a, tdmod.total_variance_td(b, c, self.td, "expected")),
            x0, v0, t0, self.td,
        )
        res = (lhs + tdmod.source_td(x0, v0, t0, self.td, "expected")).abs().max().item()
        self.assertLessEqual(res, 1e-6)

    def test_gate3_td_xi_zero(self):
        self.assertEqual(tdmod.gate3_td(self.td, self.dom_exp, n=512, seed=0), 0.0)

    def test_greeks_on_td_control_variate(self):
        x = torch.tensor([-0.2, 0.0, 0.15], dtype=torch.float64)
        v = torch.tensor([0.005, 0.04, 0.16], dtype=torch.float64)
        tau = torch.tensor([0.02, 0.25, 1.5], dtype=torch.float64)
        fn = lambda a, b, c: cvpinn.bs_call(a, tdmod.total_variance_td(b, c, self.td, "expected"))
        c, cx, gam, cxv = cvpinn.greeks(fn, x, v, tau)
        a_np, b_np = tdmod.alpha_beta_np(tau.numpy(), self.td)
        for i in range(3):
            xi = float(x[i])
            w = float(a_np[i] + float(v[i]) * b_np[i])
            self.assertAlmostEqual(float(c[i]), hr.bs(xi, w), places=12)
            self.assertAlmostEqual(float(gam[i]), hr.bs_gam(xi, w), places=11)
            self.assertAlmostEqual(float(cxv[i]), float(b_np[i]) * hr.bs_xw(xi, w), places=11)
            self.assertTrue(math.isfinite(float(cx[i])))

    def test_chained_riccati_closed_form(self):
        z = complex(3.7, -0.5)
        a_cf, b_cf = rtd.step_char_ab(z, [0.0, 0.25, 1.0], [hr.P_FELLER_BAD, hr.P_FELLER_OK])
        # Direct complex RK4 reference
        big_a, big_b = 0.0j, 0.0j
        for (t0, t1), (kap, tht, xi, rho) in zip(
            [(0.0, 0.25), (0.25, 1.0)], [hr.P_FELLER_BAD, hr.P_FELLER_OK]
        ):
            n_sub = 4000
            h = (t1 - t0) / n_sub
            for _ in range(n_sub):
                def rhs(bv):
                    return kap * tht * bv, 0.5 * xi * xi * bv * bv - (kap - 1j * rho * xi * z) * bv - 0.5 * (z * z + 1j * z)
                k1a, k1b = rhs(big_b)
                k2a, k2b = rhs(big_b + 0.5 * h * k1b)
                k3a, k3b = rhs(big_b + 0.5 * h * k2b)
                k4a, k4b = rhs(big_b + h * k3b)
                big_a += (h / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
                big_b += (h / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
        self.assertLess(abs(a_cf - big_a), 1e-12)
        self.assertLess(abs(b_cf - big_b), 1e-12)

    def test_smoke_train_all_td_arms(self):
        x_grid = tdmod.eval_grid_td(self.td)
        for arm in ("B0_td", "M1_td", "A1_td", "A2b_td"):
            fn, _net, info = tdmod.build_and_train_td(arm, self.td, seed=0, adam_steps=2, lbfgs_iter=1)
            self.assertFalse(info["lbfgs_rolled_back"])
            pred = cvpinn.evaluate_on_grid(fn, x_grid[:2, :2, :5], [0.02, 0.05], [0.005, 0.01])
            for arr in pred.values():
                self.assertEqual(arr.shape, (2, 2, 5))


if __name__ == "__main__":
    unittest.main()
