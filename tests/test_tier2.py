"""Unit and mathematical regression tests for Tier-2 upgrades, 8D parametric gates, and put parity."""
import math
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import numpy as np
import torch

import cvpinn
import cvpinn_m3 as m3
import cvpinn_param as cp8
import cvpinn_tier2 as t2
import heston_grid_vec as hgv
import heston_ref as hr
import spx_w8


torch.set_default_dtype(torch.float64)


class TestTier2AndParam(unittest.TestCase):
    def test_evaluate_all_six_greeks_bs_analytic(self):
        p = cvpinn.Heston(*hr.P_FELLER_BAD)
        fn = lambda x, v, tau: cvpinn.bs_call(x, cvpinn.total_variance(v, tau, p, "expected"))
        taus = (0.05, 0.5)
        vs = (0.01, 0.09)
        zs = (-1.0, 0.0, 1.0)
        x_g = np.zeros((len(taus), len(vs), len(zs)))
        for it, tau in enumerate(taus):
            for iv, v in enumerate(vs):
                w = hgv.w_exp_stable(v, tau, p.kappa, p.theta)
                x_g[it, iv, :] = np.asarray(zs) * math.sqrt(w)
        pred = t2.evaluate_all_greeks(fn, x_g, taus=taus, vs=vs)
        for it, tau in enumerate(taus):
            b = hr.beta(tau, p.kappa)
            for iv, v in enumerate(vs):
                w = hgv.w_exp_stable(v, tau, p.kappa, p.theta)
                for iz in range(len(zs)):
                    xv = float(x_g[it, iv, iz])
                    self.assertAlmostEqual(float(pred["c"][it, iv, iz]), hr.bs(xv, w), places=12)
                    self.assertAlmostEqual(float(pred["gam"][it, iv, iz]), hr.bs_gam(xv, w), places=11)
                    self.assertAlmostEqual(float(pred["cv"][it, iv, iz]), b * hr.bs_w(xv, w), places=11)
                    self.assertAlmostEqual(float(pred["cxv"][it, iv, iz]), b * hr.bs_xw(xv, w), places=11)
                    self.assertAlmostEqual(float(pred["cvv"][it, iv, iz]), (b ** 2) * hr.bs_ww(xv, w), places=10)

    def test_gate2_and_gate3_param(self):
        dom = cvpinn.Domain(v_max=0.5)
        box = cp8.ParamBox()
        self.assertLessEqual(cp8.gate2_param(dom, box, "expected", n=512, seed=0), 1e-6)
        self.assertLessEqual(cp8.gate2_param(cvpinn.Domain(v_min=1e-3, v_max=0.5), box, "spot", n=512, seed=0), 1e-6)
        self.assertEqual(cp8.gate3_param(dom, box, n=512, seed=0), 0.0)

    def test_european_put_call_parity(self):
        # Normalized parity: put(x, v, tau) = call(x, v, tau) - (exp(x) - 1) recovers BS put as xi -> 0
        p = cvpinn.Heston(1.5, 0.04, 0.0, -0.7)
        x = torch.tensor([-0.3, 0.0, 0.3], dtype=torch.float64)
        v = torch.full_like(x, 0.04)
        tau = torch.full_like(x, 0.25)
        call = cvpinn.bs_call(x, cvpinn.total_variance(v, tau, p, "expected"))
        put = call - (torch.exp(x) - 1.0)
        # Put-call parity bounds: put >= max(1 - exp(x), 0)
        intrinsic_put = torch.relu(1.0 - torch.exp(x))
        self.assertTrue(bool((put >= intrinsic_put - 1e-12).all().item()))

    def test_modified_mlp_gate3_and_smoke(self):
        p = cvpinn.Heston(*hr.P_FELLER_BAD)
        dom = cvpinn.Domain(v_max=0.5)
        p0 = cvpinn.Heston(p.kappa, p.theta, 0.0, p.rho)
        net = t2.ModifiedMlp(7, width=32, depth=3)
        with torch.no_grad():
            net.out_layer.weight.zero_()
            net.out_layer.bias.zero_()
        u0 = t2.make_correction_tier2(net, p0, dom, extra_scales=True, gauss_z=True)
        x = torch.linspace(-1.0, 1.0, 32)
        v = torch.linspace(0.0, 0.2, 32)
        tau = torch.linspace(0.01, 1.0, 32)
        self.assertEqual(cvpinn.operator(u0, x, v, tau, p0).abs().max().item(), 0.0)

    def test_w8_param_binding_smoke(self):
        p_cal = cvpinn.Heston(*hr.P_FELLER_BAD)
        dom, box = cvpinn.Domain(v_max=0.5), cp8.ParamBox()
        net = cvpinn.Mlp(10, width=16, depth=2)
        u = cp8.make_correction_param(net, dom, box, extra_scales=True)
        bound = cp8.bind_params(cp8.make_price_param(u, "expected"), p_cal)
        quotes = [(0.1, -0.1, 0.03, 0.002), (0.1, 0.0, 0.05, 0.002), (0.25, 0.1, 0.12, 0.003)]
        res = spx_w8.recalibrate_day2_v0(
            quotes,
            {"kappa": p_cal.kappa, "theta": p_cal.theta, "xi": p_cal.xi, "rho": p_cal.rho},
            pinn_price_fn=bound,
        )
        self.assertIn("pinn_vs_fourier_rmse_t2", res)
        self.assertTrue(math.isfinite(res["pinn_vs_fourier_rmse_t2"]))

    def test_m3_operator_and_smoke(self):
        p = cvpinn.Heston(*hr.P_FELLER_BAD)
        dom = cvpinn.Domain(v_max=0.5)
        net = m3.FlexMlp(6, width=16, depth=2, act="tanh")
        u = t2.make_correction_tier2(net, p, dom, extra_scales=True, z2=False)
        x = torch.linspace(-0.5, 0.5, 16)
        v = torch.linspace(0.01, 0.16, 16)
        tau = torch.linspace(0.02, 1.0, 16)
        lu_fast, c_fast, gam_fast = m3.operator_with_gam_and_c(u, x, v, tau, p, is_td=False)
        lu_ref = cvpinn.operator(u, x, v, tau, p)
        c_ref, _, gam_ref, _ = cvpinn.greeks(cvpinn.make_price(u, p, "expected"), x, v, tau)
        self.assertLess((lu_fast - lu_ref).abs().max().item(), 1e-12)
        self.assertLess((c_fast - c_ref).abs().max().item(), 1e-12)
        self.assertLess((gam_fast - gam_ref).abs().max().item(), 1e-11)
        ov = {"adam_steps": 2, "lbfgs_rounds": 2, "lbfgs_per_round": 2, "n_lbfgs_base": 128, "rar_pool": 256, "rar_add": 32}
        _, _, info = m3.train_m3("M3_rar_80", p, 0, cfg_override=ov)
        self.assertFalse(info["lbfgs_rolled_back"])
        self.assertEqual(len(info["lbfgs_rounds"]), 2)


if __name__ == "__main__":
    unittest.main()
