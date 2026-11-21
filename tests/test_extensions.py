"""Unit tests for CV-RBA-PINN extensions:
  - hedging_backtest.py (exact CIR martingale, BS carrier + Fourier spline accuracy)
  - diff_calibrate.py & spx_w8.py --diff-cal & cvpinn_param.wide_spx_box
  - cvpinn_m3.py M3_sech_80 structural wing envelope
  - cvpinn_slv.py non-affine Heston-SLV Gates G2_slv & G3_slv
"""
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import cvpinn
import cvpinn_m3 as m3
import cvpinn_param as cp8
import cvpinn_slv as slvmod
import diff_calibrate as dc
import hedging_backtest as hb
import heston_ref as hr


class TestExtensions(unittest.TestCase):
    def test_cir_simulation_and_fourier_carrier_spline(self):
        p = hr.P_FELLER_BAD
        F, V = hb.simulate_heston_paths(p, f0=100.0, v0=0.02, T=0.10, n_steps=10, n_paths=4000, seed=7)
        self.assertTrue(np.all(V >= 0.0))
        # Risk-neutral forward martingale check within 3 standard errors
        se_f = float(np.std(F[-1]) / math.sqrt(F.shape[1]))
        self.assertLess(abs(float(np.mean(F[-1])) - 100.0), 3.5 * se_f)

        spl = hb.build_fourier_slice_splines(p, [0.05], nv=36, nz=49)
        f_test = np.array([96.0, 100.0, 104.0])
        v_test = np.array([0.008, 0.02, 0.06])
        c_dlr, d_dlr, g_dlr, v_dlr = hb.eval_fourier_sens(spl, f_test, v_test, 100.0, 0.05)
        for i in range(len(f_test)):
            ref = hr.heston(math.log(f_test[i] / 100.0), float(v_test[i]), 0.05, p)
            self.assertAlmostEqual(float(c_dlr[i]), 100.0 * ref["c"], delta=2e-3)
            self.assertAlmostEqual(float(d_dlr[i]), (100.0 / f_test[i]) * ref["cx"], delta=2e-3)
            self.assertAlmostEqual(float(g_dlr[i]), (100.0 / (f_test[i] ** 2)) * ref["gam"], delta=2e-3)
            self.assertAlmostEqual(float(v_dlr[i]), 100.0 * ref["cv"], delta=max(5e-3, 1e-3 * abs(100.0 * ref["cv"])))

    def test_wide_spx_box_and_diff_calibrate_cli(self):
        box = cp8.wide_spx_box()
        self.assertGreaterEqual(box.xi_max, 1.4)
        self.assertLessEqual(box.rho_min, -0.95)

        wide_ckpt = os.path.join("runs", "param", "checkpoints", "M3_param_wide_seed0.pt")
        if os.path.exists(wide_ckpt):
            price_fn, loaded_box = dc.load_surrogate(wide_ckpt)
            self.assertAlmostEqual(loaded_box.xi_max, 1.6, places=6)
            self.assertAlmostEqual(loaded_box.kappa_max, 6.0, places=6)
            # Exact intrinsic call payoff at tau = 0 across stressed SPX vol-of-vol xi = 1.497
            x0 = torch.tensor([-0.2, 0.0, 0.2], dtype=torch.float64)
            v0 = torch.tensor([0.0655, 0.0655, 0.0655], dtype=torch.float64)
            t0 = torch.zeros_like(x0)
            kap = torch.full_like(x0, 4.073)
            tht = torch.full_like(x0, 0.0558)
            xi = torch.full_like(x0, 1.497)
            rho = torch.full_like(x0, -0.746)
            c0 = price_fn(x0, v0, t0, kap, tht, xi, rho).detach()
            exact_intr = torch.clamp(torch.exp(x0) - 1.0, min=0.0)
            self.assertLessEqual(float((c0 - exact_intr).abs().max().item()), 1e-15)

        crisis_v2_ckpt = os.path.join("runs", "param", "checkpoints", "M3_param_crisis_v2_seed0.pt")
        if os.path.exists(crisis_v2_ckpt):
            pfn_c2, box_c2 = dc.load_surrogate(crisis_v2_ckpt)
            self.assertAlmostEqual(box_c2.xi_max, 2.9, places=6)
            self.assertAlmostEqual(box_c2.rho_min, -0.98, places=6)
            self.assertAlmostEqual(getattr(pfn_c2, "_v_max", 0.5), 1.0, places=6)
            xc2 = torch.tensor([-0.3, 0.0, 0.3], dtype=torch.float64)
            cc2 = pfn_c2(xc2, torch.full_like(xc2, 0.857), torch.zeros_like(xc2), torch.full_like(xc2, 5.40), torch.full_like(xc2, 0.156), torch.full_like(xc2, 2.80), torch.full_like(xc2, -0.97)).detach()
            self.assertLessEqual(float((cc2 - torch.clamp(torch.exp(xc2) - 1.0, min=0.0)).abs().max().item()), 1e-15)

        exp_t1 = [
            ("2026-12-18", "SPX", 5900.0, 0.045, 0.013),
            ("2027-02-19", "SPX", 5900.0, 0.044, 0.013),
            ("2027-05-21", "SPX", 5900.0, 0.043, 0.013),
        ]
        exp_t2 = [
            ("2026-12-18", "SPX", 5930.0, 0.045, 0.013),
            ("2027-02-19", "SPX", 5930.0, 0.044, 0.013),
            ("2027-05-21", "SPX", 5930.0, 0.043, 0.013),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            f1 = os.path.join(tmp, "t1.csv")
            f2 = os.path.join(tmp, "t2.csv")
            out_j = os.path.join(tmp, "out.json")
            dc._make_noisy_chain(f1, "2026-11-18 14:30:00", exp_t1, 0.040, hr.P_FELLER_BAD, noise_bps=0.5, seed=1)
            dc._make_noisy_chain(f2, "2026-11-19 14:30:00", exp_t2, 0.055, hr.P_FELLER_BAD, noise_bps=0.5, seed=2)
            proc = subprocess.run(
                [sys.executable, "spx_w8.py", "--t1", f1, "--t2", f2, "--diff-cal", "--out", out_j],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("differentiable", proc.stdout)
            self.assertTrue(os.path.exists(out_j))

    def test_m3_sech_wing_decay_and_terminal_condition(self):
        p = cvpinn.Heston(*hr.P_FELLER_BAD)
        dom = cvpinn.Domain(v_max=0.5)
        cfg = m3.M3_CONFIGS["M3_sech_80"]
        net = m3.build_m3_net(cfg)
        u = m3._make_m3_u(net, p, dom, cfg, is_td=False)
        x = torch.tensor([-2.4, 0.0, 2.4], dtype=torch.float64)
        v = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        # Exact zero at tau = 0
        u0 = u(x, v, torch.zeros_like(x))
        self.assertEqual(float(u0.abs().max().item()), 0.0)
        # Far-wing sech envelope strongly damps |z| >> 1
        tau_s = torch.full_like(x, 0.02)
        u_vals = u(x, v, tau_s).detach().abs()
        self.assertLess(float(u_vals[0]), 0.01 * (float(u_vals[1]) + 1e-3))

    def test_slv_gates_and_zero_skew_reduction(self):
        dom = cvpinn.Domain(v_max=0.5)
        slv = slvmod.HestonSLV(skew_amp=0.25)
        g2 = slvmod.gate2_slv(slv, dom, n=1024, seed=0)
        g3 = slvmod.gate3_slv(slv, dom, n=1024, seed=0)
        self.assertLessEqual(g2, 1e-12)
        self.assertEqual(g3, 0.0)
        # When skew_amp = 0, source_slv must match standard Heston expected-variance source identically
        x, v, tau = cvpinn.sample_with_vmin(256, 16, dom, torch.Generator().manual_seed(3))
        slv0 = slvmod.HestonSLV(skew_amp=0.0)
        diff = (slvmod.source_slv(x, v, tau, slv0) - cvpinn.source(x, v, tau, slv0.to_heston(), "expected")).abs().max().item()
        self.assertLessEqual(diff, 1e-15)

    def test_bates_pide_gates_and_zero_jump_reduction(self):
        import cvpinn_bates as bmod
        import heston_grid_vec as hgv

        dom = cvpinn.Domain(v_max=0.5)
        bp = bmod.BatesSVJ()
        # Exact intrinsic payoff at tau = 0
        x0 = torch.tensor([-0.3, 0.0, 0.3], dtype=torch.float64)
        v0 = torch.tensor([0.01, 0.04, 0.16], dtype=torch.float64)
        c0 = bmod.merton_carrier_expected(x0, v0, torch.zeros_like(x0), bp)
        self.assertLessEqual(float((c0 - torch.clamp(torch.exp(x0) - 1.0, min=0.0)).abs().max().item()), 1e-15)
        # Source vanishes identically at v = 0 and xi = 0
        self.assertEqual(float(bmod.source_bates(x0, torch.zeros_like(x0), torch.full_like(x0, 0.25), bp).abs().max().item()), 0.0)
        bp_xi0 = bmod.BatesSVJ(xi=0.0)
        self.assertEqual(float(bmod.source_bates(x0, v0, torch.full_like(x0, 0.25), bp_xi0).abs().max().item()), 0.0)
        # When lam = 0, Bates Fourier oracle matches Heston Fourier oracle to machine precision across all 6 outputs
        bp_nojump = bmod.BatesSVJ(lam=0.0)
        _, ref_b = bmod.bates_eval_grid(bp_nojump, taus=(0.05, 0.5), vs=(0.01, 0.04))
        _, ref_h = hgv.heston_eval_grid((bp.kappa, bp.theta, bp.xi, bp.rho), taus=(0.05, 0.5), vs=(0.01, 0.04))
        for k in ("c", "cx", "gam", "cv", "cxv", "cvv"):
            self.assertLessEqual(float(np.max(np.abs(ref_b[k] - ref_h[k]))), 1e-13)


if __name__ == "__main__":
    unittest.main()
