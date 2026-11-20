"""cvpinn_slv.py - Non-Affine 2D Heston Stochastic-Local Volatility (Heston-SLV) Extension.

Solves the genuinely non-affine Heston-SLV PDE in normalized forward log-moneyness x = ln(F/K):
  c_tau = 0.5 * L(x, tau)^2 * v * (c_xx - c_x) + rho * xi * L(x, tau) * v * c_xv
          + 0.5 * xi^2 * v * c_vv + kappa * (theta - v) * c_v,
with downside equity leverage surface L(x, tau) = 1 + skew_amp * tanh(2*x) * exp(-0.5*tau).
No closed-form characteristic function or Riccati ODE exists when skew_amp != 0, yet
subtracting the ATM expected-variance carrier b(x, w_exp(v, tau)) still removes the tau=0
payoff kink identically and vanishes on the degenerate v=0 boundary:
  F_SLV = 0.5*(L^2 - 1)*v*b_gam + rho*xi*L*v*beta*b_xw + 0.5*xi^2*v*beta^2*b_ww.

Includes a validated 2D Hundsdorfer-Verwer ADI SLV solver, Gates G2_slv & G3_slv,
and B0_slv vs. M3_slv (RAR-L-BFGS) training & evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import json
import math
import os
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import solve_banded
import torch

import cvpinn
import cvpinn_m3 as m3
import cvpinn_tier2 as t2
import heston_adi_2d as adi
import heston_grid_vec as hgv
import heston_ref as hr

torch.set_default_dtype(torch.float64)
SLV_REF_PATH = os.path.join("runs", "ref_grids", "SLV_ADI.npz")
SLV_DIR = os.path.join("runs", "slv")
SLV_CKPT_DIR = os.path.join(SLV_DIR, "checkpoints")


@dataclasses.dataclass
class HestonSLV:
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    skew_amp: float = 0.25
    decay: float = 0.5

    def to_heston(self):
        return cvpinn.Heston(self.kappa, self.theta, self.xi, self.rho)

    def lev_np(self, x, tau):
        return 1.0 + self.skew_amp * np.tanh(2.0 * x) * math.exp(-self.decay * float(tau))

    def lev_t(self, x, tau):
        return 1.0 + self.skew_amp * torch.tanh(2.0 * x) * torch.exp(-self.decay * tau)


def source_slv(x, v, tau, slv: HestonSLV):
    p = slv.to_heston()
    w_safe = torch.clamp(cvpinn.total_variance(v, tau, p, "expected"), min=1e-30)
    s = torch.sqrt(w_safe)
    d1 = x / s + 0.5 * s
    d2 = d1 - s
    n_d2 = torch.where(torch.abs(d2) <= 38.0, cvpinn.npdf(d2), torch.zeros_like(d2))
    b_gam = n_d2 / s
    b_xw = -d2 * n_d2 / (2.0 * w_safe)
    b_ww = n_d2 * (d1 * d2 - 1.0) / (4.0 * w_safe * s)
    b = cvpinn.beta(tau, p)
    lev = slv.lev_t(x, tau)
    return (
        0.5 * (lev * lev - 1.0) * v * b_gam
        + slv.rho * slv.xi * lev * v * b * b_xw
        + 0.5 * (slv.xi ** 2) * v * (b ** 2) * b_ww
    )


def operator_slv(fn, x, v, tau, slv: HestonSLV):
    x, v, tau = (t.detach().clone().requires_grad_(True) for t in (x, v, tau))
    f = fn(x, v, tau)
    if not f.requires_grad:
        return torch.zeros_like(f)
    ones = torch.ones_like(f)
    f_x, f_v, f_tau = torch.autograd.grad(f, (x, v, tau), ones, create_graph=True, allow_unused=True)
    f_x = torch.zeros_like(x) if f_x is None else f_x
    f_v = torch.zeros_like(v) if f_v is None else f_v
    f_tau = torch.zeros_like(tau) if f_tau is None else f_tau
    if f_x.requires_grad:
        f_xx, f_xv = torch.autograd.grad(f_x, (x, v), torch.ones_like(f_x), create_graph=True, allow_unused=True)
        f_xx = torch.zeros_like(x) if f_xx is None else f_xx
        f_xv = torch.zeros_like(v) if f_xv is None else f_xv
    else:
        f_xx, f_xv = torch.zeros_like(x), torch.zeros_like(v)
    f_vv = cvpinn.grad(f_v, v)
    lev = slv.lev_t(x, tau)
    return (
        f_tau
        - 0.5 * (lev * lev) * v * (f_xx - f_x)
        - slv.rho * slv.xi * lev * v * f_xv
        - 0.5 * (slv.xi ** 2) * v * f_vv
        - slv.kappa * (slv.theta - v) * f_v
    )


def gate2_slv(slv: HestonSLV, dom: cvpinn.Domain, n=4096, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x, v, tau = cvpinn.sample_with_vmin(n, n // 16, dom, gen)
    p = slv.to_heston()
    lhs = operator_slv(lambda a, b, c: cvpinn.bs_call(a, cvpinn.total_variance(b, c, p, "expected")), x, v, tau, slv)
    return float((lhs + source_slv(x, v, tau, slv)).abs().max().item())


def gate3_slv(slv: HestonSLV, dom: cvpinn.Domain, n=4096, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x, v, tau = cvpinn.sample_with_vmin(n, n // 16, dom, gen)
    slv0 = HestonSLV(slv.kappa, slv.theta, 0.0, slv.rho, skew_amp=0.0, decay=slv.decay)
    zero_u = lambda a, b, c: torch.zeros_like(a)
    src_max = source_slv(x, v, tau, slv0).abs().max().item()
    res_max = operator_slv(zero_u, x, v, tau, slv0).abs().max().item()
    return float(max(src_max, res_max))


def solve_adi_slv(slv: HestonSLV, snapshot_taus, nx=600, nv=300, dt_scale=1,
                  x_max=4.0, v_max=1.0, cx=0.03, cv=0.01, rannacher_steps=4):
    """2D Hundsdorfer-Verwer ADI finite-difference solver for Heston-SLV."""
    segments = [
        (0.02, 200 * dt_scale),
        (0.05, 150 * dt_scale),
        (0.10, 125 * dt_scale),
        (0.25, 150 * dt_scale),
        (0.50, 125 * dt_scale),
        (1.00, 200 * dt_scale),
        (2.00, 400 * dt_scale),
    ]
    tau_nodes = [0.0]
    cur_t = 0.0
    for end_t, m in segments:
        tau_nodes.extend(np.linspace(cur_t, end_t, m + 1)[1:].tolist())
        cur_t = end_t

    snap_steps = {}
    for t_s in snapshot_taus:
        diffs = [abs(t - float(t_s)) for t in tau_nodes]
        k_step = int(np.argmin(diffs))
        snap_steps[k_step] = float(t_s)

    x = adi.make_sinh_grid(-x_max, x_max, 0.0, cx, nx)
    v = adi.make_sinh_grid(0.0, v_max, 0.0, cv, nv)
    wx = adi.fd1_nonuniform_coeffs(x)
    wxx = adi.fd2_nonuniform_coeffs(x)
    wv = adi.fd1_nonuniform_coeffs(v)
    wvv = adi.fd2_nonuniform_coeffs(v)

    base_a = (wxx[1:-1, 0] - wx[1:-1, 0])[None, :]
    base_b = (wxx[1:-1, 1] - wx[1:-1, 1])[None, :]
    base_c = (wxx[1:-1, 2] - wx[1:-1, 2])[None, :]
    x_int = x[1:-1]

    def apply_a1(U, tau_val):
        lev2 = (slv.lev_np(x, tau_val) ** 2)[None, :]
        return 0.5 * v[:, None] * lev2 * (adi.apply_fd2_x(U, wxx) - adi.apply_fd1_x(U, wx))

    def apply_a2(U):
        d1v = adi.apply_fd1_v(U, wv)
        d2v = adi.apply_fd2_v(U, wvv)
        out = 0.5 * (slv.xi ** 2) * v[:, None] * d2v + slv.kappa * (slv.theta - v[:, None]) * d1v
        out[:, 0] = 0.0
        out[:, -1] = 0.0
        return out

    def apply_a0(U, tau_val):
        lev = slv.lev_np(x, tau_val)[None, :]
        d1x = adi.apply_fd1_x(U, wx)
        dxv = adi.apply_fd1_v(d1x, wv)
        out = slv.rho * slv.xi * v[:, None] * lev * dxv
        out[0, :] = 0.0
        out[:, 0] = 0.0
        out[:, -1] = 0.0
        return out

    U = np.broadcast_to(adi._cell_averaged_payoff(x)[None, :], (nv + 1, nx + 1)).copy()
    theta_hv = 0.5 + math.sqrt(3.0) / 6.0
    snapshots = {}

    def step_hv(U_cur, t0, h, th):
        t_l = t0 + 1e-9 * h
        t_r = t0 + (1.0 - 1e-9) * h
        a1_u0 = apply_a1(U_cur, t_l)
        a2_u0 = apply_a2(U_cur)
        a0_u0 = apply_a0(U_cur, t_l)
        a_u0 = a0_u0 + a1_u0 + a2_u0

        Y0 = U_cur + h * a_u0
        fac = th * h
        lev2_r = (slv.lev_np(x_int, t_r) ** 2)[None, :]
        coeff_vr = 0.5 * v[:, None] * lev2_r
        tri_a = -fac * coeff_vr * base_a
        tri_b = 1.0 - fac * coeff_vr * base_b
        tri_c = -fac * coeff_vr * base_c
        dY1_int = adi._solve_tridiag_batch(tri_a, tri_b, tri_c, (Y0 - U_cur + fac * (apply_a1(U_cur, t_r) - a1_u0))[:, 1:-1])
        Y1 = U_cur.copy()
        Y1[:, 1:-1] += dY1_int

        ab_v = adi._build_v_banded(v, wv, wvv, slv.kappa, slv.theta, slv.xi, fac)
        Y2 = U_cur.copy()
        Y2[:, 1:-1] = solve_banded((2, 2), ab_v, (Y1 - fac * a2_u0)[:, 1:-1])

        if th >= 0.999999:
            return Y2

        a_y2 = apply_a0(Y2, t_r) + apply_a1(Y2, t_r) + apply_a2(Y2)
        Y0_t = Y0 + 0.5 * h * (a_y2 - a_u0)
        dY1_t_int = adi._solve_tridiag_batch(tri_a, tri_b, tri_c, (Y0_t - Y2)[:, 1:-1])
        Y1_t = Y2.copy()
        Y1_t[:, 1:-1] += dY1_t_int

        a2_y2 = apply_a2(Y2)
        U_next = U_cur.copy()
        U_next[:, 1:-1] = solve_banded((2, 2), ab_v, (Y1_t - fac * a2_y2)[:, 1:-1])
        return U_next

    for step_idx in range(1, len(tau_nodes)):
        t0 = tau_nodes[step_idx - 1]
        h = tau_nodes[step_idx] - t0
        if step_idx <= rannacher_steps:
            U = step_hv(U, t0, 0.5 * h, 1.0)
            U = step_hv(U, t0 + 0.5 * h, 0.5 * h, 1.0)
        else:
            U = step_hv(U, t0, h, theta_hv)
        if step_idx in snap_steps:
            snapshots[snap_steps[step_idx]] = U.copy()
    return x, v, snapshots


def eval_adi_slv_on_grid(slv: HestonSLV, x_eval_grid, nx=600, nv=300, dt_scale=1, rannacher_steps=4):
    t0 = time.perf_counter()
    xg, vg, snaps = solve_adi_slv(slv, hgv.EVAL_TAUS, nx=nx, nv=nv, dt_scale=dt_scale, rannacher_steps=rannacher_steps)
    out = {k: np.zeros(x_eval_grid.shape, dtype=np.float64) for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
    for it, tau in enumerate(hgv.EVAL_TAUS):
        spl = RectBivariateSpline(vg, xg, snaps[float(tau)], kx=3, ky=3)
        for iv, vv in enumerate(hgv.EVAL_VS):
            xs_row = x_eval_grid[it, iv, :]
            vv_arr = np.full_like(xs_row, float(vv))
            c_val = spl(vv_arr, xs_row, dx=0, dy=0, grid=False)
            cx_val = spl(vv_arr, xs_row, dx=0, dy=1, grid=False)
            cxx_val = spl(vv_arr, xs_row, dx=0, dy=2, grid=False)
            cv_val = spl(vv_arr, xs_row, dx=1, dy=0, grid=False)
            cxv_val = spl(vv_arr, xs_row, dx=1, dy=1, grid=False)
            cvv_val = spl(vv_arr, xs_row, dx=2, dy=0, grid=False)
            out["c"][it, iv, :] = c_val
            out["cx"][it, iv, :] = cx_val
            out["gam"][it, iv, :] = cxx_val - cx_val
            out["cv"][it, iv, :] = cv_val
            out["cxv"][it, iv, :] = cxv_val
            out["cvv"][it, iv, :] = cvv_val
    return out, time.perf_counter() - t0


def cmd_oracle(_args):
    os.makedirs(os.path.dirname(SLV_REF_PATH), exist_ok=True)
    os.makedirs(SLV_DIR, exist_ok=True)
    dom = cvpinn.Domain(v_max=0.5)
    slv = HestonSLV()
    g2 = gate2_slv(slv, dom, n=4096)
    g3 = gate3_slv(slv, dom, n=4096)
    print(f"Gates: G2_slv={g2:.3e}, G3_slv={g3:.3e}")
    assert g2 <= 1e-6 and g3 == 0.0

    d_p2 = np.load(os.path.join("runs", "ref_grids", "P2.npz"))
    x_grid = d_p2["x_grid"]
    ref_p2 = {k: d_p2[k] for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}

    # 1. Verify skew_amp=0 recovers P2 Fourier including Vega & Volga
    slv_zero = HestonSLV(skew_amp=0.0)
    adi_z, sec_z = eval_adi_slv_on_grid(slv_zero, x_grid, nx=600, nv=300, dt_scale=1, rannacher_steps=4)
    sc_z = t2.score_all_six_buckets(adi_z, ref_p2)
    ov_z = sc_z["overall"]
    print(
        f"[SLV-Check-1] skew_amp=0 vs P2 Fourier ({sec_z:.1f}s): "
        f"c={ov_z['c']['rel_rmse']:.2e} gam={ov_z['gam']['rel_rmse']:.2e} "
        f"cxv={ov_z['cxv']['rel_rmse']:.2e} cv={ov_z['cv']['rel_rmse']:.2e} cvv={ov_z['cvv']['rel_rmse']:.2e}"
    )

    # 2. Grid refinement on skew_amp=0.25 (600x300 vs 800x400, 2700t)
    adi_med, sec_m = eval_adi_slv_on_grid(slv, x_grid, nx=600, nv=300, dt_scale=1, rannacher_steps=4)
    adi_fine, sec_f = eval_adi_slv_on_grid(slv, x_grid, nx=800, nv=400, dt_scale=2, rannacher_steps=8)
    diff_mf = t2.score_all_six_buckets(adi_med, adi_fine)
    ov_mf = diff_mf["overall"]
    print(
        f"[SLV-Check-2] Med (600x300, {sec_m:.1f}s) vs Fine (800x400, {sec_f:.1f}s): "
        f"c={ov_mf['c']['rel_rmse']:.2e} gam={ov_mf['gam']['rel_rmse']:.2e} "
        f"cxv={ov_mf['cxv']['rel_rmse']:.2e} cv={ov_mf['cv']['rel_rmse']:.2e} cvv={ov_mf['cvv']['rel_rmse']:.2e}"
    )

    np.savez(
        SLV_REF_PATH, x_grid=x_grid, taus=hgv.EVAL_TAUS, vs=hgv.EVAL_VS, zs=hgv.EVAL_ZS,
        elapsed_sec=np.asarray([sec_f]), **adi_fine,
    )
    val_rep = {
        "gate2_slv": g2, "gate3_slv": g3,
        "zero_skew_vs_p2_fourier": sc_z,
        "med_vs_fine_slv": diff_mf,
    }
    with open(os.path.join(SLV_DIR, "oracle_validation.json"), "w") as f:
        json.dump(val_rep, f, indent=2)
    print(f"Cached fine SLV ADI oracle -> {SLV_REF_PATH}")


def train_slv_job(arm, seed, overwrite=False):
    os.makedirs(SLV_DIR, exist_ok=True)
    os.makedirs(SLV_CKPT_DIR, exist_ok=True)
    jpath = os.path.join(SLV_DIR, f"{arm}_SLV_seed{seed}.json")
    cpath = os.path.join(SLV_CKPT_DIR, f"{arm}_SLV_seed{seed}.pt")
    if (not overwrite) and os.path.exists(jpath):
        with open(jpath) as f:
            return json.load(f), True

    slv = HestonSLV()
    p = slv.to_heston()
    dom = cvpinn.Domain(v_max=0.5)
    d = np.load(SLV_REF_PATH)
    x_grid = d["x_grid"]
    ref = {k: d[k] for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()

    if arm == "B0_slv":
        net = cvpinn.Mlp(3, width=64, depth=4)
        c_fn = lambda x, v, tau: net(cvpinn.base_features(x, v, tau, dom))
        payoff = lambda x: torch.relu(torch.exp(x) - 1.0)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10000, eta_min=1e-4)
        for _ in range(10000):
            x, v, tau = cvpinn.sample_with_vmin(2048, 128, dom, gen)
            x0, v0, _ = cvpinn.sample_with_vmin(512, 0, dom, gen)
            xe, ve, te = cvpinn.sample_edges(256, dom, gen)
            loss = (
                (operator_slv(c_fn, x, v, tau, slv) ** 2).mean()
                + 100.0 * ((c_fn(x0, v0, torch.zeros_like(x0)) - payoff(x0)) ** 2).mean()
                + ((c_fn(xe, ve, te) - payoff(xe)) ** 2).mean()
            )
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        lx, lv, lt = cvpinn.sample_with_vmin(8192, 512, dom, gen)
        lx0, lv0, _ = cvpinn.sample_with_vmin(2048, 0, dom, gen)
        lt0 = torch.zeros_like(lx0)
        pay0 = payoff(lx0).detach()
        lxe, lve, lte = cvpinn.sample_edges(1024, dom, gen)
        pay_e = payoff(lxe).detach()
        def lloss_b0():
            return (
                (operator_slv(c_fn, lx, lv, lt, slv) ** 2).mean()
                + 100.0 * ((c_fn(lx0, lv0, lt0) - pay0) ** 2).mean()
                + ((c_fn(lxe, lve, lte) - pay_e) ** 2).mean()
            )
        linfo = cvpinn._run_lbfgs_safe(net, lloss_b0, 500)
        price_fn = c_fn
        info = {"n_params": m3.count_parameters(net), **linfo}
    else:
        # M3_slv: 4x80 Tanh + sech_alpha=0.35 wing envelope + 4k Adam + 3x1200 RAR-L-BFGS
        net = m3.FlexMlp(6, width=80, depth=4, act="tanh")
        cfg_u = {"extra_scales": True, "z2": False, "sech_alpha": 0.35}
        u = m3._make_m3_u(net, p, dom, cfg_u, is_td=False)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=1e-4)
        for _ in range(4000):
            x, v, tau = t2.sample_stratified_z(2048, 128, p, dom, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2)
            xe, ve, te = cvpinn.sample_edges(256, dom, gen)
            r = operator_slv(u, x, v, tau, slv) - source_slv(x, v, tau, slv)
            with torch.no_grad():
                rw = 1.0 / torch.sqrt(v + 0.02)
                rw = rw / torch.sqrt((rw ** 2).mean() + 1e-30)
            loss = ((rw * r) ** 2).mean() + (u(xe, ve, te) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        rounds = []
        for rd in range(1, 4):
            lx, lv, lt = t2.sample_stratified_z(16384, 1024, p, dom, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2)
            if rd >= 2:
                xp, vp, tp = t2.sample_stratified_z(49152, 3072, p, dom, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2)
                scs = []
                for i0 in range(0, len(xp), 8192):
                    sli = slice(i0, i0 + 8192)
                    rc = operator_slv(u, xp[sli], vp[sli], tp[sli], slv) - source_slv(xp[sli], vp[sli], tp[sli], slv)
                    with torch.no_grad():
                        scs.append(((1.0 / torch.sqrt(vp[sli] + 0.02)) * rc.detach()).abs())
                top_i = torch.topk(torch.cat(scs), k=4096).indices
                lx = torch.cat([lx, xp[top_i].detach()])
                lv = torch.cat([lv, vp[top_i].detach()])
                lt = torch.cat([lt, tp[top_i].detach()])
            lxe, lve, lte = cvpinn.sample_edges(1024, dom, gen)
            src_c = source_slv(lx, lv, lt, slv).detach()
            lw_raw = 1.0 / torch.sqrt(lv + 0.02)
            lw = (lw_raw / torch.sqrt((lw_raw ** 2).mean() + 1e-30)).detach()
            def lloss_m3():
                rl = operator_slv(u, lx, lv, lt, slv) - src_c
                return ((lw * rl) ** 2).mean() + (u(lxe, lve, lte) ** 2).mean()
            rinfo = m3._run_lbfgs_logged(net, lloss_m3, 1200, history_size=100, log_every=200)
            rounds.append(rinfo)
        price_fn = cvpinn.make_price(u, p, "expected")
        info = {"n_params": m3.count_parameters(net), "lbfgs_post_loss": rounds[-1]["lbfgs_post_loss"], "rounds": rounds}

    train_sec = time.perf_counter() - t0
    t_inf0 = time.perf_counter()
    pred = t2.evaluate_all_greeks(price_fn, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)
    infer_ms = 1000.0 * (time.perf_counter() - t_inf0)

    metrics = t2.score_all_six_buckets(pred, ref)
    arb, dist = m3.audit_arbitrage_and_percentiles(pred, ref, x_grid)
    torch.save({"state_dict": net.state_dict(), "arm": arm, "seed": seed}, cpath)
    rec = {
        "arm": arm, "seed": seed, "train_sec": round(train_sec, 4), "infer_ms_875pt": round(infer_ms, 3),
        "train_info": info, "metrics": metrics, "arbitrage": arb, "percentiles": dist,
    }
    with open(jpath, "w") as f:
        json.dump(rec, f, indent=2)
    return rec, False


def _w_slv(kw):
    return train_slv_job(**kw)


def cmd_benchmark(args):
    jobs = [{"arm": a.strip(), "seed": int(s.strip()), "overwrite": args.overwrite} for a in args.arms.split(",") for s in args.seeds.split(",")]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_w_slv, j) for j in jobs]
        for fut in as_completed(futs):
            rec, skip = fut.result()
            ov, sl, arb = rec["metrics"]["overall"], rec["metrics"]["short_low_var"], rec["arbitrage"]
            tag = "SKIP" if skip else f"{rec['train_sec']:.1f}s"
            print(
                f"[{tag}] {rec['arm']} s{rec['seed']} | "
                f"ov c={ov['c']['rel_rmse']:.4f} gam={ov['gam']['rel_rmse']:.4f} cxv={ov['cxv']['rel_rmse']:.4f} "
                f"cv={ov['cv']['rel_rmse']:.4f} cvv={ov['cvv']['rel_rmse']:.4f} | "
                f"slv gam={sl['gam']['rel_rmse']:.4f} | neg_gam={arb['neg_gam_count']}(core={arb['neg_gam_core_count']}) flr={arb['floor_viol_count']}"
            )


def cmd_plot(_args):
    d = np.load(SLV_REF_PATH)
    x_grid = d["x_grid"]
    ref = {k: d[k] for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
    slv = HestonSLV()
    p = slv.to_heston()
    dom = cvpinn.Domain(v_max=0.5)

    cb0 = torch.load(os.path.join(SLV_CKPT_DIR, "B0_slv_SLV_seed0.pt"), map_location="cpu", weights_only=True)
    nb0 = cvpinn.Mlp(3, width=64, depth=4); nb0.load_state_dict(cb0["state_dict"]); nb0.eval()
    pb0 = t2.evaluate_all_greeks(lambda x, v, tau: nb0(cvpinn.base_features(x, v, tau, dom)), x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)

    cm3 = torch.load(os.path.join(SLV_CKPT_DIR, "M3_slv_SLV_seed0.pt"), map_location="cpu", weights_only=True)
    nm3 = m3.FlexMlp(6, width=80, depth=4, act="tanh"); nm3.load_state_dict(cm3["state_dict"]); nm3.eval()
    um3 = m3._make_m3_u(nm3, p, dom, {"extra_scales": True, "z2": False, "sech_alpha": 0.35})
    pm3 = t2.evaluate_all_greeks(cvpinn.make_price(um3, p, "expected"), x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)

    zs = np.array(hgv.EVAL_ZS)
    vs = hgv.EVAL_VS
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.1), gridspec_kw={"width_ratios": [1.0, 1.0, 1.08], "wspace": 0.30})

    # Panel 1 & 2: tau=0.05 Gamma error surface B0_slv vs M3_slv
    e0 = np.abs(pb0["gam"][1] - ref["gam"][1])
    e3 = np.abs(pm3["gam"][1] - ref["gam"][1])
    vmax = max(float(e0.max()), float(e3.max()))
    vmin = max(vmax * 1e-4, 1e-4)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for ax, err, lbl, pred_g in ((axes[0], e0, "B0_slv", pb0["gam"][1]), (axes[1], e3, "M3_slv (CV+Sech+RAR)", pm3["gam"][1])):
        im = ax.pcolormesh(zs, np.arange(len(vs)), np.clip(err, vmin, None), norm=norm, cmap="magma", shading="auto")
        ax.set_yticks(np.arange(len(vs)))
        ax.set_yticklabels([f"{v:g}" for v in vs])
        rmse = float(np.sqrt(np.mean((pred_g - ref["gam"][1]) ** 2)))
        ax.set_title(f"{lbl} ($\\tau=0.05$, RMSE={rmse:.3f})", fontsize=10.4)
        ax.set_xlabel("Scaled moneyness $z$")
        if ax is axes[0]:
            ax.set_ylabel("Spot variance $v$")
    cbar = fig.colorbar(im, ax=[axes[0], axes[1]], orientation="horizontal", fraction=0.055, pad=0.18)
    cbar.set_label("Absolute Normalized Gamma Error (log scale)", fontsize=9.5)

    # Panel 3: 1D SLV Vanna profile at (tau=0.05, v=0.005)
    ax3 = axes[2]
    ax3.plot(zs, ref["cxv"][1, 0], "k-", lw=2.5, label="ADI SLV Oracle")
    ax3.plot(zs, pb0["cxv"][1, 0], color="#d62728", ls=":", lw=2.0, label="B0_slv")
    ax3.plot(zs, pm3["cxv"][1, 0], color="#2ca02c", ls="-.", lw=2.2, label="M3_slv")
    ax3.set_xlabel("Scaled moneyness $z$")
    ax3.set_ylabel("Normalized Vanna ($c_{xv}$)")
    ax3.set_title("SLV Vanna Slice ($\\tau=0.05, v=0.005$)", fontsize=10.4)
    ax3.grid(alpha=0.25)
    ax3.legend(frameon=True, fontsize=8.8)

    fig.suptitle("Non-Affine 2D Heston Stochastic-Local Volatility (Heston-SLV): B0_slv vs. M3_slv", fontsize=12.0, y=1.02)
    out = os.path.join("paper", "fig14_slv_benchmark.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("oracle")
    pb = sub.add_parser("benchmark")
    pb.add_argument("--arms", default="B0_slv,M3_slv")
    pb.add_argument("--seeds", default="0")
    pb.add_argument("--workers", type=int, default=2)
    pb.add_argument("--overwrite", action="store_true")
    sub.add_parser("plot")
    args = ap.parse_args()
    {"oracle": cmd_oracle, "benchmark": cmd_benchmark, "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    main()
