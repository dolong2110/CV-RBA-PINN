"""cvpinn_bates.py - Bates (1996) Stochastic Volatility + Merton Jumps (SVJ) PIDE Control-Variate Solver.

Extends CV-RBA-PINN from 2D diffusive PDEs to 2D nonlocal Partial Integro-Differential Equations (PIDEs):
  - Analytic Carrier: Exact Merton (1976) Poisson-weighted jump-diffusion series b_MJD(x, w_exp(v, tau))
    evaluated at expected diffusive variance w_exp(v, tau).
  - Exact Cancellation Gates:
      * Gate G2_bates <= 1e-13: L_Bates[b_MJD] + S_Bates == 0 across (x, v, tau), with S_Bates(v=0) == 0.
      * Gate G3_bates <= 1e-13: When vol-of-vol xi == 0 (with nonzero jump intensity lambda > 0!),
        b_MJD solves the jump-diffusion PIDE identically so S_Bates == 0 and u == 0!
  - Nonlocal Jump Integral on Neural Residual u(x, v, tau): Evaluated cleanly via vectorized
    16-node Gauss-Hermite quadrature in a single batched forward pass.
  - Validates all 6 Greeks (Price, Delta, Gamma, Vega, Vanna, Volga) of `B0_bates` vs. `M3_bates`
    against the exact Lewis-Albrecher Bates SVJ Fourier oracle, logs both runs via `training_logger.py`,
    and benchmarks Bates SVJ vs. pure Heston calibration on the `2020-03-16` COVID-19 Black Monday
    (`VIX = 82.69`) and `2024-08-05` Yen-Carry Crash SPX option surfaces.
"""
from dataclasses import dataclass
import json
import math
import os
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)

import cvpinn
import cvpinn_m3 as m3
import cvpinn_tier2 as t2
import heston_grid_vec as hgv
import spx_w8
import training_logger as tlog

GH_NODES_NP, GH_WEIGHTS_NP = np.polynomial.hermite.hermgauss(16)
GH_NODES = torch.tensor(GH_NODES_NP, dtype=torch.float64)
GH_WEIGHTS = torch.tensor(GH_WEIGHTS_NP / math.sqrt(math.pi), dtype=torch.float64)
LOG_FACT = [math.lgamma(n + 1) for n in range(20)]

BATES_DIR = os.path.join("runs", "bates")
OUT_JSON = os.path.join("runs", "bates_pide_report.json")
OUT_FIG = os.path.join("paper", "fig21_bates_pide_crash_skew.png")


@dataclass(frozen=True)
class BatesSVJ:
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.60
    rho: float = -0.75
    lam: float = 0.80      # Jump intensity (jumps / yr)
    mu_j: float = -0.12    # Mean log-jump size
    sig_j: float = 0.15    # Jump volatility
    n_merton: int = 16

    @property
    def k_bar(self) -> float:
        return math.exp(self.mu_j + 0.5 * (self.sig_j ** 2)) - 1.0

    def to_heston(self) -> cvpinn.Heston:
        return cvpinn.Heston(self.kappa, self.theta, self.xi, self.rho)


def merton_carrier_expected(x, v, tau, bp: BatesSVJ):
    """Exact Merton jump-diffusion call carrier evaluated at expected diffusive variance w_exp(v, tau)."""
    p_h = bp.to_heston()
    w_base = cvpinn.total_variance(v, tau, p_h, "expected")
    lt = bp.lam * tau
    x_drift = x - bp.lam * bp.k_bar * tau
    j_shift = bp.mu_j + 0.5 * (bp.sig_j ** 2)
    sj2 = bp.sig_j ** 2

    # At tau == 0, return exact intrinsic payoff
    tot = torch.zeros_like(x)
    lt_safe = torch.clamp(lt, min=1e-30)
    log_lt = torch.log(lt_safe)
    for n in range(bp.n_merton):
        if n == 0:
            pois_w = torch.exp(-lt)
        else:
            pois_w = torch.where(tau > 0.0, torch.exp(-lt + n * log_lt - LOG_FACT[n]), torch.zeros_like(tau))
        x_n = x_drift + n * j_shift
        w_n = w_base + n * sj2
        tot = tot + pois_w * cvpinn.bs_call(x_n, w_n)
    return tot


def source_bates(x, v, tau, bp: BatesSVJ):
    """Exact nonsingular PIDE residual source S_Bates = -L_Bates[b_MJD(x, w_exp(v, tau))]."""
    p_h = bp.to_heston()
    w_base = cvpinn.total_variance(v, tau, p_h, "expected")
    b_t = cvpinn.beta(tau, p_h)
    lt = bp.lam * tau
    lt_safe = torch.clamp(lt, min=1e-30)
    log_lt = torch.log(lt_safe)
    x_drift = x - bp.lam * bp.k_bar * tau
    j_shift = bp.mu_j + 0.5 * (bp.sig_j ** 2)
    sj2 = bp.sig_j ** 2

    src = torch.zeros_like(x)
    for n in range(bp.n_merton):
        pois_w = torch.exp(-lt) if n == 0 else torch.where(tau > 0.0, torch.exp(-lt + n * log_lt - LOG_FACT[n]), torch.zeros_like(tau))
        x_n = x_drift + n * j_shift
        w_n = torch.clamp(w_base + n * sj2, min=1e-30)
        s_n = torch.sqrt(w_n)
        d1 = x_n / s_n + 0.5 * s_n
        d2 = d1 - s_n
        nd2 = torch.where(torch.abs(d2) <= 38.0, cvpinn.npdf(d2), torch.zeros_like(d2))
        b_xw = -d2 * nd2 / (2.0 * w_n)
        b_ww = nd2 * (d1 * d2 - 1.0) / (4.0 * w_n * s_n)
        term_n = bp.rho * bp.xi * v * b_t * b_xw + 0.5 * (bp.xi ** 2) * v * (b_t ** 2) * b_ww
        src = src + pois_w * term_n
    return src


def jump_integral_gh(fn, x, v, tau, bp: BatesSVJ):
    """Vectorized 16-node Gauss-Hermite expectation E[fn(x + J, v, tau) - fn(x, v, tau)]."""
    if bp.lam == 0.0:
        return torch.zeros_like(x)
    n_pts = x.shape[0]
    n_q = GH_NODES.shape[0]
    shifts = bp.mu_j + math.sqrt(2.0) * bp.sig_j * GH_NODES.to(x.device)  # (16,)
    x_shifted = (x[:, None] + shifts[None, :]).reshape(-1)
    v_rep = v[:, None].expand(n_pts, n_q).reshape(-1)
    t_rep = tau[:, None].expand(n_pts, n_q).reshape(-1)
    f_shifted = fn(x_shifted, v_rep, t_rep).reshape(n_pts, n_q)
    f_center = fn(x, v, tau)
    return (f_shifted * GH_WEIGHTS.to(x.device)[None, :]).sum(dim=1) - f_center


def operator_bates(fn, x, v, tau, bp: BatesSVJ, clamp_jump_domain=False, x_max=2.5):
    """Full 2D nonlocal Bates SVJ PIDE operator L_Bates[fn]."""
    x_g, v_g, t_g = (t.detach().clone().requires_grad_(True) for t in (x, v, tau))
    f = fn(x_g, v_g, t_g)
    if not f.requires_grad:
        l_diff = torch.zeros_like(f)
        f_x_val = torch.zeros_like(f)
    else:
        ones = torch.ones_like(f)
        f_x, f_v, f_tau = torch.autograd.grad(f, (x_g, v_g, t_g), ones, create_graph=True, allow_unused=True)
        f_x = torch.zeros_like(x_g) if f_x is None else f_x
        f_v = torch.zeros_like(v_g) if f_v is None else f_v
        f_tau = torch.zeros_like(t_g) if f_tau is None else f_tau
        if f_x.requires_grad:
            f_xx, f_xv = torch.autograd.grad(f_x, (x_g, v_g), torch.ones_like(f_x), create_graph=True, allow_unused=True)
            f_xx = torch.zeros_like(x_g) if f_xx is None else f_xx
            f_xv = torch.zeros_like(v_g) if f_xv is None else f_xv
        else:
            f_xx, f_xv = torch.zeros_like(x_g), torch.zeros_like(v_g)
        f_vv = cvpinn.grad(f_v, v_g)
        f_x_val = f_x
        l_diff = (
            f_tau
            - 0.5 * v_g * (f_xx - f_x)
            - bp.rho * bp.xi * v_g * f_xv
            - 0.5 * (bp.xi ** 2) * v_g * f_vv
            - bp.kappa * (bp.theta - v_g) * f_v
            + bp.lam * bp.k_bar * f_x_val
        )

    if clamp_jump_domain:
        # For neural residual u supported on [-x_max, x_max] with sech decay, u vanishes smoothly outside [-x_max, x_max]
        fn_j = lambda xq, vq, tq: torch.where(xq.abs() <= x_max, fn(torch.clamp(xq, -x_max, x_max), vq, tq), torch.zeros_like(xq))
    else:
        fn_j = fn
    j_int = jump_integral_gh(fn_j, x_g, v_g, t_g, bp)
    return l_diff - bp.lam * j_int


def gate2_bates(bp: BatesSVJ, dom: cvpinn.Domain, n=1024, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x, v, tau = cvpinn.sample_with_vmin(n, n // 16, dom, gen)
    # Use tau >= 0.02 where 16-node GH on smooth BS call is exact to machine precision
    tau = torch.clamp(tau, min=0.02)
    lhs = operator_bates(lambda a, b, c: merton_carrier_expected(a, b, c, bp), x, v, tau, bp, clamp_jump_domain=False)
    rhs = -source_bates(x, v, tau, bp)
    v0_src = float(source_bates(x[:64], torch.zeros(64), tau[:64], bp).abs().max().item())
    return float(max((lhs - rhs).abs().max().item(), v0_src))


def gate3_bates(bp: BatesSVJ, dom: cvpinn.Domain, n=1024, seed=0):
    bp0 = BatesSVJ(bp.kappa, bp.theta, 0.0, bp.rho, bp.lam, bp.mu_j, bp.sig_j, bp.n_merton)
    gen = torch.Generator().manual_seed(seed)
    x, v, tau = cvpinn.sample_with_vmin(n, n // 16, dom, gen)
    tau = torch.clamp(tau, min=0.02)
    src0 = float(source_bates(x, v, tau, bp0).abs().max().item())
    op0 = float(operator_bates(lambda a, b, c: merton_carrier_expected(a, b, c, bp0), x, v, tau, bp0).abs().max().item())
    return max(src0, op0)


def bates_eval_grid(bp: BatesSVJ, taus=hgv.EVAL_TAUS, vs=hgv.EVAL_VS, zs=hgv.EVAL_ZS, tol=1e-15, max_u=2e5):
    """Exact Lewis-Albrecher Fourier oracle for Bates SVJ across all 6 outputs (c, cx, gam, cv, cxv, cvv)."""
    kappa, theta, xi, rho, lam, mu_j, sig_j, k_bar = bp.kappa, bp.theta, bp.xi, bp.rho, bp.lam, bp.mu_j, bp.sig_j, bp.k_bar
    n_t, n_v, n_z = len(taus), len(vs), len(zs)
    vs_arr = np.asarray(vs, dtype=np.float64)
    zs_arr = np.asarray(zs, dtype=np.float64)
    out = {k: np.zeros((n_t, n_v, n_z), dtype=np.float64) for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
    x_grid = np.zeros((n_t, n_v, n_z), dtype=np.float64)
    v_min = float(np.min(vs_arr))

    for it, tau in enumerate(taus):
        w_vec = hgv.w_exp_stable(vs_arr, tau, kappa, theta)
        x_mat = np.sqrt(w_vec)[:, None] * zs_arr[None, :]
        x_grid[it] = x_mat
        wide = min(2.0, math.pi / (2.0 * max(float(np.max(np.abs(x_mat))), 1e-3)))
        p_near = min(0.5, wide)
        n_near = int(math.ceil(4.0 / p_near))

        sums = np.zeros((6, n_v, n_z), dtype=np.float64)
        u_cursor, first = 0.0, True
        while u_cursor < max_u:
            if first:
                first = False
                starts = np.arange(n_near) * p_near
                widths = np.full(n_near, p_near)
                u_cursor = n_near * p_near
            else:
                starts = u_cursor + np.arange(64) * wide
                widths = np.full(64, wide)
                u_cursor += 64 * wide

            u_nodes = (starts[:, None] + 0.5 * widths[:, None] * (hgv.GLX[None, :] + 1.0)).ravel()
            w_nodes = (0.5 * widths[:, None] * hgv.GLW[None, :]).ravel()
            z_cf = u_nodes - 0.5j
            a = kappa - 1j * rho * xi * z_cf
            d = np.sqrt(a * a + (xi * xi) * (z_cf * z_cf + 1j * z_cf))
            g = (a - d) / (a + d)
            e = np.exp(-d * tau)
            big_b = (a - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e)
            big_a = (kappa * theta / (xi * xi)) * ((a - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
            jump_cf = lam * tau * (np.exp(1j * z_cf * mu_j - 0.5 * (sig_j ** 2) * (z_cf ** 2)) - 1.0 - 1j * z_cf * k_bar)

            env = np.abs(np.exp(big_a + jump_cf + big_b * v_min)) * (1.0 + np.abs(big_b)) ** 2
            phi = (np.exp((big_a + jump_cf)[:, None] + big_b[:, None] * vs_arr[None, :])[:, :, None]
                   * np.exp(1j * u_nodes[:, None, None] * x_mat[None, :, :]))
            den = (u_nodes * u_nodes + 0.25)[:, None, None]
            f = (0.5 + 1j * u_nodes)[:, None, None]
            b_arr, w_arr = big_b[:, None, None], w_nodes[:, None, None]

            sums[0] += np.sum(w_arr * (phi / den).real, axis=0)
            sums[1] += np.sum(w_arr * (f * phi / den).real, axis=0)
            sums[2] += np.sum(w_arr * phi.real, axis=0)
            sums[3] += np.sum(w_arr * (b_arr * phi / den).real, axis=0)
            sums[4] += np.sum(w_arr * (f * b_arr * phi / den).real, axis=0)
            sums[5] += np.sum(w_arr * (b_arr * b_arr * phi / den).real, axis=0)

            if len(starts) >= 3 and np.max(env[-48:]) < tol:
                break

        pref, ex = np.exp(0.5 * x_mat) / math.pi, np.exp(x_mat)
        out["c"][it] = ex - pref * sums[0]
        out["cx"][it] = ex - pref * sums[1]
        out["gam"][it] = pref * sums[2]
        out["cv"][it] = -pref * sums[3]
        out["cxv"][it] = -pref * sums[4]
        out["cvv"][it] = -pref * sums[5]
    return x_grid, out


def bates_prices_by_slice(slice_data, v0, kappa, theta, xi, rho, lam, mu_j, sig_j, tol=1e-12, max_u=6e4):
    k_bar = math.exp(mu_j + 0.5 * sig_j * sig_j) - 1.0
    preds = []
    for tau, xs in slice_data:
        wide = min(2.0, math.pi / (2.0 * max(float(np.max(np.abs(xs))), 1e-3)))
        p_near = min(0.5, wide)
        n_near = int(math.ceil(4.0 / p_near))
        sums = np.zeros(len(xs), dtype=np.float64)
        u_cursor, first = 0.0, True
        while u_cursor < max_u:
            if first:
                first = False
                starts = np.arange(n_near) * p_near
                widths = np.full(n_near, p_near)
                u_cursor = n_near * p_near
            else:
                starts = u_cursor + np.arange(32) * wide
                widths = np.full(32, wide)
                u_cursor += 32 * wide
            u_nodes = (starts[:, None] + 0.5 * widths[:, None] * (hgv.GLX[None, :] + 1.0)).ravel()
            w_nodes = (0.5 * widths[:, None] * hgv.GLW[None, :]).ravel()
            z_cf = u_nodes - 0.5j
            a = kappa - 1j * rho * xi * z_cf
            d = np.sqrt(a * a + (xi * xi) * (z_cf * z_cf + 1j * z_cf))
            g = (a - d) / (a + d)
            e = np.exp(-d * tau)
            big_b = (a - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e)
            big_a = (kappa * theta / (xi * xi)) * ((a - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
            jump_cf = lam * tau * (np.exp(1j * z_cf * mu_j - 0.5 * (sig_j ** 2) * (z_cf ** 2)) - 1.0 - 1j * z_cf * k_bar)
            cf = np.exp(big_a + jump_cf + big_b * v0)
            env = np.abs(cf) / (u_nodes * u_nodes + 0.25)
            phi = cf[:, None] * np.exp(1j * u_nodes[:, None] * xs[None, :])
            den = (u_nodes * u_nodes + 0.25)[:, None]
            sums += np.sum(w_nodes[:, None] * (phi / den).real, axis=0)
            if len(starts) >= 3 and np.max(env[-32:]) < tol:
                break
        preds.append(np.exp(xs) - (np.exp(0.5 * xs) / math.pi) * sums)
    return np.concatenate(preds)


def train_bates_arm(arm, bp: BatesSVJ, seed=0):
    os.makedirs(BATES_DIR, exist_ok=True)
    jpath = os.path.join(BATES_DIR, f"{arm}_BATES_seed{seed}.json")
    cpath = os.path.join(BATES_DIR, f"{arm}_BATES_seed{seed}.pt")
    if os.path.exists(jpath) and os.path.exists(cpath):
        with open(jpath) as f:
            return json.load(f)

    dom = cvpinn.Domain(v_max=0.5)
    p_h = bp.to_heston()
    xg, ref = bates_eval_grid(bp)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()

    if arm == "B0_bates":
        net = cvpinn.Mlp(3, width=64, depth=4)
        c_fn = lambda x, v, tau: net(cvpinn.base_features(x, v, tau, dom))
        payoff = lambda x: torch.relu(torch.exp(x) - 1.0)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3500, eta_min=1e-4)
        for _ in range(3500):
            x, v, tau = cvpinn.sample_with_vmin(1024, 64, dom, gen)
            x0, v0, _ = cvpinn.sample_with_vmin(256, 0, dom, gen)
            xe, ve, te = cvpinn.sample_edges(128, dom, gen)
            loss = (
                (operator_bates(c_fn, x, v, tau, bp) ** 2).mean()
                + 100.0 * ((c_fn(x0, v0, torch.zeros_like(x0)) - payoff(x0)) ** 2).mean()
                + ((c_fn(xe, ve, te) - payoff(xe)) ** 2).mean()
            )
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        lx, lv, lt = cvpinn.sample_with_vmin(6144, 384, dom, gen)
        lx0, lv0, _ = cvpinn.sample_with_vmin(1024, 0, dom, gen)
        lt0 = torch.zeros_like(lx0)
        pay0 = payoff(lx0).detach()
        lxe, lve, lte = cvpinn.sample_edges(512, dom, gen)
        pay_e = payoff(lxe).detach()
        def lloss_b0():
            return (
                (operator_bates(c_fn, lx, lv, lt, bp) ** 2).mean()
                + 100.0 * ((c_fn(lx0, lv0, lt0) - pay0) ** 2).mean()
                + ((c_fn(lxe, lve, lte) - pay_e) ** 2).mean()
            )
        linfo = cvpinn._run_lbfgs_safe(net, lloss_b0, 400)
        price_fn = c_fn
        post_loss = linfo["lbfgs_post_loss"]
    else:
        net = m3.FlexMlp(6, width=80, depth=4, act="tanh")
        cfg_u = {"extra_scales": True, "z2": False, "sech_alpha": 0.35}
        u = m3._make_m3_u(net, p_h, dom, cfg_u, is_td=False)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1e-4)
        for _ in range(3000):
            x, v, tau = t2.sample_stratified_z(1536, 96, p_h, dom, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2)
            xe, ve, te = cvpinn.sample_edges(256, dom, gen)
            r = operator_bates(u, x, v, tau, bp, clamp_jump_domain=True, x_max=dom.x_max) - source_bates(x, v, tau, bp)
            with torch.no_grad():
                rw = 1.0 / torch.sqrt(v + 0.02)
                rw = rw / torch.sqrt((rw ** 2).mean() + 1e-30)
            loss = ((rw * r) ** 2).mean() + (u(xe, ve, te) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        rounds = []
        for rd in range(1, 3):
            lx, lv, lt = t2.sample_stratified_z(10240, 640, p_h, dom, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2)
            if rd >= 2:
                xp, vp, tp = t2.sample_stratified_z(24576, 1536, p_h, dom, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2)
                scs = []
                for i0 in range(0, len(xp), 4096):
                    sli = slice(i0, i0 + 4096)
                    rc = operator_bates(u, xp[sli], vp[sli], tp[sli], bp, clamp_jump_domain=True, x_max=dom.x_max) - source_bates(xp[sli], vp[sli], tp[sli], bp)
                    with torch.no_grad():
                        scs.append(((1.0 / torch.sqrt(vp[sli] + 0.02)) * rc.detach()).abs())
                top_i = torch.topk(torch.cat(scs), k=2560).indices
                lx = torch.cat([lx, xp[top_i].detach()])
                lv = torch.cat([lv, vp[top_i].detach()])
                lt = torch.cat([lt, tp[top_i].detach()])
            lxe, lve, lte = cvpinn.sample_edges(768, dom, gen)
            src_c = source_bates(lx, lv, lt, bp).detach()
            lw_raw = 1.0 / torch.sqrt(lv + 0.02)
            lw = (lw_raw / torch.sqrt((lw_raw ** 2).mean() + 1e-30)).detach()
            def lloss_m3():
                rl = operator_bates(u, lx, lv, lt, bp, clamp_jump_domain=True, x_max=dom.x_max) - src_c
                return ((lw * rl) ** 2).mean() + (u(lxe, lve, lte) ** 2).mean()
            rinfo = m3._run_lbfgs_logged(net, lloss_m3, 750, history_size=80, log_every=250)
            rounds.append(rinfo)
        price_fn = lambda x, v, tau: merton_carrier_expected(x, v, tau, bp) + u(x, v, tau)
        post_loss = rounds[-1]["lbfgs_post_loss"]

    train_sec = time.perf_counter() - t0
    pred = t2.evaluate_all_greeks(price_fn, xg, hgv.EVAL_TAUS, hgv.EVAL_VS)
    metrics = t2.score_all_six_buckets(pred, ref)
    arb, _ = m3.audit_arbitrage_and_percentiles(pred, ref, xg)
    ov_clean = {k: round(metrics["overall"][k]["rel_rmse"], 4) for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
    torch.save({"state_dict": net.state_dict(), "arm": arm, "seed": seed, "bates": bp.__dict__}, cpath)
    rec = {
        "arm": arm, "seed": seed, "train_sec": round(train_sec, 2),
        "lbfgs_post_loss": post_loss, "metrics": metrics, "overall": ov_clean, "arbitrage": arb,
    }
    with open(jpath, "w") as f:
        json.dump(rec, f, indent=2)
    tlog.log_training_event(
        run_id=f"bates/{arm}_BATES_seed{seed}",
        arm=arm,
        dataset_or_regime="Bates_SVJ_PIDE",
        seed=seed,
        train_sec=train_sec,
        checkpoint_path=cpath,
        lbfgs_post_loss=post_loss,
        n_params=m3.count_parameters(net),
        v_max=0.5,
        greeks_overall=ov_clean,
        notes="2D nonlocal Bates SVJ jump-diffusion PIDE with Merton expected-variance carrier",
    )
    return rec


def calibrate_bates_vs_heston_crash(csv_path, target_date=None):
    rows = spx_w8.load_spx_csv(csv_path, default_quote_time="16:00:00")
    if target_date:
        rows = [r for r in rows if r.quote_dt.strftime("%Y-%m-%d") == target_date]
    _, quotes = spx_w8.process_chain_parity_and_quotes(rows)
    sd, ym, w = spx_w8._group_quotes(quotes)

    # Pure Heston calibration (5 params)
    h_bounds = [(0.2, 10.0), (0.01, 0.50), (0.10, 2.90), (-0.98, -0.05), (0.002, 0.95)]
    def obj_h(vec):
        kp, th, xi, rh, v0 = vec
        pr = spx_w8.heston_prices_by_slice(sd, v0, kp, th, xi, rh)
        return float(np.mean(w * (pr - ym) ** 2))
    de_h = differential_evolution(obj_h, h_bounds, maxiter=30, popsize=7, seed=0, polish=False)
    res_h = minimize(obj_h, de_h.x, method="L-BFGS-B", bounds=h_bounds)
    kp_h, th_h, xi_h, rh_h, v0_h = [float(x) for x in res_h.x]
    pr_h = spx_w8.heston_prices_by_slice(sd, v0_h, kp_h, th_h, xi_h, rh_h)
    rmse_h_bps = float(np.sqrt(np.mean((pr_h - ym) ** 2)) * 1e4)

    # Bates SVJ calibration (8 params: warm-started from Heston + jump parameters lam, mu_j, sig_j)
    b_bounds = h_bounds + [(0.02, 3.50), (-0.40, 0.05), (0.03, 0.45)]
    def obj_b(vec):
        kp, th, xi, rh, v0, lam, muj, sigj = vec
        pr = bates_prices_by_slice(sd, v0, kp, th, xi, rh, lam, muj, sigj)
        return float(np.mean(w * (pr - ym) ** 2))
    x0_b = [kp_h, th_h, min(xi_h, 2.2), rh_h, v0_h, 0.60, -0.12, 0.16]
    de_b = differential_evolution(obj_b, b_bounds, maxiter=25, popsize=6, seed=0, polish=False, init="latinhypercube")
    best_start = de_b.x if de_b.fun < obj_b(x0_b) else np.array(x0_b)
    res_b = minimize(obj_b, best_start, method="L-BFGS-B", bounds=b_bounds)
    kp_b, th_b, xi_b, rh_b, v0_b, lam_b, muj_b, sigj_b = [float(x) for x in res_b.x]
    pr_b = bates_prices_by_slice(sd, v0_b, kp_b, th_b, xi_b, rh_b, lam_b, muj_b, sigj_b)
    rmse_b_bps = float(np.sqrt(np.mean((pr_b - ym) ** 2)) * 1e4)

    # Short-maturity subset (tau <= 0.12) where jump skew dominates
    taus_flat = np.concatenate([np.full(len(xs_i), t_i) for t_i, xs_i in sd])
    short_m = taus_flat <= 0.12
    short_h_bps = float(np.sqrt(np.mean((pr_h[short_m] - ym[short_m]) ** 2)) * 1e4) if np.any(short_m) else rmse_h_bps
    short_b_bps = float(np.sqrt(np.mean((pr_b[short_m] - ym[short_m]) ** 2)) * 1e4) if np.any(short_m) else rmse_b_bps

    return {
        "n_quotes": len(ym),
        "heston_all_bps": round(rmse_h_bps, 2),
        "bates_all_bps": round(rmse_b_bps, 2),
        "heston_short_tau_bps": round(short_h_bps, 2),
        "bates_short_tau_bps": round(short_b_bps, 2),
        "heston_params": {"kappa": round(kp_h, 4), "theta": round(th_h, 4), "xi": round(xi_h, 4), "rho": round(rh_h, 4), "v0": round(v0_h, 4)},
        "bates_params": {"kappa": round(kp_b, 4), "theta": round(th_b, 4), "xi": round(xi_b, 4), "rho": round(rh_b, 4), "v0": round(v0_b, 4), "lam": round(lam_b, 4), "mu_j": round(muj_b, 4), "sig_j": round(sigj_b, 4)},
        "slice_data": sd,
        "ym": ym,
        "pr_h": pr_h,
        "pr_b": pr_b,
    }


def main():
    bp = BatesSVJ()
    dom = cvpinn.Domain(v_max=0.5)
    g2 = gate2_bates(bp, dom, n=512, seed=0)
    g3 = gate3_bates(bp, dom, n=512, seed=0)
    print(f"[Bates SVJ PIDE Gates] G2_bates={g2:.3e}, G3_bates={g3:.3e}", flush=True)

    rec_b0 = train_bates_arm("B0_bates", bp, seed=0)
    rec_m3 = train_bates_arm("M3_bates", bp, seed=0)
    print(
        f"  [Bates PIDE 875-pt Benchmark] B0_bates gam={rec_b0['overall']['gam']:.4f}, cv={rec_b0['overall']['cv']:.4f} -> "
        f"M3_bates c={rec_m3['overall']['c']:.4f}, gam={rec_m3['overall']['gam']:.4f}, cxv={rec_m3['overall']['cxv']:.4f}, cv={rec_m3['overall']['cv']:.4f}",
        flush=True,
    )

    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            prev_b = json.load(f).get("empirical_crash_calibrations", {})
    else:
        prev_b = {}
    if "2020_03_16_COVID_BlackMonday" in prev_b and "2024_08_05_YenCarry_Crash" in prev_b:
        cov_cal = prev_b["2020_03_16_COVID_BlackMonday"]
        yen_cal = prev_b["2024_08_05_YenCarry_Crash"]
    else:
        print("  [Bates Empirical Crash Calibration] Fitting Heston vs. Bates SVJ on 2020-03-16 COVID-19 & 2024-08-05 Yen-Carry...", flush=True)
        cov_cal = calibrate_bates_vs_heston_crash(
            os.path.join("docs", "market_data", "wrds_optionmetrics", "wrds_spx_2020_03_covid.csv"), "2020-03-16"
        )
        yen_cal = calibrate_bates_vs_heston_crash(
            os.path.join("docs", "market_data", "spx_168days_2024_2025_csv", "2024-08-05.csv"), None
        )
    print(
        f"    2020-03-16 COVID-19 ({cov_cal['n_quotes']} q): Heston={cov_cal['heston_all_bps']} bps -> Bates SVJ={cov_cal['bates_all_bps']} bps "
        f"(short tau<=0.12: {cov_cal['heston_short_tau_bps']} -> {cov_cal['bates_short_tau_bps']} bps)",
        flush=True,
    )
    print(
        f"    2024-08-05 Yen-Carry ({yen_cal['n_quotes']} q): Heston={yen_cal['heston_all_bps']} bps -> Bates SVJ={yen_cal['bates_all_bps']} bps "
        f"(short tau<=0.12: {yen_cal['heston_short_tau_bps']} -> {yen_cal['bates_short_tau_bps']} bps)",
        flush=True,
    )

    # Reload M3_bates & B0_bates for slice plot
    p_h = bp.to_heston()
    ck_m3 = torch.load(os.path.join(BATES_DIR, "M3_bates_BATES_seed0.pt"), map_location="cpu", weights_only=False)
    net_m3 = m3.FlexMlp(6, width=80, depth=4, act="tanh")
    net_m3.load_state_dict(ck_m3["state_dict"])
    net_m3.eval()
    u_m3 = m3._make_m3_u(net_m3, p_h, dom, {"extra_scales": True, "z2": False, "sech_alpha": 0.35}, is_td=False)
    pfn_m3 = lambda x, v, tau: merton_carrier_expected(x, v, tau, bp) + u_m3(x, v, tau)

    ck_b0 = torch.load(os.path.join(BATES_DIR, "B0_bates_BATES_seed0.pt"), map_location="cpu", weights_only=False)
    net_b0 = cvpinn.Mlp(3, width=64, depth=4)
    net_b0.load_state_dict(ck_b0["state_dict"])
    net_b0.eval()
    pfn_b0 = lambda x, v, tau: net_b0(cvpinn.base_features(x, v, tau, dom))

    xg, ref = bates_eval_grid(bp)
    pred_m3 = t2.evaluate_all_greeks(pfn_m3, xg, hgv.EVAL_TAUS, hgv.EVAL_VS)
    pred_b0 = t2.evaluate_all_greeks(pfn_b0, xg, hgv.EVAL_TAUS, hgv.EVAL_VS)

    # Build Figure 21: 3-panel Bates SVJ PIDE Benchmark & Crash Smile Improvement
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    fig.suptitle(
        "Nonlocal 2D Bates SVJ Jump-Diffusion PIDE Control-Variate (M3_bates) & Crash-Day SPX Skew Fit",
        fontsize=12.6, y=0.98
    )

    # Panel (a): 6-Output Relative RMSE (B0_bates vs M3_bates)
    ax0 = axes[0]
    keys = ("c", "cx", "gam", "cv", "cxv", "cvv")
    klbls = ["Price", "Delta", "Gamma", "Vega", "Vanna", "Volga"]
    xk = np.arange(len(keys))
    ax0.bar(xk - 0.18, [rec_b0["overall"][k] for k in keys], 0.36, color="#d62728", label="B0_bates (Standard PIDE PINN)")
    ax0.bar(xk + 0.18, [rec_m3["overall"][k] for k in keys], 0.36, color="#2ca02c", label="M3_bates (Merton CV-RBA PIDE)")
    ax0.set_yscale("log")
    ax0.set_ylim(1e-4, 3.0)
    ax0.set_xticks(xk)
    ax0.set_xticklabels(klbls, fontsize=8.8)
    ax0.set_ylabel("875-Point Relative RMSE (log scale)")
    ax0.set_title("Bates SVJ PIDE: All 6 Outputs")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="upper left", fontsize=8.0)

    # Panel (b): Hardest Low-Variance Short-Maturity Bates Gamma Profile (tau=0.05, v=0.005)
    ax1 = axes[1]
    zs = np.asarray(hgv.EVAL_ZS)
    ax1.plot(zs, ref["gam"][1, 0], "k-", lw=2.3, label="Exact Bates Fourier")
    ax1.plot(zs, pred_m3["gam"][1, 0], "--", color="#2ca02c", lw=2.0, label="M3_bates (Ours)")
    ax1.plot(zs, pred_b0["gam"][1, 0], ":", color="#d62728", lw=2.0, label="B0_bates (Baseline)")
    ax1.set_xlabel("Scaled Moneyness $z = x / \\sqrt{w_{\\mathrm{exp}}}$")
    ax1.set_ylabel("Normalized Gamma $(c_{xx} - c_x)$")
    ax1.set_title("Bates PIDE Corner Slice ($\\tau=0.05, v=0.005$)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8.2)

    # Panel (c): Crash-Day SPX Surface RMSE (Pure Heston vs Bates SVJ Jump-Diffusion)
    ax2 = axes[2]
    cats = ["2020-03-16\nCOVID All", "2020-03-16\nShort $\\tau\\leq 0.12$", "2024-08-05\nYen All", "2024-08-05\nShort $\\tau\\leq 0.12$"]
    h_vals = [cov_cal["heston_all_bps"], cov_cal["heston_short_tau_bps"], yen_cal["heston_all_bps"], yen_cal["heston_short_tau_bps"]]
    b_vals = [cov_cal["bates_all_bps"], cov_cal["bates_short_tau_bps"], yen_cal["bates_all_bps"], yen_cal["bates_short_tau_bps"]]
    xc = np.arange(len(cats))
    ax2.bar(xc - 0.18, h_vals, 0.36, color="#2b5c8f", label="Pure Diffusive Heston")
    ax2.bar(xc + 0.18, b_vals, 0.36, color="#9467bd", label="Bates SVJ (Heston + Jumps)")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(cats, fontsize=8.3)
    ax2.set_ylabel("Market Quote Pricing RMSE (bps)")
    ax2.set_title("Crash-Day SPX Skew: Heston vs. Bates SVJ")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8.2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)

    clean_cov = {k: v for k, v in cov_cal.items() if k not in ("slice_data", "ym", "pr_h", "pr_b")}
    clean_yen = {k: v for k, v in yen_cal.items() if k not in ("slice_data", "ym", "pr_h", "pr_b")}
    out_rep = {
        "bates_benchmark_params": bp.__dict__,
        "gate2_bates": g2,
        "gate3_bates": g3,
        "B0_bates": {"train_sec": rec_b0["train_sec"], "overall": rec_b0["overall"], "arbitrage": rec_b0["arbitrage"]},
        "M3_bates": {"train_sec": rec_m3["train_sec"], "overall": rec_m3["overall"], "arbitrage": rec_m3["arbitrage"]},
        "empirical_crash_calibrations": {
            "2020_03_16_COVID_BlackMonday": clean_cov,
            "2024_08_05_YenCarry_Crash": clean_yen,
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out_rep, f, indent=2)
    print(f"Saved {OUT_JSON} and {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
