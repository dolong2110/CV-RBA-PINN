"""hedging_backtest.py - Dynamic Delta & Delta-Gamma-Vega Hedging Replication Experiment.

Demonstrates the concrete economic value of accurate second-order PINN Greeks under
Feller-violating Heston dynamics (P2: 2*kappa*theta = 0.12 < xi^2 = 0.25):
  - Simulates N_paths exact CIR (Poisson-Gamma non-central chi-square) + Broadie-Kaya
    correlated log-forward trajectories with zero negative-variance truncation bias.
  - Marks the hedged liability and benchmark hedging options at TRUE market prices
    (Exact Lewis-Albrecher Fourier), while computing portfolio hedge ratios from each
    competing model: Exact Fourier, B0, M1, M2_zv_lw, M3_rar_80, and zero-shot 10D M3_param_plus.
  - Evaluates both discrete Delta-only hedging and Delta-Gamma-Vega multi-instrument hedging.
  - Saves runs/hedging_report.json and paper/fig13_dynamic_hedging_pnl.png.
"""
import json
import math
import os
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
import torch

import cvpinn
import cvpinn_m3 as m3
import cvpinn_param as cp8
import cvpinn_tier2 as t2
import heston_grid_vec as hgv
import heston_ref as hr

torch.set_default_dtype(torch.float64)


def simulate_heston_paths(p_tup, f0, v0, T, n_steps, n_paths, seed=42):
    """Exact Poisson-Gamma CIR variance + Broadie-Kaya correlated log-forward simulation."""
    kappa, theta, xi, rho = p_tup
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    ek = math.exp(-kappa * dt)
    c_cir = (xi * xi * (1.0 - ek)) / (4.0 * kappa)
    df = (4.0 * kappa * theta) / (xi * xi)

    F = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    V = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    F[0, :] = f0
    V[0, :] = v0
    logF = np.full(n_paths, math.log(f0), dtype=np.float64)

    rho_comp = math.sqrt(max(1.0 - rho * rho, 0.0))
    for k in range(n_steps):
        vt = V[k]
        lam = (ek / c_cir) * vt
        pois = rng.poisson(0.5 * lam)
        v_next = c_cir * rng.gamma(0.5 * df + pois, 2.0)

        v_bar = 0.5 * (vt + v_next)
        dw_v_xi = v_next - vt - kappa * (theta - v_bar) * dt  # = xi * int sqrt(v) dW_v
        z_perp = rng.standard_normal(n_paths)
        logF = (
            logF
            - 0.5 * v_bar * dt
            + (rho / xi) * dw_v_xi
            + rho_comp * np.sqrt(np.maximum(v_bar * dt, 0.0)) * z_perp
        )
        V[k + 1] = v_next
        F[k + 1] = np.exp(logF)
    return F, V


def build_fourier_slice_splines(p_tup, taus_unique, x_min=-0.6, x_max=0.6, nx=121, v_max=0.45, nv=90):
    """Precomputes cubic splines of Exact Fourier (c, cx, gam, cv) at each discrete tau > 0."""
    xs = np.linspace(x_min, x_max, nx)
    # Cluster v near 0 via quadratic spacing
    vs = np.maximum(v_max * (np.linspace(0.0, 1.0, nv) ** 2), 1e-5)
    splines = {}
    kappa, theta, xi, rho = p_tup

    for tau in taus_unique:
        if tau <= 1e-10:
            continue
        # Evaluate vectorized Lewis-Albrecher Fourier on fixed rectangular (vs, xs) grid
        wide = min(2.0, math.pi / (2.0 * max(abs(x_min), abs(x_max), 1e-3)))
        p_near = min(0.5, wide)
        n_near = int(math.ceil(4.0 / p_near))
        sums = np.zeros((4, nv, nx), dtype=np.float64)
        u_cur, first = 0.0, True
        v_min = float(vs[0])
        while u_cur < 2e5:
            if first:
                first = False
                starts = np.arange(n_near) * p_near
                widths = np.full(n_near, p_near)
                u_cur = n_near * p_near
            else:
                starts = u_cur + np.arange(64) * wide
                widths = np.full(64, wide)
                u_cur += 64 * wide
            u_nodes = (starts[:, None] + 0.5 * widths[:, None] * (hgv.GLX[None, :] + 1.0)).ravel()
            w_nodes = (0.5 * widths[:, None] * hgv.GLW[None, :]).ravel()
            z_cf = u_nodes - 0.5j
            a = kappa - 1j * rho * xi * z_cf
            d = np.sqrt(a * a + (xi * xi) * (z_cf * z_cf + 1j * z_cf))
            g = (a - d) / (a + d)
            e = np.exp(-d * tau)
            big_b = (a - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e)
            big_a = (kappa * theta / (xi * xi)) * ((a - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
            env = np.abs(np.exp(big_a + big_b * v_min)) * (1.0 + np.abs(big_b))
            phi = (np.exp(big_a[:, None] + big_b[:, None] * vs[None, :])[:, :, None]
                   * np.exp(1j * u_nodes[:, None, None] * xs[None, None, :]))
            den = (u_nodes * u_nodes + 0.25)[:, None, None]
            f = (0.5 + 1j * u_nodes)[:, None, None]
            b_arr, w_arr = big_b[:, None, None], w_nodes[:, None, None]
            sums[0] += np.sum(w_arr * (phi / den).real, axis=0)
            sums[1] += np.sum(w_arr * (f * phi / den).real, axis=0)
            sums[2] += np.sum(w_arr * phi.real, axis=0)
            sums[3] += np.sum(w_arr * (b_arr * phi / den).real, axis=0)
            if len(starts) >= 3 and np.max(env[-32:]) < 1e-14:
                break
        pref = np.exp(0.5 * xs)[None, :] / math.pi
        ex = np.exp(xs)[None, :]
        c_mat = ex - pref * sums[0]
        cx_mat = ex - pref * sums[1]
        gam_mat = pref * sums[2]
        cv_mat = -pref * sums[3]
        splines[round(tau, 10)] = {
            "c": RectBivariateSpline(vs, xs, c_mat, kx=3, ky=3),
            "cx": RectBivariateSpline(vs, xs, cx_mat, kx=3, ky=3),
            "gam": RectBivariateSpline(vs, xs, gam_mat, kx=3, ky=3),
            "cv": RectBivariateSpline(vs, xs, cv_mat, kx=3, ky=3),
            "x_bounds": (x_min, x_max),
            "v_bounds": (float(vs[0]), float(vs[-1])),
        }
    return splines


def eval_fourier_sens(splines, F_vec, V_vec, K, tau):
    """Returns dollar (C, Delta_F, Gamma_F, Vega_v) from precomputed Fourier splines."""
    if tau <= 1e-10:
        pay = np.maximum(F_vec - K, 0.0)
        return pay, (F_vec > K).astype(np.float64), np.zeros_like(F_vec), np.zeros_like(F_vec)
    sp = splines[round(tau, 10)]
    x = np.clip(np.log(F_vec / K), sp["x_bounds"][0], sp["x_bounds"][1])
    v = np.clip(V_vec, sp["v_bounds"][0], sp["v_bounds"][1])
    c = sp["c"](v, x, grid=False)
    cx = sp["cx"](v, x, grid=False)
    gam = sp["gam"](v, x, grid=False)
    cv = sp["cv"](v, x, grid=False)
    return K * c, (K / F_vec) * cx, (K / (F_vec ** 2)) * gam, K * cv


def eval_pinn_sens(price_fn, F_vec, V_vec, K, tau, chunk=8192):
    """Returns dollar (C, Delta_F, Gamma_F, Vega_v) from PyTorch autodiff on a PINN price_fn."""
    n = len(F_vec)
    c_out = np.empty(n, dtype=np.float64)
    cx_out = np.empty(n, dtype=np.float64)
    gam_out = np.empty(n, dtype=np.float64)
    cv_out = np.empty(n, dtype=np.float64)
    x_np = np.clip(np.log(F_vec / K), -2.49, 2.49)
    v_np = np.clip(V_vec, 0.0, 0.499)

    for i0 in range(0, n, chunk):
        sl = slice(i0, min(i0 + chunk, n))
        xt = torch.tensor(x_np[sl], requires_grad=True)
        vt = torch.tensor(v_np[sl], requires_grad=True)
        tt = torch.full_like(xt, max(tau, 1e-4))
        c = price_fn(xt, vt, tt)
        cx, cv = torch.autograd.grad(c, (xt, vt), torch.ones_like(c), create_graph=True, allow_unused=True)
        cx = torch.zeros_like(xt) if cx is None else cx
        cv = torch.zeros_like(vt) if cv is None else cv
        if cx.requires_grad:
            cxx = torch.autograd.grad(cx, xt, torch.ones_like(cx), create_graph=False, allow_unused=True)[0]
            cxx = torch.zeros_like(xt) if cxx is None else cxx
        else:
            cxx = torch.zeros_like(xt)
        c_out[sl] = c.detach().numpy()
        cx_out[sl] = cx.detach().numpy()
        gam_out[sl] = (cxx - cx).detach().numpy()
        cv_out[sl] = cv.detach().numpy()

    return K * c_out, (K / F_vec) * cx_out, (K / (F_vec ** 2)) * gam_out, K * cv_out


def load_models(p_tup):
    p = cvpinn.Heston(*p_tup)
    models = {}
    # B0 & M1 (median seed 2 on P2)
    for arm in ("B0", "M1"):
        ckpt = torch.load(f"runs/checkpoints/{arm}_P2_seed2.pt", map_location="cpu", weights_only=True)
        net = cvpinn.Mlp(3 if arm == "B0" else 4)
        net.load_state_dict(ckpt["state_dict"]); net.eval()
        models[arm] = cvpinn.make_price_from_net(arm, net, p, v_max=ckpt["v_max"])
    # M2_zv_lw (seed 0)
    ckpt2 = torch.load("runs/tier2/checkpoints/M2_zv_lw_P2_seed0.pt", map_location="cpu", weights_only=True)
    cfg2 = ckpt2["info"]["config"]
    net2 = t2.build_net_from_cfg(cfg2); net2.load_state_dict(ckpt2["state_dict"]); net2.eval()
    models["M2_zv_lw"] = t2._make_price_from_net_t2(cfg2, net2, p)
    # M3_rar_80 (seed 0)
    ckpt3 = torch.load("runs/m3/checkpoints/M3_rar_80_P2_seed0.pt", map_location="cpu", weights_only=True)
    cfg3 = ckpt3["config"]
    net3 = m3.build_m3_net(cfg3); net3.load_state_dict(ckpt3["state_dict"]); net3.eval()
    models["M3_rar_80"] = m3.make_m3_price_from_net(cfg3, net3, p)
    # Zero-shot M3_param_plus
    ckptp = torch.load("runs/param/checkpoints/M3_param_plus_seed0.pt", map_location="cpu", weights_only=True)
    netp = cvpinn.Mlp(10, width=int(ckptp["width"]), depth=4)
    netp.load_state_dict(ckptp["state_dict"]); netp.eval()
    up = cp8.make_correction_param(netp, cvpinn.Domain(v_max=0.5), cp8.ParamBox(), extra_scales=True)
    models["M3_param_plus"] = cp8.bind_params(cp8.make_price_param(up, "expected"), p)
    return models


def summarize_pnl(pnl, ref_pnl=None):
    rmse = float(np.sqrt(np.mean(pnl ** 2)))
    sd = float(np.std(pnl))
    mae = float(np.mean(np.abs(pnl)))
    q05 = float(np.percentile(pnl, 5))
    cvar05 = float(np.mean(pnl[pnl <= q05]))
    out = {
        "mean_pnl": float(np.mean(pnl)),
        "rmse_pnl": rmse,
        "std_pnl": sd,
        "mae_pnl": mae,
        "var_95_loss": -q05,
        "cvar_95_loss": -cvar05,
    }
    if ref_pnl is not None:
        out["tracking_rmse_vs_exact_hedge"] = float(np.sqrt(np.mean((pnl - ref_pnl) ** 2)))
    return out


def run_hedging_scenario(p_tup, models, f0=100.0, v0=0.01, T_liab=0.05, T_bench=0.25,
                         K_liab=100.0, K_b1=97.0, K_b2=103.0, n_steps=25, n_paths=15000, seed=42):
    t0 = time.perf_counter()
    F, V = simulate_heston_paths(p_tup, f0, v0, T_liab, n_steps, n_paths, seed=seed)
    dt = T_liab / n_steps

    # Precompute Exact Fourier splines at all required tau slices for liability and benchmarks
    taus_needed = set()
    for k in range(n_steps + 1):
        tk = k * dt
        if T_liab - tk > 1e-10:
            taus_needed.add(round(T_liab - tk, 10))
        taus_needed.add(round(T_bench - tk, 10))
    splines = build_fourier_slice_splines(p_tup, sorted(taus_needed))

    # Verify martingale and unhedged liability MC convergence
    c0_exact = hr.heston(math.log(f0 / K_liab), v0, T_liab, p_tup)["c"] * K_liab
    payoff_T = np.maximum(F[-1] - K_liab, 0.0)
    unhedged_pnl = c0_exact - payoff_T

    # Market prices of benchmark options along all paths at all steps k=0..n_steps
    b1_mkt = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    b2_mkt = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    for k in range(n_steps + 1):
        tk = k * dt
        b1_mkt[k] = eval_fourier_sens(splines, F[k], V[k], K_b1, T_bench - tk)[0]
        b2_mkt[k] = eval_fourier_sens(splines, F[k], V[k], K_b2, T_bench - tk)[0]

    all_names = ["Exact Fourier"] + list(models.keys())
    delta_pnls = {name: np.zeros(n_paths, dtype=np.float64) for name in all_names}
    dgv_pnls = {name: np.zeros(n_paths, dtype=np.float64) for name in all_names}

    for name in all_names:
        pnl_d = np.full(n_paths, c0_exact, dtype=np.float64)
        pnl_dgv = np.full(n_paths, c0_exact, dtype=np.float64)

        for k in range(n_steps):
            tk = k * dt
            tau_l = T_liab - tk
            tau_b = T_bench - tk
            fk, vk = F[k], V[k]
            df_step = F[k + 1] - fk
            db1_step = b1_mkt[k + 1] - b1_mkt[k]
            db2_step = b2_mkt[k + 1] - b2_mkt[k]

            if name == "Exact Fourier":
                _, d0, g0, v0_s = eval_fourier_sens(splines, fk, vk, K_liab, tau_l)
                _, d1, g1, v1_s = eval_fourier_sens(splines, fk, vk, K_b1, tau_b)
                _, d2, g2, v2_s = eval_fourier_sens(splines, fk, vk, K_b2, tau_b)
            else:
                pfn = models[name]
                _, d0, g0, v0_s = eval_pinn_sens(pfn, fk, vk, K_liab, tau_l)
                _, d1, g1, v1_s = eval_pinn_sens(pfn, fk, vk, K_b1, tau_b)
                _, d2, g2, v2_s = eval_pinn_sens(pfn, fk, vk, K_b2, tau_b)

            # 1. Delta-only hedge
            pnl_d += d0 * df_step

            # 2. Delta-Gamma-Vega 2x2 damped least-squares solve per path:
            #    [g1, g2; v1, v2] [w1; w2] = [g0; v0]
            # Scale Gamma by F^2/100 (=100) so Gamma and Vega rows have comparable O(10) units
            sc_g = 100.0
            a11, a12 = sc_g * g1, sc_g * g2
            a21, a22 = v1_s, v2_s
            y1, y2 = sc_g * g0, v0_s
            # 2x2 normal equations (A^T A + reg I) w = A^T y with mild reg=1e-2 for B0 stability
            reg = 1e-2
            m11 = a11 * a11 + a21 * a21 + reg
            m12 = a11 * a12 + a21 * a22
            m22 = a12 * a12 + a22 * a22 + reg
            rhs1 = a11 * y1 + a21 * y2
            rhs2 = a12 * y1 + a22 * y2
            det = m11 * m22 - m12 * m12
            w1 = np.clip((m22 * rhs1 - m12 * rhs2) / det, -5.0, 5.0)
            w2 = np.clip((-m12 * rhs1 + m11 * rhs2) / det, -5.0, 5.0)
            phi_f = d0 - w1 * d1 - w2 * d2

            pnl_dgv += phi_f * df_step + w1 * db1_step + w2 * db2_step

        pnl_d -= payoff_T
        pnl_dgv -= payoff_T
        delta_pnls[name] = pnl_d
        dgv_pnls[name] = pnl_dgv

    elapsed = time.perf_counter() - t0
    res = {
        "scenario": {"f0": f0, "v0": v0, "K_liab": K_liab, "T_liab": T_liab, "T_bench": T_bench, "n_steps": n_steps, "n_paths": n_paths},
        "mc_checks": {
            "exact_c0": c0_exact,
            "mc_payoff_mean": float(np.mean(payoff_T)),
            "mc_payoff_se": float(np.std(payoff_T) / math.sqrt(n_paths)),
            "mc_F_T_mean": float(np.mean(F[-1])),
        },
        "elapsed_sec": round(elapsed, 2),
        "unhedged": summarize_pnl(unhedged_pnl),
        "delta_hedge": {k: summarize_pnl(v, delta_pnls["Exact Fourier"]) for k, v in delta_pnls.items()},
        "dgv_hedge": {k: summarize_pnl(v, dgv_pnls["Exact Fourier"]) for k, v in dgv_pnls.items()},
    }
    return res, delta_pnls, dgv_pnls


def plot_hedging(res_low, dgv_low, res_mod):
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
    order = ["B0", "M1", "M2_zv_lw", "M3_param_plus", "M3_rar_80", "Exact Fourier"]
    colors = {
        "B0": "#d62728", "M1": "#1f77b4", "M2_zv_lw": "#ff7f0e",
        "M3_param_plus": "#9467bd", "M3_rar_80": "#2ca02c", "Exact Fourier": "#222222",
    }

    # Panel 1: PnL density histogram on low-variance P2 (v0=0.01) for Delta-Gamma-Vega hedge
    ax = axes[0]
    bins = np.linspace(-0.8, 0.8, 81)
    for k in ("B0", "M1", "M3_rar_80", "Exact Fourier"):
        ls = "--" if k == "Exact Fourier" else "-"
        lw = 2.2 if k in ("M3_rar_80", "Exact Fourier") else 1.6
        ax.hist(np.clip(dgv_low[k], -0.8, 0.8), bins=bins, density=True, histtype="step", lw=lw, ls=ls, color=colors[k], label=k)
    ax.set_xlabel("Terminal Hedging PnL ($ on $100 notional)")
    ax.set_ylabel("Empirical Density")
    ax.set_title("Delta-Gamma-Vega PnL Distribution ($v_0=0.01, T=0.05$)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.2, frameon=True)

    # Panel 2: Low-variance (v0=0.01) Delta vs Delta-Gamma-Vega RMSE comparison
    ax2 = axes[1]
    x = np.arange(len(order))
    w = 0.36
    d_vals = [res_low["delta_hedge"][k]["rmse_pnl"] for k in order]
    dgv_vals = [res_low["dgv_hedge"][k]["rmse_pnl"] for k in order]
    ax2.bar(x - 0.5 * w, d_vals, width=w, color="#7f7f7f", alpha=0.75, label="Delta-Only Hedge")
    ax2.bar(x + 0.5 * w, dgv_vals, width=w, color=[colors[k] for k in order], alpha=0.92, label="Delta-Gamma-Vega Hedge")
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=22, ha="right", fontsize=8.8)
    ax2.set_ylabel("Replication PnL RMSE ($)")
    ax2.set_title("Low-Variance Corner ($v_0=0.01, T=0.05$): Hedge RMSE")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(fontsize=8.5, frameon=True)

    # Panel 3: Moderate-variance (v0=0.04) Delta vs Delta-Gamma-Vega + CVaR_95
    ax3 = axes[2]
    dgv_m = [res_mod["dgv_hedge"][k]["rmse_pnl"] for k in order]
    cvar_m = [res_mod["dgv_hedge"][k]["cvar_95_loss"] for k in order]
    ax3.bar(x - 0.5 * w, dgv_m, width=w, color="#1f77b4", alpha=0.88, label="DGV PnL RMSE ($)")
    ax3.bar(x + 0.5 * w, cvar_m, width=w, color="#d62728", alpha=0.82, label="DGV 95% CVaR Loss ($)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(order, rotation=22, ha="right", fontsize=8.8)
    ax3.set_ylabel("Dollars ($ per $100 ATM Call)")
    ax3.set_title("Moderate Variance ($v_0=0.04, T=0.05$): RMSE & Tail Risk")
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend(fontsize=8.5, frameon=True)

    fig.suptitle("Economic Value of Second-Order Greek Accuracy: Discrete Hedging Under Feller-Violated Heston (P2)", fontsize=12.2)
    out = os.path.join("paper", "fig13_dynamic_hedging_pnl.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p_tup = hr.P_FELLER_BAD
    models = load_models(p_tup)
    print("Running Scenario 1: Low-variance short-dated ATM call (v0=0.01, T=0.05, 15,000 paths, 25 steps)...")
    res_low, _, dgv_low = run_hedging_scenario(p_tup, models, v0=0.01, T_liab=0.05, n_steps=25, n_paths=15000, seed=42)
    print("Running Scenario 2: Moderate-variance short-dated ATM call (v0=0.04, T=0.05, 15,000 paths, 25 steps)...")
    res_mod, _, _ = run_hedging_scenario(p_tup, models, v0=0.04, T_liab=0.05, n_steps=25, n_paths=15000, seed=108)

    report = {"P2_low_var_v001": res_low, "P2_mod_var_v004": res_mod}
    out_json = os.path.join("runs", "hedging_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out_json}")

    for label, r in (("v0=0.01 (Low Var)", res_low), ("v0=0.04 (Mod Var)", res_mod)):
        mc = r["mc_checks"]
        print(f"\n=== {label} | Exact C0={mc['exact_c0']:.4f}, MC Payoff={mc['mc_payoff_mean']:.4f}±{mc['mc_payoff_se']:.4f}, Unhedged RMSE={r['unhedged']['rmse_pnl']:.4f} ===")
        for k in ("Exact Fourier", "M3_rar_80", "M3_param_plus", "M2_zv_lw", "M1", "B0"):
            dh = r["delta_hedge"][k]
            gh = r["dgv_hedge"][k]
            print(
                f"  {k:14s}: Delta RMSE={dh['rmse_pnl']:.4f} | "
                f"DGV RMSE={gh['rmse_pnl']:.4f} (CVaR95={gh['cvar_95_loss']:.4f}, "
                f"vsExact={gh.get('tracking_rmse_vs_exact_hedge', 0.0):.4f})"
            )

    plot_hedging(res_low, dgv_low, res_mod)


if __name__ == "__main__":
    main()
