"""diff_calibrate.py - Millisecond Differentiable Heston Calibration via 10D M3_param_plus.

Implements exact PyTorch autograd calibration of (kappa, theta, xi, rho, v0) directly
through the amortized neural surrogate M3_param_plus:
  - Batches K=8 multi-start trajectories simultaneously in one vectorized forward/backward pass.
  - Polishes the best start with Strong-Wolfe L-BFGS in unconstrained logit coordinates.
  - Benchmarks speed, parameter recovery, and Day-2 out-of-sample fit against SciPy
    Differential Evolution + Lewis-Albrecher Fourier (spx_w8.calibrate_day1).
  - Exports calibrate_day1_differentiable() for live SPX/SPXW chains in spx_w8.py --diff-cal.
"""
import csv
import json
import math
import os
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import cvpinn
import cvpinn_param as cp8
import heston_ref as hr
import spx_w8

torch.set_default_dtype(torch.float64)
DEFAULT_PARAM_CKPT = os.path.join("runs", "param", "checkpoints", "M3_param_plus_seed0.pt")


def load_surrogate(ckpt_path=DEFAULT_PARAM_CKPT, box=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    arm_base = ckpt.get("arm", "M_param_plus")
    extra = bool(ckpt.get("extra_scales", ("plus" in arm_base) or ("wide" in arm_base) or ("crisis" in arm_base) or ("plus" in ckpt.get("upgraded_arm", ""))))
    width = int(ckpt["width"])
    v_max = float(ckpt.get("v_max", 0.5))
    dom = cvpinn.Domain(v_max=v_max)
    if box is None:
        box = cp8.ParamBox(**ckpt["box_dict"]) if "box_dict" in ckpt else cp8.ParamBox()
    net = cvpinn.Mlp(8 + (2 if extra else 0), width=width, depth=4)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    base_u = cp8.make_correction_param(net, dom, box, kind="expected", use_z=True, extra_scales=extra)
    sa = float(ckpt.get("sech_alpha", 0.0))
    if sa > 0.0:
        def u_fn(x, v, tau, kap, tht, xi, rho):
            w = torch.clamp(cp8.total_variance_param(v, tau, kap, tht, "expected"), min=1e-30)
            z = x / torch.sqrt(w)
            return (1.0 / torch.cosh(torch.clamp(sa * z, -40.0, 40.0))) * base_u(x, v, tau, kap, tht, xi, rho)
    else:
        u_fn = base_u
    p_fn = cp8.make_price_param(u_fn, "expected")
    p_fn._v_max = v_max
    return p_fn, box


def _map_params(raw, box: cp8.ParamBox, v0_bounds=(0.002, 0.25)):
    s = torch.sigmoid(raw)
    kap = box.kappa_min + (box.kappa_max - box.kappa_min) * s[..., 0]
    tht = box.theta_min + (box.theta_max - box.theta_min) * s[..., 1]
    xi = box.xi_min + (box.xi_max - box.xi_min) * s[..., 2]
    rho = box.rho_min + (box.rho_max - box.rho_min) * s[..., 3]
    v0 = v0_bounds[0] + (v0_bounds[1] - v0_bounds[0]) * s[..., 4]
    return kap, tht, xi, rho, v0


def calibrate_day1_differentiable(quotes, ckpt_path=DEFAULT_PARAM_CKPT, n_starts=8,
                                  adam_steps=60, lbfgs_iter=30, seed=0, preloaded=None, v0_bounds=None):
    """Fits (kappa, theta, xi, rho, v0(t1)) by exact PyTorch autograd through M3_param_plus / wide / crisis."""
    price_fn_param, box = load_surrogate(ckpt_path) if preloaded is None else preloaded
    if v0_bounds is None:
        vm = float(getattr(price_fn_param, "_v_max", 0.5))
        v0_bounds = (0.002, round(0.95 * vm, 4)) if vm > 0.5 else (0.002, 0.25)
    t0 = time.perf_counter()

    taus = torch.tensor([q[0] for q in quotes], dtype=torch.float64)
    xs = torch.tensor([q[1] for q in quotes], dtype=torch.float64)
    ym = torch.tensor([q[2] for q in quotes], dtype=torch.float64)
    ws = torch.tensor([1.0 / max(q[3], 1e-4) ** 2 for q in quotes], dtype=torch.float64)
    ws = ws / ws.mean()

    n_q = len(quotes)
    gen = torch.Generator().manual_seed(seed)
    # Stratified logit multi-starts in [-1.8, 1.8]^5 + 2 model-free ATM term-structure warm-starts
    raw_init = 3.6 * torch.rand((n_starts, 5), generator=gen) - 1.8
    if n_starts >= 4 and n_q >= 8:
        def _inv_sig(val, lo, hi):
            p = min(max((val - lo) / max(hi - lo, 1e-12), 0.03), 0.97)
            return math.log(p / (1.0 - p))

        q_sorted_t = sorted(quotes, key=lambda item: item[0])
        t_min, t_max = q_sorted_t[0][0], q_sorted_t[-1][0]
        short_atm = [q for q in quotes if q[0] <= t_min * 1.6 and abs(q[1]) <= 0.06] or q_sorted_t[:5]
        long_atm = [q for q in quotes if q[0] >= max(t_max * 0.6, t_min) and abs(q[1]) <= 0.10] or q_sorted_t[-5:]
        q_s = min(short_atm, key=lambda item: abs(item[1]))
        q_l = min(long_atm, key=lambda item: abs(item[1]))
        v0_hat = (max(q_s[2] - max(math.exp(q_s[1]) - 1.0, 0.0), 0.005)) ** 2 * (2.0 * math.pi / max(q_s[0], 0.01))
        th_hat = (max(q_l[2] - max(math.exp(q_l[1]) - 1.0, 0.0), 0.005)) ** 2 * (2.0 * math.pi / max(q_l[0], 0.02))
        raw_init[0] = torch.tensor([
            _inv_sig(2.2, box.kappa_min, box.kappa_max),
            _inv_sig(th_hat, box.theta_min, box.theta_max),
            _inv_sig(0.45 * box.xi_max, box.xi_min, box.xi_max),
            _inv_sig(-0.72, box.rho_min, box.rho_max),
            _inv_sig(v0_hat, v0_bounds[0], v0_bounds[1]),
        ], dtype=torch.float64)
        raw_init[1] = torch.tensor([
            _inv_sig(4.5, box.kappa_min, box.kappa_max),
            _inv_sig(th_hat, box.theta_min, box.theta_max),
            _inv_sig(0.80 * box.xi_max, box.xi_min, box.xi_max),
            _inv_sig(-0.82, box.rho_min, box.rho_max),
            _inv_sig(v0_hat, v0_bounds[0], v0_bounds[1]),
        ], dtype=torch.float64)
    raw = raw_init.requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=0.12)

    taus_b = taus[None, :].expand(n_starts, n_q).reshape(-1)
    xs_b = xs[None, :].expand(n_starts, n_q).reshape(-1)

    for _ in range(adam_steps):
        kap, tht, xi, rho, v0 = _map_params(raw, box, v0_bounds=v0_bounds)
        pred = price_fn_param(
            xs_b,
            v0[:, None].expand(n_starts, n_q).reshape(-1),
            taus_b,
            kap[:, None].expand(n_starts, n_q).reshape(-1),
            tht[:, None].expand(n_starts, n_q).reshape(-1),
            xi[:, None].expand(n_starts, n_q).reshape(-1),
            rho[:, None].expand(n_starts, n_q).reshape(-1),
        ).reshape(n_starts, n_q)
        losses = (ws[None, :] * (pred - ym[None, :]) ** 2).mean(dim=1)
        tot = losses.sum()
        opt.zero_grad()
        tot.backward()
        opt.step()

    with torch.no_grad():
        kap, tht, xi, rho, v0 = _map_params(raw, box, v0_bounds=v0_bounds)
        pred = price_fn_param(
            xs_b, v0[:, None].expand(n_starts, n_q).reshape(-1), taus_b,
            kap[:, None].expand(n_starts, n_q).reshape(-1),
            tht[:, None].expand(n_starts, n_q).reshape(-1),
            xi[:, None].expand(n_starts, n_q).reshape(-1),
            rho[:, None].expand(n_starts, n_q).reshape(-1),
        ).reshape(n_starts, n_q)
        losses = (ws[None, :] * (pred - ym[None, :]) ** 2).mean(dim=1)
        best_idx = int(torch.argmin(losses).item())

    best_raw = raw[best_idx].detach().clone().requires_grad_(True)
    lbfgs = torch.optim.LBFGS([best_raw], lr=1.0, max_iter=lbfgs_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-11, tolerance_change=1e-14)

    def closure():
        lbfgs.zero_grad()
        kp, th, xv, rh, vv = _map_params(best_raw, box, v0_bounds=v0_bounds)
        pr = price_fn_param(
            xs, vv.expand(n_q), taus, kp.expand(n_q), th.expand(n_q), xv.expand(n_q), rh.expand(n_q)
        )
        loss_1 = (ws * (pr - ym) ** 2).mean()
        loss_1.backward()
        return loss_1

    lbfgs.step(closure)
    cal_sec = time.perf_counter() - t0

    with torch.no_grad():
        kp, th, xv, rh, vv = [float(t.item()) for t in _map_params(best_raw, box, v0_bounds=v0_bounds)]
        pr_surr = price_fn_param(
            xs, torch.full_like(xs, vv), taus,
            torch.full_like(xs, kp), torch.full_like(xs, th),
            torch.full_like(xs, xv), torch.full_like(xs, rh),
        ).numpy()

    slice_data, y_mkt, _ = spx_w8._group_quotes(quotes)
    pr_fourier = spx_w8.heston_prices_by_slice(slice_data, vv, kp, th, xv, rh)
    return {
        "kappa": kp, "theta": th, "xi": xv, "rho": rh, "v0_t1": vv,
        "feller_lhs": 2.0 * kp * th, "feller_rhs": xv ** 2,
        "feller_satisfied": bool(2.0 * kp * th >= xv ** 2),
        "calib_sec": round(cal_sec, 4),
        "surr_vs_mkt_rmse_t1": float(np.sqrt(np.mean((pr_surr - ym.numpy()) ** 2))),
        "fourier_vs_mkt_rmse_t1": float(np.sqrt(np.mean((pr_fourier - y_mkt) ** 2))),
    }


def recalibrate_day2_differentiable(quotes_t2, struct_params, preloaded=None, ckpt_path=DEFAULT_PARAM_CKPT, v0_max=None):
    """1D autograd L-BFGS polish of scalar v0(t2) on Day-2 quotes."""
    price_fn_param, _ = load_surrogate(ckpt_path) if preloaded is None else preloaded
    if v0_max is None:
        vm = float(getattr(price_fn_param, "_v_max", 0.5))
        v0_max = round(0.95 * vm, 4) if vm > 0.5 else 0.25
    v_span = v0_max - 0.002
    t0 = time.perf_counter()
    taus = torch.tensor([q[0] for q in quotes_t2], dtype=torch.float64)
    xs = torch.tensor([q[1] for q in quotes_t2], dtype=torch.float64)
    ym = torch.tensor([q[2] for q in quotes_t2], dtype=torch.float64)
    ws = torch.tensor([1.0 / max(q[3], 1e-4) ** 2 for q in quotes_t2], dtype=torch.float64)
    ws = ws / ws.mean()
    n_q = len(quotes_t2)

    kp = torch.full_like(xs, struct_params["kappa"])
    th = torch.full_like(xs, struct_params["theta"])
    xv = torch.full_like(xs, struct_params["xi"])
    rh = torch.full_like(xs, struct_params["rho"])

    with torch.no_grad():
        g_raw = torch.linspace(-4.5, 4.5, 32, dtype=torch.float64)
        g_vv = 0.002 + v_span * torch.sigmoid(g_raw)
        n_g = len(g_raw)
        pr_g = price_fn_param(
            xs[None, :].expand(n_g, n_q).reshape(-1),
            g_vv[:, None].expand(n_g, n_q).reshape(-1),
            taus[None, :].expand(n_g, n_q).reshape(-1),
            kp[None, :].expand(n_g, n_q).reshape(-1),
            th[None, :].expand(n_g, n_q).reshape(-1),
            xv[None, :].expand(n_g, n_q).reshape(-1),
            rh[None, :].expand(n_g, n_q).reshape(-1),
        ).reshape(n_g, n_q)
        g_losses = (ws[None, :] * (pr_g - ym[None, :]) ** 2).mean(dim=1)
        init_idx = int(torch.argmin(g_losses).item())
        best_r = float(g_raw[init_idx].item())
        best_l = float(g_losses[init_idx].item())

    raw_v = torch.tensor([best_r], dtype=torch.float64, requires_grad=True)
    lbfgs = torch.optim.LBFGS([raw_v], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        nonlocal best_r, best_l
        lbfgs.zero_grad()
        vv = 0.002 + v_span * torch.sigmoid(raw_v[0])
        pr = price_fn_param(xs, vv.expand(n_q), taus, kp, th, xv, rh)
        loss = (ws * (pr - ym) ** 2).mean()
        lv = float(loss.detach().item())
        if math.isfinite(lv) and lv < best_l:
            best_l = lv
            best_r = float(raw_v[0].detach().item())
        loss.backward()
        return loss

    lbfgs.step(closure)
    sec = time.perf_counter() - t0
    v0_t2 = float(0.002 + v_span * (1.0 / (1.0 + math.exp(-best_r))))
    with torch.no_grad():
        pr_surr = price_fn_param(xs, torch.full_like(xs, v0_t2), taus, kp, th, xv, rh).numpy()
    slice_data, y_mkt, _ = spx_w8._group_quotes(quotes_t2)
    pr_fourier = spx_w8.heston_prices_by_slice(slice_data, v0_t2, struct_params["kappa"], struct_params["theta"], struct_params["xi"], struct_params["rho"])
    taus_s = torch.tensor(np.concatenate([np.full(len(xs_i), t_i) for t_i, xs_i in slice_data]), dtype=torch.float64)
    xs_s = torch.tensor(np.concatenate([xs_i for _, xs_i in slice_data]), dtype=torch.float64)
    with torch.no_grad():
        pr_surr_grouped = price_fn_param(
            xs_s, torch.full_like(xs_s, v0_t2), taus_s,
            torch.full_like(xs_s, struct_params["kappa"]),
            torch.full_like(xs_s, struct_params["theta"]),
            torch.full_like(xs_s, struct_params["xi"]),
            torch.full_like(xs_s, struct_params["rho"]),
        ).numpy()
    return {
        "v0_t2": v0_t2,
        "calib_sec": round(sec, 4),
        "surr_vs_mkt_rmse_t2": float(np.sqrt(np.mean((pr_surr - ym.numpy()) ** 2))),
        "fourier_vs_mkt_rmse_t2": float(np.sqrt(np.mean((pr_fourier - y_mkt) ** 2))),
        "surr_vs_fourier_rmse_t2": float(np.sqrt(np.mean((pr_surr_grouped - pr_fourier) ** 2))),
    }


def _make_noisy_chain(path, quote_dt_str, expiries, v0, p_tup, noise_bps=1.5, seed=0):
    kappa, theta, xi, rho = p_tup
    rng = np.random.default_rng(seed)
    rows = []
    for exp_str, root, s0, r, q in expiries:
        settle = "09:30:00" if root == "SPX" else "16:00:00"
        qdt = spx_w8._parse_dt(quote_dt_str).replace(tzinfo=spx_w8.ET)
        edt = spx_w8._parse_dt(f"{exp_str} {settle}").replace(tzinfo=spx_w8.ET)
        tau = (edt - qdt).total_seconds() / (365.25 * 86400.0)
        d_true = math.exp(-r * tau)
        f_true = s0 * math.exp((r - q) * tau)
        strikes = np.linspace(f_true * 0.78, f_true * 1.22, 21)
        xs = np.log(f_true / strikes)
        c_norm = spx_w8.heston_prices_by_slice([(tau, xs)], v0, kappa, theta, xi, rho)
        for idx, (k, cn) in enumerate(zip(strikes, c_norm)):
            # Realistic microstructure perturbation keeping parity intact
            cn_noisy = max(cn + (noise_bps * 1e-4) * rng.standard_normal(), max(math.exp(math.log(f_true / k)) - 1.0, 0.0) + 1e-5)
            c_mid = k * d_true * cn_noisy
            p_mid = c_mid - d_true * (f_true - k)
            for cp, mid in (("C", c_mid), ("P", p_mid)):
                if (cp == "P" and idx <= 1) or (cp == "C" and idx >= len(strikes) - 2):
                    bid, ask = 0.0, max(mid * 1.2, 0.05)
                else:
                    half = max(0.012 * mid, 0.025)
                    bid, ask = max(mid - half, 0.05), mid + half
                rows.append({
                    "quote_datetime": quote_dt_str, "expiration": exp_str, "root": root,
                    "option_type": cp, "strike": f"{k:.2f}", "bid": f"{bid:.4f}", "ask": f"{ask:.4f}",
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    preloaded = load_surrogate()
    # One-step autograd warmup so first benchmarked regime excludes PyTorch graph init
    _ = calibrate_day1_differentiable([(0.1, 0.0, 0.025, 0.001), (0.25, 0.05, 0.07, 0.001)], preloaded=preloaded, n_starts=2, adam_steps=2, lbfgs_iter=2)
    expiries_t1 = [
        ("2026-12-04", "SPXW", 5900.0, 0.045, 0.013),
        ("2026-12-18", "SPX", 5900.0, 0.045, 0.013),
        ("2027-02-19", "SPX", 5900.0, 0.044, 0.013),
        ("2027-05-21", "SPX", 5900.0, 0.043, 0.013),
        ("2027-11-19", "SPX", 5900.0, 0.042, 0.013),
    ]
    expiries_t2 = [
        ("2026-12-04", "SPXW", 5935.0, 0.045, 0.013),
        ("2026-12-18", "SPX", 5935.0, 0.045, 0.013),
        ("2027-02-19", "SPX", 5935.0, 0.044, 0.013),
        ("2027-05-21", "SPX", 5935.0, 0.043, 0.013),
        ("2027-11-19", "SPX", 5935.0, 0.042, 0.013),
    ]
    regimes = {
        "P1_FellerOK": (hr.P_FELLER_OK, 0.035, 0.048),
        "P2_FellerBad": (hr.P_FELLER_BAD, 0.042, 0.060),
        "P3_SteepSkew": ((3.0, 0.025, 0.65, -0.80), 0.030, 0.052),
    }

    report = {}
    with tempfile.TemporaryDirectory() as tmp:
        for idx, (rname, (p_true, v1_true, v2_true)) in enumerate(regimes.items()):
            f1 = os.path.join(tmp, f"{rname}_t1.csv")
            f2 = os.path.join(tmp, f"{rname}_t2.csv")
            _make_noisy_chain(f1, "2026-11-18 14:30:00", expiries_t1, v1_true, p_true, seed=10 + idx)
            _make_noisy_chain(f2, "2026-11-19 14:30:00", expiries_t2, v2_true, p_true, seed=20 + idx)
            _, q1 = spx_w8.process_chain_parity_and_quotes(spx_w8.load_spx_csv(f1))
            _, q2 = spx_w8.process_chain_parity_and_quotes(spx_w8.load_spx_csv(f2))

            t_de0 = time.perf_counter()
            cal_de = spx_w8.calibrate_day1(q1, popsize=8, maxiter=40, seed=idx)
            cal_de["calib_sec"] = round(time.perf_counter() - t_de0, 4)
            t_de2 = time.perf_counter()
            recal_de = spx_w8.recalibrate_day2_v0(q2, cal_de)
            recal_de["calib_sec"] = round(time.perf_counter() - t_de2, 4)

            cal_nn = calibrate_day1_differentiable(q1, preloaded=preloaded, seed=idx)
            recal_nn = recalibrate_day2_differentiable(q2, cal_nn, preloaded=preloaded)

            speedup = cal_de["calib_sec"] / max(cal_nn["calib_sec"], 1e-6)
            report[rname] = {
                "n_quotes_t1": len(q1),
                "n_quotes_t2": len(q2),
                "true_params": {"kappa": p_true[0], "theta": p_true[1], "xi": p_true[2], "rho": p_true[3], "v0_t1": v1_true, "v0_t2": v2_true},
                "fourier_de": {"day1": cal_de, "day2": recal_de},
                "neural_autograd": {"day1": cal_nn, "day2": recal_nn},
                "day1_speedup_x": round(speedup, 2),
            }
            print(
                f"[{rname}] quotes={len(q1)} | DE+Fourier: {cal_de['calib_sec']*1000:.0f}ms "
                f"(t1_rmse={cal_de['fourier_vs_mkt_rmse_t1']:.2e}, t2_rmse={recal_de['fourier_vs_mkt_rmse_t2']:.2e}) | "
                f"Neural Autograd: {cal_nn['calib_sec']*1000:.0f}ms ({speedup:.1f}x faster, "
                f"t1_rmse={cal_nn['fourier_vs_mkt_rmse_t1']:.2e}, t2_rmse={recal_nn['fourier_vs_mkt_rmse_t2']:.2e}, "
                f"v0_t2={recal_nn['v0_t2']:.4f} vs true {v2_true:.4f})"
            )

    out_json = os.path.join("runs", "diff_calibration_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out_json}")

    # Plot calibration speedup & Day-1 / Day-2 fit parity
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), gridspec_kw={"wspace": 0.28})
    names = list(report.keys())
    x = np.arange(len(names))
    w = 0.34
    de_ms = [report[k]["fourier_de"]["day1"]["calib_sec"] * 1000.0 for k in names]
    nn_ms = [report[k]["neural_autograd"]["day1"]["calib_sec"] * 1000.0 for k in names]
    axes[0].bar(x - 0.5 * w, de_ms, width=w, color="#1f77b4", label="SciPy DE + Fourier")
    axes[0].bar(x + 0.5 * w, nn_ms, width=w, color="#2ca02c", label="Differentiable M3_param_plus")
    axes[0].set_yscale("log")
    axes[0].set_ylim(min(nn_ms) * 0.55, max(de_ms) * 3.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=10)
    axes[0].set_ylabel("Day-1 5-Parameter Calibration Time (ms, log scale)")
    axes[0].set_title("Day-1 Calibration Speed (DE vs. Autograd)", fontsize=10.8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", frameon=True, fontsize=8.8)

    de_t2 = [report[k]["fourier_de"]["day2"]["fourier_vs_mkt_rmse_t2"] * 1e4 for k in names]
    nn_t2 = [report[k]["neural_autograd"]["day2"]["fourier_vs_mkt_rmse_t2"] * 1e4 for k in names]
    axes[1].bar(x - 0.5 * w, de_t2, width=w, color="#1f77b4", label="DE + Fourier Fit")
    axes[1].bar(x + 0.5 * w, nn_t2, width=w, color="#2ca02c", label="M3_param_plus Calibrated Fit")
    axes[1].set_ylim(0.0, max(max(de_t2), max(nn_t2)) * 1.28)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=10)
    axes[1].set_ylabel("Out-of-Sample Day-2 Normalized Price RMSE (bps)")
    axes[1].set_title("Out-of-Sample Day-2 Fit (Exact Fourier)", fontsize=10.8)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right", frameon=True, fontsize=8.8)

    fig.suptitle("Differentiable Neural Calibration via 10D M3_param_plus Across SPX-Style Chains", fontsize=12, y=1.02)
    out_fig = os.path.join("paper", "fig15_diff_calibration.png")
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
