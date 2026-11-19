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
    extra = "plus" in arm_base or "plus" in ckpt.get("upgraded_arm", "")
    width = int(ckpt["width"])
    dom = cvpinn.Domain(v_max=0.5)
    box = cp8.ParamBox() if box is None else box
    net = cvpinn.Mlp(8 + (2 if extra else 0), width=width, depth=4)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    u = cp8.make_correction_param(net, dom, box, kind="expected", use_z=True, extra_scales=extra)
    return cp8.make_price_param(u, "expected"), box


def _map_params(raw, box: cp8.ParamBox, v0_bounds=(0.002, 0.25)):
    s = torch.sigmoid(raw)
    kap = box.kappa_min + (box.kappa_max - box.kappa_min) * s[..., 0]
    tht = box.theta_min + (box.theta_max - box.theta_min) * s[..., 1]
    xi = box.xi_min + (box.xi_max - box.xi_min) * s[..., 2]
    rho = box.rho_min + (box.rho_max - box.rho_min) * s[..., 3]
    v0 = v0_bounds[0] + (v0_bounds[1] - v0_bounds[0]) * s[..., 4]
    return kap, tht, xi, rho, v0


def calibrate_day1_differentiable(quotes, ckpt_path=DEFAULT_PARAM_CKPT, n_starts=8,
                                  adam_steps=60, lbfgs_iter=30, seed=0, preloaded=None):
    """Fits (kappa, theta, xi, rho, v0(t1)) by exact PyTorch autograd through M3_param_plus."""
    price_fn_param, box = load_surrogate(ckpt_path) if preloaded is None else preloaded
    t0 = time.perf_counter()

    taus = torch.tensor([q[0] for q in quotes], dtype=torch.float64)
    xs = torch.tensor([q[1] for q in quotes], dtype=torch.float64)
    ym = torch.tensor([q[2] for q in quotes], dtype=torch.float64)
    ws = torch.tensor([1.0 / max(q[3], 1e-4) ** 2 for q in quotes], dtype=torch.float64)
    ws = ws / ws.mean()

    n_q = len(quotes)
    gen = torch.Generator().manual_seed(seed)
    # Stratified logit multi-starts in [-1.8, 1.8]^5
    raw = (3.6 * torch.rand((n_starts, 5), generator=gen) - 1.8).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=0.12)

    taus_b = taus[None, :].expand(n_starts, n_q).reshape(-1)
    xs_b = xs[None, :].expand(n_starts, n_q).reshape(-1)

    for _ in range(adam_steps):
        kap, tht, xi, rho, v0 = _map_params(raw, box)
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
        kap, tht, xi, rho, v0 = _map_params(raw, box)
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
        kp, th, xv, rh, vv = _map_params(best_raw, box)
        pr = price_fn_param(
            xs, vv.expand(n_q), taus, kp.expand(n_q), th.expand(n_q), xv.expand(n_q), rh.expand(n_q)
        )
        loss_1 = (ws * (pr - ym) ** 2).mean()
        loss_1.backward()
        return loss_1

    lbfgs.step(closure)
    cal_sec = time.perf_counter() - t0

    with torch.no_grad():
        kp, th, xv, rh, vv = [float(t.item()) for t in _map_params(best_raw, box)]
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


def recalibrate_day2_differentiable(quotes_t2, struct_params, preloaded=None, ckpt_path=DEFAULT_PARAM_CKPT):
    """1D autograd L-BFGS polish of scalar v0(t2) on Day-2 quotes."""
    price_fn_param, _ = load_surrogate(ckpt_path) if preloaded is None else preloaded
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

    raw_v = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    lbfgs = torch.optim.LBFGS([raw_v], lr=1.0, max_iter=25, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        vv = 0.002 + 0.248 * torch.sigmoid(raw_v[0])
        pr = price_fn_param(xs, vv.expand(n_q), taus, kp, th, xv, rh)
        loss = (ws * (pr - ym) ** 2).mean()
        loss.backward()
        return loss

    lbfgs.step(closure)
    sec = time.perf_counter() - t0
    v0_t2 = float((0.002 + 0.248 * torch.sigmoid(raw_v[0])).detach().item())
    with torch.no_grad():
        pr_surr = price_fn_param(xs, torch.full_like(xs, v0_t2), taus, kp, th, xv, rh).numpy()
    slice_data, y_mkt, _ = spx_w8._group_quotes(quotes_t2)
    pr_fourier = spx_w8.heston_prices_by_slice(slice_data, v0_t2, struct_params["kappa"], struct_params["theta"], struct_params["xi"], struct_params["rho"])
    return {
        "v0_t2": v0_t2,
        "calib_sec": round(sec, 4),
        "surr_vs_mkt_rmse_t2": float(np.sqrt(np.mean((pr_surr - ym.numpy()) ** 2))),
        "fourier_vs_mkt_rmse_t2": float(np.sqrt(np.mean((pr_fourier - y_mkt) ** 2))),
        "surr_vs_fourier_rmse_t2": float(np.sqrt(np.mean((pr_surr - np.array([q[2] for q in quotes_t2])) ** 2))),
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
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    names = list(report.keys())
    x = np.arange(len(names))
    w = 0.34
    de_ms = [report[k]["fourier_de"]["day1"]["calib_sec"] * 1000.0 for k in names]
    nn_ms = [report[k]["neural_autograd"]["day1"]["calib_sec"] * 1000.0 for k in names]
    axes[0].bar(x - 0.5 * w, de_ms, width=w, color="#1f77b4", label="SciPy DE + Fourier")
    axes[0].bar(x + 0.5 * w, nn_ms, width=w, color="#2ca02c", label="Differentiable M3_param_plus")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=12)
    axes[0].set_ylabel("Day-1 5-Parameter Calibration Time (ms, log scale)")
    axes[0].set_title("Calibration Speed: Global DE+Fourier vs. Batched Autograd")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=True)

    de_t2 = [report[k]["fourier_de"]["day2"]["fourier_vs_mkt_rmse_t2"] * 1e4 for k in names]
    nn_t2 = [report[k]["neural_autograd"]["day2"]["fourier_vs_mkt_rmse_t2"] * 1e4 for k in names]
    axes[1].bar(x - 0.5 * w, de_t2, width=w, color="#1f77b4", label="DE + Fourier Fit")
    axes[1].bar(x + 0.5 * w, nn_t2, width=w, color="#2ca02c", label="M3_param_plus Calibrated Fit")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=12)
    axes[1].set_ylabel("Out-of-Sample Day-2 Normalized Price RMSE (bps)")
    axes[1].set_title("Out-of-Sample Day-2 Pricing Accuracy (Evaluated by Exact Fourier)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=True)

    fig.suptitle("Differentiable Neural Calibration via 10D M3_param_plus Across SPX-Style Chains", fontsize=12)
    out_fig = os.path.join("paper", "fig15_diff_calibration.png")
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
