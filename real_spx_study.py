"""real_spx_study.py - Empirical Live Cboe SPX / SPXW Option Surface Validation (Nov 19, 2026).

Executes a full empirical study on live delayed Cboe SPX & SPXW option chains
(docs/market_data/spx_cboe_2026_11_19.csv, ^SPX ~ 7,205.56, 8 expiries, 2,580 OTM quotes):
  1. Model-free put-call parity extraction of D(tau) and F(tau) across AM SPX & PM SPXW.
  2. Cross-maturity holdout calibration (5 in-sample expiries -> 3 held-out unseen expiries,
     including PM-settled SPXW), comparing Global SciPy DE + Fourier vs. Millisecond
     Differentiable PyTorch Autograd via 10D M3_param_plus.
  3. Full-chain Heston calibration and training of B0 vs. Championship M3_sech_80 on the
     live calibrated SPX Feller-violating parameter regime, auditing live 2,580-quote fit
     plus all 6 Greeks (Price, Delta, Gamma, Vanna, Vega, Volga) and no-arbitrage admissibility.
  4. Saves runs/real_spx_report.json and paper/fig16_real_spx_market.png.
"""
from concurrent.futures import ProcessPoolExecutor
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
import numpy as np
from scipy.optimize import brentq
import torch

import cvpinn
import cvpinn_m3 as m3
import cvpinn_param as cp8
import cvpinn_tier2 as t2
import diff_calibrate as dc
import heston_grid_vec as hgv
import heston_ref as hr
import spx_w8

torch.set_default_dtype(torch.float64)
CSV_PATH = os.path.join("docs", "market_data", "spx_cboe_2026_11_19.csv")
OUT_JSON = os.path.join("runs", "real_spx_report.json")
OUT_FIG = os.path.join("paper", "fig16_real_spx_market.png")
CKPT_DIR = os.path.join("runs", "spx_live")


def implied_vol_norm(c_norm, x, tau):
    """Inverts normalized call price c_norm = b(x, sigma^2 * tau) for Black-Scholes IV sigma."""
    intr = max(math.exp(x) - 1.0, 0.0)
    if c_norm <= intr + 1e-9:
        return float("nan")
    try:
        return brentq(lambda sig: hr.bs(x, (sig * sig) * tau) - c_norm, 0.02, 2.50, xtol=1e-7)
    except ValueError:
        return float("nan")


def eval_quotes_metrics(quotes, p_tup, v0, pinn_fn=None):
    """Evaluates Fourier (and optional PINN) normalized price RMSE & IV RMSE on a quote list."""
    slice_data, y_mkt, _ = spx_w8._group_quotes(quotes)
    c_four = spx_w8.heston_prices_by_slice(slice_data, v0, *p_tup)
    taus_flat = np.concatenate([np.full(len(xs), tau) for tau, xs in slice_data])
    xs_flat = np.concatenate([xs for _, xs in slice_data])

    iv_mkt, iv_four = [], []
    for c_m, c_f, x_i, t_i in zip(y_mkt, c_four, xs_flat, taus_flat):
        if abs(x_i) <= 0.35:  # Liquid smile core |x| <= 0.35 for clean IV comparison
            im = implied_vol_norm(float(c_m), float(x_i), float(t_i))
            if_v = implied_vol_norm(float(c_f), float(x_i), float(t_i))
            if math.isfinite(im) and math.isfinite(if_v):
                iv_mkt.append(im)
                iv_four.append(if_v)
    iv_mkt = np.array(iv_mkt)
    iv_four = np.array(iv_four)
    out = {
        "n_quotes": len(y_mkt),
        "fourier_vs_mkt_rmse_bps": float(np.sqrt(np.mean((c_four - y_mkt) ** 2)) * 1e4),
        "fourier_iv_rmse_volpts": float(np.sqrt(np.mean((iv_four - iv_mkt) ** 2)) * 100.0) if len(iv_mkt) else float("nan"),
    }
    if pinn_fn is not None:
        # Clamp |x| <= 2.49 to domain
        xs_c = np.clip(xs_flat, -2.49, 2.49)
        with torch.no_grad():
            c_pinn = pinn_fn(
                torch.tensor(xs_c, dtype=torch.float64),
                torch.full((len(xs_c),), float(v0), dtype=torch.float64),
                torch.tensor(taus_flat, dtype=torch.float64),
            ).numpy()
        mask_dom = np.abs(xs_flat) <= 2.45
        out["pinn_vs_fourier_rmse_bps"] = float(np.sqrt(np.mean((c_pinn[mask_dom] - c_four[mask_dom]) ** 2)) * 1e4)
        out["pinn_vs_mkt_rmse_bps"] = float(np.sqrt(np.mean((c_pinn[mask_dom] - y_mkt[mask_dom]) ** 2)) * 1e4)
    return out


def _train_spx_worker(spec):
    arm, p_tup, seed = spec
    os.makedirs(CKPT_DIR, exist_ok=True)
    cpath = os.path.join(CKPT_DIR, f"{arm}_SPX_seed{seed}.pt")
    jpath = os.path.join(CKPT_DIR, f"{arm}_SPX_seed{seed}.json")
    if os.path.exists(cpath) and os.path.exists(jpath):
        with open(jpath) as f:
            return json.load(f)
    p = cvpinn.Heston(*p_tup)
    x_grid, ref = hgv.heston_eval_grid(p_tup)
    t0 = time.perf_counter()
    if arm == "B0":
        pfn, net, info = cvpinn.build_and_train("B0", p, seed, v_max=0.5, term_weight=100.0, return_info=True)
        cfg_save = None
    else:
        pfn, net, info = m3.train_m3("M3_sech_80", p, seed, v_max=0.5)
        cfg_save = info["config"]
    train_sec = time.perf_counter() - t0
    t_i0 = time.perf_counter()
    pred = t2.evaluate_all_greeks(pfn, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)
    infer_ms = 1000.0 * (time.perf_counter() - t_i0)
    metrics = t2.score_all_six_buckets(pred, ref)
    arb, dist = m3.audit_arbitrage_and_percentiles(pred, ref, x_grid)
    torch.save({"state_dict": net.state_dict(), "arm": arm, "p_tup": p_tup, "config": cfg_save}, cpath)
    rec = {
        "arm": arm, "seed": seed, "train_sec": round(train_sec, 3), "infer_ms_875pt": round(infer_ms, 3),
        "metrics": metrics, "arbitrage": arb, "percentiles": dist,
    }
    with open(jpath, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    rows = spx_w8.load_spx_csv(CSV_PATH)
    slices, quotes_all = spx_w8.process_chain_parity_and_quotes(rows)
    # Filter to |x| <= 0.80 liquid market calibration band where market quotes carry true option vega
    quotes_cal_band = [q for q in quotes_all if abs(q[1]) <= 0.80]
    print(f"Loaded {len(rows)} QuoteRows across {len(slices)} parity slices -> {len(quotes_all)} total OTM quotes ({len(quotes_cal_band)} in |x|<=0.80)")

    # Holdout split: 5 in-sample expiries vs 3 out-of-sample unseen expiries (Jan 2027, Mar 2027, Jun 30 2027 SPXW)
    holdout_taus = {round(slices[1]["tau"], 6), round(slices[3]["tau"], 6), round(slices[5]["tau"], 6)}
    q_in = [q for q in quotes_cal_band if round(q[0], 6) not in holdout_taus]
    q_out = [q for q in quotes_cal_band if round(q[0], 6) in holdout_taus]
    print(f"Cross-maturity split: In-sample={len(q_in)} quotes (5 expiries) | Out-of-sample={len(q_out)} quotes (3 unseen expiries)")

    # Warmup & run Differentiable M3_param_plus vs SciPy DE+Fourier on In-Sample -> Out-of-Sample
    preloaded = dc.load_surrogate()
    _ = dc.calibrate_day1_differentiable(q_in[:20], preloaded=preloaded, n_starts=2, adam_steps=2, lbfgs_iter=2)

    t_de0 = time.perf_counter()
    cal_de_split = spx_w8.calibrate_day1(q_in, popsize=8, maxiter=40, seed=0)
    cal_de_split["calib_sec"] = round(time.perf_counter() - t_de0, 4)

    cal_nn_split = dc.calibrate_day1_differentiable(q_in, preloaded=preloaded, seed=0)
    speedup_split = cal_de_split["calib_sec"] / max(cal_nn_split["calib_sec"], 1e-6)

    p_de_s = (cal_de_split["kappa"], cal_de_split["theta"], cal_de_split["xi"], cal_de_split["rho"])
    p_nn_s = (cal_nn_split["kappa"], cal_nn_split["theta"], cal_nn_split["xi"], cal_nn_split["rho"])
    eval_de_in = eval_quotes_metrics(q_in, p_de_s, cal_de_split["v0_t1"])
    eval_de_out = eval_quotes_metrics(q_out, p_de_s, cal_de_split["v0_t1"])
    eval_nn_in = eval_quotes_metrics(q_in, p_nn_s, cal_nn_split["v0_t1"])
    eval_nn_out = eval_quotes_metrics(q_out, p_nn_s, cal_nn_split["v0_t1"])
    print(
        f"[Holdout Split] DE+Fourier: {cal_de_split['calib_sec']*1000:.0f}ms (in={eval_de_in['fourier_vs_mkt_rmse_bps']:.1f}bps, "
        f"out={eval_de_out['fourier_vs_mkt_rmse_bps']:.1f}bps / {eval_de_out['fourier_iv_rmse_volpts']:.2f} vol pts) | "
        f"Neural Autograd: {cal_nn_split['calib_sec']*1000:.0f}ms ({speedup_split:.1f}x faster, "
        f"in={eval_nn_in['fourier_vs_mkt_rmse_bps']:.1f}bps, out={eval_nn_out['fourier_vs_mkt_rmse_bps']:.1f}bps / {eval_nn_out['fourier_iv_rmse_volpts']:.2f} vol pts)"
    )

    # Full-surface live SPX calibration
    t_f0 = time.perf_counter()
    cal_full_de = spx_w8.calibrate_day1(quotes_cal_band, popsize=10, maxiter=45, seed=0)
    cal_full_de["calib_sec"] = round(time.perf_counter() - t_f0, 4)
    cal_full_nn = dc.calibrate_day1_differentiable(quotes_cal_band, preloaded=preloaded, seed=0)
    speedup_full = cal_full_de["calib_sec"] / max(cal_full_nn["calib_sec"], 1e-6)
    print(
        f"[Full Live SPX] DE+Fourier ({cal_full_de['calib_sec']*1000:.0f}ms): kappa={cal_full_de['kappa']:.3f}, "
        f"theta={cal_full_de['theta']:.4f}, xi={cal_full_de['xi']:.3f}, rho={cal_full_de['rho']:.3f}, v0={cal_full_de['v0_t1']:.4f} "
        f"(2*k*th={cal_full_de['feller_lhs']:.4f} vs xi^2={cal_full_de['feller_rhs']:.4f}) | "
        f"Neural Autograd ({cal_full_nn['calib_sec']*1000:.0f}ms, {speedup_full:.1f}x faster)"
    )

    p_full = (cal_full_de["kappa"], cal_full_de["theta"], cal_full_de["xi"], cal_full_de["rho"])
    v0_full = cal_full_de["v0_t1"]

    # Train B0 and M3_sech_80 in parallel on the exact live calibrated SPX regime
    print("Training B0 and M3_sech_80 on exact live calibrated SPX Heston regime...")
    with ProcessPoolExecutor(max_workers=2) as ex:
        rec_b0, rec_m3 = list(ex.map(_train_spx_worker, [("B0", p_full, 0), ("M3_sech_80", p_full, 0)]))

    # Reload trained B0 & M3_sech_80 + evaluate on live Cboe SPX quotes
    p_obj = cvpinn.Heston(*p_full)
    cb0 = torch.load(os.path.join(CKPT_DIR, "B0_SPX_seed0.pt"), map_location="cpu", weights_only=True)
    nb0 = cvpinn.Mlp(3); nb0.load_state_dict(cb0["state_dict"]); nb0.eval()
    pfn_b0 = cvpinn.make_price_from_net("B0", nb0, p_obj, v_max=0.5)

    cm3 = torch.load(os.path.join(CKPT_DIR, "M3_sech_80_SPX_seed0.pt"), map_location="cpu", weights_only=True)
    nm3 = m3.build_m3_net(cm3["config"]); nm3.load_state_dict(cm3["state_dict"]); nm3.eval()
    pfn_m3 = m3.make_m3_price_from_net(cm3["config"], nm3, p_obj, v_max=0.5)

    live_fit_b0 = eval_quotes_metrics(quotes_cal_band, p_full, v0_full, pinn_fn=pfn_b0)
    live_fit_m3 = eval_quotes_metrics(quotes_cal_band, p_full, v0_full, pinn_fn=pfn_m3)

    # Also evaluate zero-shot M3_param_plus on its own calibrated in-box SPX parameters
    pfn_param_nn = cp8.bind_params(preloaded[0], cvpinn.Heston(*p_nn_s))
    x_grid_nn, ref_nn = hgv.heston_eval_grid(p_nn_s)
    pred_param_nn = t2.evaluate_all_greeks(pfn_param_nn, x_grid_nn, hgv.EVAL_TAUS, hgv.EVAL_VS)
    metrics_param_nn = t2.score_all_six_buckets(pred_param_nn, ref_nn)
    live_fit_param_nn = eval_quotes_metrics(quotes_cal_band, p_nn_s, cal_full_nn["v0_t1"], pinn_fn=pfn_param_nn)

    report = {
        "snapshot_metadata": {
            "source_csv": CSV_PATH,
            "quote_date": "2026-11-19 (Intraday US Regular Hours, Cboe Delayed)",
            "spot_approx": 7205.56,
            "n_raw_rows": len(rows),
            "n_slices": len(slices),
            "n_otm_quotes_total": len(quotes_all),
            "n_otm_quotes_cal_band": len(quotes_cal_band),
        },
        "parity_slices": [
            {
                "expiry": s["expiry"], "series": s["series"], "tau": round(s["tau"], 6),
                "D": round(s["D"], 6), "F": round(s["F"], 2),
                "implied_r_pct": round(-100.0 * math.log(s["D"]) / s["tau"], 3),
                "r2": round(s["r2"], 8),
                "in_holdout_test": round(s["tau"], 6) in holdout_taus,
            }
            for s in slices
        ],
        "cross_maturity_holdout": {
            "de_fourier": {"calibration": cal_de_split, "in_sample": eval_de_in, "out_of_sample": eval_de_out},
            "neural_autograd": {"calibration": cal_nn_split, "in_sample": eval_nn_in, "out_of_sample": eval_nn_out},
            "speedup_x": round(speedup_split, 2),
        },
        "full_surface_calibration": {
            "de_fourier": cal_full_de,
            "neural_autograd": cal_full_nn,
            "speedup_x": round(speedup_full, 2),
        },
        "live_spx_pinn_benchmark": {
            "B0": {"grid_875": rec_b0, "live_quotes_fit": live_fit_b0},
            "M3_sech_80": {"grid_875": rec_m3, "live_quotes_fit": live_fit_m3},
            "M3_param_plus_zeroshot": {
                "grid_875_overall": metrics_param_nn["overall"],
                "live_quotes_fit": live_fit_param_nn,
            },
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {OUT_JSON}")
    plot_from_saved()


def plot_from_saved():
    with open(OUT_JSON) as f:
        report = json.load(f)
    rows = spx_w8.load_spx_csv(CSV_PATH)
    slices, quotes_all = spx_w8.process_chain_parity_and_quotes(rows)
    quotes_cal_band = [q for q in quotes_all if abs(q[1]) <= 0.80]
    holdout_taus = {round(slices[1]["tau"], 6), round(slices[3]["tau"], 6), round(slices[5]["tau"], 6)}

    cal_full_de = report["full_surface_calibration"]["de_fourier"]
    p_full = (cal_full_de["kappa"], cal_full_de["theta"], cal_full_de["xi"], cal_full_de["rho"])
    v0_full = cal_full_de["v0_t1"]
    p_obj = cvpinn.Heston(*p_full)

    cm3 = torch.load(os.path.join(CKPT_DIR, "M3_sech_80_SPX_seed0.pt"), map_location="cpu", weights_only=True)
    nm3 = m3.build_m3_net(cm3["config"]); nm3.load_state_dict(cm3["state_dict"]); nm3.eval()
    pfn_m3 = m3.make_m3_price_from_net(cm3["config"], nm3, p_obj, v_max=0.5)

    rec_b0 = report["live_spx_pinn_benchmark"]["B0"]["grid_875"]
    rec_m3 = report["live_spx_pinn_benchmark"]["M3_sech_80"]["grid_875"]
    mp_ov = report["live_spx_pinn_benchmark"]["M3_param_plus_zeroshot"]["grid_875_overall"]

    # Plot 3-panel publication figure paper/fig16_real_spx_market.png
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), gridspec_kw={"wspace": 0.28})

    # Panel 1: Implied Forward F(tau) from Put-Call Parity
    ax1 = axes[0]
    taus_s = [s["tau"] for s in slices]
    fs_s = [s["F"] for s in slices]
    cols_s = ["#d62728" if round(s["tau"], 6) in holdout_taus else "#1f77b4" for s in slices]
    ax1.scatter(taus_s, fs_s, c=cols_s, s=55, zorder=4)
    ax1.plot(taus_s, fs_s, color="#1f77b4", alpha=0.5, lw=1.6)
    ax1.scatter([], [], c="#1f77b4", s=45, label="In-Sample SPX Expiry (5)")
    ax1.scatter([], [], c="#d62728", s=45, label="Held-Out SPX/SPXW Expiry (3)")
    ax1.set_xlabel("Time to Maturity $\\tau$ (years)")
    ax1.set_ylabel("Parity-Implied Forward $F(\\tau)$ (Index Pts)")
    ax1.set_title("Live Cboe SPX/SPXW Parity ($R^2 \\geq 0.9999995$)", fontsize=10.4)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8.2, frameon=True, loc="upper left")

    # Panel 2: Live SPX Market IV Smiles vs Calibrated M3_sech_80 (In-Sample Dec 26 & Held-Out Mar 27)
    ax2 = axes[1]
    for s_idx, col, lbl_tag in ((0, "#1f77b4", "Dec 2026 (In)"), (3, "#2ca02c", "Mar 2027 (Held-Out)")):
        tau_i = slices[s_idx]["tau"]
        q_sl = sorted([q for q in quotes_cal_band if abs(q[0] - tau_i) < 1e-6 and abs(q[1]) <= 0.30], key=lambda z: z[1])
        xs_i = np.array([-q[1] for q in q_sl])
        iv_m = np.array([implied_vol_norm(q[2], q[1], tau_i) * 100.0 for q in q_sl])
        with torch.no_grad():
            c_m3 = pfn_m3(
                torch.tensor([q[1] for q in q_sl], dtype=torch.float64),
                torch.full((len(q_sl),), float(v0_full), dtype=torch.float64),
                torch.full((len(q_sl),), float(tau_i), dtype=torch.float64),
            ).numpy()
        iv_p = np.array([implied_vol_norm(float(c_v), q[1], tau_i) * 100.0 for c_v, q in zip(c_m3, q_sl)])
        ok = np.isfinite(iv_m) & np.isfinite(iv_p)
        ax2.scatter(xs_i[ok][::2], iv_m[ok][::2], s=18, color=col, alpha=0.65, label=f"Cboe Mid {lbl_tag}")
        ax2.plot(xs_i[ok], iv_p[ok], color=col, lw=2.1, label=f"M3_sech_80 {lbl_tag}")
    ax2.set_xlabel("Log-Strike Moneyness $\\ln(K/F)$")
    ax2.set_ylabel("Black-Scholes Implied Volatility (%)")
    ax2.set_title("Live Cboe SPX Smiles vs. M3_sech_80", fontsize=10.4)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7.9, frameon=True, loc="upper right")

    # Panel 3: Greek Accuracy on Live Calibrated SPX Regime (B0 vs Zero-Shot M3_param_plus vs M3_sech_80)
    ax3 = axes[2]
    g_keys = [("c", "Price"), ("gam", "Gamma"), ("cxv", "Vanna"), ("cv", "Vega"), ("cvv", "Volga")]
    x_pos = np.arange(len(g_keys))
    w = 0.26
    b0_v = [rec_b0["metrics"]["overall"][k]["rel_rmse"] for k, _ in g_keys]
    mp_v = [mp_ov[k]["rel_rmse"] for k, _ in g_keys]
    m3_v = [rec_m3["metrics"]["overall"][k]["rel_rmse"] for k, _ in g_keys]
    ax3.bar(x_pos - w, b0_v, width=w, color="#d62728", label="B0 (Live SPX)")
    ax3.bar(x_pos, mp_v, width=w, color="#9467bd", label="M3_param_plus (0-Shot)")
    ax3.bar(x_pos + w, m3_v, width=w, color="#2ca02c", label="M3_sech_80 (Live SPX)")
    ax3.set_yscale("log")
    ax3.set_ylim(1e-4, 28.0)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([lbl for _, lbl in g_keys])
    ax3.set_ylabel("Overall Relative RMSE vs. Exact Fourier (log)")
    ax3.set_title("Greeks on Live SPX Regime ($\\xi^*=1.50$)", fontsize=10.4)
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend(fontsize=8.0, frameon=True, loc="upper left")

    fig.suptitle("Empirical Validation on Live Cboe SPX / SPXW Option Chains (Nov 19, 2026, ^SPX = 7,205.56)", fontsize=12.0, y=1.02)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        plot_from_saved()
    else:
        main()
