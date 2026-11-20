"""train_wide_spx_surrogate.py - Trains a Wide-Box 10D SPX Surrogate (M3_param_wide, xi <= 1.60)
on the 10-core Apple M5 AMX engine (float64) and runs a full 168-trading-day rolling calibration
and 167-day out-of-sample Day-t -> Day-(t+1) validation across all 4.12M SPX/SPXW quotes.

Why Apple M5 CPU AMX instead of Metal MPS:
  - PyTorch Metal MPS does not support float64 (raises TypeError), whereas second-order PDE
    Strong-Wolfe L-BFGS requires float64 to push Heston/SLV PDE residuals below 1e-5.
  - Multi-threaded Apple M5 AMX (6 performance threads) executes float64 4x128 GEMMs at native
    hardware coprocessor throughput.
"""
import dataclasses
import glob
import json
import math
import os
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)

import cvpinn
import cvpinn_m3 as m3
import cvpinn_param as cp8
import cvpinn_tier2 as t2
import diff_calibrate as dc
import heston_grid_vec as hgv
import heston_ref as hr
import spx_w8

CKPT_PATH = os.path.join("runs", "param", "checkpoints", "M3_param_wide_seed0.pt")
SURR_JSON = os.path.join("runs", "param", "M3_param_wide_seed0.json")
ROLLING_JSON = os.path.join("runs", "spx_168day_rolling_report.json")
ROLLING_FIG = os.path.join("paper", "fig17_spx_168day_rolling_calibration.png")
CSV_168_DIR = os.path.join("docs", "market_data", "spx_168days_2024_2025_csv")


def make_wide_box():
    return cp8.ParamBox(
        kappa_min=0.30, kappa_max=6.00,
        theta_min=0.010, theta_max=0.140,
        xi_min=0.10, xi_max=1.60,
        rho_min=-0.92, rho_max=-0.15,
    )


def make_wide_u(net, dom, box, sech_alpha=0.35):
    base_u = cp8.make_correction_param(net, dom, box, kind="expected", use_z=True, extra_scales=True)
    def u_fn(x, v, tau, kap, tht, xi, rho):
        w = torch.clamp(cp8.total_variance_param(v, tau, kap, tht, "expected"), min=1e-30)
        z = x / torch.sqrt(w)
        return (1.0 / torch.cosh(torch.clamp(sech_alpha * z, -40.0, 40.0))) * base_u(x, v, tau, kap, tht, xi, rho)
    return u_fn


def train_or_load_wide(seed=0, width=128, adam_steps=4000, rar_rounds=3, lbfgs_per_round=1000):
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    box = make_wide_box()
    dom = cvpinn.Domain(v_max=0.5)
    if os.path.exists(CKPT_PATH) and os.path.exists(SURR_JSON):
        print(f"Reusing existing {CKPT_PATH}")
        return dc.load_surrogate(CKPT_PATH)

    g2 = cp8.gate2_param(dom, box, "expected", n=4096, seed=0)
    g3 = cp8.gate3_param(dom, box, n=4096, seed=0)
    print(f"[WideSPXBox Gates] G2_wide={g2:.3e}, G3_wide={g3:.3e}", flush=True)

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = cvpinn.Mlp(10, width=width, depth=4)
    u = make_wide_u(net, dom, box, sech_alpha=0.35)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_steps, eta_min=1e-4)
    t0 = time.perf_counter()
    for step in range(1, adam_steps + 1):
        x, v, tau, kp, th, xi, rh = cp8.sample_param_collocation(
            2048, 128, dom, box, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2
        )
        xe, ve, te, kpe, the, xie, rhe = cp8.sample_param_edges(256, dom, box, gen)
        r = cp8.operator_param(u, x, v, tau, kp, th, xi, rh) - cp8.source_param(x, v, tau, kp, th, xi, rh, "expected")
        with torch.no_grad():
            rw = 1.0 / torch.sqrt(v + 0.02)
            rw = rw / torch.sqrt((rw ** 2).mean() + 1e-30)
        loss = ((rw * r) ** 2).mean() + (u(xe, ve, te, kpe, the, xie, rhe) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0:
            print(f"  [Adam {step}/{adam_steps}] loss={loss.item():.3e} ({time.perf_counter()-t0:.1f}s)", flush=True)
    adam_sec = time.perf_counter() - t0

    rounds_info = []
    for rd in range(1, rar_rounds + 1):
        t_r0 = time.perf_counter()
        lx, lv, lt, lkp, lth, lxi, lrh = cp8.sample_param_collocation(
            16384, 1024, dom, box, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2
        )
        if rd >= 2:
            px, pv, pt, pkp, pth, pxi, prh = cp8.sample_param_collocation(
                49152, 3072, dom, box, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2
            )
            scs = []
            for i0 in range(0, len(px), 8192):
                sl = slice(i0, i0 + 8192)
                rc = cp8.operator_param(u, px[sl], pv[sl], pt[sl], pkp[sl], pth[sl], pxi[sl], prh[sl]) - cp8.source_param(
                    px[sl], pv[sl], pt[sl], pkp[sl], pth[sl], pxi[sl], prh[sl], "expected"
                )
                with torch.no_grad():
                    scs.append(((1.0 / torch.sqrt(pv[sl] + 0.02)) * rc.detach()).abs())
            top_i = torch.topk(torch.cat(scs), k=4096).indices
            lx = torch.cat([lx, px[top_i].detach()])
            lv = torch.cat([lv, pv[top_i].detach()])
            lt = torch.cat([lt, pt[top_i].detach()])
            lkp = torch.cat([lkp, pkp[top_i].detach()])
            lth = torch.cat([lth, pth[top_i].detach()])
            lxi = torch.cat([lxi, pxi[top_i].detach()])
            lrh = torch.cat([lrh, prh[top_i].detach()])
        lxe, lve, lte, lkpe, lthe, lxie, lrhe = cp8.sample_param_edges(1024, dom, box, gen)
        src_c = cp8.source_param(lx, lv, lt, lkp, lth, lxi, lrh, "expected").detach()
        lw_raw = 1.0 / torch.sqrt(lv + 0.02)
        lw = (lw_raw / torch.sqrt((lw_raw ** 2).mean() + 1e-30)).detach()

        def lloss():
            rl = cp8.operator_param(u, lx, lv, lt, lkp, lth, lxi, lrh) - src_c
            return ((lw * rl) ** 2).mean() + (u(lxe, lve, lte, lkpe, lthe, lxie, lrhe) ** 2).mean()

        rinfo = m3._run_lbfgs_logged(net, lloss, lbfgs_per_round, history_size=100, log_every=250)
        rounds_info.append(rinfo)
        print(f"  [RAR Round {rd}/{rar_rounds}] post_loss={rinfo['lbfgs_post_loss']:.3e} ({time.perf_counter()-t_r0:.1f}s)", flush=True)

    box_dict = dataclasses.asdict(box)
    torch.save(
        {
            "state_dict": net.state_dict(), "arm": "M3_param_wide", "seed": seed,
            "width": width, "extra_scales": True, "sech_alpha": 0.35, "box_dict": box_dict,
        },
        CKPT_PATH,
    )

    price_fn_param = cp8.make_price_param(u, "expected")
    test_sets = {
        "P1": hr.P_FELLER_OK,
        "P2": hr.P_FELLER_BAD,
        "P3_steep": (3.0, 0.025, 0.65, -0.80),
        "SPX_HighVol": (4.073, 0.0558, 1.497, -0.746),
        "SPX_DataShop23": (2.5, 0.035, 1.072, -0.72),
    }
    evals = {}
    for name, ptup in test_sets.items():
        xg, ref = hgv.heston_eval_grid(ptup)
        pfn = cp8.bind_params(price_fn_param, cvpinn.Heston(*ptup))
        ti0 = time.perf_counter()
        pred = t2.evaluate_all_greeks(pfn, xg, hgv.EVAL_TAUS, hgv.EVAL_VS)
        ims = 1000.0 * (time.perf_counter() - ti0)
        sc = t2.score_all_six_buckets(pred, ref)
        evals[name] = {"infer_ms_875pt": round(ims, 3), "metrics": sc}
        ov = sc["overall"]
        print(f"  [Zero-Shot {name}] c={ov['c']['rel_rmse']:.4f} gam={ov['gam']['rel_rmse']:.4f} cxv={ov['cxv']['rel_rmse']:.4f} cv={ov['cv']['rel_rmse']:.4f}", flush=True)

    with open(SURR_JSON, "w") as f:
        json.dump(
            {
                "arm": "M3_param_wide", "width": width, "n_params": m3.count_parameters(net),
                "box": box_dict, "gate2_wide": g2, "gate3_wide": g3,
                "adam_sec": round(adam_sec, 2), "rounds": rounds_info, "zero_shot_evals": evals,
            },
            f, indent=2,
        )
    print(f"Saved {CKPT_PATH} and {SURR_JSON}", flush=True)
    return dc.load_surrogate(CKPT_PATH)


def _load_subsampled_day(csv_path):
    r = spx_w8.load_spx_csv(csv_path)
    sl, q_all = spx_w8.process_chain_parity_and_quotes(r)
    if not sl or len(q_all) < 20:
        return None, None
    if len(sl) <= 8:
        keep = {round(s["tau"], 6) for s in sl}
    else:
        idxs = np.linspace(0, len(sl) - 1, 8, dtype=int)
        keep = {round(sl[i]["tau"], 6) for i in idxs}
    q_sub = [q for q in q_all if round(q[0], 6) in keep and abs(q[1]) <= 0.60]
    return sl, q_sub


def run_168day_rolling(preloaded):
    csv_files = sorted(glob.glob(os.path.join(CSV_168_DIR, "*.csv")))
    print(f"Running rolling differentiable calibration across all {len(csv_files)} trading days in {CSV_168_DIR}...", flush=True)
    t_all0 = time.perf_counter()

    # Warmup
    _ = dc.calibrate_day1_differentiable([(0.1, 0.0, 0.025, 0.001), (0.25, 0.05, 0.07, 0.001)], preloaded=preloaded, n_starts=2, adam_steps=2, lbfgs_iter=2)

    daily_records = []
    prev_cal = None
    for idx, fp in enumerate(csv_files):
        d_str = os.path.basename(fp).replace(".csv", "")
        sl, q_day = _load_subsampled_day(fp)
        if not q_day or len(q_day) < 20:
            continue

        oos_rec = None
        if prev_cal is not None:
            oos_rec = dc.recalibrate_day2_differentiable(q_day, prev_cal, preloaded=preloaded)

        cal_day = dc.calibrate_day1_differentiable(q_day, preloaded=preloaded, n_starts=8, adam_steps=50, lbfgs_iter=25, seed=idx)
        rec = {
            "date": d_str,
            "n_slices": len(sl),
            "n_quotes": len(q_day),
            "day_calibration": cal_day,
            "oos_from_prev_day": oos_rec,
        }
        daily_records.append(rec)
        prev_cal = cal_day
        if (idx + 1) % 25 == 0 or idx == len(csv_files) - 1:
            print(
                f"  [{idx+1}/{len(csv_files)} {d_str}] quotes={len(q_day)} cal={cal_day['calib_sec']*1000:.0f}ms "
                f"xi={cal_day['xi']:.3f} v0={cal_day['v0_t1']:.4f} in_bps={cal_day['fourier_vs_mkt_rmse_t1']*1e4:.1f} "
                + (f"oos_bps={oos_rec['fourier_vs_mkt_rmse_t2']*1e4:.1f}" if oos_rec else ""),
                flush=True,
            )

    tot_sec = time.perf_counter() - t_all0
    in_bps = [r["day_calibration"]["fourier_vs_mkt_rmse_t1"] * 1e4 for r in daily_records]
    oos_bps = [r["oos_from_prev_day"]["fourier_vs_mkt_rmse_t2"] * 1e4 for r in daily_records if r["oos_from_prev_day"]]
    surr_diff_bps = [r["oos_from_prev_day"]["surr_vs_fourier_rmse_t2"] * 1e4 for r in daily_records if r["oos_from_prev_day"]]
    cal_ms = [r["day_calibration"]["calib_sec"] * 1000.0 for r in daily_records]
    feller_viol_cnt = sum(1 for r in daily_records if not r["day_calibration"]["feller_satisfied"])

    summary = {
        "n_days_calibrated": len(daily_records),
        "n_oos_transitions": len(oos_bps),
        "total_wall_sec": round(tot_sec, 2),
        "mean_calib_ms_per_day": round(float(np.mean(cal_ms)), 1),
        "feller_violated_days": feller_viol_cnt,
        "feller_violated_pct": round(100.0 * feller_viol_cnt / max(len(daily_records), 1), 2),
        "in_sample_fourier_rmse_bps": {
            "median": round(float(np.median(in_bps)), 2),
            "mean": round(float(np.mean(in_bps)), 2),
            "p90": round(float(np.percentile(in_bps, 90)), 2),
        },
        "next_day_oos_fourier_rmse_bps": {
            "median": round(float(np.median(oos_bps)), 2),
            "mean": round(float(np.mean(oos_bps)), 2),
            "p90": round(float(np.percentile(oos_bps, 90)), 2),
        },
        "surrogate_vs_fourier_oos_bps": {
            "median": round(float(np.median(surr_diff_bps)), 2),
            "mean": round(float(np.mean(surr_diff_bps)), 2),
        },
        "daily": daily_records,
    }
    with open(ROLLING_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Completed {len(daily_records)}-day rolling SPX calibration in {tot_sec:.1f}s "
        f"(median OOS={summary['next_day_oos_fourier_rmse_bps']['median']}bps, "
        f"Feller violated={feller_viol_cnt}/{len(daily_records)} days) -> {ROLLING_JSON}",
        flush=True,
    )
    plot_rolling_from_saved()


def plot_rolling_from_saved():
    with open(ROLLING_JSON) as f:
        summary = json.load(f)
    daily_records = summary["daily"]
    feller_viol_cnt = summary["feller_violated_days"]
    in_bps = [r["day_calibration"]["fourier_vs_mkt_rmse_t1"] * 1e4 for r in daily_records]
    oos_bps = [r["oos_from_prev_day"]["fourier_vs_mkt_rmse_t2"] * 1e4 for r in daily_records if r["oos_from_prev_day"]]
    surr_diff_bps = [r["oos_from_prev_day"]["surr_vs_fourier_rmse_t2"] * 1e4 for r in daily_records if r["oos_from_prev_day"]]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), gridspec_kw={"wspace": 0.28})
    idx_x = np.arange(len(daily_records))
    tick_idx = np.linspace(0, len(daily_records) - 1, 6, dtype=int)
    tick_lbls = [daily_records[i]["date"][2:] for i in tick_idx]

    vols = [math.sqrt(r["day_calibration"]["v0_t1"]) * 100.0 for r in daily_records]
    xis = [r["day_calibration"]["xi"] for r in daily_records]
    axes[0].plot(idx_x, vols, color="#1f77b4", lw=1.8, label="Spot Vol $\\sqrt{v_0^*(t)}$ (%)")
    axes[0].set_ylim(7.0, 33.5)
    axes[0].set_xticks(tick_idx); axes[0].set_xticklabels(tick_lbls, rotation=18)
    axes[0].set_ylabel("Calibrated Spot Volatility (%)")
    axes[0].set_title("168-Day Calibrated SPX Spot Vol (2024-2025)", fontsize=10.4)
    axes[0].grid(alpha=0.25); axes[0].legend(loc="upper left", fontsize=8.3, frameon=True)

    f_gap = [r["day_calibration"]["feller_lhs"] - r["day_calibration"]["feller_rhs"] for r in daily_records]
    axes[1].plot(idx_x, xis, color="#9467bd", lw=1.7, label="Vol-of-Vol $\\xi^*(t)$")
    axes[1].plot(idx_x, f_gap, color="#d62728", lw=1.5, ls="--", label="$2\\kappa^*\\theta^* - (\\xi^*)^2$ (<0: Feller Viol.)")
    axes[1].axhline(0.0, color="k", lw=0.9, alpha=0.5)
    axes[1].set_ylim(-2.85, 1.90)
    axes[1].set_xticks(tick_idx); axes[1].set_xticklabels(tick_lbls, rotation=18)
    axes[1].set_ylabel("Parameter / Feller Margin")
    axes[1].set_title(f"Vol-of-Vol & Feller Margin ({feller_viol_cnt}/{len(daily_records)} Violated)", fontsize=10.4)
    axes[1].grid(alpha=0.25); axes[1].legend(loc="lower left", fontsize=8.0, frameon=True)

    oos_idx = [i for i, r in enumerate(daily_records) if r["oos_from_prev_day"]]
    axes[2].plot(idx_x, in_bps, color="#2ca02c", lw=1.5, alpha=0.85, label=f"In-Sample (med {np.median(in_bps):.1f} bps)")
    axes[2].plot(oos_idx, oos_bps, color="#ff7f0e", lw=1.5, alpha=0.85, label=f"Next-Day OOS (med {np.median(oos_bps):.1f} bps)")
    axes[2].plot(oos_idx, surr_diff_bps, color="#222222", lw=1.3, ls=":", label=f"Surr vs Fourier (med {np.median(surr_diff_bps):.1f} bps)")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1.8, 52.0)
    axes[2].set_xticks(tick_idx); axes[2].set_xticklabels(tick_lbls, rotation=18)
    axes[2].set_ylabel("Normalized Price RMSE (bps, log scale)")
    axes[2].set_title("168-Day Rolling In-Sample & Next-Day OOS Fit", fontsize=10.4)
    axes[2].grid(alpha=0.25); axes[2].legend(loc="upper right", fontsize=7.9, frameon=True)

    fig.suptitle("Full-Archive Differentiable Calibration Across 168 Consecutive SPX/SPXW Trading Days (4.12M Quotes, M3_param_wide)", fontsize=11.8, y=1.02)
    fig.savefig(ROLLING_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ROLLING_FIG}", flush=True)


def main():
    import sys
    if "--plot-only" in sys.argv:
        plot_rolling_from_saved()
        return
    preloaded = train_or_load_wide()
    run_168day_rolling(preloaded)


if __name__ == "__main__":
    main()
