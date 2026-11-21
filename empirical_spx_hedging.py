"""empirical_spx_hedging.py - Real-Market Empirical SPX/SPXW Option Dynamic Hedging Backtest.

Tracks actual traded S&P 500 option contracts (`optionid`) day-to-day across two continuous
full-month Wharton WRDS OptionMetrics IvyDB US panels:
  1. August 2025 Normal/Calm Regime (`wrds_spx_2025_08_fullmonth.csv`, 21 trading days)
  2. March 2020 COVID-19 Crash & Rebound (`wrds_spx_2020_03_fullmonth.csv`, 22 trading days, VIX up to 82.69)

All option positions (shorted target liability option + two benchmark hedging options) are marked
strictly at real WRDS closing bid-ask midpoints V^mkt = 0.5*(bid + ask), and underlying forward
hedges use the curve-free put-call parity forward move D_{t+1}(F_{t+1} - F_t).
Also reports net-of-transaction-cost hedging error under realistic institutional half-spread decay.

Compares 6 strategies on identical traded SPX contract-day transitions:
  - Unhedged (raw 1-day option mark-to-market change)
  - Vendor_BS_Delta (OptionMetrics IvyDB US reported Black-Scholes delta)
  - CV_RBA_MV_Delta (Skew-adjusted Heston minimum-variance delta via CV-RBA-PINN)
  - B0_PINN_DGV (Standard B0_param PINN 2-instrument Delta-Gamma-Vega hedge)
  - Vendor_BS_DGV (OptionMetrics reported Black-Scholes delta/gamma/vega 2-instrument hedge)
  - CV_RBA_PINN_DGV (Ours: M3_param_wide / M3_param_crisis_v2 2-instrument Delta-Gamma-Vega hedge)
  - Exact_Fourier_DGV (Exact Lewis-Albrecher Heston Fourier oracle 2-instrument DGV hedge)
"""
from collections import defaultdict
import csv
import json
import math
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)

import cvpinn
import cvpinn_param as cp8
import diff_calibrate as dc
import heston_ref as hr
import spx_w8

WRDS_DIR = os.path.join("docs", "market_data", "wrds_optionmetrics")
WRDS_REP = os.path.join("runs", "wrds_optionmetrics_report.json")
WIDE_CKPT = os.path.join("runs", "param", "checkpoints", "M3_param_wide_seed0.pt")
CRISIS_CKPT = os.path.join("runs", "param", "checkpoints", "M3_param_crisis_v2_seed0.pt")
B0_PARAM_CKPT = os.path.join("runs", "param", "checkpoints", "B0_param_seed0.pt")
OUT_JSON = os.path.join("runs", "empirical_spx_hedging_report.json")
OUT_FIG = os.path.join("paper", "fig20_empirical_spx_hedging.png")


def load_b0_param_fn():
    ckpt = torch.load(B0_PARAM_CKPT, map_location="cpu", weights_only=False)
    dom = cvpinn.Domain(v_max=0.5)
    box = cp8.ParamBox()
    net = cvpinn.Mlp(7, width=int(ckpt.get("width", 128)), depth=4)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return lambda x, v, tau, kp, th, xi, rh: net(cp8.param_features(x, v, tau, kp, th, xi, rh, dom, box))


def eval_surr_sens_dollar(pfn, contracts, p_cal):
    """Computes dollar dV/dF, d2V/dF2, and dV/dv for a list of contract dicts via PyTorch autograd."""
    n = len(contracts)
    xs = torch.tensor([c["x"] for c in contracts], dtype=torch.float64, requires_grad=True)
    vs = torch.full((n,), float(p_cal["v0"]), dtype=torch.float64, requires_grad=True)
    ts = torch.tensor([c["tau"] for c in contracts], dtype=torch.float64)
    kp = torch.full((n,), float(p_cal["kappa"]), dtype=torch.float64)
    th = torch.full((n,), float(p_cal["theta"]), dtype=torch.float64)
    xi = torch.full((n,), float(p_cal["xi"]), dtype=torch.float64)
    rh = torch.full((n,), float(p_cal["rho"]), dtype=torch.float64)

    # Clamp v inside B0_param domain if needed
    v_cap = float(getattr(pfn, "_v_max", 0.5))
    vs_in = torch.clamp(vs, max=0.98 * v_cap)
    c_val = pfn(xs, vs_in, ts, kp, th, xi, rh)
    cx, cv = torch.autograd.grad(c_val.sum(), (xs, vs_in), create_graph=True)
    cxx = torch.autograd.grad(cx.sum(), xs, retain_graph=False)[0]

    cx_np = cx.detach().numpy()
    gam_n = (cxx - cx).detach().numpy()
    cv_np = cv.detach().numpy()

    out = []
    for i, con in enumerate(contracts):
        D, F, K, ex_inv = con["D"], con["F"], con["K"], math.exp(-con["x"])
        is_put = 1.0 if con["cp"] == "P" else 0.0
        dV_dF = D * (ex_inv * cx_np[i] - is_put)
        d2V_dF2 = (D * ex_inv / F) * gam_n[i]
        dV_dv = K * D * cv_np[i]
        out.append((float(dV_dF), float(d2V_dF2), float(dV_dv)))
    return out


def eval_fourier_sens_dollar(contracts, p_cal):
    p_tup = (p_cal["kappa"], p_cal["theta"], p_cal["xi"], p_cal["rho"])
    v0 = float(p_cal["v0"])
    out = []
    for con in contracts:
        h = hr.heston(con["x"], v0, con["tau"], p_tup)
        D, F, K, ex_inv = con["D"], con["F"], con["K"], math.exp(-con["x"])
        is_put = 1.0 if con["cp"] == "P" else 0.0
        dV_dF = D * (ex_inv * h["cx"] - is_put)
        d2V_dF2 = (D * ex_inv / F) * h["gam"]
        dV_dv = K * D * h["cv"]
        out.append((float(dV_dF), float(d2V_dF2), float(dV_dv)))
    return out


def solve_dgv_weights(sens_liab, sens_b1, sens_b2, max_w=3.0, reg=1e-2):
    """Solves 2x2 Gamma-Vega linear system with scale-free Tikhonov regularization."""
    dF_L, g_L, v_L = sens_liab
    dF_1, g_1, v_1 = sens_b1
    dF_2, g_2, v_2 = sens_b2
    s_g = max(math.hypot(g_1, g_2), 1e-6)
    s_v = max(math.hypot(v_1, v_2), 1e-4)
    A = np.array([[g_1 / s_g, g_2 / s_g], [v_1 / s_v, v_2 / s_v]], dtype=np.float64)
    b = np.array([g_L / s_g, v_L / s_v], dtype=np.float64)
    ATA = A.T @ A + reg * np.eye(2)
    w1, w2 = np.linalg.solve(ATA, A.T @ b)
    w1 = float(np.clip(w1, -max_w, max_w))
    w2 = float(np.clip(w2, -max_w, max_w))
    w_fwd = float(dF_L - w1 * dF_1 - w2 * dF_2)
    return w1, w2, w_fwd


def run_panel_backtest(csv_path, daily_cal_map, wide_fn, crisis_fn, b0_fn):
    # 1. Extract daily parity curves F_t(exp, ser), D_t(exp, ser)
    rows_spx = spx_w8.load_spx_csv(csv_path, default_quote_time="16:00:00")
    by_day_spx = defaultdict(list)
    for r in rows_spx:
        by_day_spx[r.quote_dt.strftime("%Y-%m-%d")].append(r)

    parity_by_day = {}
    for d, drows in by_day_spx.items():
        sl, _ = spx_w8.process_chain_parity_and_quotes(drows)
        parity_by_day[d] = {(s["expiry"], s["series"]): s for s in sl}

    # 2. Load raw WRDS optionid records with vendor Greeks
    contracts_by_day = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d = r["date"].strip()
            if d not in parity_by_day:
                continue
            exp_s = r["exdate"].strip()
            ser = "SPX" if r["am_settlement"].strip() == "1" else "SPXW"
            if (exp_s, ser) not in parity_by_day[d]:
                continue
            bid, ask = float(r["best_bid"]), float(r["best_offer"])
            if bid <= 0.25 or ask <= bid:
                continue
            mid = 0.5 * (bid + ask)
            spr = ask - bid
            if spr / mid > 0.20:
                continue
            iv_s, del_s, gam_s, veg_s = r["impl_volatility"].strip(), r["delta"].strip(), r["gamma"].strip(), r["vega"].strip()
            if not (iv_s and del_s and gam_s and veg_s):
                continue
            iv = float(iv_s)
            if iv <= 0.03:
                continue
            sl = parity_by_day[d][(exp_s, ser)]
            k = float(r["strike_price"]) * 0.001
            x = math.log(sl["F"] / k)
            oid = r["optionid"].strip()
            contracts_by_day[d][oid] = {
                "optionid": oid,
                "expiry": exp_s,
                "series": ser,
                "cp": r["cp_flag"].strip().upper(),
                "K": k,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "half_spr": 0.5 * spr,
                "tau": sl["tau"],
                "F": sl["F"],
                "D": sl["D"],
                "x": x,
                "om_iv": iv,
                "om_delta": float(del_s),
                "om_gamma": float(gam_s),
                "om_vega_v": float(veg_s) / (2.0 * iv),  # dV/dv = (dV/dsigma) / (2*sigma)
            }

    days_sorted = sorted(d for d in contracts_by_day if d in daily_cal_map)
    strategies = ("Unhedged", "Vendor_BS_Delta", "CV_RBA_MV_Delta", "B0_PINN_DGV", "Vendor_BS_DGV", "CV_RBA_PINN_DGV", "Exact_Fourier_DGV")
    errs_gross = {k: [] for k in strategies}
    errs_tc = {k: [] for k in strategies}
    four_diff_m3 = []
    four_diff_b0 = []
    n_days_used = 0

    for i in range(len(days_sorted) - 1):
        d0, d1 = days_sorted[i], days_sorted[i + 1]
        c0_map, c1_map = contracts_by_day[d0], contracts_by_day[d1]
        shared_ids = set(c0_map) & set(c1_map)
        if len(shared_ids) < 15:
            continue

        # Select 2 liquid near-ATM benchmark options: b1 around tau ~ 0.08-0.16, b2 around tau ~ 0.25-0.65
        cand_b1 = [c0_map[oid] for oid in shared_ids if 0.07 <= c0_map[oid]["tau"] <= 0.18 and abs(c0_map[oid]["x"]) <= 0.03 and c0_map[oid]["cp"] == "C"]
        cand_b2 = [c0_map[oid] for oid in shared_ids if 0.24 <= c0_map[oid]["tau"] <= 0.75 and abs(c0_map[oid]["x"]) <= 0.04 and c0_map[oid]["cp"] == "C"]
        if not cand_b1 or not cand_b2:
            continue
        b1 = min(cand_b1, key=lambda c: abs(c["x"]) + 0.2 * abs(c["tau"] - 0.11))
        b2 = min(cand_b2, key=lambda c: abs(c["x"]) + 0.1 * abs(c["tau"] - 0.40))
        if b1["optionid"] == b2["optionid"] or abs(b1["tau"] - b2["tau"]) < 0.08:
            continue

        # Target liabilities: evenly stratified across the liquid OTM smile (0.015 <= |x| <= 0.28, 0.035 <= tau <= 0.25)
        liab_pool = sorted(
            [
                c0_map[oid] for oid in shared_ids
                if oid not in (b1["optionid"], b2["optionid"])
                and 0.035 <= c0_map[oid]["tau"] <= 0.25
                and 0.015 <= abs(c0_map[oid]["x"]) <= 0.28
                and ((c0_map[oid]["cp"] == "C" and c0_map[oid]["x"] < 0.0) or (c0_map[oid]["cp"] == "P" and c0_map[oid]["x"] > 0.0))
            ],
            key=lambda c: c["x"],
        )
        if len(liab_pool) > 40:
            idx_s = np.linspace(0, len(liab_pool) - 1, 40, dtype=int)
            liabs = [liab_pool[j] for j in idx_s]
        else:
            liabs = liab_pool
        if not liabs:
            continue

        n_days_used += 1
        p_cal = daily_cal_map[d0]
        pfn_m3 = crisis_fn if p_cal["surrogate"].startswith("M3_param_crisis") else wide_fn

        all_cons = [b1, b2] + liabs
        s_m3 = eval_surr_sens_dollar(pfn_m3, all_cons, p_cal)
        s_b0 = eval_surr_sens_dollar(b0_fn, all_cons, p_cal)
        s_fr = eval_fourier_sens_dollar(all_cons, p_cal)
        s_om = [(c["om_delta"], c["om_gamma"], c["om_vega_v"]) for c in all_cons]

        # Realized 1-day market midpoint changes & parity forward move
        dV_b1 = c1_map[b1["optionid"]]["mid"] - b1["mid"]
        dV_b2 = c1_map[b2["optionid"]]["mid"] - b2["mid"]
        # Use short-maturity parity forward move as the single liquid SPX futures/forward hedge instrument
        dF_b1 = c1_map[b1["optionid"]]["D"] * (c1_map[b1["optionid"]]["F"] - b1["F"])
        # Note: since s_m3 returns dV/dF = D * dV/d(DF), forward unit PnL per unit dF is (F_{t+1} - F_t)
        raw_dF = c1_map[b1["optionid"]]["F"] - b1["F"]
        # Effective institutional transaction cost (10% of quoted option half-spread for passive/mid-peg execution)
        tc_b1 = 0.10 * b1["half_spr"]
        tc_b2 = 0.10 * b2["half_spr"]

        rho_xi_scale = p_cal["rho"] * p_cal["xi"]

        for idx_l, con_l in enumerate(liabs):
            dV_L = c1_map[con_l["optionid"]]["mid"] - con_l["mid"]
            dF_L_move = c1_map[con_l["optionid"]]["F"] - con_l["F"]

            # 1. Unhedged
            errs_gross["Unhedged"].append(dV_L)
            errs_tc["Unhedged"].append(dV_L)

            # 2. Vendor BS Delta
            wF_bs = con_l["om_delta"] / max(con_l["D"], 0.8)
            e_bsd = dV_L - wF_bs * dF_L_move * con_l["D"]
            errs_gross["Vendor_BS_Delta"].append(e_bsd)
            errs_tc["Vendor_BS_Delta"].append(e_bsd - 0.02)

            # 3. CV-RBA Minimum-Variance Skew-Adjusted Delta: dV/dF_MV = dV/dF + (rho*xi / F) * dV/dv
            dF_m3, g_m3, v_m3 = s_m3[2 + idx_l]
            wF_mv = dF_m3 + (rho_xi_scale / con_l["F"]) * v_m3
            e_mv = dV_L - wF_mv * dF_L_move
            errs_gross["CV_RBA_MV_Delta"].append(e_mv)
            errs_tc["CV_RBA_MV_Delta"].append(e_mv - 0.02)

            # 4-7. Two-Benchmark DGV Hedges
            for name_s, s_all in (
                ("B0_PINN_DGV", s_b0),
                ("Vendor_BS_DGV", s_om),
                ("CV_RBA_PINN_DGV", s_m3),
                ("Exact_Fourier_DGV", s_fr),
            ):
                w1, w2, w_fwd = solve_dgv_weights(s_all[2 + idx_l], s_all[0], s_all[1])
                hedged_pnl = w1 * dV_b1 + w2 * dV_b2 + w_fwd * raw_dF
                err_g = dV_L - hedged_pnl
                tc_cost = abs(w1) * tc_b1 + abs(w2) * tc_b2 + 0.02
                errs_gross[name_s].append(err_g)
                errs_tc[name_s].append(err_g - tc_cost)

            four_diff_m3.append(errs_gross["CV_RBA_PINN_DGV"][-1] - errs_gross["Exact_Fourier_DGV"][-1])
            four_diff_b0.append(errs_gross["B0_PINN_DGV"][-1] - errs_gross["Exact_Fourier_DGV"][-1])

    unh_var = float(np.mean(np.square(errs_gross["Unhedged"])))
    summary = {
        "n_transitions_days": n_days_used,
        "n_contract_days": len(errs_gross["Unhedged"]),
        "m3_vs_exact_fourier_tracking_rmse_dlr": round(float(np.sqrt(np.mean(np.square(four_diff_m3)))), 4),
        "b0_vs_exact_fourier_tracking_rmse_dlr": round(float(np.sqrt(np.mean(np.square(four_diff_b0)))), 4),
        "strategies": {},
    }
    for k in strategies:
        arr_g = np.asarray(errs_gross[k], dtype=np.float64)
        arr_t = np.asarray(errs_tc[k], dtype=np.float64)
        rmse_g = float(np.sqrt(np.mean(arr_g ** 2)))
        rmse_t = float(np.sqrt(np.mean(arr_t ** 2)))
        var_red = 100.0 * (1.0 - (rmse_g ** 2) / max(unh_var, 1e-12))
        summary["strategies"][k] = {
            "gross_rmse_dlr": round(rmse_g, 4),
            "net_tc_rmse_dlr": round(rmse_t, 4),
            "gross_mae_dlr": round(float(np.mean(np.abs(arr_g))), 4),
            "p95_abs_err_dlr": round(float(np.percentile(np.abs(arr_g), 95)), 4),
            "variance_reduction_pct": round(var_red, 2),
        }
    return summary, errs_gross


def main():
    with open(WRDS_REP) as f:
        wrep = json.load(f)
    daily_map = {r["date"]: r for r in wrep["daily_rolling_78days"]}

    wide_fn, _ = dc.load_surrogate(WIDE_CKPT)
    crisis_fn, _ = dc.load_surrogate(CRISIS_CKPT)
    b0_fn = load_b0_param_fn()

    print("[Empirical SPX Hedging] Running August 2025 (21d) & March 2020 COVID-19 (22d) real-contract backtests...", flush=True)
    aug_sum, aug_errs = run_panel_backtest(
        os.path.join(WRDS_DIR, "wrds_spx_2025_08_fullmonth.csv"), daily_map, wide_fn, crisis_fn, b0_fn
    )
    mar_sum, mar_errs = run_panel_backtest(
        os.path.join(WRDS_DIR, "wrds_spx_2020_03_fullmonth.csv"), daily_map, wide_fn, crisis_fn, b0_fn
    )

    for lbl, s in (("August 2025 (Normal)", aug_sum), ("March 2020 (COVID Crash)", mar_sum)):
        st = s["strategies"]
        print(
            f"  [{lbl}: {s['n_contract_days']} contract-days] "
            f"Unhedged=${st['Unhedged']['gross_rmse_dlr']:.2f} | "
            f"BS-Delta=${st['Vendor_BS_Delta']['gross_rmse_dlr']:.2f} | "
            f"B0-DGV=${st['B0_PINN_DGV']['gross_rmse_dlr']:.2f} | "
            f"BS-DGV=${st['Vendor_BS_DGV']['gross_rmse_dlr']:.2f} | "
            f"CV-RBA-DGV=${st['CV_RBA_PINN_DGV']['gross_rmse_dlr']:.2f} "
            f"(ExactFour=${st['Exact_Fourier_DGV']['gross_rmse_dlr']:.2f}, var_red={st['CV_RBA_PINN_DGV']['variance_reduction_pct']:.1f}%)",
            flush=True,
        )

    report = {
        "August_2025_Normal_21d": aug_sum,
        "March_2020_COVID_22d": mar_sum,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # Build Figure 20: 3-panel Real-Market Empirical SPX Hedging Figure
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))
    fig.suptitle(
        "Empirical Market-Marked SPX/SPXW Option Dynamic Hedging Backtest (Wharton WRDS OptionMetrics Traded Contracts)",
        fontsize=12.6, y=0.98
    )

    strat_order = [
        ("Unhedged", "Unhedged", "#7f7f7f"),
        ("Vendor_BS_Delta", "BS $\\Delta$", "#1f77b4"),
        ("CV_RBA_MV_Delta", "Heston MV-$\\Delta$", "#17becf"),
        ("B0_PINN_DGV", "B0 $\\Delta\\Gamma$V", "#d62728"),
        ("Vendor_BS_DGV", "Vendor BS $\\Delta\\Gamma$V", "#ff7f0e"),
        ("CV_RBA_PINN_DGV", "CV-RBA $\\Delta\\Gamma$V", "#2ca02c"),
        ("Exact_Fourier_DGV", "Fourier $\\Delta\\Gamma$V", "#222222"),
    ]

    for ax_i, (s_dict, p_title) in enumerate([
        (aug_sum, f"Aug 2025 Panel ({aug_sum['n_contract_days']} Contract-Days)"),
        (mar_sum, f"Mar 2020 COVID Crash ({mar_sum['n_contract_days']} Contract-Days)"),
    ]):
        ax = axes[ax_i]
        x_idx = np.arange(len(strat_order))
        g_vals = [s_dict["strategies"][k]["gross_rmse_dlr"] for k, _, _ in strat_order]
        tc_vals = [s_dict["strategies"][k]["net_tc_rmse_dlr"] for k, _, _ in strat_order]
        cols = [c for _, _, c in strat_order]
        ax.bar(x_idx - 0.17, g_vals, 0.34, color=cols, alpha=0.92, label="Gross Midpoint RMSE ($)")
        ax.bar(x_idx + 0.17, tc_vals, 0.34, color=cols, alpha=0.45, hatch="//", label="Net of Spread TC ($)")
        ax.set_xticks(x_idx)
        ax.set_xticklabels([lbl for _, lbl, _ in strat_order], rotation=24, ha="right", fontsize=8.3)
        ax.set_ylabel("1-Day Hedging Replication RMSE ($ / SPX pt)")
        ax.set_title(p_title)
        ax.grid(axis="y", alpha=0.25)
        if ax_i == 0:
            ax.legend(loc="upper right", fontsize=8.0)

    # Panel 3: Empirical Hedging Error Distribution in August 2025 (BS Delta vs B0 DGV vs CV-RBA DGV)
    ax2 = axes[2]
    bins = np.linspace(-15.0, 15.0, 49)
    ax2.hist(np.clip(aug_errs["Vendor_BS_Delta"], -15, 15), bins=bins, density=True, alpha=0.38, color="#1f77b4", label="Vendor BS $\\Delta$")
    ax2.hist(np.clip(aug_errs["B0_PINN_DGV"], -15, 15), bins=bins, density=True, alpha=0.38, color="#d62728", label="B0 PINN $\\Delta\\Gamma$V")
    ax2.hist(np.clip(aug_errs["CV_RBA_PINN_DGV"], -15, 15), bins=bins, density=True, alpha=0.62, color="#2ca02c", label="CV-RBA-PINN $\\Delta\\Gamma$V")
    ax2.axvline(0.0, color="k", lw=1.0, ls="--")
    ax2.set_xlabel("Realized 1-Day Market Hedging Error ($)")
    ax2.set_ylabel("Empirical Density")
    ax2.set_title("Aug 2025 Hedging Error Distribution")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8.2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)
    print(f"Saved {OUT_JSON} and {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
