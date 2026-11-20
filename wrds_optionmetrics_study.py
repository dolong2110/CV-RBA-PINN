"""wrds_optionmetrics_study.py - 17-Year Wharton WRDS OptionMetrics IvyDB US SPX Study (2008-2025).

Executes comprehensive training and out-of-sample empirical validation across 5 multi-decade
Wharton WRDS OptionMetrics IvyDB US SPX/SPXW datasets (SECID = 108105, 434,763 raw rows ->
119,374 clean VIX-filtered OTM quotes across 25 trading days in 2008, 2020, 2021, 2022, 2025):

  1. Trains the Extreme-Crisis 10D Amortized Surrogate `M3_param_crisis` on Domain(v_max=1.00)
     with kappa in [0.30, 8.00], theta in [0.015, 0.450], xi in [0.15, 2.60], rho in [-0.95, -0.15]
     and structural sech(0.35*z) Lee-moment wing decay on 6 Apple M5 AMX float64 threads.
  2. Benchmarks SciPy Differential Evolution + Fourier vs. Dual-Regime Differentiable Autograd
     (`M3_param_wide` for normal/elevated v_ATM <= 0.15; `M3_param_crisis` for crisis v_ATM > 0.15)
     across all 5 eras, evaluating full-week out-of-sample generalization across 96,172 held-out quotes.
  3. Runs daily rolling differentiable calibration across all 25 WRDS trading days (2008-2025).
  4. Trains single-surface `B0` vs. Championship `M3_sech_80` (v_max=1.00) directly on the
     all-time VIX closing peak (`2020-03-16` COVID-19 Black Monday, VIX = 82.69, 2,670 SPX quotes),
     auditing live crisis pricing, 6-Greek accuracy on extreme variance slices, and OptionMetrics
     IvyDB US vendor-reported implied volatility and Greek alignment.
"""
from collections import defaultdict
import csv
import dataclasses
import json
import math
import os
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
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

WRDS_DIR = os.path.join("docs", "market_data", "wrds_optionmetrics")
WIDE_CKPT = os.path.join("runs", "param", "checkpoints", "M3_param_wide_seed0.pt")
CRISIS_CKPT = os.path.join("runs", "param", "checkpoints", "M3_param_crisis_seed0.pt")
CRISIS_JSON = os.path.join("runs", "param", "M3_param_crisis_seed0.json")
COVID_DIR = os.path.join("runs", "wrds_covid")
OUT_JSON = os.path.join("runs", "wrds_optionmetrics_report.json")
OUT_FIG = os.path.join("paper", "fig18_wrds_multidecade_study.png")

ERAS = [
    ("2008_Oct_GFC", "wrds_spx_2008_10_gfc.csv", "2008-10-14"),
    ("2020_Mar_COVID_Crash", "wrds_spx_2020_03_covid.csv", "2020-03-16"),
    ("2021_Jul_LowVol_Bull", "wrds_spx_2021_07_bull.csv", "2021-07-12"),
    ("2022_Oct_Inflation_Bear", "wrds_spx_2022_10_bear.csv", "2022-10-10"),
    ("2025_Aug_Latest_IvyDB", "wrds_spx_2025_08_latest.csv", "2025-08-25"),
]


def make_crisis_box():
    return cp8.ParamBox(
        kappa_min=0.30, kappa_max=8.00,
        theta_min=0.015, theta_max=0.450,
        xi_min=0.15, xi_max=2.60,
        rho_min=-0.95, rho_max=-0.15,
    )


def make_crisis_u(net, dom, box, sech_alpha=0.35):
    base_u = cp8.make_correction_param(net, dom, box, kind="expected", use_z=True, extra_scales=True)

    def u_fn(x, v, tau, kap, tht, xi, rho):
        w = torch.clamp(cp8.total_variance_param(v, tau, kap, tht, "expected"), min=1e-30)
        z = x / torch.sqrt(w)
        return (1.0 / torch.cosh(torch.clamp(sech_alpha * z, -40.0, 40.0))) * base_u(x, v, tau, kap, tht, xi, rho)

    return u_fn


def train_or_load_crisis_surrogate(seed=0, width=96, adam_steps=3000, rar_rounds=2, lbfgs_per_round=750):
    os.makedirs(os.path.dirname(CRISIS_CKPT), exist_ok=True)
    box = make_crisis_box()
    dom = cvpinn.Domain(v_max=1.00)
    if os.path.exists(CRISIS_CKPT) and os.path.exists(CRISIS_JSON):
        print(f"[Stage 1/4] Reusing existing {CRISIS_CKPT}", flush=True)
        return dc.load_surrogate(CRISIS_CKPT)

    g2 = cp8.gate2_param(dom, box, "expected", n=4096, seed=0)
    g3 = cp8.gate3_param(dom, box, n=4096, seed=0)
    print(f"[Stage 1/4] Training M3_param_crisis (v_max=1.0, xi_max=2.6) | G2={g2:.3e}, G3={g3:.3e}", flush=True)

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = cvpinn.Mlp(10, width=width, depth=4)
    u = make_crisis_u(net, dom, box, sech_alpha=0.35)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_steps, eta_min=1e-4)
    t0 = time.perf_counter()
    for step in range(1, adam_steps + 1):
        x, v, tau, kp, th, xi, rh = cp8.sample_param_collocation(
            2048, 128, dom, box, gen, z_frac=0.60, z_max=4.5, v_power=1.8, t_power=2.0
        )
        xe, ve, te, kpe, the, xie, rhe = cp8.sample_param_edges(256, dom, box, gen)
        r = cp8.operator_param(u, x, v, tau, kp, th, xi, rh) - cp8.source_param(x, v, tau, kp, th, xi, rh, "expected")
        with torch.no_grad():
            rw = 1.0 / torch.sqrt(v + 0.04)
            rw = rw / torch.sqrt((rw ** 2).mean() + 1e-30)
        loss = ((rw * r) ** 2).mean() + (u(xe, ve, te, kpe, the, xie, rhe) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"  [Crisis Adam {step}/{adam_steps}] loss={loss.item():.3e} ({time.perf_counter()-t0:.1f}s)", flush=True)
    adam_sec = time.perf_counter() - t0

    rounds_info = []
    for rd in range(1, rar_rounds + 1):
        t_r0 = time.perf_counter()
        lx, lv, lt, lkp, lth, lxi, lrh = cp8.sample_param_collocation(
            12288, 768, dom, box, gen, z_frac=0.60, z_max=4.5, v_power=1.8, t_power=2.0
        )
        if rd >= 2:
            px, pv, pt, pkp, pth, pxi, prh = cp8.sample_param_collocation(
                32768, 2048, dom, box, gen, z_frac=0.60, z_max=4.5, v_power=1.8, t_power=2.0
            )
            scs = []
            for i0 in range(0, len(px), 8192):
                sl = slice(i0, i0 + 8192)
                rc = cp8.operator_param(u, px[sl], pv[sl], pt[sl], pkp[sl], pth[sl], pxi[sl], prh[sl]) - cp8.source_param(
                    px[sl], pv[sl], pt[sl], pkp[sl], pth[sl], pxi[sl], prh[sl], "expected"
                )
                with torch.no_grad():
                    scs.append(((1.0 / torch.sqrt(pv[sl] + 0.04)) * rc.detach()).abs())
            top_i = torch.topk(torch.cat(scs), k=3072).indices
            lx = torch.cat([lx, px[top_i].detach()])
            lv = torch.cat([lv, pv[top_i].detach()])
            lt = torch.cat([lt, pt[top_i].detach()])
            lkp = torch.cat([lkp, pkp[top_i].detach()])
            lth = torch.cat([lth, pth[top_i].detach()])
            lxi = torch.cat([lxi, pxi[top_i].detach()])
            lrh = torch.cat([lrh, prh[top_i].detach()])

        lxe, lve, lte, lkpe, lthe, lxie, lrhe = cp8.sample_param_edges(1536, dom, box, gen)
        src_l = cp8.source_param(lx, lv, lt, lkp, lth, lxi, lrh, "expected").detach()
        lw_l = (1.0 / torch.sqrt(lv + 0.04)).detach()
        lw_l = (lw_l / torch.sqrt((lw_l ** 2).mean() + 1e-30)).detach()

        def loss_fn():
            res = cp8.operator_param(u, lx, lv, lt, lkp, lth, lxi, lrh) - src_l
            return ((lw_l * res) ** 2).mean() + (u(lxe, lve, lte, lkpe, lthe, lxie, lrhe) ** 2).mean()

        lb = m3._run_lbfgs_logged(net, loss_fn, lbfgs_per_round, history_size=80, log_every=250)
        lb["round_sec"] = round(time.perf_counter() - t_r0, 2)
        rounds_info.append(lb)
        print(f"  [Crisis RAR-LBFGS Rd {rd}/{rar_rounds}] post_loss={lb['lbfgs_post_loss']:.3e} ({lb['round_sec']}s)", flush=True)

    torch.save({
        "arm": "M3_param_crisis",
        "width": width,
        "v_max": 1.00,
        "extra_scales": True,
        "sech_alpha": 0.35,
        "box_dict": dataclasses.asdict(box),
        "state_dict": net.state_dict(),
    }, CRISIS_CKPT)

    # Zero-shot evaluation on extreme GFC & COVID-like synthetic stress points
    zs_eval = tuple(np.linspace(-2.4, 2.4, 25).tolist())
    vs_crisis = (0.04, 0.16, 0.36, 0.64, 0.84)
    p_covid_test = (3.5, 0.18, 2.10, -0.78)
    xg_c, ref_c = hgv.heston_eval_grid(p_covid_test, vs=vs_crisis, zs=zs_eval)
    p_fn_crisis = cp8.make_price_param(u, "expected")
    p_bound_crisis = cp8.bind_params(p_fn_crisis, cvpinn.Heston(*p_covid_test))
    pred_c = t2.evaluate_all_greeks(p_bound_crisis, xg_c, hgv.EVAL_TAUS, vs_crisis)
    sc_c = t2.score_all_six_buckets(pred_c, ref_c)["overall"]

    meta = {
        "arm": "M3_param_crisis",
        "width": width,
        "v_max": 1.00,
        "n_params": m3.count_parameters(net),
        "box": dataclasses.asdict(box),
        "gate2_crisis": g2,
        "gate3_crisis": g3,
        "adam_sec": round(adam_sec, 2),
        "rounds": rounds_info,
        "zeroshot_covid_stress_overall": {k: round(sc_c[k]["rel_rmse"], 4) for k in ("c", "cx", "gam", "cv", "cxv", "cvv")},
    }
    with open(CRISIS_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {CRISIS_CKPT} | zero-shot crisis test c={sc_c['c']['rel_rmse']:.4f}, gam={sc_c['gam']['rel_rmse']:.4f}, cv={sc_c['cv']['rel_rmse']:.4f}", flush=True)
    return dc.load_surrogate(CRISIS_CKPT)


def stratify_quotes(quotes, max_q=1600):
    if len(quotes) <= max_q:
        return quotes
    by_t = defaultdict(list)
    for q in quotes:
        by_t[q[0]].append(q)
    out = []
    taus = sorted(by_t)
    per_t = max(8, max_q // len(taus))
    for t in taus:
        arr = sorted(by_t[t], key=lambda item: item[1])
        if len(arr) <= per_t:
            out.extend(arr)
        else:
            idx = np.linspace(0, len(arr) - 1, per_t, dtype=int)
            out.extend(arr[i] for i in idx)
    return out[:max_q]


def estimate_atm_var(quotes):
    for tau, x, cm, _ in quotes:
        if tau >= 0.04 and abs(x) < 0.025:
            sig = cm * math.sqrt(2.0 * math.pi / tau)
            return sig * sig
    for tau, x, cm, _ in quotes:
        if abs(x) < 0.05:
            sig = cm * math.sqrt(2.0 * math.pi / tau)
            return sig * sig
    return 0.04


def implied_vol_norm(c_norm, x, tau):
    intr = max(math.exp(x) - 1.0, 0.0)
    if c_norm <= intr + 1e-9:
        return float("nan")
    try:
        return brentq(lambda sig: hr.bs(x, (sig * sig) * tau) - c_norm, 0.02, 3.50, xtol=1e-7)
    except ValueError:
        return float("nan")


def audit_optionmetrics_vendor_greeks(csv_path, target_date, p_tup, v0, pinn_fn, b0_fn):
    """Compares Exact Heston Fourier, M3_sech_80, and B0 against OptionMetrics IvyDB US reported IV & Greeks."""
    rows_spx = spx_w8.load_spx_csv(csv_path, default_quote_time="16:00:00")
    day_rows = [r for r in rows_spx if r.quote_dt.strftime("%Y-%m-%d") == target_date]
    clean_slices, _ = spx_w8.process_chain_parity_and_quotes(day_rows)
    slice_map = {(s["expiry"], s["series"]): s for s in clean_slices}

    om_records = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["date"].strip() != target_date:
                continue
            iv_s, d_s, g_s, v_s = r["impl_volatility"].strip(), r["delta"].strip(), r["gamma"].strip(), r["vega"].strip()
            if not (iv_s and d_s and g_s and v_s):
                continue
            exp_s = r["exdate"].strip()
            am = r["am_settlement"].strip()
            ser = "SPX" if am == "1" else "SPXW"
            if (exp_s, ser) not in slice_map:
                continue
            sl = slice_map[(exp_s, ser)]
            k = float(r["strike_price"]) * 0.001
            bid, ask = float(r["best_bid"]), float(r["best_offer"])
            cp = r["cp_flag"].strip().upper()
            if bid <= 0.0 or ask <= bid:
                continue
            mid = 0.5 * (bid + ask)
            if (ask - bid) / mid > 0.25:
                continue
            x = math.log(sl["F"] / k)
            # Liquid OTM smile core |x| <= 0.35
            if abs(x) > 0.35:
                continue
            if (cp == "C" and k < sl["F"]) or (cp == "P" and k >= sl["F"]):
                continue
            om_records.append({
                "tau": sl["tau"], "F": sl["F"], "D": sl["D"], "K": k, "x": x, "cp": cp,
                "om_iv": float(iv_s), "om_delta": float(d_s), "om_gamma": float(g_s), "om_vega": float(v_s),
            })

    if not om_records:
        return {}

    xs_t = torch.tensor([rec["x"] for rec in om_records], dtype=torch.float64, requires_grad=True)
    vs_t = torch.full((len(om_records),), float(v0), dtype=torch.float64, requires_grad=True)
    ts_t = torch.tensor([rec["tau"] for rec in om_records], dtype=torch.float64)

    def _pinn_chain_greeks(fn):
        x_q = xs_t.detach().clone().requires_grad_(True)
        v_q = vs_t.detach().clone().requires_grad_(True)
        c_val = fn(x_q, v_q, ts_t)
        cx, cv = torch.autograd.grad(c_val.sum(), (x_q, v_q), create_graph=True)
        cxx = torch.autograd.grad(cx.sum(), x_q, retain_graph=False)[0]
        gam_n = (cxx - cx).detach().numpy()
        return c_val.detach().numpy(), cx.detach().numpy(), gam_n, cv.detach().numpy()

    c_m3, cx_m3, gam_m3, cv_m3 = _pinn_chain_greeks(pinn_fn)
    c_b0, cx_b0, gam_b0, cv_b0 = _pinn_chain_greeks(b0_fn)

    # Exact scalar/vectorized Fourier reference at each quote
    f_c, f_cx, f_gam, f_cv = [], [], [], []
    for rec in om_records:
        h = hr.heston(rec["x"], float(v0), rec["tau"], p_tup)
        f_c.append(h["c"])
        f_cx.append(h["cx"])
        f_gam.append(h["gam"])
        f_cv.append(h["cv"])
    f_c, f_cx, f_gam, f_cv = map(np.asarray, (f_c, f_cx, f_gam, f_cv))

    # Convert normalized call sensitivities to forward/spot option Delta & Dollar Gamma
    # Call forward price C = K * D * c(x, v, tau), where x = ln(F/K):
    #   dC/dS (spot S ~ D*F assuming flat q_div captured in parity F = S*exp((r-q)tau))
    #   Specifically dC/d(D*F) = (K / F) * c_x = exp(-x) * c_x; for Put: exp(-x)*c_x - 1 (times D_q ~ 1)
    ex_inv = np.exp(-np.array([rec["x"] for rec in om_records]))
    is_put = np.array([1.0 if rec["cp"] == "P" else 0.0 for rec in om_records])
    F_arr = np.array([rec["F"] for rec in om_records])
    D_arr = np.array([rec["D"] for rec in om_records])
    om_del = np.array([rec["om_delta"] for rec in om_records])
    om_gam = np.array([rec["om_gamma"] for rec in om_records])
    om_iv = np.array([rec["om_iv"] for rec in om_records])

    del_four = ex_inv * f_cx - is_put
    del_m3 = ex_inv * cx_m3 - is_put
    del_b0 = ex_inv * cx_b0 - is_put

    gam_spot_four = ex_inv * f_gam / (D_arr * F_arr)
    gam_spot_m3 = ex_inv * gam_m3 / (D_arr * F_arr)
    gam_spot_b0 = ex_inv * gam_b0 / (D_arr * F_arr)

    iv_m3 = np.array([implied_vol_norm(float(c_i), rec["x"], rec["tau"]) for c_i, rec in zip(c_m3, om_records)])
    valid_iv = np.isfinite(iv_m3)

    return {
        "date": target_date,
        "n_vendor_matched_quotes": len(om_records),
        "m3_vs_fourier_delta_rmse": float(np.sqrt(np.mean((del_m3 - del_four) ** 2))),
        "b0_vs_fourier_delta_rmse": float(np.sqrt(np.mean((del_b0 - del_four) ** 2))),
        "m3_vs_fourier_gamma_rel_rmse": float(np.sqrt(np.mean((gam_spot_m3 - gam_spot_four) ** 2)) / (np.sqrt(np.mean(gam_spot_four ** 2)) + 1e-12)),
        "b0_vs_fourier_gamma_rel_rmse": float(np.sqrt(np.mean((gam_spot_b0 - gam_spot_four) ** 2)) / (np.sqrt(np.mean(gam_spot_four ** 2)) + 1e-12)),
        "m3_vs_optionmetrics_iv_rmse_volpts": float(np.sqrt(np.mean((iv_m3[valid_iv] - om_iv[valid_iv]) ** 2)) * 100.0),
        "heston_vs_bs_vendor_delta_corr": float(np.corrcoef(del_m3, om_del)[0, 1]),
        "heston_vs_bs_vendor_gamma_corr": float(np.corrcoef(gam_spot_m3, om_gam)[0, 1]),
        "b0_vs_bs_vendor_gamma_corr": float(np.corrcoef(gam_spot_b0, om_gam)[0, 1]),
    }


def main():
    os.makedirs(COVID_DIR, exist_ok=True)
    wide_preloaded = dc.load_surrogate(WIDE_CKPT)
    crisis_preloaded = train_or_load_crisis_surrogate()

    print("\n[Stage 2/4] Running 5-Era WRDS Anchor DE+Fourier vs. Differentiable Surrogate + Full-Week OOS...", flush=True)
    era_reports = []
    all_daily_rolling = []
    covid_anchor_quotes = None
    covid_anchor_slices = None
    covid_de_cal = None

    for era_name, fname, anchor_date in ERAS:
        fpath = os.path.join(WRDS_DIR, fname)
        rows = spx_w8.load_spx_csv(fpath, default_quote_time="16:00:00")
        by_day = defaultdict(list)
        for r in rows:
            by_day[r.quote_dt.strftime("%Y-%m-%d")].append(r)

        days_sorted = sorted(by_day)
        day_chains = {}
        for d in days_sorted:
            sl, q = spx_w8.process_chain_parity_and_quotes(by_day[d])
            day_chains[d] = (sl, q)

        sl_a, q_a = day_chains[anchor_date]
        q_a_cal = stratify_quotes(q_a, max_q=1600)
        v_atm_est = estimate_atm_var(q_a)
        is_crisis = v_atm_est > 0.15
        surr_use = crisis_preloaded if is_crisis else wide_preloaded
        surr_tag = "M3_param_crisis" if is_crisis else "M3_param_wide"
        de_bounds = (
            [(0.2, 10.0), (0.01, 0.50), (0.10, 2.80), (-0.98, -0.05), (0.002, 0.95)]
            if is_crisis else
            [(0.2, 10.0), (0.005, 0.25), (0.05, 1.65), (-0.98, -0.05), (0.002, 0.25)]
        )
        v0_max_oos = 0.95 if is_crisis else 0.40

        t_de0 = time.perf_counter()
        de_res = spx_w8.calibrate_day1(q_a_cal, popsize=7, maxiter=35, seed=0, bounds=de_bounds)
        de_sec = time.perf_counter() - t_de0

        dc_res = dc.calibrate_day1_differentiable(
            q_a_cal, n_starts=10, adam_steps=65, lbfgs_iter=30, seed=0, preloaded=surr_use
        )
        speedup = de_sec / max(dc_res["calib_sec"], 1e-6)

        # Evaluate full-chain Day-1 RMSE (all quotes, not just 1,600 stratified)
        sd_a_full, ym_a_full, _ = spx_w8._group_quotes(q_a)
        de_pred_a = spx_w8.heston_prices_by_slice(sd_a_full, de_res["v0_t1"], de_res["kappa"], de_res["theta"], de_res["xi"], de_res["rho"])
        dc_pred_a = spx_w8.heston_prices_by_slice(sd_a_full, dc_res["v0_t1"], dc_res["kappa"], dc_res["theta"], dc_res["xi"], dc_res["rho"])
        de_is_bps = float(np.sqrt(np.mean((de_pred_a - ym_a_full) ** 2)) * 1e4)
        dc_is_bps = float(np.sqrt(np.mean((dc_pred_a - ym_a_full) ** 2)) * 1e4)

        if era_name == "2020_Mar_COVID_Crash":
            covid_anchor_quotes = q_a
            covid_anchor_slices = sl_a
            covid_de_cal = de_res

        # Out-of-sample across all subsequent days in that week (freeze structural params, 1D v0)
        oos_days = [d for d in days_sorted if d > anchor_date and len(day_chains[d][1]) >= 50]
        de_sq_errs, dc_sq_errs, surr_four_sq_errs = [], [], []
        oos_breakdown = []
        for od in oos_days:
            _, q_o = day_chains[od]
            de_o = spx_w8.recalibrate_day2_v0(q_o, de_res, v0_max=v0_max_oos)
            dc_o = dc.recalibrate_day2_differentiable(q_o, dc_res, preloaded=surr_use)
            n_o = len(q_o)
            de_sq_errs.append((de_o["fourier_vs_mkt_rmse_t2"] ** 2) * n_o)
            dc_sq_errs.append((dc_o["fourier_vs_mkt_rmse_t2"] ** 2) * n_o)
            surr_four_sq_errs.append((dc_o["surr_vs_fourier_rmse_t2"] ** 2) * n_o)
            oos_breakdown.append({
                "date": od,
                "n_quotes": n_o,
                "de_v0": round(de_o["v0_t2"], 5),
                "dc_v0": round(dc_o["v0_t2"], 5),
                "de_oos_bps": round(de_o["fourier_vs_mkt_rmse_t2"] * 1e4, 2),
                "dc_oos_bps": round(dc_o["fourier_vs_mkt_rmse_t2"] * 1e4, 2),
                "surr_vs_fourier_bps": round(dc_o["surr_vs_fourier_rmse_t2"] * 1e4, 2),
            })

        tot_oos_q = sum(item["n_quotes"] for item in oos_breakdown)
        de_week_oos_bps = float(math.sqrt(sum(de_sq_errs) / tot_oos_q) * 1e4)
        dc_week_oos_bps = float(math.sqrt(sum(dc_sq_errs) / tot_oos_q) * 1e4)
        surr_four_week_bps = float(math.sqrt(sum(surr_four_sq_errs) / tot_oos_q) * 1e4)

        era_rec = {
            "era": era_name,
            "anchor_date": anchor_date,
            "surrogate_dispatched": surr_tag,
            "n_anchor_quotes_total": len(q_a),
            "n_anchor_quotes_cal": len(q_a_cal),
            "n_week_oos_quotes": tot_oos_q,
            "de_calib_sec": round(de_sec, 2),
            "autograd_calib_sec": round(dc_res["calib_sec"], 3),
            "speedup_x": round(speedup, 1),
            "de_params": {k: round(de_res[k], 5) for k in ("kappa", "theta", "xi", "rho", "v0_t1")},
            "de_feller_satisfied": de_res["feller_satisfied"],
            "autograd_params": {k: round(dc_res[k], 5) for k in ("kappa", "theta", "xi", "rho", "v0_t1")},
            "autograd_feller_satisfied": dc_res["feller_satisfied"],
            "de_anchor_full_bps": round(de_is_bps, 2),
            "autograd_anchor_full_bps": round(dc_is_bps, 2),
            "de_week_oos_bps": round(de_week_oos_bps, 2),
            "autograd_week_oos_bps": round(dc_week_oos_bps, 2),
            "surr_vs_fourier_week_bps": round(surr_four_week_bps, 2),
            "oos_days": oos_breakdown,
        }
        era_reports.append(era_rec)
        print(
            f"  [{era_name}] {surr_tag} | DE {de_sec:.1f}s vs Autograd {dc_res['calib_sec']:.2f}s ({speedup:.1f}x) | "
            f"Week OOS ({tot_oos_q} q): DE={de_week_oos_bps:.2f} bps vs Autograd={dc_week_oos_bps:.2f} bps (Surr-Four={surr_four_week_bps:.2f} bps)",
            flush=True,
        )

        # Stage 3: Daily rolling across every trading day in this WRDS file
        for idx_d, d in enumerate(days_sorted):
            sl_d, q_d = day_chains[d]
            if len(q_d) < 15:
                continue
            q_d_cal = stratify_quotes(q_d, max_q=1600)
            v_est_d = estimate_atm_var(q_d)
            s_d = crisis_preloaded if v_est_d > 0.15 else wide_preloaded
            cal_d = dc.calibrate_day1_differentiable(q_d_cal, n_starts=8, adam_steps=55, lbfgs_iter=25, seed=0, preloaded=s_d)
            sd_d_full, ym_d_full, _ = spx_w8._group_quotes(q_d)
            pr_f_d = spx_w8.heston_prices_by_slice(sd_d_full, cal_d["v0_t1"], cal_d["kappa"], cal_d["theta"], cal_d["xi"], cal_d["rho"])
            is_bps_d = float(np.sqrt(np.mean((pr_f_d - ym_d_full) ** 2)) * 1e4)
            next_oos_bps = None
            if idx_d + 1 < len(days_sorted) and len(day_chains[days_sorted[idx_d + 1]][1]) >= 50:
                q_next = day_chains[days_sorted[idx_d + 1]][1]
                s_next = crisis_preloaded if estimate_atm_var(q_next) > 0.15 else s_d
                o_next = dc.recalibrate_day2_differentiable(q_next, cal_d, preloaded=s_next)
                next_oos_bps = round(o_next["fourier_vs_mkt_rmse_t2"] * 1e4, 2)
            all_daily_rolling.append({
                "era": era_name,
                "date": d,
                "n_slices": len(sl_d),
                "n_quotes": len(q_d),
                "surrogate": "M3_param_crisis" if v_est_d > 0.15 else "M3_param_wide",
                "v0": round(cal_d["v0_t1"], 5),
                "spot_vol_pct": round(100.0 * math.sqrt(cal_d["v0_t1"]), 2),
                "kappa": round(cal_d["kappa"], 4),
                "theta": round(cal_d["theta"], 5),
                "xi": round(cal_d["xi"], 4),
                "rho": round(cal_d["rho"], 4),
                "feller_satisfied": cal_d["feller_satisfied"],
                "calib_sec": cal_d["calib_sec"],
                "in_sample_bps": round(is_bps_d, 2),
                "next_day_oos_bps": next_oos_bps,
            })

    n_feller_viol = sum(1 for r in all_daily_rolling if not r["feller_satisfied"])
    print(f"\n[Stage 3/4] Completed 25-Day WRDS Rolling Calibration: {n_feller_viol}/{len(all_daily_rolling)} days violate Feller!", flush=True)

    # Stage 4: Single-surface B0 vs M3_sech_80 on 2020-03-16 COVID-19 All-Time VIX Peak (VIX = 82.69)
    p_covid = (covid_de_cal["kappa"], covid_de_cal["theta"], covid_de_cal["xi"], covid_de_cal["rho"])
    v0_covid = covid_de_cal["v0_t1"]
    print(f"\n[Stage 4/4] Training B0 vs M3_sech_80 on 2020-03-16 COVID-19 Peak: p={tuple(round(x,4) for x in p_covid)}, v0*={v0_covid:.4f} (vol={100*math.sqrt(v0_covid):.1f}%)", flush=True)

    vs_covid_grid = (0.04, 0.16, 0.36, 0.64, min(0.90, max(0.75, round(v0_covid, 2))))
    zs_covid_grid = tuple(np.linspace(-2.3, 2.3, 25).tolist())
    xg_cov, ref_cov = hgv.heston_eval_grid(p_covid, vs=vs_covid_grid, zs=zs_covid_grid)
    p_obj_cov = cvpinn.Heston(*p_covid)

    b0_ckpt = os.path.join(COVID_DIR, "B0_COVID_seed0.pt")
    b0_json = os.path.join(COVID_DIR, "B0_COVID_seed0.json")
    if os.path.exists(b0_ckpt) and os.path.exists(b0_json):
        ck_b0 = torch.load(b0_ckpt, map_location="cpu", weights_only=True)
        net_b0 = cvpinn.Mlp(3, width=64, depth=4)
        net_b0.load_state_dict(ck_b0["state_dict"])
        net_b0.eval()
        dom_c = cvpinn.Domain(v_max=1.0)
        b0_fn = lambda x, v, tau: net_b0(cvpinn.base_features(x, v, tau, dom_c))
        with open(b0_json) as f:
            b0_rec = json.load(f)
    else:
        t0 = time.perf_counter()
        b0_fn, net_b0, _ = cvpinn.build_and_train("B0", p_obj_cov, seed=0, v_max=1.0, term_weight=100.0, return_info=True)
        b0_sec = time.perf_counter() - t0
        pred_b0 = t2.evaluate_all_greeks(b0_fn, xg_cov, hgv.EVAL_TAUS, vs_covid_grid)
        sc_b0 = t2.score_all_six_buckets(pred_b0, ref_cov)
        torch.save({"state_dict": net_b0.state_dict(), "p_covid": p_covid, "v0_covid": v0_covid}, b0_ckpt)
        b0_rec = {"arm": "B0", "train_sec": round(b0_sec, 2), "overall": sc_b0["overall"]}
        with open(b0_json, "w") as f:
            json.dump(b0_rec, f, indent=2)

    m3_ckpt = os.path.join(COVID_DIR, "M3_sech_80_COVID_seed0.pt")
    m3_json = os.path.join(COVID_DIR, "M3_sech_80_COVID_seed0.json")
    cfg_cov_ovr = {"adam_steps": 3500, "lbfgs_rounds": 2, "lbfgs_per_round": 800, "n_lbfgs_base": 12288, "rar_pool": 32768, "rar_add": 3072}
    if os.path.exists(m3_ckpt) and os.path.exists(m3_json):
        ck_m3 = torch.load(m3_ckpt, map_location="cpu", weights_only=False)
        net_m3 = m3.build_m3_net(ck_m3["config"])
        net_m3.load_state_dict(ck_m3["state_dict"])
        net_m3.eval()
        m3_fn = m3.make_m3_price_from_net(ck_m3["config"], net_m3, p_obj_cov, v_max=1.0)
        with open(m3_json) as f:
            m3_rec = json.load(f)
    else:
        t0 = time.perf_counter()
        m3_fn, net_m3, info_m3 = m3.train_m3("M3_sech_80", p_obj_cov, seed=0, v_max=1.0, cfg_override=cfg_cov_ovr)
        m3_sec = time.perf_counter() - t0
        pred_m3 = t2.evaluate_all_greeks(m3_fn, xg_cov, hgv.EVAL_TAUS, vs_covid_grid)
        sc_m3 = t2.score_all_six_buckets(pred_m3, ref_cov)
        torch.save({"state_dict": net_m3.state_dict(), "config": info_m3["config"], "p_covid": p_covid, "v0_covid": v0_covid}, m3_ckpt)
        m3_rec = {"arm": "M3_sech_80", "train_sec": round(m3_sec, 2), "overall": sc_m3["overall"]}
        with open(m3_json, "w") as f:
            json.dump(m3_rec, f, indent=2)

    # Evaluate live 2,670-quote 2020-03-16 COVID-19 pricing
    sd_cov, ym_cov, _ = spx_w8._group_quotes(covid_anchor_quotes)
    cf_cov = spx_w8.heston_prices_by_slice(sd_cov, v0_covid, *p_covid)
    taus_f = np.concatenate([np.full(len(xs_i), t_i) for t_i, xs_i in sd_cov])
    xs_f = np.concatenate([xs_i for _, xs_i in sd_cov])
    mask_d = np.abs(xs_f) <= 2.45
    with torch.no_grad():
        xt_d = torch.tensor(np.clip(xs_f, -2.45, 2.45), dtype=torch.float64)
        vt_d = torch.full_like(xt_d, float(v0_covid))
        tt_d = torch.tensor(taus_f, dtype=torch.float64)
        cb0_cov = b0_fn(xt_d, vt_d, tt_d).numpy()
        cm3_cov = m3_fn(xt_d, vt_d, tt_d).numpy()

    covid_surface_summary = {
        "date": "2020-03-16",
        "n_slices": len(covid_anchor_slices),
        "n_quotes": len(ym_cov),
        "calibrated_params": {k: round(covid_de_cal[k], 5) for k in ("kappa", "theta", "xi", "rho", "v0_t1")},
        "spot_vol_pct": round(100.0 * math.sqrt(v0_covid), 2),
        "fourier_vs_mkt_bps": round(float(np.sqrt(np.mean((cf_cov - ym_cov) ** 2)) * 1e4), 2),
        "b0_vs_fourier_bps": round(float(np.sqrt(np.mean((cb0_cov[mask_d] - cf_cov[mask_d]) ** 2)) * 1e4), 2),
        "m3_vs_fourier_bps": round(float(np.sqrt(np.mean((cm3_cov[mask_d] - cf_cov[mask_d]) ** 2)) * 1e4), 2),
        "b0_vs_mkt_bps": round(float(np.sqrt(np.mean((cb0_cov[mask_d] - ym_cov[mask_d]) ** 2)) * 1e4), 2),
        "m3_vs_mkt_bps": round(float(np.sqrt(np.mean((cm3_cov[mask_d] - ym_cov[mask_d]) ** 2)) * 1e4), 2),
        "b0_crisis_greeks_overall": {k: round(b0_rec["overall"][k]["rel_rmse"], 4) for k in ("c", "cx", "gam", "cv", "cxv", "cvv")},
        "m3_crisis_greeks_overall": {k: round(m3_rec["overall"][k]["rel_rmse"], 4) for k in ("c", "cx", "gam", "cv", "cxv", "cvv")},
    }

    # OptionMetrics Vendor IV & Greek cross-check on 2020-03-16
    vendor_audit_covid = audit_optionmetrics_vendor_greeks(
        os.path.join(WRDS_DIR, "wrds_spx_2020_03_covid.csv"), "2020-03-16", p_covid, v0_covid, m3_fn, b0_fn
    )
    covid_surface_summary["optionmetrics_vendor_audit"] = vendor_audit_covid
    print(
        f"  [2020-03-16 COVID Surface] M3 vs Fourier={covid_surface_summary['m3_vs_fourier_bps']} bps "
        f"(B0={covid_surface_summary['b0_vs_fourier_bps']} bps) | M3 Crisis Gamma={covid_surface_summary['m3_crisis_greeks_overall']['gam']:.4f}, "
        f"Vega={covid_surface_summary['m3_crisis_greeks_overall']['cv']:.4f}",
        flush=True,
    )

    report = {
        "dataset": "Wharton WRDS OptionMetrics IvyDB US (SECID=108105, S&P 500 Index SPX/SPXW)",
        "coverage_years": "2008-2025 (5 Historical Regimes, 25 Trading Days, 434,763 Raw Rows, 119,374 Clean OTM Quotes)",
        "total_held_out_week_oos_quotes": sum(r["n_week_oos_quotes"] for r in era_reports),
        "feller_violated_days": n_feller_viol,
        "total_evaluated_days": len(all_daily_rolling),
        "era_benchmarks": era_reports,
        "daily_rolling_25days": all_daily_rolling,
        "covid_2020_03_16_pinn_study": covid_surface_summary,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # Build Figure 18: 3-panel multi-decade WRDS OptionMetrics empirical figure
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.1))
    fig.suptitle(
        "Wharton WRDS OptionMetrics IvyDB US Study (2008–2025, 434,763 SPX Quotes Across 5 Regimes)",
        fontsize=13, y=0.98
    )

    # Panel (a): 5-Era Full-Week Out-of-Sample RMSE (DE+Fourier vs Differentiable Surrogate)
    ax0 = axes[0]
    labels = ["2008 Oct\nGFC", "2020 Mar\nCOVID-19", "2021 Jul\nCalm Bull", "2022 Oct\nBear CPI", "2025 Aug\nLatest"]
    x_pos = np.arange(len(era_reports))
    w_bar = 0.36
    de_bars = [r["de_week_oos_bps"] for r in era_reports]
    dc_bars = [r["autograd_week_oos_bps"] for r in era_reports]
    b1 = ax0.bar(x_pos - w_bar / 2, de_bars, w_bar, label="SciPy DE + Exact Fourier", color="#2b5c8f")
    b2 = ax0.bar(x_pos + w_bar / 2, dc_bars, w_bar, label="PyTorch Autograd Surrogate", color="#e07b39")
    for idx_b, r in enumerate(era_reports):
        top_y = max(de_bars[idx_b], dc_bars[idx_b])
        ax0.text(x_pos[idx_b], top_y * 1.04, f"{r['speedup_x']:.0f}x", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#222222")
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(labels, fontsize=8.5)
    ax0.set_ylabel("Full-Week Held-Out OOS RMSE (bps)")
    ax0.set_title("5-Decade/Era Week OOS (96,172 Quotes)")
    ax0.set_ylim(0, max(max(de_bars), max(dc_bars)) * 1.24)
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8.0)

    # Panel (b): 2020-03-16 COVID-19 Black Monday (VIX = 82.69) IV Smiles across 3 maturities
    ax1 = axes[1]
    target_taus = [0.06, 0.25, 0.75]
    avail_taus = [t_i for t_i, _ in sd_cov]
    chosen_idx = []
    for tt in target_taus:
        best_i = int(np.argmin([abs(t_i - tt) for t_i in avail_taus]))
        if best_i not in chosen_idx:
            chosen_idx.append(best_i)
    colors_s = ["#1f77b4", "#2ca02c", "#9467bd"]
    for c_col, s_i in zip(colors_s, chosen_idx):
        tau_s, xs_s = sd_cov[s_i]
        # Market quotes for this tau
        ym_s = np.array([q[2] for q in covid_anchor_quotes if abs(q[0] - tau_s) < 1e-9 and abs(q[1]) <= 0.45])
        xm_s = np.array([q[1] for q in covid_anchor_quotes if abs(q[0] - tau_s) < 1e-9 and abs(q[1]) <= 0.45])
        ord_i = np.argsort(xm_s)
        xm_s, ym_s = xm_s[ord_i], ym_s[ord_i]
        iv_m = np.array([implied_vol_norm(float(c_v), float(x_v), float(tau_s)) for c_v, x_v in zip(ym_s, xm_s)])
        x_dense = np.linspace(max(float(xm_s.min()), -0.42), min(float(xm_s.max()), 0.35), 45)
        cf_d = spx_w8.heston_prices_by_slice([(tau_s, x_dense)], v0_covid, *p_covid)
        with torch.no_grad():
            xd_t = torch.tensor(x_dense, dtype=torch.float64)
            cm3_d = m3_fn(xd_t, torch.full_like(xd_t, float(v0_covid)), torch.full_like(xd_t, float(tau_s))).numpy()
        iv_f = np.array([implied_vol_norm(float(c_v), float(x_v), float(tau_s)) for c_v, x_v in zip(cf_d, x_dense)])
        iv_p = np.array([implied_vol_norm(float(c_v), float(x_v), float(tau_s)) for c_v, x_v in zip(cm3_d, x_dense)])
        step_pts = max(1, len(xm_s) // 18)
        ax1.scatter(xm_s[::step_pts], 100.0 * iv_m[::step_pts], s=16, alpha=0.65, color=c_col, label=f"WRDS $\\tau={tau_s:.2f}$y")
        ax1.plot(x_dense, 100.0 * iv_f, "-", lw=1.8, color=c_col)
        ax1.plot(x_dense, 100.0 * iv_p, "--", lw=1.5, color="#111111" if s_i == chosen_idx[0] else c_col,
                 label="M3_sech_80 (dashed)" if s_i == chosen_idx[0] else None)
    ax1.set_xlabel("Log-Forward Moneyness $x = \\ln(F/K)$")
    ax1.set_ylabel("Implied Volatility (%)")
    ax1.set_title("2020-03-16 COVID Crash Smiles (VIX = 82.69)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", fontsize=7.8, ncol=1)

    # Panel (c): 2020-03-16 Extreme-Variance Short-Maturity Gamma Profile (tau = 0.05, v = v0_covid)
    ax2 = axes[2]
    tau_g = 0.05
    w_g = hgv.w_exp_stable(v0_covid, tau_g, p_covid[0], p_covid[1])
    z_line = np.linspace(-2.2, 2.2, 61)
    x_line = np.sqrt(w_g) * z_line
    _, ref_g = hgv.heston_eval_grid(p_covid, taus=(tau_g,), vs=(v0_covid,), zs=tuple(z_line.tolist()))
    pred_m3_g = t2.evaluate_all_greeks(m3_fn, x_line[None, None, :], (tau_g,), (v0_covid,))
    pred_b0_g = t2.evaluate_all_greeks(b0_fn, x_line[None, None, :], (tau_g,), (v0_covid,))
    ax2.plot(z_line, ref_g["gam"][0, 0], "k-", lw=2.2, label="Exact Heston Fourier")
    ax2.plot(z_line, pred_m3_g["gam"][0, 0], "--", color="#2ca02c", lw=2.0, label="M3_sech_80 (Ours)")
    ax2.plot(z_line, pred_b0_g["gam"][0, 0], ":", color="#d62728", lw=2.0, label="B0 (Tuned Baseline)")
    ax2.set_xlabel("Scaled Moneyness $z = x / \\sqrt{w_{\\mathrm{exp}}}$")
    ax2.set_ylabel("Normalized Gamma $(c_{xx} - c_x)$")
    ax2.set_title(f"COVID Peak Gamma ($\\tau=0.05, v_0^*={v0_covid:.2f}$)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8.2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)
    print(f"Saved {OUT_JSON} and {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
