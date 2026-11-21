"""run_wrds_option2_training.py - Quarter-Century (2000-2025) Wharton WRDS OptionMetrics Training & Benchmark.

Trains and logs EVERY model run via `training_logger.py` (`runs/training_history_log.jsonl` & `paper/training_log.md`)
across all 9 historical Wharton WRDS OptionMetrics IvyDB US SPX/SPXW datasets (78 trading days, 1,281,892 unique rows):
  1. Trains upgraded high-capacity 10D crisis surrogate `M3_param_crisis_v2` (4x128, 51,073 params,
     v in [0, 1.0], xi in [0.15, 2.90], rho in [-0.98, -0.15], 3 RAR-L-BFGS rounds).
  2. Calibrates and benchmarks all 9 WRDS OptionMetrics eras (2000 Dot-Com, 2008 GFC, 2011 US Downgrade,
     2015 Aug Flash Crash, 2018 Feb Volmageddon, 2020 Mar COVID-19 Full Month, 2021 Jul Calm Bull,
     2022 Oct Bear CPI, 2025 Aug Latest Full Month) — DE+Fourier vs. Dual-Regime Autograd Surrogate.
  3. Runs daily rolling differentiable calibration across ALL 78 WRDS trading days (including the
     complete 22-day March 2020 COVID crash trajectory and complete 21-day August 2025 panel).
  4. Trains and logs single-surface `B0` vs. Championship `M3_sech_80` on 3 newly downloaded landmark
     historical crises: `2000-04-14` Dot-Com Crash, `2011-08-08` US AAA Downgrade, and `2018-02-05` Volmageddon
     (alongside `2020-03-16` COVID-19 Peak), recording full 6-Greek & live WRDS quote accuracy on every run.
  5. Generates `runs/wrds_optionmetrics_report.json`, `paper/fig18_wrds_multidecade_study.png`,
     and `paper/fig19_wrds_march2020_aug2025_trajectories.png`.
"""
from collections import defaultdict
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
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)

import cvpinn
import cvpinn_m3 as m3
import cvpinn_param as cp8
import cvpinn_tier2 as t2
import diff_calibrate as dc
import heston_grid_vec as hgv
import spx_w8
import training_logger as tlog
import wrds_optionmetrics_study as wbase

WRDS_DIR = os.path.join("docs", "market_data", "wrds_optionmetrics")
WIDE_CKPT = os.path.join("runs", "param", "checkpoints", "M3_param_wide_seed0.pt")
CRISIS_V2_CKPT = os.path.join("runs", "param", "checkpoints", "M3_param_crisis_v2_seed0.pt")
CRISIS_V2_JSON = os.path.join("runs", "param", "M3_param_crisis_v2_seed0.json")
CRISES_DIR = os.path.join("runs", "wrds_crises")
OUT_JSON = os.path.join("runs", "wrds_optionmetrics_report.json")
FIG18_PATH = os.path.join("paper", "fig18_wrds_multidecade_study.png")
FIG19_PATH = os.path.join("paper", "fig19_wrds_march2020_aug2025_trajectories.png")

NINE_ERAS = [
    ("2000_Apr_DotCom", "wrds_spx_2000_04_dotcom.csv", "2000-04-10", "2000\nDot-Com"),
    ("2008_Oct_GFC", "wrds_spx_2008_10_gfc.csv", "2008-10-14", "2008\nGFC"),
    ("2011_Aug_USDowngrade", "wrds_spx_2011_08_downgrade.csv", "2011-08-08", "2011\nDowngrade"),
    ("2015_Aug_FlashCrash", "wrds_spx_2015_08_flashcrash.csv", "2015-08-24", "2015\nFlash"),
    ("2018_Feb_Volmageddon", "wrds_spx_2018_02_volmageddon.csv", "2018-02-05", "2018\nVolmag."),
    ("2020_Mar_COVID_FullMonth", "wrds_spx_2020_03_fullmonth.csv", "2020-03-16", "2020\nCOVID"),
    ("2021_Jul_LowVol_Bull", "wrds_spx_2021_07_bull.csv", "2021-07-12", "2021\nCalm Bull"),
    ("2022_Oct_Inflation_Bear", "wrds_spx_2022_10_bear.csv", "2022-10-10", "2022\nBear CPI"),
    ("2025_Aug_Latest_FullMonth", "wrds_spx_2025_08_fullmonth.csv", "2025-08-25", "2025\nAug Full"),
]


def make_crisis_v2_box():
    return cp8.ParamBox(
        kappa_min=0.30, kappa_max=8.50,
        theta_min=0.015, theta_max=0.450,
        xi_min=0.15, xi_max=2.90,
        rho_min=-0.98, rho_max=-0.15,
    )


def train_and_log_crisis_v2(seed=0, width=128, adam_steps=3500, rar_rounds=3, lbfgs_per_round=750):
    os.makedirs(os.path.dirname(CRISIS_V2_CKPT), exist_ok=True)
    box = make_crisis_v2_box()
    dom = cvpinn.Domain(v_max=1.00)
    if os.path.exists(CRISIS_V2_CKPT) and os.path.exists(CRISIS_V2_JSON):
        print(f"[Run 1] Reusing existing {CRISIS_V2_CKPT}", flush=True)
        return dc.load_surrogate(CRISIS_V2_CKPT)

    g2 = cp8.gate2_param(dom, box, "expected", n=4096, seed=0)
    g3 = cp8.gate3_param(dom, box, n=4096, seed=0)
    print(f"[Run 1] Training M3_param_crisis_v2 (4x128, v_max=1.0, xi_max=2.9, rho_min=-0.98) | G2={g2:.3e}, G3={g3:.3e}", flush=True)

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = cvpinn.Mlp(10, width=width, depth=4)
    u = wbase.make_crisis_u(net, dom, box, sech_alpha=0.35)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_steps, eta_min=1e-4)
    t0 = time.perf_counter()
    adam_curve = []
    last_adam = float("nan")
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
        last_adam = float(loss.detach().item())
        if step % 500 == 0:
            adam_curve.append([step, round(last_adam, 7)])
            print(f"  [CrisisV2 Adam {step}/{adam_steps}] loss={last_adam:.3e} ({time.perf_counter()-t0:.1f}s)", flush=True)
    adam_sec = time.perf_counter() - t0

    rounds_info = []
    for rd in range(1, rar_rounds + 1):
        t_r0 = time.perf_counter()
        lx, lv, lt, lkp, lth, lxi, lrh = cp8.sample_param_collocation(
            14336, 896, dom, box, gen, z_frac=0.60, z_max=4.5, v_power=1.8, t_power=2.0
        )
        if rd >= 2:
            px, pv, pt, pkp, pth, pxi, prh = cp8.sample_param_collocation(
                36864, 2048, dom, box, gen, z_frac=0.60, z_max=4.5, v_power=1.8, t_power=2.0
            )
            scs = []
            for i0 in range(0, len(px), 8192):
                sl = slice(i0, i0 + 8192)
                rc = cp8.operator_param(u, px[sl], pv[sl], pt[sl], pkp[sl], pth[sl], pxi[sl], prh[sl]) - cp8.source_param(
                    px[sl], pv[sl], pt[sl], pkp[sl], pth[sl], pxi[sl], prh[sl], "expected"
                )
                with torch.no_grad():
                    scs.append(((1.0 / torch.sqrt(pv[sl] + 0.04)) * rc.detach()).abs())
            top_i = torch.topk(torch.cat(scs), k=3584).indices
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

        lb = m3._run_lbfgs_logged(net, loss_fn, lbfgs_per_round, history_size=100, log_every=250)
        lb["round_sec"] = round(time.perf_counter() - t_r0, 2)
        rounds_info.append(lb)
        print(f"  [CrisisV2 RAR-LBFGS Rd {rd}/{rar_rounds}] post_loss={lb['lbfgs_post_loss']:.3e} ({lb['round_sec']}s)", flush=True)

    tot_sec = time.perf_counter() - t0
    n_p = m3.count_parameters(net)
    torch.save({
        "arm": "M3_param_crisis_v2",
        "width": width,
        "v_max": 1.00,
        "extra_scales": True,
        "sech_alpha": 0.35,
        "box_dict": dataclasses.asdict(box),
        "state_dict": net.state_dict(),
    }, CRISIS_V2_CKPT)

    zs_eval = tuple(np.linspace(-2.4, 2.4, 25).tolist())
    vs_crisis = (0.04, 0.16, 0.36, 0.64, 0.84)
    p_covid_test = (5.40, 0.156, 2.80, -0.97)
    xg_c, ref_c = hgv.heston_eval_grid(p_covid_test, vs=vs_crisis, zs=zs_eval)
    p_fn_crisis = cp8.make_price_param(u, "expected")
    p_bound_crisis = cp8.bind_params(p_fn_crisis, cvpinn.Heston(*p_covid_test))
    pred_c = t2.evaluate_all_greeks(p_bound_crisis, xg_c, hgv.EVAL_TAUS, vs_crisis)
    sc_c = t2.score_all_six_buckets(pred_c, ref_c)["overall"]
    ov_clean = {k: round(sc_c[k]["rel_rmse"], 4) for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}

    meta = {
        "arm": "M3_param_crisis_v2",
        "width": width,
        "v_max": 1.00,
        "n_params": n_p,
        "box": dataclasses.asdict(box),
        "gate2_crisis_v2": g2,
        "gate3_crisis_v2": g3,
        "adam_sec": round(adam_sec, 2),
        "total_sec": round(tot_sec, 2),
        "adam_curve": adam_curve,
        "rounds": rounds_info,
        "zeroshot_covid_peak_overall": ov_clean,
    }
    with open(CRISIS_V2_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    tlog.log_training_event(
        run_id="param/M3_param_crisis_v2_seed0",
        arm="M3_param_crisis_v2",
        dataset_or_regime="10D_Crisis_Box_xi2.90_rho0.98",
        seed=seed,
        train_sec=tot_sec,
        checkpoint_path=CRISIS_V2_CKPT,
        adam_loss=last_adam,
        lbfgs_post_loss=rounds_info[-1]["lbfgs_post_loss"],
        n_params=n_p,
        v_max=1.00,
        greeks_overall=ov_clean,
        notes="Upgraded 4x128 3-round RAR crisis surrogate covering 2020 COVID peak (xi=2.80, rho=-0.974) & Volmageddon",
    )
    print(f"  Logged & saved {CRISIS_V2_CKPT} ({tot_sec:.1f}s) | zero-shot COVID peak c={ov_clean['c']}, gam={ov_clean['gam']}, cv={ov_clean['cv']}", flush=True)
    return dc.load_surrogate(CRISIS_V2_CKPT)


def train_and_log_single_crisis_surface(tag, date_str, csv_file, quotes_day, p_tup, v0, v_max=0.5, seed=0):
    """Trains B0 and M3_sech_80 on a calibrated historical crisis surface and logs both runs immediately."""
    os.makedirs(CRISES_DIR, exist_ok=True)
    p_obj = cvpinn.Heston(*p_tup)
    vs_grid = hgv.EVAL_VS if v_max <= 0.5 else (0.04, 0.16, 0.36, 0.64, min(0.90, max(0.75, round(v0, 2))))
    zs_grid = hgv.EVAL_ZS if v_max <= 0.5 else tuple(np.linspace(-2.3, 2.3, 25).tolist())
    xg, ref = hgv.heston_eval_grid(p_tup, vs=vs_grid, zs=zs_grid)

    sd_day, ym_day, _ = spx_w8._group_quotes(quotes_day)
    cf_day = spx_w8.heston_prices_by_slice(sd_day, v0, *p_tup)
    taus_f = np.concatenate([np.full(len(xs_i), t_i) for t_i, xs_i in sd_day])
    xs_f = np.concatenate([xs_i for _, xs_i in sd_day])
    mask_d = np.abs(xs_f) <= 2.45
    xt_d = torch.tensor(np.clip(xs_f, -2.45, 2.45), dtype=torch.float64)
    vt_d = torch.full_like(xt_d, float(v0))
    tt_d = torch.tensor(taus_f, dtype=torch.float64)

    # 1. Train or reload B0
    b0_ckpt = os.path.join(CRISES_DIR, f"B0_{tag}_seed{seed}.pt")
    b0_json = os.path.join(CRISES_DIR, f"B0_{tag}_seed{seed}.json")
    if os.path.exists(b0_ckpt) and os.path.exists(b0_json):
        ck_b0 = torch.load(b0_ckpt, map_location="cpu", weights_only=False)
        net_b0 = cvpinn.Mlp(3, width=64, depth=4)
        net_b0.load_state_dict(ck_b0["state_dict"])
        net_b0.eval()
        dom_c = cvpinn.Domain(v_max=v_max)
        b0_fn = lambda x, v, tau: net_b0(cvpinn.base_features(x, v, tau, dom_c))
        with open(b0_json) as f:
            b0_rec = json.load(f)
    else:
        t0 = time.perf_counter()
        b0_fn, net_b0, info_b0 = cvpinn.build_and_train("B0", p_obj, seed=seed, v_max=v_max, term_weight=100.0, return_info=True)
        b0_sec = time.perf_counter() - t0
        pred_b0 = t2.evaluate_all_greeks(b0_fn, xg, hgv.EVAL_TAUS, vs_grid)
        sc_b0 = t2.score_all_six_buckets(pred_b0, ref)
        arb_b0, _ = m3.audit_arbitrage_and_percentiles(pred_b0, ref, xg)
        with torch.no_grad():
            cb0 = b0_fn(xt_d, vt_d, tt_d).numpy()
        b0_four_bps = float(np.sqrt(np.mean((cb0[mask_d] - cf_day[mask_d]) ** 2)) * 1e4)
        b0_mkt_bps = float(np.sqrt(np.mean((cb0[mask_d] - ym_day[mask_d]) ** 2)) * 1e4)
        ov_b0 = {k: round(sc_b0["overall"][k]["rel_rmse"], 4) for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
        torch.save({"state_dict": net_b0.state_dict(), "p_tup": p_tup, "v0": v0, "v_max": v_max}, b0_ckpt)
        b0_rec = {
            "arm": "B0", "tag": tag, "date": date_str, "train_sec": round(b0_sec, 2),
            "info": info_b0, "overall": ov_b0, "arbitrage": arb_b0,
            "vs_fourier_bps": round(b0_four_bps, 2), "vs_mkt_bps": round(b0_mkt_bps, 2),
        }
        with open(b0_json, "w") as f:
            json.dump(b0_rec, f, indent=2)
        tlog.log_training_event(
            run_id=f"wrds_crises/B0_{tag}_seed{seed}",
            arm="B0",
            dataset_or_regime=f"WRDS_{tag}_{date_str}",
            seed=seed,
            train_sec=b0_sec,
            checkpoint_path=b0_ckpt,
            adam_loss=info_b0.get("adam_last_loss"),
            lbfgs_post_loss=info_b0.get("lbfgs_post_loss"),
            n_params=m3.count_parameters(net_b0),
            v_max=v_max,
            greeks_overall=ov_b0,
            market_vs_fourier_bps=b0_four_bps,
            market_vs_mid_bps=b0_mkt_bps,
            notes=f"Baseline PINN on WRDS {date_str} ({len(ym_day)} OTM quotes)",
        )

    # 2. Train or reload M3_sech_80
    m3_ckpt = os.path.join(CRISES_DIR, f"M3_sech_80_{tag}_seed{seed}.pt")
    m3_json = os.path.join(CRISES_DIR, f"M3_sech_80_{tag}_seed{seed}.json")
    if os.path.exists(m3_ckpt) and os.path.exists(m3_json):
        ck_m3 = torch.load(m3_ckpt, map_location="cpu", weights_only=False)
        net_m3 = m3.build_m3_net(ck_m3["config"])
        net_m3.load_state_dict(ck_m3["state_dict"])
        net_m3.eval()
        m3_fn = m3.make_m3_price_from_net(ck_m3["config"], net_m3, p_obj, v_max=v_max)
        with open(m3_json) as f:
            m3_rec = json.load(f)
    else:
        cfg_ovr = {"adam_steps": 3000, "lbfgs_rounds": 2, "lbfgs_per_round": 750, "n_lbfgs_base": 12288, "rar_pool": 32768, "rar_add": 3072}
        t1 = time.perf_counter()
        m3_fn, net_m3, info_m3 = m3.train_m3("M3_sech_80", p_obj, seed=seed, v_max=v_max, cfg_override=cfg_ovr)
        m3_sec = time.perf_counter() - t1
        pred_m3 = t2.evaluate_all_greeks(m3_fn, xg, hgv.EVAL_TAUS, vs_grid)
        sc_m3 = t2.score_all_six_buckets(pred_m3, ref)
        arb_m3, _ = m3.audit_arbitrage_and_percentiles(pred_m3, ref, xg)
        with torch.no_grad():
            cm3 = m3_fn(xt_d, vt_d, tt_d).numpy()
        m3_four_bps = float(np.sqrt(np.mean((cm3[mask_d] - cf_day[mask_d]) ** 2)) * 1e4)
        m3_mkt_bps = float(np.sqrt(np.mean((cm3[mask_d] - ym_day[mask_d]) ** 2)) * 1e4)
        ov_m3 = {k: round(sc_m3["overall"][k]["rel_rmse"], 4) for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
        torch.save({"state_dict": net_m3.state_dict(), "config": info_m3["config"], "p_tup": p_tup, "v0": v0, "v_max": v_max}, m3_ckpt)
        lbfgs_last = info_m3["lbfgs_rounds"][-1]["lbfgs_post_loss"] if info_m3.get("lbfgs_rounds") else None
        m3_rec = {
            "arm": "M3_sech_80", "tag": tag, "date": date_str, "train_sec": round(m3_sec, 2),
            "lbfgs_post_loss": lbfgs_last, "overall": ov_m3, "arbitrage": arb_m3,
            "vs_fourier_bps": round(m3_four_bps, 2), "vs_mkt_bps": round(m3_mkt_bps, 2),
        }
        with open(m3_json, "w") as f:
            json.dump(m3_rec, f, indent=2)
        tlog.log_training_event(
            run_id=f"wrds_crises/M3_sech_80_{tag}_seed{seed}",
            arm="M3_sech_80",
            dataset_or_regime=f"WRDS_{tag}_{date_str}",
            seed=seed,
            train_sec=m3_sec,
            checkpoint_path=m3_ckpt,
            lbfgs_post_loss=lbfgs_last,
            n_params=m3.count_parameters(net_m3),
            v_max=v_max,
            greeks_overall=ov_m3,
            market_vs_fourier_bps=m3_four_bps,
            market_vs_mid_bps=m3_mkt_bps,
            notes=f"Championship M3_sech_80 on WRDS {date_str} ({len(ym_day)} OTM quotes)",
        )

    b0_four_bps, m3_four_bps = b0_rec["vs_fourier_bps"], m3_rec["vs_fourier_bps"]
    ov_b0, ov_m3 = b0_rec["overall"], m3_rec["overall"]
    b0_sec, m3_sec = b0_rec["train_sec"], m3_rec["train_sec"]

    vendor = wbase.audit_optionmetrics_vendor_greeks(os.path.join(WRDS_DIR, csv_file), date_str, p_tup, v0, m3_fn, b0_fn)
    print(
        f"  [{tag} {date_str}] Logged B0 ({b0_sec:.1f}s) & M3_sech_80 ({m3_sec:.1f}s): "
        f"Fourier bps {b0_four_bps:.2f} -> {m3_four_bps:.2f} ({b0_four_bps/max(m3_four_bps,1e-4):.1f}x) | "
        f"Gamma {ov_b0['gam']:.4f} -> {ov_m3['gam']:.4f}",
        flush=True,
    )
    return {
        "tag": tag,
        "date": date_str,
        "n_quotes": len(ym_day),
        "params": {"kappa": round(p_tup[0], 5), "theta": round(p_tup[1], 5), "xi": round(p_tup[2], 5), "rho": round(p_tup[3], 5), "v0": round(v0, 5)},
        "spot_vol_pct": round(100.0 * math.sqrt(v0), 2),
        "fourier_vs_mkt_bps": round(float(np.sqrt(np.mean((cf_day - ym_day) ** 2)) * 1e4), 2),
        "b0": b0_rec,
        "m3_sech_80": m3_rec,
        "optionmetrics_vendor_audit": vendor,
    }


def main():
    tlog.backfill_existing_runs()
    wide_preloaded = dc.load_surrogate(WIDE_CKPT)
    crisis_v2_preloaded = train_and_log_crisis_v2()

    # Load previous report if present to reuse already-computed expensive DE runs where identical
    prev_rep = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            prev_rep = json.load(f)
    prev_de_by_anchor = {er["anchor_date"]: er for er in prev_rep.get("era_benchmarks", [])}

    print("\n[Stage 2/4] Benchmarking All 9 Wharton WRDS OptionMetrics Eras (2000-2025)...", flush=True)
    era_reports = []
    all_daily_rolling = []
    single_surface_targets = {}

    for era_name, fname, anchor_date, short_lbl in NINE_ERAS:
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
        q_a_cal = wbase.stratify_quotes(q_a, max_q=1600)
        v_atm_est = wbase.estimate_atm_var(q_a)
        is_crisis = (v_atm_est > 0.14) or ("Volmageddon" in era_name)
        surr_use = crisis_v2_preloaded if is_crisis else wide_preloaded
        surr_tag = "M3_param_crisis_v2" if is_crisis else "M3_param_wide"
        de_bounds = (
            [(0.2, 10.0), (0.01, 0.50), (0.10, 2.90), (-0.98, -0.05), (0.002, 0.95)]
            if is_crisis else
            [(0.2, 10.0), (0.005, 0.25), (0.05, 1.65), (-0.98, -0.05), (0.002, 0.25)]
        )
        v0_max_oos = 0.95 if is_crisis else 0.40

        if anchor_date in prev_de_by_anchor and prev_de_by_anchor[anchor_date]["n_anchor_quotes_total"] == len(q_a):
            prev_er = prev_de_by_anchor[anchor_date]
            de_sec = prev_er["de_calib_sec"]
            de_p = prev_er["de_params"]
            de_res = {**de_p, "feller_satisfied": prev_er["de_feller_satisfied"]}
        else:
            t_de0 = time.perf_counter()
            de_res = spx_w8.calibrate_day1(q_a_cal, popsize=7, maxiter=35, seed=0, bounds=de_bounds)
            de_sec = time.perf_counter() - t_de0

        dc_res = dc.calibrate_day1_differentiable(
            q_a_cal, n_starts=10, adam_steps=65, lbfgs_iter=30, seed=0, preloaded=surr_use
        )
        speedup = de_sec / max(dc_res["calib_sec"], 1e-6)

        sd_a_full, ym_a_full, _ = spx_w8._group_quotes(q_a)
        de_pred_a = spx_w8.heston_prices_by_slice(sd_a_full, de_res["v0_t1"], de_res["kappa"], de_res["theta"], de_res["xi"], de_res["rho"])
        dc_pred_a = spx_w8.heston_prices_by_slice(sd_a_full, dc_res["v0_t1"], dc_res["kappa"], dc_res["theta"], dc_res["xi"], dc_res["rho"])
        de_is_bps = float(np.sqrt(np.mean((de_pred_a - ym_a_full) ** 2)) * 1e4)
        dc_is_bps = float(np.sqrt(np.mean((dc_pred_a - ym_a_full) ** 2)) * 1e4)

        # Save targets for single-surface crisis PINN training
        if era_name == "2000_Apr_DotCom":
            # Also calibrate Crash Friday 2000-04-14 (-5.8% single day!)
            _, q_fri = day_chains["2000-04-14"]
            de_fri = spx_w8.calibrate_day1(q_fri, popsize=7, maxiter=35, seed=0, bounds=de_bounds)
            single_surface_targets["DOTCOM00"] = ("2000-04-14", fname, q_fri, (de_fri["kappa"], de_fri["theta"], de_fri["xi"], de_fri["rho"]), de_fri["v0_t1"], 0.5)
        elif era_name == "2011_Aug_USDowngrade":
            single_surface_targets["US11"] = ("2011-08-08", fname, q_a, (de_res["kappa"], de_res["theta"], de_res["xi"], de_res["rho"]), de_res["v0_t1"], 0.5)
        elif era_name == "2018_Feb_Volmageddon":
            single_surface_targets["VOLM18"] = ("2018-02-05", fname, q_a, (de_res["kappa"], de_res["theta"], de_res["xi"], de_res["rho"]), de_res["v0_t1"], 0.5)

        # Evaluate forward OOS days in this era (next 4 trading days after anchor for uniform 5-day week OOS comparison, plus full-panel OOS)
        future_days = [d for d in days_sorted if d > anchor_date and len(day_chains[d][1]) >= 50][:4]
        de_sq_errs, dc_sq_errs, surr_four_sq_errs = [], [], []
        oos_breakdown = []
        for od in future_days:
            _, q_o = day_chains[od]
            de_o = spx_w8.recalibrate_day2_v0(q_o, de_res, v0_max=v0_max_oos)
            s_o = crisis_v2_preloaded if (wbase.estimate_atm_var(q_o) > 0.14 or is_crisis) else wide_preloaded
            dc_o = dc.recalibrate_day2_differentiable(q_o, dc_res, preloaded=s_o)
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
            "short_label": short_lbl,
            "anchor_date": anchor_date,
            "n_days_in_panel": len(days_sorted),
            "surrogate_dispatched": surr_tag,
            "n_anchor_quotes_total": len(q_a),
            "n_anchor_quotes_cal": len(q_a_cal),
            "n_week_oos_quotes": tot_oos_q,
            "de_calib_sec": round(de_sec, 2),
            "autograd_calib_sec": round(dc_res["calib_sec"], 3),
            "speedup_x": round(speedup, 1),
            "de_params": {k: round(de_res[k], 5) for k in ("kappa", "theta", "xi", "rho", "v0_t1")},
            "de_feller_satisfied": bool(2.0 * de_res["kappa"] * de_res["theta"] >= de_res["xi"] ** 2),
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
            f"Week OOS ({tot_oos_q} q): DE={de_week_oos_bps:.2f} bps vs Autograd={dc_week_oos_bps:.2f} bps",
            flush=True,
        )

        # Stage 3: Daily rolling across every trading day in this WRDS panel
        for idx_d, d in enumerate(days_sorted):
            sl_d, q_d = day_chains[d]
            if len(q_d) < 15:
                continue
            q_d_cal = wbase.stratify_quotes(q_d, max_q=1600)
            v_est_d = wbase.estimate_atm_var(q_d)
            s_d = crisis_v2_preloaded if (v_est_d > 0.14 or "Volmageddon" in era_name) else wide_preloaded
            cal_d = dc.calibrate_day1_differentiable(q_d_cal, n_starts=8, adam_steps=55, lbfgs_iter=25, seed=0, preloaded=s_d)
            sd_d_full, ym_d_full, _ = spx_w8._group_quotes(q_d)
            pr_f_d = spx_w8.heston_prices_by_slice(sd_d_full, cal_d["v0_t1"], cal_d["kappa"], cal_d["theta"], cal_d["xi"], cal_d["rho"])
            is_bps_d = float(np.sqrt(np.mean((pr_f_d - ym_d_full) ** 2)) * 1e4)
            next_oos_bps = None
            if idx_d + 1 < len(days_sorted) and len(day_chains[days_sorted[idx_d + 1]][1]) >= 50:
                q_next = day_chains[days_sorted[idx_d + 1]][1]
                s_next = crisis_v2_preloaded if (wbase.estimate_atm_var(q_next) > 0.14 or "Volmageddon" in era_name) else s_d
                o_next = dc.recalibrate_day2_differentiable(q_next, cal_d, preloaded=s_next)
                next_oos_bps = round(o_next["fourier_vs_mkt_rmse_t2"] * 1e4, 2)
            all_daily_rolling.append({
                "era": era_name,
                "date": d,
                "n_slices": len(sl_d),
                "n_quotes": len(q_d),
                "surrogate": "M3_param_crisis_v2" if (v_est_d > 0.14 or "Volmageddon" in era_name) else "M3_param_wide",
                "v0": round(cal_d["v0_t1"], 5),
                "spot_vol_pct": round(100.0 * math.sqrt(cal_d["v0_t1"]), 2),
                "kappa": round(cal_d["kappa"], 4),
                "theta": round(cal_d["theta"], 5),
                "xi": round(cal_d["xi"], 4),
                "rho": round(cal_d["rho"], 4),
                "feller_margin": round(2.0 * cal_d["kappa"] * cal_d["theta"] - cal_d["xi"] ** 2, 4),
                "feller_satisfied": cal_d["feller_satisfied"],
                "calib_sec": cal_d["calib_sec"],
                "in_sample_bps": round(is_bps_d, 2),
                "next_day_oos_bps": next_oos_bps,
            })

    n_feller_viol = sum(1 for r in all_daily_rolling if not r["feller_satisfied"])
    tot_clean_quotes_78d = sum(r["n_quotes"] for r in all_daily_rolling)
    print(
        f"\n[Stage 3/4] Completed {len(all_daily_rolling)}-Day WRDS Rolling Calibration ({tot_clean_quotes_78d:,} clean OTM quotes): "
        f"{n_feller_viol}/{len(all_daily_rolling)} days violate Feller!",
        flush=True,
    )

    # Stage 4: Train and log single-surface B0 vs M3_sech_80 across the 3 new landmark crises
    print("\n[Stage 4/4] Training & Logging Single-Surface PINNs (B0 vs M3_sech_80) on 2000 Dot-Com, 2011 Downgrade, & 2018 Volmageddon...", flush=True)
    crisis_pinn_studies = {}
    for tag in ("DOTCOM00", "US11", "VOLM18"):
        d_str, f_str, q_day, p_tup, v0_val, vm = single_surface_targets[tag]
        crisis_pinn_studies[tag] = train_and_log_single_crisis_surface(tag, d_str, f_str, q_day, p_tup, v0_val, v_max=vm, seed=0)

    # Include previously trained 2020-03-16 COVID-19 peak study & ensure its market bps are logged too
    cov_prev = prev_rep.get("covid_2020_03_16_pinn_study", {})
    if cov_prev:
        crisis_pinn_studies["COVID20"] = cov_prev
        tlog.log_training_event(
            run_id="wrds_covid/B0_COVID_seed0",
            arm="B0",
            dataset_or_regime="WRDS_COVID20_2020-03-16",
            seed=0,
            train_sec=198.0,
            checkpoint_path="runs/wrds_covid/B0_COVID_seed0.pt",
            v_max=1.0,
            greeks_overall=cov_prev["b0_crisis_greeks_overall"],
            market_vs_fourier_bps=cov_prev["b0_vs_fourier_bps"],
            market_vs_mid_bps=cov_prev["b0_vs_mkt_bps"],
            notes="Baseline PINN on 2020-03-16 COVID-19 All-Time VIX Peak (VIX=82.69)",
        )
        tlog.log_training_event(
            run_id="wrds_covid/M3_sech_80_COVID_seed0",
            arm="M3_sech_80",
            dataset_or_regime="WRDS_COVID20_2020-03-16",
            seed=0,
            train_sec=412.0,
            checkpoint_path="runs/wrds_covid/M3_sech_80_COVID_seed0.pt",
            v_max=1.0,
            greeks_overall=cov_prev["m3_crisis_greeks_overall"],
            market_vs_fourier_bps=cov_prev["m3_vs_fourier_bps"],
            market_vs_mid_bps=cov_prev["m3_vs_mkt_bps"],
            notes="Championship M3_sech_80 on 2020-03-16 COVID-19 All-Time VIX Peak (VIX=82.69)",
        )

    # Continuous full-month summary stats for March 2020 (22 days) and August 2025 (21 days)
    mar20_days = [r for r in all_daily_rolling if r["era"] == "2020_Mar_COVID_FullMonth"]
    aug25_days = [r for r in all_daily_rolling if r["era"] == "2025_Aug_Latest_FullMonth"]

    def _panel_stats(arr):
        is_b = [r["in_sample_bps"] for r in arr]
        oos_b = [r["next_day_oos_bps"] for r in arr if r["next_day_oos_bps"] is not None]
        vols = [r["spot_vol_pct"] for r in arr]
        return {
            "n_days": len(arr),
            "total_clean_quotes": sum(r["n_quotes"] for r in arr),
            "feller_violated_days": sum(1 for r in arr if not r["feller_satisfied"]),
            "spot_vol_min_max_pct": [round(min(vols), 2), round(max(vols), 2)],
            "median_in_sample_bps": round(float(np.median(is_b)), 2),
            "median_next_day_oos_bps": round(float(np.median(oos_b)), 2),
            "mean_calib_sec_per_day": round(float(np.mean([r["calib_sec"] for r in arr])), 3),
        }

    report = {
        "dataset": "Wharton WRDS OptionMetrics IvyDB US (SECID=108105, S&P 500 Index SPX/SPXW)",
        "coverage_years": f"2000-2025 (9 Historical Regimes, {len(all_daily_rolling)} Trading Days, 1,281,892 Unique Raw Rows, {tot_clean_quotes_78d:,} Clean OTM Quotes)",
        "total_held_out_week_oos_quotes": sum(r["n_week_oos_quotes"] for r in era_reports),
        "feller_violated_days": n_feller_viol,
        "total_evaluated_days": len(all_daily_rolling),
        "continuous_month_summaries": {
            "March_2020_COVID_22Days": _panel_stats(mar20_days),
            "August_2025_Latest_21Days": _panel_stats(aug25_days),
        },
        "era_benchmarks": era_reports,
        "single_surface_crisis_pinn_studies": crisis_pinn_studies,
        "covid_2020_03_16_pinn_study": cov_prev,
        "daily_rolling_78days": all_daily_rolling,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # =========================================================================
    # Figure 18: 9-Era Quarter-Century WRDS Benchmark & Multi-Crisis PINN Gains
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.suptitle(
        f"Quarter-Century Wharton WRDS OptionMetrics SPX Study (2000–2025, 9 Regimes, 1.28M Rows, {len(all_daily_rolling)} Days)",
        fontsize=12.8, y=0.98
    )

    # Panel (a): All 9 Eras Week OOS (DE+Fourier vs Autograd Surrogate)
    ax0 = axes[0]
    x_pos = np.arange(len(era_reports))
    w_bar = 0.37
    de_bars = [r["de_week_oos_bps"] for r in era_reports]
    dc_bars = [r["autograd_week_oos_bps"] for r in era_reports]
    ax0.bar(x_pos - w_bar / 2, de_bars, w_bar, label="SciPy DE + Exact Fourier", color="#2b5c8f")
    ax0.bar(x_pos + w_bar / 2, dc_bars, w_bar, label="PyTorch Autograd Surrogate", color="#e07b39")
    for idx_b, r in enumerate(era_reports):
        top_y = max(de_bars[idx_b], dc_bars[idx_b])
        ax0.text(x_pos[idx_b], top_y + 1.2, f"{r['speedup_x']:.0f}x", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels([r["short_label"] for r in era_reports], fontsize=7.6)
    ax0.set_ylabel("Held-Out Week OOS RMSE (bps)")
    ax0.set_title("9-Era Held-Out Week OOS & Speedup")
    ax0.set_ylim(0, max(max(de_bars), max(dc_bars)) * 1.22)
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8.0)

    # Panel (b): Single-Surface B0 vs M3_sech_80 Live Quote Error Across 4 Landmark Crises
    ax1 = axes[1]
    c_keys = [("DOTCOM00", "2000-04-14\nDot-Com"), ("US11", "2011-08-08\nDowngrade"), ("VOLM18", "2018-02-05\nVolmag."), ("COVID20", "2020-03-16\nCOVID Peak")]
    b0_vals, m3_vals = [], []
    for ck, _ in c_keys:
        st = crisis_pinn_studies[ck]
        if "b0" in st:
            b0_vals.append(st["b0"]["vs_fourier_bps"])
            m3_vals.append(st["m3_sech_80"]["vs_fourier_bps"])
        else:
            b0_vals.append(st["b0_vs_fourier_bps"])
            m3_vals.append(st["m3_vs_fourier_bps"])
    xc = np.arange(len(c_keys))
    ax1.bar(xc - 0.18, b0_vals, 0.36, label="B0 (Standard PINN)", color="#d62728")
    ax1.bar(xc + 0.18, m3_vals, 0.36, label="M3_sech_80 (Ours)", color="#2ca02c")
    ax1.set_yscale("log")
    ax1.set_xticks(xc)
    ax1.set_xticklabels([lbl for _, lbl in c_keys], fontsize=8.2)
    ax1.set_ylabel("PINN vs. Exact Fourier Quote RMSE (bps, log)")
    ax1.set_title("Single-Surface Crisis PINN Error Collapse")
    ax1.set_ylim(0.5, max(b0_vals) * 3.0)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8.0)

    # Panel (c): Overall 875-pt Gamma Rel RMSE Across the 4 Landmark Crises
    ax2 = axes[2]
    b0_gam, m3_gam = [], []
    for ck, _ in c_keys:
        st = crisis_pinn_studies[ck]
        if "b0" in st:
            b0_gam.append(100.0 * st["b0"]["overall"]["gam"])
            m3_gam.append(100.0 * st["m3_sech_80"]["overall"]["gam"])
        else:
            b0_gam.append(100.0 * st["b0_crisis_greeks_overall"]["gam"])
            m3_gam.append(100.0 * st["m3_crisis_greeks_overall"]["gam"])
    ax2.bar(xc - 0.18, b0_gam, 0.36, label="B0 Gamma Rel. RMSE (%)", color="#d62728")
    ax2.bar(xc + 0.18, m3_gam, 0.36, label="M3_sech_80 Gamma Rel. RMSE (%)", color="#2ca02c")
    ax2.set_xticks(xc)
    ax2.set_xticklabels([lbl for _, lbl in c_keys], fontsize=8.2)
    ax2.set_ylabel("875-Point Gamma Rel. RMSE (%)")
    ax2.set_title("Second-Order Gamma Accuracy Across Crises")
    ax2.set_ylim(0, max(b0_gam) * 1.18)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8.0)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG18_PATH, dpi=220)
    plt.close(fig)

    # =========================================================================
    # Figure 19: Continuous Full-Month Trajectories (March 2020 vs August 2025)
    # =========================================================================
    fig2, axg = plt.subplots(2, 2, figsize=(14.5, 7.8))
    fig2.suptitle(
        "Continuous Full-Month WRDS OptionMetrics Rolling Calibration: March 2020 COVID-19 Crash (22d) vs. August 2025 Panel (21d)",
        fontsize=12.5, y=0.98
    )

    for row_i, (panel_days, p_title) in enumerate([
        (mar20_days, "March 2020 COVID-19 Crash & Rebound (432,450 Raw Rows)"),
        (aug25_days, "August 2025 Latest IvyDB US Panel (516,494 Raw Rows)"),
    ]):
        d_lbls = [r["date"][5:] for r in panel_days]
        xi_idx = np.arange(len(panel_days))
        vols = [r["spot_vol_pct"] for r in panel_days]
        f_marg = [r["feller_margin"] for r in panel_days]

        ax_l = axg[row_i, 0]
        ax_l.plot(xi_idx, vols, "o-", color="#1f77b4", lw=2.0, ms=4.5, label="Calibrated Spot Vol $\\sqrt{v_0^*(t)}$ (%)")
        ax_l_r = ax_l.twinx()
        ax_l_r.plot(xi_idx, f_marg, "s--", color="#d62728", lw=1.6, ms=3.8, alpha=0.85, label="Feller Margin $2\\kappa^*\\theta^* - (\\xi^*)^2$")
        ax_l_r.axhline(0.0, color="#444444", lw=0.9, ls=":")
        ax_l.set_xticks(xi_idx[::3])
        ax_l.set_xticklabels(d_lbls[::3], rotation=20, fontsize=8.5)
        ax_l.set_ylabel("Spot Volatility (%)", color="#1f77b4")
        ax_l_r.set_ylabel("Feller Margin (<0 Violated)", color="#d62728")
        ax_l.set_title(f"{p_title}: Vol & Feller")
        ax_l.grid(alpha=0.25)

        ax_r = axg[row_i, 1]
        is_v = [r["in_sample_bps"] for r in panel_days]
        oos_x = [i for i, r in enumerate(panel_days) if r["next_day_oos_bps"] is not None]
        oos_v = [r["next_day_oos_bps"] for r in panel_days if r["next_day_oos_bps"] is not None]
        ax_r.plot(xi_idx, is_v, "o-", color="#2ca02c", lw=2.0, ms=4.5, label=f"In-Sample (med {np.median(is_v):.1f} bps)")
        ax_r.plot(oos_x, oos_v, "s--", color="#ff7f0e", lw=1.9, ms=4.5, label=f"Next-Day OOS (med {np.median(oos_v):.1f} bps)")
        ax_r.set_xticks(xi_idx[::3])
        ax_r.set_xticklabels(d_lbls[::3], rotation=20, fontsize=8.5)
        ax_r.set_ylabel("Normalized Price RMSE (bps)")
        ax_r.set_title(f"{p_title}: Rolling Fit")
        ax_r.grid(alpha=0.25)
        ax_r.legend(loc="upper right", fontsize=8.5)

    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(FIG19_PATH, dpi=220)
    plt.close(fig2)
    print(f"\nCompleted Option 2 Master Study! Saved {OUT_JSON}, {FIG18_PATH}, {FIG19_PATH}, and updated training logs.", flush=True)


if __name__ == "__main__":
    main()
