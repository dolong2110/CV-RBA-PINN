"""unpack_and_eval_historical_spx.py - Unpacks 168 trading days of historical SPX/SPXW Parquet
data into standardized daily CSVs and runs Workstream W8 multi-day (t1 -> t2) out-of-sample
calibration & differentiable neural calibration across 4 historical SPX regimes + DataShop 2023.
"""
import csv
import io
import json
import os
import time
import zipfile

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pyarrow.parquet as pq
import torch

import diff_calibrate as dc
import spx_w8

torch.set_default_dtype(torch.float64)

ZIP_PATH = os.path.join("docs", "market_data", "historical_archives", "OptionData_SPX_2024_06_to_2025_03_parquet.zip")
CSV_OUT_DIR = os.path.join("docs", "market_data", "spx_168days_2024_2025_csv")
DATASHOP_CSV = os.path.join("docs", "market_data", "spx_cboe_datashop_2023_08_25.csv")
REPORT_OUT = os.path.join("runs", "historical_spx_multiday_report.json")


def unpack_all_168_days():
    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()
    total_rows = 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        by_date = {}
        for name in zf.namelist():
            if name.endswith(".parquet") and "Options_Data/date=" in name:
                dt_str = name.split("/")[2].replace("date=", "")
                by_date.setdefault(dt_str, []).append(name)

        dates_sorted = sorted(by_date)
        for d_str in dates_sorted:
            out_csv = os.path.join(CSV_OUT_DIR, f"{d_str}.csv")
            if os.path.exists(out_csv) and os.path.getsize(out_csv) > 1000:
                continue
            day_rows = []
            for p_name in sorted(by_date[d_str]):
                with zf.open(p_name) as fp:
                    tbl = pq.read_table(io.BytesIO(fp.read()), columns=["exp", "K", "isAM", "option", "bid", "ask"])
                exps = tbl["exp"].to_pylist()
                ks = tbl["K"].to_pylist()
                ams = tbl["isAM"].to_pylist()
                opts = tbl["option"].to_pylist()
                bids = tbl["bid"].to_pylist()
                asks = tbl["ask"].to_pylist()
                q_dt = f"{d_str} 15:45:00"
                for exp_v, k_v, am_v, op_v, b_v, a_v in zip(exps, ks, ams, opts, bids, asks):
                    if None in (exp_v, k_v, op_v, b_v, a_v):
                        continue
                    root = "SPX" if str(am_v).upper() == "AM" else "SPXW"
                    cp = "C" if str(op_v).lower().startswith("c") else "P"
                    day_rows.append((q_dt, str(exp_v), root, cp, f"{float(k_v):.2f}", f"{float(b_v):.4f}", f"{float(a_v):.4f}"))
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["quote_datetime", "expiration", "root", "option_type", "strike", "bid", "ask"])
                w.writerows(day_rows)
            total_rows += len(day_rows)
    print(f"Unpacked {len(dates_sorted)} daily SPX/SPXW CSVs -> {CSV_OUT_DIR} ({total_rows:,} newly written rows in {time.perf_counter()-t0:.1f}s)")
    return dates_sorted


def eval_two_day_pair(label, t1_csv, t2_csv, preloaded):
    r1 = spx_w8.load_spx_csv(t1_csv)
    r2 = spx_w8.load_spx_csv(t2_csv)
    sl1, q1_all = spx_w8.process_chain_parity_and_quotes(r1)
    sl2, q2_all = spx_w8.process_chain_parity_and_quotes(r2)
    # Keep liquid moneyness |x| <= 0.60 across up to 8 representative expiries spanning [0.03, 1.5]
    def _subsample_slices(slices, quotes):
        if len(slices) <= 8:
            keep_t = {round(s["tau"], 6) for s in slices}
        else:
            idxs = np.linspace(0, len(slices) - 1, 8, dtype=int)
            keep_t = {round(slices[i]["tau"], 6) for i in idxs}
        return [q for q in quotes if round(q[0], 6) in keep_t and abs(q[1]) <= 0.60]

    q1 = _subsample_slices(sl1, q1_all)
    q2 = _subsample_slices(sl2, q2_all)

    t_de0 = time.perf_counter()
    cal_de1 = spx_w8.calibrate_day1(q1, popsize=8, maxiter=35, seed=0)
    cal_de1["calib_sec"] = round(time.perf_counter() - t_de0, 4)
    t_de2 = time.perf_counter()
    cal_de2 = spx_w8.recalibrate_day2_v0(q2, cal_de1)
    cal_de2["calib_sec"] = round(time.perf_counter() - t_de2, 4)

    cal_nn1 = dc.calibrate_day1_differentiable(q1, preloaded=preloaded, seed=0)
    cal_nn2 = dc.recalibrate_day2_differentiable(q2, cal_nn1, preloaded=preloaded)
    speedup = cal_de1["calib_sec"] / max(cal_nn1["calib_sec"], 1e-6)

    print(
        f"[{label}] slices={len(sl1)}/{len(sl2)} quotes={len(q1)}/{len(q2)} | "
        f"DE+Fourier: {cal_de1['calib_sec']*1000:.0f}ms (xi*={cal_de1['xi']:.3f}, v0_t1={cal_de1['v0_t1']:.4f}->v0_t2={cal_de2['v0_t2']:.4f}, "
        f"t1={cal_de1['fourier_vs_mkt_rmse_t1']*1e4:.1f}bps, t2_oos={cal_de2['fourier_vs_mkt_rmse_t2']*1e4:.1f}bps) | "
        f"Neural Autograd: {cal_nn1['calib_sec']*1000:.0f}ms ({speedup:.1f}x faster, "
        f"t1={cal_nn1['fourier_vs_mkt_rmse_t1']*1e4:.1f}bps, t2_oos={cal_nn2['fourier_vs_mkt_rmse_t2']*1e4:.1f}bps, "
        f"surr_vs_four_t2={cal_nn2['surr_vs_fourier_rmse_t2']*1e4:.1f}bps)",
        flush=True,
    )
    return {
        "t1_csv": t1_csv, "t2_csv": t2_csv,
        "n_slices_t1": len(sl1), "n_slices_t2": len(sl2),
        "min_parity_r2_t1": float(min(s["r2"] for s in sl1)),
        "min_parity_r2_t2": float(min(s["r2"] for s in sl2)),
        "n_quotes_cal_t1": len(q1), "n_quotes_oos_t2": len(q2),
        "fourier_de": {"day1": cal_de1, "day2_oos": cal_de2},
        "neural_autograd": {"day1": cal_nn1, "day2_oos": cal_nn2},
        "day1_speedup_x": round(speedup, 2),
    }


def main():
    dates = unpack_all_168_days()
    preloaded = dc.load_surrogate()
    _ = dc.calibrate_day1_differentiable([(0.1, 0.0, 0.025, 0.001), (0.25, 0.05, 0.07, 0.001)], preloaded=preloaded, n_starts=2, adam_steps=2, lbfgs_iter=2)

    pairs = [
        ("2024_Jun_CalmBull", "2024-06-12", "2024-06-13"),
        ("2024_Aug_YenCarrySpike", "2024-08-05", "2024-08-06"),
        ("2024_Nov_USElection", "2024-11-05", "2024-11-06"),
        ("2025_Mar_Q1Selloff", "2025-03-10", "2025-03-11"),
    ]
    results = {"n_unpacked_2024_2025_days": len(dates), "date_range_2024_2025": [dates[0], dates[-1]], "two_day_w8_benchmarks": {}}
    for lbl, d1, d2 in pairs:
        p1 = os.path.join(CSV_OUT_DIR, f"{d1}.csv")
        p2 = os.path.join(CSV_OUT_DIR, f"{d2}.csv")
        if os.path.exists(p1) and os.path.exists(p2):
            results["two_day_w8_benchmarks"][lbl] = eval_two_day_pair(lbl, p1, p2, preloaded)

    # Also evaluate Official Cboe DataShop 2023-08-25 snapshot parity & calibration
    r_ds = spx_w8.load_spx_csv(DATASHOP_CSV)
    sl_ds, q_ds_all = spx_w8.process_chain_parity_and_quotes(r_ds)
    idxs_ds = np.linspace(0, len(sl_ds) - 1, 8, dtype=int)
    keep_ds = {round(sl_ds[i]["tau"], 6) for i in idxs_ds}
    q_ds = [q for q in q_ds_all if round(q[0], 6) in keep_ds and abs(q[1]) <= 0.60]
    t0 = time.perf_counter()
    cal_ds_de = spx_w8.calibrate_day1(q_ds, popsize=8, maxiter=35, seed=0)
    cal_ds_de["calib_sec"] = round(time.perf_counter() - t0, 4)
    cal_ds_nn = dc.calibrate_day1_differentiable(q_ds, preloaded=preloaded, seed=0)
    sp_ds = cal_ds_de["calib_sec"] / max(cal_ds_nn["calib_sec"], 1e-6)
    print(
        f"[2023_Aug25_CboeDataShop] slices={len(sl_ds)} quotes={len(q_ds)} | "
        f"DE+Fourier: {cal_ds_de['calib_sec']*1000:.0f}ms (xi*={cal_ds_de['xi']:.3f}, rmse={cal_ds_de['fourier_vs_mkt_rmse_t1']*1e4:.1f}bps) | "
        f"Neural Autograd: {cal_ds_nn['calib_sec']*1000:.0f}ms ({sp_ds:.1f}x faster, rmse={cal_ds_nn['fourier_vs_mkt_rmse_t1']*1e4:.1f}bps)"
    )
    results["cboe_datashop_2023_08_25"] = {
        "n_slices": len(sl_ds), "n_otm_quotes_total": len(q_ds_all), "n_quotes_cal": len(q_ds),
        "min_parity_r2": float(min(s["r2"] for s in sl_ds)),
        "fourier_de": cal_ds_de, "neural_autograd": cal_ds_nn, "speedup_x": round(sp_ds, 2),
    }

    with open(REPORT_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {REPORT_OUT}")


if __name__ == "__main__":
    main()
