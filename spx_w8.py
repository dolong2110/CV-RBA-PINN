"""spx_w8.py - Workstream W8: European-style SPX/SPXW parity & Heston calibration (Tab 7.5).

Implements the exact 5-step market validation protocol:
  1. Ingest two daily CSV option snapshots (t1 calibration day, t2 out-of-sample v0 day).
  2. Separate AM-settled SPX (09:30 ET SOQ) vs PM-settled SPXW (16:00 ET close) in ACT/365.25;
     keep 0.02 <= tau <= 2.0.
  3. Quote hygiene (bid >= 0, ask > bid, rel_spread <= 0.25) + Cboe VIX two-zero-bid wing cutoff.
  4. Curve-free forward F(tau) and discount D(tau) via put-call parity OLS (R^2 >= 0.999,
     0 < D <= 1.005, monotone D, >= 8 paired strikes inside |x| <= 1.2).
  5. Day-1 structural calibration of (kappa, theta, xi, rho, v0(t1)) via differential evolution +
     L-BFGS-B, optional 5-arm PINN evaluation, and Day-2 1D Brent v0(t2) recalibration.
"""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, time as dt_time
import json
import math
import os
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

import heston_grid_vec as hgv

ET = ZoneInfo("America/New_York")
GLX, GLW = hgv.GLX, hgv.GLW


@dataclass
class QuoteRow:
    quote_dt: datetime
    expiry_date: datetime
    series: str          # "SPX" (AM) or "SPXW" (PM)
    cp: str              # "C" or "P"
    strike: float
    bid: float
    ask: float
    tau: float


def _parse_dt(val):
    val = val.strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date/datetime format: {val!r}")


def _col(row, *candidates):
    lower = {k.strip().lower(): v for k, v in row.items()}
    for c in candidates:
        if c in lower and lower[c] != "":
            return lower[c].strip()
    return None


def load_spx_csv(csv_path, default_quote_time="14:30:00", strike_scale=None):
    """Loads OptionMetrics / DataShop / broker SPX/SPXW CSV into normalized QuoteRow records."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        return []

    # Auto-detect OptionMetrics strike_price * 1000 convention if strike_scale is None
    if strike_scale is None:
        sample_strikes = []
        for r in raw_rows[:200]:
            s = _col(r, "strike", "strike_price", "k")
            if s is not None:
                sample_strikes.append(float(s))
        strike_scale = 0.001 if (sample_strikes and np.median(sample_strikes) > 50000.0) else 1.0

    for r in raw_rows:
        q_raw = _col(r, "quote_datetime", "quote_date", "date", "timestamp")
        exp_raw = _col(r, "expiration", "exdate", "expiry", "maturity_date")
        cp_raw = _col(r, "cp_flag", "option_type", "type", "call_put", "right")
        k_raw = _col(r, "strike", "strike_price", "k")
        b_raw = _col(r, "bid", "best_bid", "bid_price")
        a_raw = _col(r, "ask", "best_offer", "ask_price", "offer")
        if None in (q_raw, exp_raw, cp_raw, k_raw, b_raw, a_raw):
            continue

        cp = cp_raw.upper()[0]
        if cp not in ("C", "P"):
            continue
        strike = float(k_raw) * strike_scale
        bid = float(b_raw)
        ask = float(a_raw)

        q_dt = _parse_dt(q_raw)
        if q_dt.hour == 0 and q_dt.minute == 0 and q_dt.second == 0 and " " not in q_raw and "T" not in q_raw:
            hh, mm, ss = (int(x) for x in default_quote_time.split(":"))
            q_dt = q_dt.replace(hour=hh, minute=mm, second=ss)
        q_dt = q_dt.replace(tzinfo=ET)

        exp_dt_d = _parse_dt(exp_raw).date()
        root_raw = (_col(r, "root", "symbol", "option_symbol", "ticker") or "").upper()
        am_raw = _col(r, "am_settlement", "settlement_type", "settle")
        if "SPXW" in root_raw or (am_raw is not None and am_raw.upper() in ("0", "PM", "P")):
            series = "SPXW"
            settle_t = dt_time(16, 0, 0)
        elif "SPX" in root_raw or (am_raw is not None and am_raw.upper() in ("1", "AM", "A")):
            series = "SPX"
            settle_t = dt_time(9, 30, 0)
        else:
            # Default third-Friday AM assumption if unmarked, else SPXW
            is_third_fri = (exp_dt_d.weekday() == 4) and (15 <= exp_dt_d.day <= 21)
            series = "SPX" if is_third_fri else "SPXW"
            settle_t = dt_time(9, 30, 0) if series == "SPX" else dt_time(16, 0, 0)

        exp_dt = datetime.combine(exp_dt_d, settle_t, tzinfo=ET)
        tau = (exp_dt - q_dt).total_seconds() / (365.25 * 86400.0)
        if 0.02 <= tau <= 2.0:
            rows.append(QuoteRow(q_dt, exp_dt, series, cp, strike, bid, ask, tau))
    return rows


def _vix_wing_filter(strikes_sorted, quotes_by_k, k_atm, direction):
    """Applies Cboe VIX wing cutoff: drop bid<=0 and stop after 2 consecutive bid==0 strikes."""
    if direction == "put":
        ordered = [k for k in reversed(strikes_sorted) if k < k_atm]
    else:
        ordered = [k for k in strikes_sorted if k > k_atm]
    kept = []
    consec_zero = 0
    for k in ordered:
        q = quotes_by_k[k]
        if q.bid <= 0.0:
            consec_zero += 1
            if consec_zero >= 2:
                break
            continue
        consec_zero = 0
        if q.ask <= q.bid:
            continue
        mid = 0.5 * (q.bid + q.ask)
        if mid > 0.0 and (q.ask - q.bid) / mid <= 0.25:
            kept.append(q)
    return kept


def process_chain_parity_and_quotes(rows):
    """Steps 3 & 4: VIX wing hygiene, put-call parity OLS, and normalized call-equivalent quotes."""
    slices = {}
    for r in rows:
        key = (r.expiry_date.strftime("%Y-%m-%d"), r.series)
        slices.setdefault(key, []).append(r)

    raw_slices = []
    for (exp_str, series), srows in sorted(slices.items(), key=lambda kv: kv[1][0].tau):
        calls = {r.strike: r for r in srows if r.cp == "C"}
        puts = {r.strike: r for r in srows if r.cp == "P"}
        paired_k = sorted(set(calls) & set(puts))
        valid_pairs = []
        for k in paired_k:
            c, p = calls[k], puts[k]
            if c.bid > 0.0 and p.bid > 0.0 and c.ask > c.bid and p.ask > p.bid:
                cm = 0.5 * (c.bid + c.ask)
                pm = 0.5 * (p.bid + p.ask)
                if (c.ask - c.bid) / cm <= 0.25 and (p.ask - p.bid) / pm <= 0.25:
                    valid_pairs.append((k, cm, pm))
        if len(valid_pairs) < 8:
            continue

        # Minimal |C_mid - P_mid| ATM anchor for VIX wing cutoff and parity OLS
        k_atm = min(valid_pairs, key=lambda tup: abs(tup[1] - tup[2]))[0]
        ks_arr = np.array([t[0] for t in valid_pairs], dtype=np.float64)
        diff_arr = np.array([t[1] - t[2] for t in valid_pairs], dtype=np.float64)
        # OLS: C - P = a - b * K
        slope, intercept = np.polyfit(ks_arr, diff_arr, 1)
        b_hat = -float(slope)
        a_hat = float(intercept)
        pred = a_hat - b_hat * ks_arr
        ss_res = float(np.sum((diff_arr - pred) ** 2))
        ss_tot = float(np.sum((diff_arr - np.mean(diff_arr)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

        if r2 < 0.999 or b_hat <= 0.0 or b_hat > 1.005 or a_hat <= 0.0:
            continue
        d_tau = b_hat
        f_tau = a_hat / b_hat

        # Count pairs inside |x| <= 1.2
        in_band = [t for t in valid_pairs if abs(math.log(f_tau / t[0])) <= 1.2]
        if len(in_band) < 8:
            continue

        # VIX wing cutoff on OTM puts (K < F) and OTM calls (K >= F)
        put_strikes = sorted(puts)
        call_strikes = sorted(calls)
        otm_puts = _vix_wing_filter(put_strikes, puts, f_tau, "put")
        otm_calls = _vix_wing_filter(call_strikes, calls, f_tau, "call")
        # Include ATM call/put if K == f_tau exact
        if f_tau in calls and calls[f_tau].bid > 0.0 and calls[f_tau].ask > calls[f_tau].bid:
            q_atm = calls[f_tau]
            if (q_atm.ask - q_atm.bid) / (0.5 * (q_atm.bid + q_atm.ask)) <= 0.25:
                otm_calls.insert(0, q_atm)

        tau_m = float(np.mean([r.tau for r in srows]))
        raw_slices.append({
            "expiry": exp_str,
            "series": series,
            "tau": tau_m,
            "D": d_tau,
            "F": f_tau,
            "r2": r2,
            "n_paired": len(valid_pairs),
            "otm_puts": otm_puts,
            "otm_calls": otm_calls,
        })

    # Enforce monotone discount D(tau_m) across increasing maturities
    clean_slices = []
    prev_d = float("inf")
    for sl in sorted(raw_slices, key=lambda s: s["tau"]):
        if sl["D"] <= prev_d + 1e-4:  # tiny numerical noise guard; reject material inversions
            prev_d = min(prev_d, sl["D"])
            clean_slices.append(sl)

    quotes = []
    for sl in clean_slices:
        d_tau, f_tau, tau_m = sl["D"], sl["F"], sl["tau"]
        for q in sl["otm_puts"]:
            x = math.log(f_tau / q.strike)
            if abs(x) > 1.2:
                continue
            p_mid = 0.5 * (q.bid + q.ask)
            c_equiv = p_mid + d_tau * (f_tau - q.strike)
            c_mkt = c_equiv / (q.strike * d_tau)
            spr = (q.ask - q.bid) / (q.strike * d_tau)
            if c_mkt > 0.0 and spr > 0.0:
                quotes.append((tau_m, x, c_mkt, spr))
        for q in sl["otm_calls"]:
            x = math.log(f_tau / q.strike)
            if abs(x) > 1.2:
                continue
            c_mid = 0.5 * (q.bid + q.ask)
            c_mkt = c_mid / (q.strike * d_tau)
            spr = (q.ask - q.bid) / (q.strike * d_tau)
            if c_mkt > 0.0 and spr > 0.0:
                quotes.append((tau_m, x, c_mkt, spr))

    return clean_slices, quotes


def heston_prices_by_slice(slice_data, v0, kappa, theta, xi, rho, tol=1e-13, max_u=1e5):
    """Fast vectorized Lewis-Albrecher Fourier prices grouped by maturity slice."""
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
            u_nodes = (starts[:, None] + 0.5 * widths[:, None] * (GLX[None, :] + 1.0)).ravel()
            w_nodes = (0.5 * widths[:, None] * GLW[None, :]).ravel()
            z_cf = u_nodes - 0.5j
            a = kappa - 1j * rho * xi * z_cf
            d = np.sqrt(a * a + (xi * xi) * (z_cf * z_cf + 1j * z_cf))
            g = (a - d) / (a + d)
            e = np.exp(-d * tau)
            big_b = (a - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e)
            big_a = (kappa * theta / (xi * xi)) * (
                (a - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g))
            )
            cf = np.exp(big_a + big_b * v0)
            env = np.abs(cf) / (u_nodes * u_nodes + 0.25)
            phi = cf[:, None] * np.exp(1j * u_nodes[:, None] * xs[None, :])
            den = (u_nodes * u_nodes + 0.25)[:, None]
            sums += np.sum(w_nodes[:, None] * (phi / den).real, axis=0)
            if len(starts) >= 3 and np.max(env[-32:]) < tol:
                break
        preds.append(np.exp(xs) - (np.exp(0.5 * xs) / math.pi) * sums)
    return np.concatenate(preds)


def _group_quotes(quotes):
    by_tau = {}
    for tau, x, cm, spr in quotes:
        by_tau.setdefault(tau, []).append((x, cm, spr))
    slice_data = []
    ys = []
    weights = []
    for tau in sorted(by_tau):
        arr = by_tau[tau]
        slice_data.append((tau, np.array([a[0] for a in arr], dtype=np.float64)))
        ys.extend(a[1] for a in arr)
        weights.extend(1.0 / max(a[2], 1e-4) ** 2 for a in arr)
    w_arr = np.array(weights, dtype=np.float64)
    w_arr /= np.mean(w_arr)
    return slice_data, np.array(ys, dtype=np.float64), w_arr


def calibrate_day1(quotes, popsize=8, maxiter=40, seed=0):
    """Fits (kappa, theta, xi, rho, v0(t1)) by inverse-spread-weighted normalized price MSE."""
    slice_data, y_mkt, w = _group_quotes(quotes)
    bounds = [(0.2, 10.0), (0.005, 0.25), (0.05, 1.5), (-0.98, -0.05), (0.002, 0.25)]

    def obj(vec):
        kappa, theta, xi, rho, v0 = vec
        pred = heston_prices_by_slice(slice_data, v0, kappa, theta, xi, rho)
        return float(np.mean(w * (pred - y_mkt) ** 2))

    de = differential_evolution(obj, bounds, maxiter=maxiter, popsize=popsize, seed=seed, polish=False)
    res = minimize(obj, de.x, method="L-BFGS-B", bounds=bounds)
    kappa, theta, xi, rho, v0 = [float(x) for x in res.x]
    pred = heston_prices_by_slice(slice_data, v0, kappa, theta, xi, rho)
    rmse_fourier = float(np.sqrt(np.mean((pred - y_mkt) ** 2)))
    return {
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
        "v0_t1": v0,
        "feller_lhs": 2.0 * kappa * theta,
        "feller_rhs": xi ** 2,
        "feller_satisfied": bool(2.0 * kappa * theta >= xi ** 2),
        "weighted_mse": float(res.fun),
        "fourier_vs_mkt_rmse_t1": rmse_fourier,
    }


def recalibrate_day2_v0(quotes_t2, struct_params, pinn_price_fn=None):
    """Step 5b: Freeze (kappa*, theta*, xi*, rho*) and solve 1D Brent for v0(t2)."""
    slice_data, y_mkt, w = _group_quotes(quotes_t2)
    kappa, theta, xi, rho = (
        struct_params["kappa"],
        struct_params["theta"],
        struct_params["xi"],
        struct_params["rho"],
    )

    def obj_v0(v0):
        pred = heston_prices_by_slice(slice_data, v0, kappa, theta, xi, rho)
        return float(np.mean(w * (pred - y_mkt) ** 2))

    opt = minimize_scalar(obj_v0, bounds=(0.001, 0.40), method="bounded")
    v0_t2 = float(opt.x)
    c_fourier = heston_prices_by_slice(slice_data, v0_t2, kappa, theta, xi, rho)
    out = {
        "v0_t2": v0_t2,
        "n_quotes_t2": len(y_mkt),
        "fourier_vs_mkt_rmse_t2": float(np.sqrt(np.mean((c_fourier - y_mkt) ** 2))),
    }
    if pinn_price_fn is not None:
        import torch
        taus_flat = np.concatenate([np.full(len(xs), tau) for tau, xs in slice_data])
        xs_flat = np.concatenate([xs for _, xs in slice_data])
        vs_flat = np.full_like(xs_flat, v0_t2)
        with torch.no_grad():
            c_pinn = pinn_price_fn(
                torch.tensor(xs_flat, dtype=torch.float64),
                torch.tensor(vs_flat, dtype=torch.float64),
                torch.tensor(taus_flat, dtype=torch.float64),
            ).cpu().numpy()
        out["pinn_vs_fourier_rmse_t2"] = float(np.sqrt(np.mean((c_pinn - c_fourier) ** 2)))
        out["pinn_vs_mkt_rmse_t2"] = float(np.sqrt(np.mean((c_pinn - y_mkt) ** 2)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--out", default="runs/w8_spx_report.json")
    parser.add_argument("--train-pinn", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    r1 = load_spx_csv(args.t1)
    r2 = load_spx_csv(args.t2)
    sl1, q1 = process_chain_parity_and_quotes(r1)
    sl2, q2 = process_chain_parity_and_quotes(r2)
    if not sl1 or not sl2:
        raise SystemExit("Insufficient clean parity slices (require R^2 >= 0.999 and >= 8 paired strikes).")

    cal1 = calibrate_day1(q1)
    pinn_fn = None
    if args.train_pinn:
        import cvpinn
        p_cal = cvpinn.Heston(cal1["kappa"], cal1["theta"], cal1["xi"], cal1["rho"])
        pinn_fn = cvpinn.build_and_train("M1", p_cal, args.seed)
    cal2 = recalibrate_day2_v0(q2, cal1, pinn_price_fn=pinn_fn)

    report = {
        "t1_slices": [{"expiry": s["expiry"], "series": s["series"], "tau": s["tau"], "D": s["D"], "F": s["F"], "r2": s["r2"]} for s in sl1],
        "t2_slices": [{"expiry": s["expiry"], "series": s["series"], "tau": s["tau"], "D": s["D"], "F": s["F"], "r2": s["r2"]} for s in sl2],
        "day1_calibration": cal1,
        "day2_recalibration": cal2,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
