"""Synthetic end-to-end tests for Workstream W8 (spx_w8.py)."""
import csv
import math
import os
import tempfile
import unittest

import numpy as np

import heston_ref as hr
import spx_w8


def _generate_synthetic_chain(path, quote_dt_str, expiries, v0, p_tuple):
    kappa, theta, xi, rho = p_tuple
    rows = []
    for exp_str, root, s0, r, q in expiries:
        # Approximate tau for generating synthetic prices using exact spx_w8 timestamp rules
        settle = "09:30:00" if root == "SPX" else "16:00:00"
        qdt = spx_w8._parse_dt(quote_dt_str).replace(tzinfo=spx_w8.ET)
        edt = spx_w8._parse_dt(f"{exp_str} {settle}").replace(tzinfo=spx_w8.ET)
        tau = (edt - qdt).total_seconds() / (365.25 * 86400.0)
        d_true = math.exp(-r * tau)
        f_true = s0 * math.exp((r - q) * tau)
        strikes = np.linspace(f_true * 0.72, f_true * 1.28, 19)
        xs = np.log(f_true / strikes)
        c_norm = spx_w8.heston_prices_by_slice([(tau, xs)], v0, kappa, theta, xi, rho)
        for idx, (k, cn) in enumerate(zip(strikes, c_norm)):
            c_mid = k * d_true * cn
            p_mid = c_mid - d_true * (f_true - k)
            # Inject 2 consecutive zero-bid OTM puts on far left (idx 0, 1) and far right calls
            for cp, mid in (("C", c_mid), ("P", p_mid)):
                if (cp == "P" and idx <= 1) or (cp == "C" and idx >= len(strikes) - 2):
                    bid, ask = 0.0, max(mid * 1.2, 0.05)
                else:
                    half = max(0.01 * mid, 0.02)
                    bid, ask = max(mid - half, 0.05), mid + half
                rows.append({
                    "quote_datetime": quote_dt_str,
                    "expiration": exp_str,
                    "root": root,
                    "option_type": cp,
                    "strike": f"{k:.2f}",
                    "bid": f"{bid:.4f}",
                    "ask": f"{ask:.4f}",
                })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestSPXW8(unittest.TestCase):
    def test_5_step_parity_and_calibration(self):
        p_true = hr.P_FELLER_BAD  # (1.5, 0.04, 0.5, -0.7)
        expiries_t1 = [
            ("2026-12-18", "SPX", 5900.0, 0.045, 0.013),
            ("2027-01-15", "SPX", 5900.0, 0.045, 0.013),
            ("2027-03-19", "SPXW", 5900.0, 0.044, 0.013),
            ("2027-06-18", "SPX", 5900.0, 0.043, 0.013),
            ("2027-12-17", "SPX", 5900.0, 0.042, 0.013),
        ]
        expiries_t2 = [
            ("2026-12-18", "SPX", 5940.0, 0.045, 0.013),
            ("2027-01-15", "SPX", 5940.0, 0.045, 0.013),
            ("2027-03-19", "SPXW", 5940.0, 0.044, 0.013),
            ("2027-06-18", "SPX", 5940.0, 0.043, 0.013),
            ("2027-12-17", "SPX", 5940.0, 0.042, 0.013),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p1_csv = os.path.join(tmp, "t1.csv")
            p2_csv = os.path.join(tmp, "t2.csv")
            _generate_synthetic_chain(p1_csv, "2026-11-18 14:30:00", expiries_t1, 0.045, p_true)
            _generate_synthetic_chain(p2_csv, "2026-11-19 14:30:00", expiries_t2, 0.062, p_true)

            r1 = spx_w8.load_spx_csv(p1_csv)
            r2 = spx_w8.load_spx_csv(p2_csv)
            sl1, q1 = spx_w8.process_chain_parity_and_quotes(r1)
            sl2, q2 = spx_w8.process_chain_parity_and_quotes(r2)
            self.assertEqual(len(sl1), 5)
            self.assertEqual(len(sl2), 5)
            for sl in sl1:
                self.assertGreaterEqual(sl["r2"], 0.9999)
                self.assertGreater(sl["D"], 0.9)
                self.assertLessEqual(sl["D"], 1.005)

            cal1 = spx_w8.calibrate_day1(q1, popsize=6, maxiter=20, seed=0)
            self.assertLess(cal1["fourier_vs_mkt_rmse_t1"], 5e-4)
            self.assertAlmostEqual(cal1["v0_t1"], 0.045, delta=0.005)

            cal2 = spx_w8.recalibrate_day2_v0(q2, cal1)
            self.assertLess(cal2["fourier_vs_mkt_rmse_t2"], 5e-4)
            self.assertAlmostEqual(cal2["v0_t2"], 0.062, delta=0.005)


if __name__ == "__main__":
    unittest.main()
