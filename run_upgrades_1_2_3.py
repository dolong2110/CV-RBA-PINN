"""run_upgrades_1_2_3.py - Sequential 6-thread Apple M5 AMX runner for Upgrades #1, #2, and #3.

Executes with zero CPU thread contention and logs EVERY training run via `training_logger.py`:
  1. Re-runs 9-era WRDS OptionMetrics autograd calibration with ATM term-structure warm-starts.
  2. Runs real-market empirical SPX option dynamic hedging backtest (`empirical_spx_hedging.py`).
  3. Completes 3-seed non-affine `Heston-SLV` (`seeds 0, 1, 2` for `B0_slv` & `M3_slv`) and writes `runs/slv/slv_3seed_summary.json`.
  4. Runs 2D nonlocal `Bates SVJ` jump-diffusion PIDE control-variate benchmark & crash-skew calibration (`cvpinn_bates.py`).
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)

import cvpinn_bates
import cvpinn_slv
import empirical_spx_hedging
import run_wrds_option2_training


def build_slv_3seed_summary():
    summary = {}
    for arm in ("B0_slv", "M3_slv"):
        recs = []
        for s in (0, 1, 2):
            rec, skip = cvpinn_slv.train_slv_job(arm, s, overwrite=False)
            ov = rec["metrics"]["overall"]
            t_lbl = "reused" if skip else f"{rec['train_sec']:.1f}s"
            print(
                f"  [SLV {arm} seed={s} ({t_lbl})] "
                f"c={ov['c']['rel_rmse']:.4f}, gam={ov['gam']['rel_rmse']:.4f}, cxv={ov['cxv']['rel_rmse']:.4f}, cv={ov['cv']['rel_rmse']:.4f}",
                flush=True,
            )
            recs.append(rec)

        arm_sum = {"n_seeds": 3, "overall": {}, "short_low_var": {}, "arbitrage_mean": {}}
        for bkt in ("overall", "short_low_var"):
            for k in ("c", "cx", "gam", "cv", "cxv", "cvv"):
                vals = [r["metrics"][bkt][k]["rel_rmse"] for r in recs]
                arm_sum[bkt][k] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals, ddof=1)), 4),
                }
        for ak in ("neg_gam_count", "neg_gam_core_count", "floor_viol_count"):
            avals = [r["arbitrage"][ak] for r in recs]
            arm_sum["arbitrage_mean"][ak] = round(float(np.mean(avals)), 2)
        summary[arm] = arm_sum

    out_p = os.path.join("runs", "slv", "slv_3seed_summary.json")
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved 3-seed SLV summary -> {out_p}", flush=True)


def main():
    print("=== [Upgrade Step 1/4] Term-Structure Warm-Started 9-Era WRDS Calibration ===", flush=True)
    run_wrds_option2_training.main()

    print("\n=== [Upgrade Step 2/4] Empirical Market-Marked SPX Dynamic Hedging Backtest ===", flush=True)
    empirical_spx_hedging.main()

    print("\n=== [Upgrade Step 3/4] 3-Seed Non-Affine Heston-SLV Benchmark (Seeds 0, 1, 2) ===", flush=True)
    torch.set_num_threads(6)
    build_slv_3seed_summary()

    print("\n=== [Upgrade Step 4/4] Nonlocal 2D Bates SVJ Jump-Diffusion PIDE Study ===", flush=True)
    torch.set_num_threads(6)
    cvpinn_bates.main()

    print("\nALL UPGRADES #1, #2, AND #3 COMPLETED CLEANLY!", flush=True)


if __name__ == "__main__":
    main()
