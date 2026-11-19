"""run_m3_experiments.py - Multi-round RAR-L-BFGS M3 championship runner with full trajectory,
arbitrage admissibility, percentile, and per-slice error logging for the final report.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import numpy as np
import torch

import cvpinn
import cvpinn_m3 as m3
import cvpinn_td as tdmod
import cvpinn_tier2 as t2
import heston_grid_vec as hgv
import heston_ref as hr

torch.set_default_dtype(torch.float64)
M3_DIR = os.path.join("runs", "m3")
M3_CKPT_DIR = os.path.join(M3_DIR, "checkpoints")


def _load_ref(pname):
    if pname == "TD":
        d = np.load(os.path.join("runs", "ref_grids", "TD_ADI.npz"))
        return d["x_grid"], {k: d[k] for k in ("c", "cx", "gam", "cxv") if k in d}
    d = np.load(os.path.join("runs", "ref_grids", f"{pname}.npz"))
    return d["x_grid"], {k: d[k] for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}


def _slice_rmse_matrix(pred, ref):
    out = {}
    for k in ("c", "cx", "gam", "cv", "cxv", "cvv"):
        if k not in pred or k not in ref:
            continue
        mat = []
        for it in range(len(hgv.EVAL_TAUS)):
            row = [
                float(np.sqrt(np.mean((pred[k][it, iv] - ref[k][it, iv]) ** 2)))
                for iv in range(len(hgv.EVAL_VS))
            ]
            mat.append(row)
        out[k] = mat
    return out


def run_m3_job(arm, pname, seed, overwrite=False):
    os.makedirs(M3_DIR, exist_ok=True)
    os.makedirs(M3_CKPT_DIR, exist_ok=True)
    stem = f"{arm}_{pname}_seed{seed}"
    json_path = os.path.join(M3_DIR, f"{stem}.json")
    ckpt_path = os.path.join(M3_CKPT_DIR, f"{stem}.pt")
    if (not overwrite) and os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f), True

    x_grid, ref = _load_ref(pname)
    is_td = pname == "TD"
    p_or_td = tdmod.default_td_heston() if is_td else cvpinn.Heston(*hr.P_FELLER_BAD if pname == "P2" else hr.P_FELLER_OK)

    t0 = time.perf_counter()
    price_fn, net, info = m3.train_m3(
        arm, p_or_td, seed, x_grid_eval=x_grid, ref_eval=ref, is_td=is_td, v_max=0.5, device="cpu",
    )
    train_sec = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    pred = t2.evaluate_all_greeks(price_fn, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)
    infer_ms = 1000.0 * (time.perf_counter() - t_inf0)

    metrics = t2.score_all_six_buckets(pred, ref)
    arb, dist = m3.audit_arbitrage_and_percentiles(pred, ref, x_grid)
    slices = _slice_rmse_matrix(pred, ref)

    torch.save(
        {
            "state_dict": net.state_dict(), "arm": arm, "pname": pname, "seed": seed,
            "config": info["config"], "n_params": info["n_params"],
        },
        ckpt_path,
    )
    rec = {
        "arm": arm, "param_set": pname, "seed": seed,
        "n_params": info["n_params"], "train_sec": round(train_sec, 4),
        "infer_ms_875pt": round(infer_ms, 3),
        "train_info": info, "metrics": metrics,
        "arbitrage": arb, "percentiles": dist, "slice_rmse": slices,
    }
    with open(json_path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec, False


def _worker(kw):
    return run_m3_job(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--params", default="P2")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--jobs", default="", help="Optional comma-separated arm:param:seed specs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.jobs.strip():
        jobs = []
        for spec in args.jobs.split(","):
            spec = spec.strip()
            if not spec:
                continue
            a, p, s = spec.split(":")
            jobs.append({"arm": a.strip(), "pname": p.strip(), "seed": int(s.strip()), "overwrite": args.overwrite})
    else:
        jobs = [
            {"arm": a.strip(), "pname": p.strip(), "seed": int(s.strip()), "overwrite": args.overwrite}
            for a in args.arms.split(",") if a.strip()
            for p in args.params.split(",") if p.strip()
            for s in args.seeds.split(",") if s.strip()
        ]

    def _log(rec, skip):
        ov, sl = rec["metrics"]["overall"], rec["metrics"]["short_low_var"]
        arb = rec["arbitrage"]
        cv_s = f" cv={ov['cv']['rel_rmse']:.4f} cvv={ov['cvv']['rel_rmse']:.4f}" if "cv" in ov else ""
        tag = "SKIP" if skip else f"{rec['train_sec']:.1f}s"
        print(
            f"[{tag}] {rec['arm']:16s} ({rec['param_set']} s{rec['seed']}) loss={rec['train_info']['lbfgs_post_loss']:.2e} | "
            f"ov c={ov['c']['rel_rmse']:.4f} gam={ov['gam']['rel_rmse']:.4f}(max={ov['gam']['max_abs']:.2f}) "
            f"cxv={ov['cxv']['rel_rmse']:.4f}{cv_s} | slv gam={sl['gam']['rel_rmse']:.4f} cxv={sl['cxv']['rel_rmse']:.4f} | "
            f"neg_gam={arb['neg_gam_count']}(core={arb['neg_gam_core_count']},min={arb['min_gam']:+.2e}) flr={arb['floor_viol_count']}"
        )

    print(f"Running {len(jobs)} M3 jobs with workers={args.workers}...")
    t0 = time.perf_counter()
    if args.workers <= 1:
        for j in jobs:
            _log(*_worker(j))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker, j) for j in jobs]
            for fut in as_completed(futs):
                _log(*fut.result())
    print(f"Done in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
