"""run_param_experiments.py - Train & zero-shot evaluate 8D amortized parametric Heston surrogates."""
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
import cvpinn_param as cp8
import cvpinn_tier2 as t2
import heston_grid_vec as hgv
import heston_ref as hr

torch.set_default_dtype(torch.float64)
PARAM_DIR = os.path.join("runs", "param")
PARAM_CKPT_DIR = os.path.join(PARAM_DIR, "checkpoints")

EVAL_REGIMES = {
    "P1": hr.P_FELLER_OK,
    "P2": hr.P_FELLER_BAD,
    "P3_steep": (3.0, 0.025, 0.65, -0.80),
    "P4_calm": (1.0, 0.080, 0.25, -0.35),
}


def _get_regime_ref(name, p_tup):
    cache_path = os.path.join("runs", "ref_grids", f"{name}.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["x_grid"], {k: d[k] for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
    x_grid, ref = hgv.heston_eval_grid(p_tup)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path, x_grid=x_grid, taus=hgv.EVAL_TAUS, vs=hgv.EVAL_VS, zs=hgv.EVAL_ZS,
        **ref,
    )
    return x_grid, ref


def run_param_job(arm, seed, adam_steps, lbfgs_iter, width, overwrite):
    os.makedirs(PARAM_DIR, exist_ok=True)
    os.makedirs(PARAM_CKPT_DIR, exist_ok=True)
    json_path = os.path.join(PARAM_DIR, f"{arm}_seed{seed}.json")
    ckpt_path = os.path.join(PARAM_CKPT_DIR, f"{arm}_seed{seed}.pt")
    if (not overwrite) and os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f), True

    t0 = time.perf_counter()
    price_fn_param, net, info = cp8.train_param_model(
        arm=arm, seed=seed, adam_steps=adam_steps, lbfgs_iter=lbfgs_iter,
        width=width, depth=4, v_max=0.5, term_weight=100.0, device="cpu",
    )
    train_sec = time.perf_counter() - t0
    torch.save({"state_dict": net.state_dict(), "arm": arm, "seed": seed, "width": width, "info": info}, ckpt_path)

    evals = {}
    for pname, p_tup in EVAL_REGIMES.items():
        x_grid, ref = _get_regime_ref(pname, p_tup)
        bound_fn = cp8.bind_params(price_fn_param, cvpinn.Heston(*p_tup))
        t_inf = time.perf_counter()
        pred = t2.evaluate_all_greeks(bound_fn, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)
        infer_ms = 1000.0 * (time.perf_counter() - t_inf)
        evals[pname] = {
            "params": {"kappa": p_tup[0], "theta": p_tup[1], "xi": p_tup[2], "rho": p_tup[3]},
            "infer_ms_875pt": round(infer_ms, 3),
            "metrics": t2.score_all_six_buckets(pred, ref),
        }

    rec = {
        "arm": arm, "seed": seed, "width": width, "adam_steps": adam_steps,
        "lbfgs_iter": lbfgs_iter, "train_sec": round(train_sec, 4),
        "train_info": info, "zero_shot_evals": evals,
    }
    with open(json_path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec, False


def _worker(kw):
    return run_param_job(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B0_param,M_param,M_param_plus")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--adam-steps", type=int, default=12000)
    ap.add_argument("--lbfgs-iter", type=int, default=500)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Pre-cache reference grids synchronously so parallel workers only read them
    for pname, p_tup in EVAL_REGIMES.items():
        _get_regime_ref(pname, p_tup)

    jobs = [
        {
            "arm": a.strip(), "seed": int(s.strip()), "adam_steps": args.adam_steps,
            "lbfgs_iter": args.lbfgs_iter, "width": args.width, "overwrite": args.overwrite,
        }
        for a in args.arms.split(",") if a.strip()
        for s in args.seeds.split(",") if s.strip()
    ]
    print(f"Training {len(jobs)} amortized 8D models with workers={args.workers}...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for fut in as_completed(futs):
            rec, skip = fut.result()
            tag = "SKIP" if skip else f"{rec['train_sec']:.1f}s"
            print(f"[{tag}] {rec['arm']} s{rec['seed']} loss={rec['train_info']['lbfgs_post_loss']:.3e}")
            for pname, ev in rec["zero_shot_evals"].items():
                ov = ev["metrics"]["overall"]
                print(
                    f"  zero-shot {pname:8s}: c={ov['c']['rel_rmse']:.3e} gam={ov['gam']['rel_rmse']:.3e} "
                    f"cxv={ov['cxv']['rel_rmse']:.3e} cv={ov['cv']['rel_rmse']:.3e} cvv={ov['cvv']['rel_rmse']:.3e}"
                )


if __name__ == "__main__":
    main()
