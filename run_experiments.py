"""run_experiments.py - Gate verification, tuning, S2 pilot, and 50-run orchestrator.

Usage:
  .venv/bin/python run_experiments.py gates
  .venv/bin/python run_experiments.py day4_tune --workers 4
  .venv/bin/python run_experiments.py s2_pilot --term-weight 10.0 --v-max 0.5 --workers 4
  .venv/bin/python run_experiments.py full_matrix --workers 4
  .venv/bin/python run_experiments.py summarize
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import statistics
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
if "--workers" in sys.argv:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

import cvpinn
import heston_grid_vec as hgv
import heston_ref as hr


PARAMS = {
    "P1": hr.P_FELLER_OK,   # (2.0, 0.04, 0.3, -0.5)
    "P2": hr.P_FELLER_BAD,  # (1.5, 0.04, 0.5, -0.7)
}
ARMS = ("B0", "M1", "A1", "A2a", "A2b")
SEEDS = (0, 1, 2, 3, 4)


def _to_heston(p_tuple):
    return cvpinn.Heston(*p_tuple)


def load_or_build_ref_grid(pname, force=False):
    os.makedirs("runs/ref_grids", exist_ok=True)
    path = f"runs/ref_grids/{pname}.npz"
    if os.path.exists(path) and not force:
        data = np.load(path)
        x_grid = data["x_grid"]
        ref = {k: data[k] for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
        return x_grid, ref, None
    res = hgv.verify_against_scalar(PARAMS[pname])
    np.savez(
        path,
        x_grid=res["x_grid"],
        taus=np.array(hgv.EVAL_TAUS, dtype=np.float64),
        vs=np.array(hgv.EVAL_VS, dtype=np.float64),
        zs=np.array(hgv.EVAL_ZS, dtype=np.float64),
        **res["grid"],
    )
    g1_info = {
        k: res[k] for k in ("t_vec_sec", "t_scalar_sec", "speedup", "max_abs_diff", "passed")
    }
    return res["x_grid"], res["grid"], g1_info


def cmd_gates(args):
    rep = {"G1": {}, "G2": {}, "G3": {}, "greeks_bs_check": {}}
    for pname, p_tuple in PARAMS.items():
        _, _, g1 = load_or_build_ref_grid(pname, force=args.force)
        if g1 is None:
            res = hgv.verify_against_scalar(p_tuple)
            g1 = {
                k: res[k]
                for k in ("t_vec_sec", "t_scalar_sec", "speedup", "max_abs_diff", "passed")
            }
        rep["G1"][pname] = g1

        p = _to_heston(p_tuple)
        g2_exp = cvpinn.gate2(p, cvpinn.Domain(v_min=0.0), "expected")
        g2_spot = cvpinn.gate2(p, cvpinn.Domain(v_min=1e-3), "spot")

        gen = torch.Generator().manual_seed(0)
        x0 = 2.5 * (2.0 * torch.rand(1024, generator=gen) - 1.0)
        v0 = torch.zeros(1024)
        t0 = 1e-4 + (2.0 - 1e-4) * torch.rand(1024, generator=gen) ** 2
        lhs0 = cvpinn.operator(
            lambda a, b, c: cvpinn.bs_call(a, cvpinn.total_variance(b, c, p, "expected")),
            x0, v0, t0, p,
        )
        src0 = cvpinn.source(x0, v0, t0, p, "expected")
        g2_v0_strip = (lhs0 + src0).abs().max().item()
        g3_val = cvpinn.gate3(p, cvpinn.Domain(v_min=0.0))

        rep["G2"][pname] = {
            "expected_all": float(g2_exp),
            "expected_v0_strip": float(g2_v0_strip),
            "spot_all": float(g2_spot),
            "passed": bool((g2_exp <= 1e-6) and (g2_v0_strip <= 1e-6) and (g2_spot <= 1e-6)),
        }
        rep["G3"][pname] = {
            "max_abs_xi0": float(g3_val),
            "passed": bool(g3_val == 0.0),
        }

        # Verify optimized greeks() against closed-form Black-Scholes Gamma & Vanna
        x_grid, _ = load_or_build_ref_grid(pname)[:2]
        price_bs = lambda a, b, c: cvpinn.bs_call(a, cvpinn.total_variance(b, c, p, "expected"))
        pred_bs = cvpinn.evaluate_on_grid(price_bs, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)
        err_gam = 0.0
        err_cxv = 0.0
        for it, tau in enumerate(hgv.EVAL_TAUS):
            b_tau = -math.expm1(-p.kappa * tau) / p.kappa
            for iv, v in enumerate(hgv.EVAL_VS):
                w = hgv.w_exp_stable(v, tau, p.kappa, p.theta)
                for iz in range(len(hgv.EVAL_ZS)):
                    xq = float(x_grid[it, iv, iz])
                    err_gam = max(err_gam, abs(float(pred_bs["gam"][it, iv, iz]) - hr.bs_gam(xq, w)))
                    err_cxv = max(err_cxv, abs(float(pred_bs["cxv"][it, iv, iz]) - b_tau * hr.bs_xw(xq, w)))
        rep["greeks_bs_check"][pname] = {
            "max_gam_diff": float(err_gam),
            "max_cxv_diff": float(err_cxv),
            "passed": bool((float(err_gam) <= 1e-12) and (float(err_cxv) <= 1e-12)),
        }

    os.makedirs("runs", exist_ok=True)
    with open("runs/gates_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    all_ok = all(
        rep[g][pn]["passed"]
        for g in ("G1", "G2", "G3", "greeks_bs_check")
        for pn in PARAMS
    )
    if not all_ok:
        raise SystemExit("One or more pre-training gates failed!")


def run_single_job(arm, pname, seed, term_weight=10.0, v_max=0.5, adam_steps=10000,
                   lbfgs_iter=500, out_json=None, ckpt_path=None, num_threads=1):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    p = _to_heston(PARAMS[pname])
    x_grid, ref_grid, _ = load_or_build_ref_grid(pname)

    t0 = time.perf_counter()
    price_fn, net, train_info = cvpinn.build_and_train(
        arm, p, seed, adam_steps=adam_steps, lbfgs_iter=lbfgs_iter,
        term_weight=term_weight, v_max=v_max, device="cpu", return_info=True,
    )
    train_sec = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    pred = cvpinn.evaluate_on_grid(price_fn, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS, device="cpu")
    infer_ms = 1000.0 * (time.perf_counter() - t_inf0)

    metrics = hgv.score_buckets(pred, ref_grid)
    record = {
        "arm": arm,
        "param_set": pname,
        "params": dict(zip(("kappa", "theta", "xi", "rho"), PARAMS[pname])),
        "seed": seed,
        "v_min": 1e-3 if arm == "A1" else 0.0,
        "v_max": v_max,
        "x_max": 2.5,
        "tau_min": 1e-4,
        "tau_max": 2.0,
        "term_weight": term_weight if arm == "B0" else None,
        "train_sec": round(train_sec, 4),
        "infer_ms_875pt": round(infer_ms, 3),
        "train_info": train_info,
        "metrics": metrics,
    }

    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(
            {
                "arm": arm,
                "param_set": pname,
                "seed": seed,
                "v_max": v_max,
                "term_weight": term_weight,
                "state_dict": net.state_dict(),
            },
            ckpt_path,
        )
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(record, f, indent=2)
    return record


def _execute_tasks(tasks, workers):
    results = []
    if workers <= 1 or len(tasks) <= 1:
        for kw in tasks:
            results.append(run_single_job(**kw))
        return results
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_single_job, **kw): kw for kw in tasks}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            sm = rec["metrics"]["short_mat"]
            print(
                f"Done {rec['arm']:3s} {rec['param_set']} seed={rec['seed']} "
                f"({rec['train_sec']:.1f}s) | short_mat Gam rel={sm['gam']['rel_rmse']:.4f} "
                f"Vanna rel={sm['cxv']['rel_rmse']:.4f}",
                flush=True,
            )
    return results


def cmd_day4_tune(args):
    os.makedirs("runs/day4", exist_ok=True)
    specs = [
        ("M1", vm, 10.0, f"runs/day4/M1_P2_vmax{vm}_seed0.json", f"runs/day4/M1_P2_vmax{vm}_seed0.pt")
        for vm in (0.5, 0.75)
    ] + [
        ("B0", 0.5, tw, f"runs/day4/B0_P2_tw{int(tw)}_seed0.json", f"runs/day4/B0_P2_tw{int(tw)}_seed0.pt")
        for tw in (1.0, 10.0, 100.0)
    ]
    tasks = []
    recs = []
    for arm, vm, tw, out_p, ckpt_p in specs:
        if os.path.exists(out_p) and not args.force:
            with open(out_p) as f:
                recs.append(json.load(f))
            continue
        tasks.append({
            "arm": arm, "pname": "P2", "seed": 0, "v_max": vm, "term_weight": tw,
            "adam_steps": args.adam_steps, "lbfgs_iter": args.lbfgs_iter,
            "out_json": out_p, "ckpt_path": ckpt_p,
            "num_threads": 1 if args.workers > 1 else args.threads,
        })
    recs.extend(_execute_tasks(tasks, args.workers))
    m1_runs = sorted([r for r in recs if r["arm"] == "M1"], key=lambda r: r["v_max"])
    b0_runs = sorted([r for r in recs if r["arm"] == "B0"], key=lambda r: r["term_weight"])
    summary = {"m1_vmax_check": m1_runs, "b0_term_weight_sweep": b0_runs}
    with open("runs/day4_tuning.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDay-4 summary saved to runs/day4_tuning.json")
    for r in m1_runs + b0_runs:
        tag = f"M1 vmax={r['v_max']}" if r["arm"] == "M1" else f"B0 tw={int(r['term_weight'])}"
        sm = r["metrics"]["short_mat"]
        ov = r["metrics"]["overall"]
        print(
            f"  {tag:12s} | time={r['train_sec']:6.1f}s | "
            f"short_mat: gam_rel={sm['gam']['rel_rmse']:.4f} cxv_rel={sm['cxv']['rel_rmse']:.4f} "
            f"c_rel={sm['c']['rel_rmse']:.4f} | overall: gam_rel={ov['gam']['rel_rmse']:.4f} "
            f"cxv_rel={ov['cxv']['rel_rmse']:.4f}"
        )


def _run_path(arm, pname, seed):
    return f"runs/{arm}_{pname}_seed{seed}.json"


def _ckpt_path(arm, pname, seed):
    return f"runs/checkpoints/{arm}_{pname}_seed{seed}.pt"


def cmd_s2_pilot(args):
    pre_reg = {
        "term_weight": args.term_weight,
        "v_max": args.v_max,
        "adam_steps": args.adam_steps,
        "lbfgs_iter": args.lbfgs_iter,
        "s2_rule": {
            "param_set": "P2",
            "bucket": "short_mat",
            "min_gamma_wins": 4,
            "min_vanna_wins": 4,
            "max_median_gamma_ratio": 0.5,
        },
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs("runs", exist_ok=True)
    with open("runs/pre_registration.json", "w") as f:
        json.dump(pre_reg, f, indent=2)

    tasks = []
    for arm in ("B0", "M1"):
        for seed in SEEDS:
            out_p = _run_path(arm, "P2", seed)
            if os.path.exists(out_p) and not args.force:
                continue
            tasks.append({
                "arm": arm, "pname": "P2", "seed": seed,
                "term_weight": args.term_weight, "v_max": args.v_max,
                "adam_steps": args.adam_steps, "lbfgs_iter": args.lbfgs_iter,
                "out_json": out_p, "ckpt_path": _ckpt_path(arm, "P2", seed),
                "num_threads": 1 if args.workers > 1 else args.threads,
            })
    _execute_tasks(tasks, args.workers)

    b0_recs = [json.load(open(_run_path("B0", "P2", s))) for s in SEEDS]
    m1_recs = [json.load(open(_run_path("M1", "P2", s))) for s in SEEDS]
    gam_ratios, cxv_ratios = [], []
    gam_wins = cxv_wins = both_wins = 0
    for b, m in zip(b0_recs, m1_recs):
        bg = b["metrics"]["short_mat"]["gam"]["rmse"]
        mg = m["metrics"]["short_mat"]["gam"]["rmse"]
        bv = b["metrics"]["short_mat"]["cxv"]["rmse"]
        mv = m["metrics"]["short_mat"]["cxv"]["rmse"]
        gam_ratios.append(mg / bg)
        cxv_ratios.append(mv / bv)
        gw = mg < bg
        vw = mv < bv
        gam_wins += int(gw)
        cxv_wins += int(vw)
        both_wins += int(gw and vw)

    med_gam_ratio = statistics.median(gam_ratios)
    s2_passed = (gam_wins >= 4) and (cxv_wins >= 4) and (med_gam_ratio <= 0.5)
    mean_b0_t = statistics.mean(r["train_sec"] for r in b0_recs)
    mean_m1_t = statistics.mean(r["train_sec"] for r in m1_recs)
    est_50_runs_hr = (10 * mean_b0_t + 40 * mean_m1_t) / 3600.0 / max(args.workers, 1)

    report = {
        "s2_passed": s2_passed,
        "gamma_wins_out_of_5": gam_wins,
        "vanna_wins_out_of_5": cxv_wins,
        "both_wins_out_of_5": both_wins,
        "median_gamma_rmse_ratio": med_gam_ratio,
        "median_vanna_rmse_ratio": statistics.median(cxv_ratios),
        "per_seed_gamma_ratios": gam_ratios,
        "per_seed_vanna_ratios": cxv_ratios,
        "mean_b0_train_sec": round(mean_b0_t, 2),
        "mean_m1_train_sec": round(mean_m1_t, 2),
        "est_remaining_40_runs_hr_at_workers": round((10 * mean_b0_t + 30 * mean_m1_t) / 3600.0 / max(args.workers, 1), 3),
        "est_total_50_runs_hr_at_workers": round(est_50_runs_hr, 3),
    }
    with open("runs/s2_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def cmd_full_matrix(args):
    with open("runs/pre_registration.json") as f:
        pre = json.load(f)
    seeds = SEEDS[: args.ablation_seeds]
    tasks = []
    for pname in ("P2", "P1"):
        for arm in ARMS:
            arm_seeds = SEEDS if arm in ("B0", "M1") else seeds
            for seed in arm_seeds:
                out_p = _run_path(arm, pname, seed)
                if os.path.exists(out_p) and not args.force:
                    continue
                tasks.append({
                    "arm": arm, "pname": pname, "seed": seed,
                    "term_weight": pre["term_weight"], "v_max": pre["v_max"],
                    "adam_steps": pre["adam_steps"], "lbfgs_iter": pre["lbfgs_iter"],
                    "out_json": out_p, "ckpt_path": _ckpt_path(arm, pname, seed),
                    "num_threads": 1 if args.workers > 1 else args.threads,
                })
    print(f"Queued {len(tasks)} runs across workers={args.workers}")
    _execute_tasks(tasks, args.workers)


def cmd_summarize(args):
    summary = {}
    for pname in PARAMS:
        summary[pname] = {}
        for arm in ARMS:
            recs = []
            for seed in SEEDS:
                p = _run_path(arm, pname, seed)
                if os.path.exists(p):
                    with open(p) as f:
                        recs.append(json.load(f))
            if not recs:
                continue
            buckets = recs[0]["metrics"].keys()
            summary[pname][arm] = {"n_seeds": len(recs), "buckets": {}}
            for b in buckets:
                summary[pname][arm]["buckets"][b] = {}
                for k in ("c", "cx", "gam", "cxv"):
                    rmses = [r["metrics"][b][k]["rmse"] for r in recs]
                    rels = [r["metrics"][b][k]["rel_rmse"] for r in recs]
                    maxs = [r["metrics"][b][k]["max_abs"] for r in recs]
                    summary[pname][arm]["buckets"][b][k] = {
                        "rmse_mean": statistics.mean(rmses),
                        "rmse_std": statistics.stdev(rmses) if len(rmses) > 1 else 0.0,
                        "rel_rmse_mean": statistics.mean(rels),
                        "rel_rmse_std": statistics.stdev(rels) if len(rels) > 1 else 0.0,
                        "max_abs_mean": statistics.mean(maxs),
                    }
    with open("runs/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved runs/summary.json")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_g = sub.add_parser("gates")
    p_g.add_argument("--force", action="store_true")

    p_d4 = sub.add_parser("day4_tune")
    p_d4.add_argument("--workers", type=int, default=4)
    p_d4.add_argument("--threads", type=int, default=4)
    p_d4.add_argument("--adam-steps", type=int, default=10000)
    p_d4.add_argument("--lbfgs-iter", type=int, default=500)
    p_d4.add_argument("--force", action="store_true")

    p_s2 = sub.add_parser("s2_pilot")
    p_s2.add_argument("--term-weight", type=float, required=True)
    p_s2.add_argument("--v-max", type=float, default=0.5)
    p_s2.add_argument("--workers", type=int, default=4)
    p_s2.add_argument("--threads", type=int, default=4)
    p_s2.add_argument("--adam-steps", type=int, default=10000)
    p_s2.add_argument("--lbfgs-iter", type=int, default=500)
    p_s2.add_argument("--force", action="store_true")

    p_fm = sub.add_parser("full_matrix")
    p_fm.add_argument("--workers", type=int, default=4)
    p_fm.add_argument("--threads", type=int, default=4)
    p_fm.add_argument("--ablation-seeds", type=int, default=5)
    p_fm.add_argument("--force", action="store_true")

    sub.add_parser("summarize")
    args = parser.parse_args()

    if args.cmd == "gates":
        cmd_gates(args)
    elif args.cmd == "day4_tune":
        cmd_day4_tune(args)
    elif args.cmd == "s2_pilot":
        cmd_s2_pilot(args)
    elif args.cmd == "full_matrix":
        cmd_full_matrix(args)
    elif args.cmd == "summarize":
        cmd_summarize(args)


if __name__ == "__main__":
    main()
