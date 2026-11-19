"""upgrade_param_rar.py - Multi-round RAR Strong-Wolfe L-BFGS upgrade for 8D/10D amortized parametric models."""
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
import cvpinn_param as cp8
import cvpinn_tier2 as t2
import heston_grid_vec as hgv
import run_param_experiments as rpe

torch.set_default_dtype(torch.float64)


def _select_param_rar(u_fn, dom, box, gen, extra, n_pool=49152, n_rar=4096, device="cpu"):
    xp, vp, tp, kp, th, xi, rh = cp8.sample_param_collocation(
        n_pool, n_pool // 16, dom, box, gen, z_frac=0.60, z_max=5.0,
        v_power=2.5, t_power=2.2, device=device,
    )
    scores = []
    chunk = 8192
    for i0 in range(0, len(xp), chunk):
        sl = slice(i0, i0 + chunk)
        rc = (
            cp8.operator_param(u_fn, xp[sl], vp[sl], tp[sl], kp[sl], th[sl], xi[sl], rh[sl])
            - cp8.source_param(xp[sl], vp[sl], tp[sl], kp[sl], th[sl], xi[sl], rh[sl], "expected")
        )
        with torch.no_grad():
            w = 1.0 / torch.sqrt(vp[sl] + 0.02) if extra else torch.ones_like(vp[sl])
            scores.append((w * rc.detach()).abs())
    all_sc = torch.cat(scores)
    idx = torch.topk(all_sc, k=min(n_rar, len(all_sc))).indices
    return tuple(t[idx].detach() for t in (xp, vp, tp, kp, th, xi, rh))


def _eval_zero_shot(price_fn_param):
    evals = {}
    for pname, p_tup in rpe.EVAL_REGIMES.items():
        x_grid, ref = rpe._get_regime_ref(pname, p_tup)
        bound = cp8.bind_params(price_fn_param, cvpinn.Heston(*p_tup))
        t0 = time.perf_counter()
        pred = t2.evaluate_all_greeks(bound, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)
        infer_ms = 1000.0 * (time.perf_counter() - t0)
        arb, dist = m3.audit_arbitrage_and_percentiles(pred, ref, x_grid)
        evals[pname] = {
            "params": {"kappa": p_tup[0], "theta": p_tup[1], "xi": p_tup[2], "rho": p_tup[3]},
            "infer_ms_875pt": round(infer_ms, 3),
            "metrics": t2.score_all_six_buckets(pred, ref),
            "arbitrage": arb,
            "percentiles": dist,
        }
    return evals


def upgrade_one(base_arm, out_arm, seed=0, rounds=2, iters_per_round=1000, overwrite=False):
    json_path = os.path.join(rpe.PARAM_DIR, f"{out_arm}_seed{seed}.json")
    ckpt_path = os.path.join(rpe.PARAM_CKPT_DIR, f"{out_arm}_seed{seed}.pt")
    if (not overwrite) and os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f), True

    base_ckpt_path = os.path.join(rpe.PARAM_CKPT_DIR, f"{base_arm}_seed{seed}.pt")
    ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=True)
    extra = base_arm == "M_param_plus"
    width = int(ckpt["width"])
    n_in = 8 + (2 if extra else 0)
    dom, box = cvpinn.Domain(v_max=0.5), cp8.ParamBox()
    net = cvpinn.Mlp(n_in, width=width, depth=4)
    net.load_state_dict(ckpt["state_dict"])
    u = cp8.make_correction_param(net, dom, box, kind="expected", use_z=True, extra_scales=extra)
    price_fn_param = cp8.make_price_param(u, "expected")

    gen = torch.Generator().manual_seed(1000 + seed)
    t0 = time.perf_counter()
    round_infos = []
    stage_evals = []

    for rd in range(1, rounds + 1):
        lx, lv, lt, lkp, lth, lxi, lrh = cp8.sample_param_collocation(
            16384, 1024, dom, box, gen, z_frac=0.60, z_max=5.0, v_power=2.5, t_power=2.2
        )
        rar = _select_param_rar(u, dom, box, gen, extra=extra, n_pool=49152, n_rar=4096)
        lx, lv, lt, lkp, lth, lxi, lrh = (torch.cat([a, b]) for a, b in zip((lx, lv, lt, lkp, lth, lxi, lrh), rar))
        lxe, lve, lte, lkpe, lthe, lxie, lrhe = cp8.sample_param_edges(1024, dom, box, gen)
        src_cached = cp8.source_param(lx, lv, lt, lkp, lth, lxi, lrh, "expected").detach()
        lw = None
        if extra:
            lw_raw = 1.0 / torch.sqrt(lv + 0.02)
            lw = (lw_raw / torch.sqrt((lw_raw ** 2).mean() + 1e-30)).detach()

        def lloss():
            r_l = cp8.operator_param(u, lx, lv, lt, lkp, lth, lxi, lrh) - src_cached
            lp = ((lw * r_l) ** 2).mean() if lw is not None else (r_l ** 2).mean()
            return lp + (u(lxe, lve, lte, lkpe, lthe, lxie, lrhe) ** 2).mean()

        rinfo = m3._run_lbfgs_logged(net, lloss, iters_per_round, history_size=100, log_every=100)
        rinfo["round"] = rd
        rinfo["n_points"] = int(len(lx))
        round_infos.append(rinfo)
        ev = _eval_zero_shot(price_fn_param)
        stage_evals.append({
            "round": rd,
            "loss": rinfo["lbfgs_post_loss"],
            "P1_gam": ev["P1"]["metrics"]["overall"]["gam"]["rel_rmse"],
            "P2_gam": ev["P2"]["metrics"]["overall"]["gam"]["rel_rmse"],
            "P2_cxv": ev["P2"]["metrics"]["overall"]["cxv"]["rel_rmse"],
            "P3_gam": ev["P3_steep"]["metrics"]["overall"]["gam"]["rel_rmse"],
        })

    extra_sec = time.perf_counter() - t0
    final_evals = _eval_zero_shot(price_fn_param)
    torch.save(
        {
            "state_dict": net.state_dict(), "arm": base_arm, "upgraded_arm": out_arm,
            "seed": seed, "width": width, "n_params": m3.count_parameters(net),
        },
        ckpt_path,
    )
    rec = {
        "arm": out_arm, "base_arm": base_arm, "seed": seed, "width": width,
        "n_params": m3.count_parameters(net), "extra_rar_sec": round(extra_sec, 4),
        "train_info": {
            "base_info": ckpt.get("info", {}),
            "rar_rounds": round_infos,
            "lbfgs_post_loss": round_infos[-1]["lbfgs_post_loss"],
            "stage_evals": stage_evals,
        },
        "zero_shot_evals": final_evals,
    }
    with open(json_path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec, False


def _worker(kw):
    return upgrade_one(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for pname, p_tup in rpe.EVAL_REGIMES.items():
        rpe._get_regime_ref(pname, p_tup)

    jobs = [
        {"base_arm": "M_param", "out_arm": "M3_param", "seed": 0, "rounds": args.rounds, "iters_per_round": args.iters, "overwrite": args.overwrite},
        {"base_arm": "M_param_plus", "out_arm": "M3_param_plus", "seed": 0, "rounds": args.rounds, "iters_per_round": args.iters, "overwrite": args.overwrite},
    ]
    print(f"Upgrading {len(jobs)} amortized parametric models via {args.rounds}x{args.iters} RAR-L-BFGS...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for fut in as_completed(futs):
            rec, skip = fut.result()
            tag = "SKIP" if skip else f"{rec['extra_rar_sec']:.1f}s"
            print(f"[{tag}] {rec['arm']} loss={rec['train_info']['lbfgs_post_loss']:.3e}")
            for pname, ev in rec["zero_shot_evals"].items():
                ov = ev["metrics"]["overall"]
                arb = ev["arbitrage"]
                print(
                    f"  zero-shot {pname:8s}: c={ov['c']['rel_rmse']:.4f} gam={ov['gam']['rel_rmse']:.4f} "
                    f"cxv={ov['cxv']['rel_rmse']:.4f} cv={ov['cv']['rel_rmse']:.4f} cvv={ov['cvv']['rel_rmse']:.4f} "
                    f"| neg_gam={arb['neg_gam_count']}(core={arb['neg_gam_core_count']})"
                )


if __name__ == "__main__":
    main()
