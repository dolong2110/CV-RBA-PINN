"""run_tier2_experiments.py - Tier-2 pilot ablations, M2 stratified-z/feature upgrades, and 6-Greek evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import statistics
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import numpy as np
import torch

import cvpinn
import cvpinn_td as tdmod
import cvpinn_tier2 as t2
import heston_grid_vec as hgv
import heston_ref as hr

torch.set_default_dtype(torch.float64)
TIER2_DIR = os.path.join("runs", "tier2")
TIER2_CKPT_DIR = os.path.join(TIER2_DIR, "checkpoints")


def _load_ref(pname):
    if pname == "TD":
        d = np.load(os.path.join("runs", "ref_grids", "TD_ADI.npz"))
        return d["x_grid"], {k: d[k] for k in ("c", "cx", "gam", "cxv") if k in d}
    d = np.load(os.path.join("runs", "ref_grids", f"{pname}.npz"))
    return d["x_grid"], {k: d[k] for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}


def train_td_tier2(arm_name, seed, adam_steps=10000, lbfgs_iter=500, v_max=0.5, device="cpu"):
    """Time-dependent counterpart of Tier-2 M2 variants."""
    configs = {
        "M2_zv_td": {"extra_scales": False, "z2": False, "z_frac": 0.65, "vp": 4.0, "tp": 3.0, "width": 64, "depth": 4, "n_lbfgs": 16384, "lowv_weight": False},
        "M2_zv_feat_td": {"extra_scales": True, "z2": True, "z_frac": 0.65, "vp": 4.0, "tp": 3.0, "width": 96, "depth": 4, "n_lbfgs": 16384, "lowv_weight": False},
        "M2_zv_lw_td": {"extra_scales": True, "z2": False, "z_frac": 0.55, "vp": 2.5, "tp": 2.2, "width": 80, "depth": 4, "n_lbfgs": 16384, "lowv_weight": True},
    }
    cfg = configs[arm_name]
    td = tdmod.default_td_heston()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dom = cvpinn.Domain(v_max=v_max)
    n_in = 4 + (2 if cfg["extra_scales"] else 0) + (1 if cfg.get("z2", False) else 0)
    net = cvpinn.Mlp(n_in, width=cfg["width"], depth=cfg["depth"]).to(device)
    u = t2.make_correction_tier2(
        net, td, dom, kind="expected", use_z=True,
        extra_scales=cfg["extra_scales"], z2=cfg.get("z2", False), is_td=True,
    )
    n_col = 2048
    n_strip = n_col // 16
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=1e-4)

    t0 = time.perf_counter()
    last_loss = float("nan")
    for _ in range(adam_steps):
        x, v, tau = t2.sample_stratified_z(
            n_col, n_strip, td, dom, gen, z_frac=cfg["z_frac"], z_max=4.0,
            v_power=cfg["vp"], t_power=cfg["tp"], is_td=True, device=device,
        )
        xe, ve, te = cvpinn.sample_edges(n_col // 8, dom, gen, device=device)
        r = tdmod.operator_td(u, x, v, tau, td) - tdmod.source_td(x, v, tau, td, "expected")
        if cfg.get("lowv_weight", False):
            rw = 1.0 / torch.sqrt(v + 0.02)
            rw = rw / torch.sqrt((rw ** 2).mean() + 1e-30)
            loss = ((rw * r) ** 2).mean() + (u(xe, ve, te) ** 2).mean()
        else:
            loss = (r ** 2).mean() + (u(xe, ve, te) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        last_loss = float(loss.detach().item())
    adam_sec = time.perf_counter() - t0

    n_lbfgs = cfg["n_lbfgs"]
    lx, lv, lt = t2.sample_stratified_z(
        n_lbfgs, n_lbfgs // 16, td, dom, gen, z_frac=cfg["z_frac"], z_max=4.0,
        v_power=cfg["vp"], t_power=cfg["tp"], is_td=True, device=device,
    )
    lxe, lve, lte = cvpinn.sample_edges(1024, dom, gen, device=device)
    src_cached = tdmod.source_td(lx, lv, lt, td, "expected").detach()
    lw = None
    if cfg.get("lowv_weight", False):
        lw_raw = 1.0 / torch.sqrt(lv + 0.02)
        lw = (lw_raw / torch.sqrt((lw_raw ** 2).mean() + 1e-30)).detach()

    def lbfgs_loss():
        r_l = tdmod.operator_td(u, lx, lv, lt, td) - src_cached
        lp = ((lw * r_l) ** 2).mean() if lw is not None else (r_l ** 2).mean()
        return lp + (u(lxe, lve, lte) ** 2).mean()

    linfo = cvpinn._run_lbfgs_safe(net, lbfgs_loss, lbfgs_iter)
    price_fn = tdmod.make_price_td(u, td, kind="expected")
    info = {
        "arm": arm_name, "config": cfg, "adam_steps": adam_steps, "adam_sec": round(adam_sec, 4),
        "adam_last_loss": last_loss, **{k: (round(v, 4) if k.endswith("_sec") else v) for k, v in linfo.items()},
    }
    return price_fn, net, info


def run_job(arm, pname, seed, adam_steps, lbfgs_iter, tag, overwrite):
    os.makedirs(TIER2_DIR, exist_ok=True)
    os.makedirs(TIER2_CKPT_DIR, exist_ok=True)
    stem = f"{arm}_{pname}_seed{seed}" + (f"_{tag}" if tag else "")
    json_path = os.path.join(TIER2_DIR, f"{stem}.json")
    ckpt_path = os.path.join(TIER2_CKPT_DIR, f"{stem}.pt")
    if (not overwrite) and os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f), True

    x_grid, ref = _load_ref(pname)
    t0 = time.perf_counter()
    if pname == "TD":
        price_fn, net, info = train_td_tier2(arm, seed, adam_steps=adam_steps, lbfgs_iter=lbfgs_iter)
    else:
        p = cvpinn.Heston(*hr.P_FELLER_BAD if pname == "P2" else hr.P_FELLER_OK)
        price_fn, net, info = t2.train_tier2(
            arm, p, seed, adam_steps=adam_steps, lbfgs_iter=lbfgs_iter, return_info=True
        )
    train_sec = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    pred = t2.evaluate_all_greeks(price_fn, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)
    infer_ms = 1000.0 * (time.perf_counter() - t_inf0)

    metrics = t2.score_all_six_buckets(pred, ref)
    torch.save({"state_dict": net.state_dict(), "arm": arm, "pname": pname, "seed": seed, "info": info}, ckpt_path)
    rec = {
        "arm": arm, "param_set": pname, "seed": seed, "tag": tag,
        "adam_steps": adam_steps, "lbfgs_iter": lbfgs_iter,
        "train_sec": round(train_sec, 4), "infer_ms_875pt": round(infer_ms, 3),
        "train_info": info, "metrics": metrics,
    }
    with open(json_path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec, False


def _worker(kw):
    return run_job(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--params", default="P2")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--jobs", default="", help="Optional comma-separated arm:param:seed specs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--adam-steps", type=int, default=10000)
    ap.add_argument("--lbfgs-iter", type=int, default=500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.jobs.strip():
        jobs = []
        for spec in args.jobs.split(","):
            spec = spec.strip()
            if not spec:
                continue
            a, p, s = spec.split(":")
            jobs.append({
                "arm": a.strip(), "pname": p.strip(), "seed": int(s.strip()),
                "adam_steps": args.adam_steps, "lbfgs_iter": args.lbfgs_iter,
                "tag": args.tag, "overwrite": args.overwrite,
            })
    else:
        if not args.arms.strip():
            raise SystemExit("Provide either --arms or --jobs.")
        jobs = [
            {
                "arm": a.strip(), "pname": p.strip(), "seed": int(s.strip()),
                "adam_steps": args.adam_steps, "lbfgs_iter": args.lbfgs_iter,
                "tag": args.tag, "overwrite": args.overwrite,
            }
            for a in args.arms.split(",") if a.strip()
            for p in args.params.split(",") if p.strip()
            for s in args.seeds.split(",") if s.strip()
        ]
    print(f"Running {len(jobs)} Tier-2 jobs with workers={args.workers}...")
    def _log(rec, skip):
        ov, sl = rec["metrics"]["overall"], rec["metrics"]["short_low_var"]
        cv_s = f" cv={ov['cv']['rel_rmse']:.3e} cvv={ov['cvv']['rel_rmse']:.3e}" if "cv" in ov else ""
        tag = "SKIP" if skip else f"{rec['train_sec']:.1f}s"
        print(
            f"[{tag}] {rec['arm']} ({rec['param_set']} s{rec['seed']}) | "
            f"ov c={ov['c']['rel_rmse']:.3e} gam={ov['gam']['rel_rmse']:.3e} cxv={ov['cxv']['rel_rmse']:.3e}{cv_s} | "
            f"slv gam={sl['gam']['rel_rmse']:.3e} cxv={sl['cxv']['rel_rmse']:.3e}"
        )

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
