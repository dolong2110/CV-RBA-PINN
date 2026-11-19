"""run_td_experiments.py - Tier-1 Time-Dependent Heston validation & benchmark runner.

Subcommands:
  oracle     Validate 2D ADI against chained piecewise-constant Riccati Fourier and
             grid refinement on Check T15 smooth parameters, caching runs/ref_grids/TD_ADI.npz.
  gates      Verify TD Chebyshev ODE vs RK4, Gate G2_td (interior + v=0 strip), and Gate G3_td.
  benchmark  Train B0_td, M1_td, A1_td, A2b_td across seeds in float64 using ProcessPoolExecutor.
  summarize  Aggregate runs/td/*_TD_seed*.json into runs/td_summary.json.
"""
import argparse
import cmath
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import statistics
import time

# Force 1 thread per worker BEFORE importing NumPy/PyTorch so --workers N scales cleanly.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import numpy as np
import torch

import cvpinn
import cvpinn_td as tdmod
import heston_adi_2d as adi
import heston_grid_vec as hgv
import heston_ref as hr


torch.set_default_dtype(torch.float64)

DEFAULT_TD_ARMS = ("B0_td", "M1_td", "A1_td", "A2b_td")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
TD_REF_PATH = os.path.join("runs", "ref_grids", "TD_ADI.npz")
TD_OUT_DIR = os.path.join("runs", "td")
TD_CKPT_DIR = os.path.join(TD_OUT_DIR, "checkpoints")


def step_char_ab(z, tau_edges, params_list):
    """Exact chained Riccati ODE solution across piecewise-constant parameter regimes."""
    big_a, big_b = 0.0j, 0.0j
    for (t0, t1), p in zip(zip(tau_edges[:-1], tau_edges[1:]), params_list):
        s = t1 - t0
        if s <= 0.0:
            continue
        kappa, theta, xi, rho = p
        a = kappa - 1j * rho * xi * z
        d = cmath.sqrt(a * a + xi * xi * (z * z + 1j * z))
        if d.real < 0.0:
            d = -d
        xi2 = xi * xi
        g = (a - d - xi2 * big_b) / (a + d - xi2 * big_b)
        e = cmath.exp(-d * s)
        big_b_new = ((a - d) - (a + d) * g * e) / (xi2 * (1.0 - g * e))
        big_a = big_a + kappa * theta / xi2 * (
            (a - d) * s - 2.0 * cmath.log((1.0 - g * e) / (1.0 - g))
        )
        big_b = big_b_new
    return big_a, big_b


def heston_pwc(x, v, tau, switch_tau=0.25, p_early=hr.P_FELLER_BAD, p_late=hr.P_FELLER_OK):
    """Exact Lewis Fourier inversion for a 2-regime step-change Heston model."""
    wide = min(2.0, math.pi / (2.0 * max(abs(x), 1e-3)))
    sums = [0.0] * 4
    u0, quiet = 0.0, 0
    if tau <= switch_tau:
        edges, plist = [0.0, tau], [p_early]
    else:
        edges, plist = [0.0, switch_tau, tau], [p_early, p_late]
    while u0 < 4e5:
        panel = min(0.5, wide) if u0 < 4.0 else wide
        biggest = 0.0
        for xk, wk in zip(hr.GLX, hr.GLW):
            u = u0 + 0.5 * panel * (xk + 1.0)
            big_a, big_b = step_char_ab(complex(u, -0.5), edges, plist)
            phi = cmath.exp(big_a + big_b * v + 1j * u * x)
            den = u * u + 0.25
            f = complex(0.5, u)
            wgt = 0.5 * panel * wk
            sums[0] += wgt * (phi / den).real
            sums[1] += wgt * (f * phi / den).real
            sums[2] += wgt * phi.real
            sums[3] += wgt * (f * big_b * phi / den).real
            biggest = max(biggest, abs(phi) * (1.0 + abs(big_b)) ** 2)
        u0 += panel
        quiet = quiet + 1 if biggest < 1e-17 else 0
        if quiet >= 3:
            break
    pref = math.exp(0.5 * x) / math.pi
    ex = math.exp(x)
    return {
        "c": ex - pref * sums[0],
        "cx": ex - pref * sums[1],
        "gam": pref * sums[2],
        "cxv": -pref * sums[3],
    }


def rk4_ab_reference(td, target_taus, n_steps=8000):
    """Independent 4th-order Runge-Kutta integrator for beta'(t)=1-k*b, alpha'(t)=k*th*b."""
    t_max = float(max(target_taus))
    dt = t_max / n_steps
    alpha, beta = 0.0, 0.0
    out = {}
    targets = sorted(set(float(t) for t in target_taus))
    idx = 0
    for step in range(n_steps):
        t = step * dt
        t_next = (step + 1) * dt
        while idx < len(targets) and targets[idx] <= t_next + 1e-12:
            # Fine local RK4 to hit exact target without interpolation error
            h = targets[idx] - t
            if h <= 0.0:
                out[targets[idx]] = (alpha, beta)
            else:
                k1a, k1b = td.kappa_np(t) * td.theta_np(t) * beta, 1.0 - td.kappa_np(t) * beta
                tm = t + 0.5 * h
                bm = beta + 0.5 * h * k1b
                k2a, k2b = td.kappa_np(tm) * td.theta_np(tm) * bm, 1.0 - td.kappa_np(tm) * bm
                bm2 = beta + 0.5 * h * k2b
                k3a, k3b = td.kappa_np(tm) * td.theta_np(tm) * bm2, 1.0 - td.kappa_np(tm) * bm2
                b_end = beta + h * k3b
                te = t + h
                k4a, k4b = td.kappa_np(te) * td.theta_np(te) * b_end, 1.0 - td.kappa_np(te) * b_end
                out[targets[idx]] = (
                    alpha + (h / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a),
                    beta + (h / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b),
                )
            idx += 1

        k1a, k1b = td.kappa_np(t) * td.theta_np(t) * beta, 1.0 - td.kappa_np(t) * beta
        tm = t + 0.5 * dt
        bm = beta + 0.5 * dt * k1b
        k2a, k2b = td.kappa_np(tm) * td.theta_np(tm) * bm, 1.0 - td.kappa_np(tm) * bm
        bm2 = beta + 0.5 * dt * k2b
        k3a, k3b = td.kappa_np(tm) * td.theta_np(tm) * bm2, 1.0 - td.kappa_np(tm) * bm2
        b_end = beta + dt * k3b
        te = t_next
        k4a, k4b = td.kappa_np(te) * td.theta_np(te) * b_end, 1.0 - td.kappa_np(te) * b_end
        alpha += (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
        beta += (dt / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
    return out


def cmd_oracle(_args):
    """Validates 2D ADI on piecewise-constant Riccati Fourier + TD grid refinement & caches oracle."""
    os.makedirs(os.path.dirname(TD_REF_PATH), exist_ok=True)
    os.makedirs(TD_OUT_DIR, exist_ok=True)

    # 1. Chained piecewise-constant Riccati cross-check: P2 on [0, 0.25], P1 on (0.25, 2.0]
    zs = np.asarray(hgv.EVAL_ZS, dtype=np.float64)
    x_pwc = np.zeros((len(hgv.EVAL_TAUS), len(hgv.EVAL_VS), len(zs)), dtype=np.float64)
    for it, tau in enumerate(hgv.EVAL_TAUS):
        for iv, vv in enumerate(hgv.EVAL_VS):
            x_pwc[it, iv, :] = zs * math.sqrt(max(0.04, float(vv)) * float(tau))

    ref_pwc = {k: np.zeros_like(x_pwc) for k in ("c", "cx", "gam", "cxv")}
    for it, tau in enumerate(hgv.EVAL_TAUS):
        for iv, vv in enumerate(hgv.EVAL_VS):
            for iz in range(len(zs)):
                res = heston_pwc(float(x_pwc[it, iv, iz]), float(vv), float(tau))
                for k in ref_pwc:
                    ref_pwc[k][it, iv, iz] = res[k]

    pwc_fn = lambda t: hr.P_FELLER_BAD if t <= 0.25 else hr.P_FELLER_OK
    adi_pwc, pwc_sec = adi.eval_adi_on_grid(pwc_fn, x_pwc)
    pwc_scores = hgv.score_buckets(adi_pwc, ref_pwc)
    print(f"[Oracle-1] ADI vs Chained Riccati PWC (P2->P1 at tau=0.25) in {pwc_sec:.2f}s:")
    for b in ("overall", "short_low_var", "short_mod_var"):
        s = pwc_scores[b]
        print(
            f"  {b:14s}: c={s['c']['rel_rmse']:.3e}, cx={s['cx']['rel_rmse']:.3e}, "
            f"gam={s['gam']['rel_rmse']:.3e} (max={s['gam']['max_abs']:.3e}), "
            f"cxv={s['cxv']['rel_rmse']:.3e} (max={s['cxv']['max_abs']:.3e})"
        )

    # 2. Smooth Check T15 grid-refinement stability
    td = tdmod.default_td_heston()
    x_td = tdmod.eval_grid_td(td)
    print("\n[Oracle-2] Solving smooth T15 ADI at coarse / medium / fine resolutions...")
    adi_coarse, t_coarse = adi.eval_adi_on_grid(td.params_at, x_td, nx=400, nv=200, dt_scale=1, rannacher_steps=4)
    adi_med, t_med = adi.eval_adi_on_grid(td.params_at, x_td, nx=600, nv=300, dt_scale=1, rannacher_steps=4)
    adi_fine, t_fine = adi.eval_adi_on_grid(td.params_at, x_td, nx=800, nv=400, dt_scale=2, rannacher_steps=8)

    diff_cm = hgv.score_buckets(adi_coarse, adi_med)
    diff_mf = hgv.score_buckets(adi_med, adi_fine)
    print(f"  Times: coarse={t_coarse:.2f}s, med={t_med:.2f}s, fine={t_fine:.2f}s")
    print("  Coarse (400x200) vs Med (600x300):")
    for b in ("overall", "short_low_var", "short_mod_var"):
        s = diff_cm[b]
        print(f"    {b:14s}: c={s['c']['rel_rmse']:.3e}, gam={s['gam']['rel_rmse']:.3e}, cxv={s['cxv']['rel_rmse']:.3e}")
    print("  Med (600x300, 1350t) vs Fine (800x400, 2700t):")
    for b in ("overall", "short_low_var", "short_mod_var"):
        s = diff_mf[b]
        print(f"    {b:14s}: c={s['c']['rel_rmse']:.3e}, gam={s['gam']['rel_rmse']:.3e}, cxv={s['cxv']['rel_rmse']:.3e}")

    # Cache the fine oracle grid for zero-bias PINN evaluation
    np.savez(
        TD_REF_PATH,
        x_grid=x_td,
        taus=np.asarray(hgv.EVAL_TAUS, dtype=np.float64),
        vs=np.asarray(hgv.EVAL_VS, dtype=np.float64),
        zs=np.asarray(hgv.EVAL_ZS, dtype=np.float64),
        c=adi_fine["c"],
        cx=adi_fine["cx"],
        gam=adi_fine["gam"],
        cxv=adi_fine["cxv"],
        elapsed_sec=np.asarray([t_fine], dtype=np.float64),
    )
    report = {
        "pwc_riccati_vs_adi": {"elapsed_sec": round(pwc_sec, 4), "buckets": pwc_scores},
        "td_refinement": {
            "times_sec": {"coarse": round(t_coarse, 3), "med": round(t_med, 3), "fine": round(t_fine, 3)},
            "coarse_vs_med": diff_cm,
            "med_vs_fine": diff_mf,
            "cached_oracle_path": TD_REF_PATH,
            "cached_resolution": "nx=800, nv=400, 2700 graded steps, rannacher=8",
        },
    }
    val_path = os.path.join("runs", "td_oracle_validation.json")
    with open(val_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nCached fine ADI reference -> {TD_REF_PATH}")
    print(f"Saved validation report   -> {val_path}")


def cmd_gates(_args):
    """Verifies TD spectral ODE precision vs RK4, Gate G2_td (incl. v=0 strip), and Gate G3_td."""
    os.makedirs(TD_OUT_DIR, exist_ok=True)
    td = tdmod.default_td_heston()
    check_taus = [1e-4, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    rk = rk4_ab_reference(td, check_taus, n_steps=8000)
    a_cheb, b_cheb = tdmod.alpha_beta_np(check_taus, td)
    ode_errs = {}
    max_rel_a, max_rel_b = 0.0, 0.0
    for i, tau in enumerate(check_taus):
        a_rk, b_rk = rk[float(tau)]
        ra = abs(float(a_cheb[i]) - a_rk) / max(abs(a_rk), 1e-30)
        rb = abs(float(b_cheb[i]) - b_rk) / max(abs(b_rk), 1e-30)
        ode_errs[str(tau)] = {"rel_alpha": ra, "rel_beta": rb}
        max_rel_a = max(max_rel_a, ra)
        max_rel_b = max(max_rel_b, rb)

    dom_exp = cvpinn.Domain(v_min=0.0, v_max=0.5)
    dom_spot = cvpinn.Domain(v_min=1e-3, v_max=0.5)
    g2_exp = tdmod.gate2_td(td, dom_exp, kind="expected", n=4096, seed=0)
    g2_spot = tdmod.gate2_td(td, dom_spot, kind="spot", n=4096, seed=0)

    # Explicit v=0 strip check for Gate G2_td expected
    gen = torch.Generator().manual_seed(0)
    x_v0 = torch.empty(1024).uniform_(-dom_exp.x_max, dom_exp.x_max, generator=gen)
    v_v0 = torch.zeros(1024)
    u1, u2 = torch.rand(1024, generator=gen), torch.rand(1024, generator=gen)
    log_half = math.log(dom_exp.tau_min) + (math.log(dom_exp.tau_max) - math.log(dom_exp.tau_min)) * u1[:512]
    t_v0 = torch.cat([torch.exp(log_half), dom_exp.tau_min + (dom_exp.tau_max - dom_exp.tau_min) * u2[512:]])
    lhs_v0 = tdmod.operator_td(
        lambda a, b, c: cvpinn.bs_call(a, tdmod.total_variance_td(b, c, td, "expected")),
        x_v0, v_v0, t_v0, td,
    )
    g2_v0_strip = float((lhs_v0 + tdmod.source_td(x_v0, v_v0, t_v0, td, "expected")).abs().max().item())
    g3 = tdmod.gate3_td(td, dom_exp, n=4096, seed=0)

    report = {
        "ode_max_rel_alpha": max_rel_a,
        "ode_max_rel_beta": max_rel_b,
        "ode_by_tau": ode_errs,
        "gate2_td_expected": g2_exp,
        "gate2_td_expected_v0_strip": g2_v0_strip,
        "gate2_td_spot": g2_spot,
        "gate3_td_xi_zero": g3,
    }
    out_path = os.path.join(TD_OUT_DIR, "gates_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"ODE max rel error: alpha={max_rel_a:.3e}, beta={max_rel_b:.3e}")
    print(f"Gate G2_td (expected): {g2_exp:.3e} | v=0 strip: {g2_v0_strip:.3e}")
    print(f"Gate G2_td (spot):     {g2_spot:.3e}")
    print(f"Gate G3_td (xi=0):     {g3:.3e}")
    assert max(g2_exp, g2_v0_strip, g2_spot) <= 1e-6
    assert g3 == 0.0
    print(f"Saved -> {out_path}")


def _load_td_oracle():
    if not os.path.exists(TD_REF_PATH):
        raise FileNotFoundError(f"Missing cached TD oracle {TD_REF_PATH}. Run `oracle` subcommand first.")
    d = np.load(TD_REF_PATH)
    ref = {k: d[k] for k in ("c", "cx", "gam", "cxv")}
    return d["x_grid"], ref


def run_single_td_job(arm, seed, adam_steps, lbfgs_iter, term_weight, v_max, overwrite):
    os.makedirs(TD_OUT_DIR, exist_ok=True)
    os.makedirs(TD_CKPT_DIR, exist_ok=True)
    json_path = os.path.join(TD_OUT_DIR, f"{arm}_TD_seed{seed}.json")
    ckpt_path = os.path.join(TD_CKPT_DIR, f"{arm}_TD_seed{seed}.pt")
    if (not overwrite) and os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        return existing, True

    td = tdmod.default_td_heston()
    x_grid, ref = _load_td_oracle()
    expected_x = tdmod.eval_grid_td(td)
    if np.max(np.abs(x_grid - expected_x)) > 1e-12:
        raise ValueError("Cached TD_ADI.npz x_grid mismatch with default_td_heston()")

    t0 = time.perf_counter()
    price_fn, net, info = tdmod.build_and_train_td(
        arm, td, seed=seed, adam_steps=adam_steps, lbfgs_iter=lbfgs_iter,
        term_weight=term_weight, v_max=v_max, device="cpu",
    )
    train_sec = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    pred = cvpinn.evaluate_on_grid(price_fn, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS, device="cpu")
    infer_ms = 1000.0 * (time.perf_counter() - t_inf0)

    metrics = hgv.score_buckets(pred, ref)
    torch.save(
        {"state_dict": net.state_dict(), "arm": arm, "seed": seed, "v_max": v_max, "train_info": info},
        ckpt_path,
    )

    rec = {
        "arm": arm,
        "param_set": "TD",
        "params": {
            "kappa": "1.5 + 0.8*sin(2*tau)",
            "theta": "0.04 + 0.02*tau",
            "xi": "0.5 - 0.1*tau",
            "rho": "-0.7 + 0.2*tau",
        },
        "seed": seed,
        "v_min": 1e-3 if arm == "A1_td" else 0.0,
        "v_max": v_max,
        "x_max": 2.5,
        "tau_min": 1e-4,
        "tau_max": 2.0,
        "term_weight": term_weight if arm == "B0_td" else None,
        "train_sec": round(train_sec, 4),
        "infer_ms_875pt": round(infer_ms, 3),
        "train_info": info,
        "metrics": metrics,
    }
    with open(json_path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec, False


def _worker_td_job(kwargs):
    return run_single_td_job(**kwargs)


def cmd_benchmark(args):
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    jobs = [
        {
            "arm": arm,
            "seed": seed,
            "adam_steps": args.adam_steps,
            "lbfgs_iter": args.lbfgs_iter,
            "term_weight": args.term_weight,
            "v_max": args.v_max,
            "overwrite": args.overwrite,
        }
        for arm in arms
        for seed in seeds
    ]
    print(f"Executing {len(jobs)} TD jobs with workers={args.workers}...")
    t0 = time.perf_counter()
    if args.workers <= 1:
        for job in jobs:
            rec, skipped = run_single_td_job(**job)
            s = rec["metrics"]["overall"]
            tag = "SKIP" if skipped else f"{rec['train_sec']:.1f}s"
            print(
                f"[{tag}] {rec['arm']} seed={rec['seed']} "
                f"c={s['c']['rel_rmse']:.3e} gam={s['gam']['rel_rmse']:.3e} "
                f"cxv={s['cxv']['rel_rmse']:.3e} "
                f"short_low_gam={rec['metrics']['short_low_var']['gam']['rel_rmse']:.3e}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker_td_job, j) for j in jobs]
            for fut in as_completed(futs):
                rec, skipped = fut.result()
                s = rec["metrics"]["overall"]
                tag = "SKIP" if skipped else f"{rec['train_sec']:.1f}s"
                print(
                    f"[{tag}] {rec['arm']} seed={rec['seed']} "
                    f"c={s['c']['rel_rmse']:.3e} gam={s['gam']['rel_rmse']:.3e} "
                    f"cxv={s['cxv']['rel_rmse']:.3e} "
                    f"short_low_gam={rec['metrics']['short_low_var']['gam']['rel_rmse']:.3e}"
                )
    print(f"Benchmark finished in {time.perf_counter() - t0:.1f}s.")


def _iqr(vals):
    if len(vals) < 2:
        return 0.0
    s = sorted(vals)
    return float(np.percentile(s, 75) - np.percentile(s, 25))


def cmd_summarize(_args):
    """Aggregates runs/td/*_TD_seed*.json into runs/td_summary.json."""
    files = sorted(
        os.path.join(TD_OUT_DIR, fn)
        for fn in os.listdir(TD_OUT_DIR)
        if fn.endswith(".json") and "_TD_seed" in fn
    )
    if not files:
        raise FileNotFoundError(f"No *_TD_seed*.json found in {TD_OUT_DIR}")

    grouped = {}
    for fp in files:
        with open(fp) as f:
            rec = json.load(f)
        grouped.setdefault(rec["arm"], []).append(rec)

    summary = {"arms": {}, "ratios_vs_B0_td": {}, "ratios_A2b_vs_M1_td": {}}
    buckets = ("overall", "short_mat", "low_var", "mod_var", "short_low_var", "short_mod_var")
    fields = ("c", "cx", "gam", "cxv")
    for arm, recs in sorted(grouped.items()):
        arm_s = {
            "n_seeds": len(recs),
            "seeds": [r["seed"] for r in recs],
            "median_train_sec": float(statistics.median(r["train_sec"] for r in recs)),
            "median_infer_ms": float(statistics.median(r["infer_ms_875pt"] for r in recs)),
            "buckets": {},
        }
        for b in buckets:
            arm_s["buckets"][b] = {}
            for fld in fields:
                rels = [r["metrics"][b][fld]["rel_rmse"] for r in recs]
                maxs = [r["metrics"][b][fld]["max_abs"] for r in recs]
                arm_s["buckets"][b][fld] = {
                    "median_rel_rmse": float(statistics.median(rels)),
                    "iqr_rel_rmse": _iqr(rels),
                    "min_rel_rmse": float(min(rels)),
                    "max_rel_rmse": float(max(rels)),
                    "median_max_abs": float(statistics.median(maxs)),
                }
        summary["arms"][arm] = arm_s

    if "B0_td" in summary["arms"]:
        b0 = summary["arms"]["B0_td"]
        for arm, arm_s in summary["arms"].items():
            if arm == "B0_td":
                continue
            summary["ratios_vs_B0_td"][arm] = {
                b: {
                    fld: arm_s["buckets"][b][fld]["median_rel_rmse"] / b0["buckets"][b][fld]["median_rel_rmse"]
                    for fld in fields
                }
                for b in buckets
            }
    if "M1_td" in summary["arms"] and "A2b_td" in summary["arms"]:
        m1 = summary["arms"]["M1_td"]
        a2b = summary["arms"]["A2b_td"]
        summary["ratios_A2b_vs_M1_td"] = {
            b: {
                fld: a2b["buckets"][b][fld]["median_rel_rmse"] / m1["buckets"][b][fld]["median_rel_rmse"]
                for fld in fields
            }
            for b in buckets
        }

    out_path = os.path.join("runs", "td_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("TD Summary (median rel_rmse across seeds):")
    for arm, arm_s in summary["arms"].items():
        ov = arm_s["buckets"]["overall"]
        sl = arm_s["buckets"]["short_low_var"]
        sm = arm_s["buckets"]["short_mod_var"]
        print(
            f"  {arm:6s} (n={arm_s['n_seeds']}, {arm_s['median_train_sec']:.1f}s) | "
            f"ov c={ov['c']['median_rel_rmse']:.3e} cx={ov['cx']['median_rel_rmse']:.3e} "
            f"gam={ov['gam']['median_rel_rmse']:.3e} cxv={ov['cxv']['median_rel_rmse']:.3e} | "
            f"slv gam={sl['gam']['median_rel_rmse']:.3e} cxv={sl['cxv']['median_rel_rmse']:.3e} | "
            f"smv gam={sm['gam']['median_rel_rmse']:.3e}"
        )
    print(f"Saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("oracle")
    sub.add_parser("gates")
    pb = sub.add_parser("benchmark")
    pb.add_argument("--arms", default=",".join(DEFAULT_TD_ARMS))
    pb.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    pb.add_argument("--workers", type=int, default=4)
    pb.add_argument("--adam-steps", type=int, default=10000)
    pb.add_argument("--lbfgs-iter", type=int, default=500)
    pb.add_argument("--term-weight", type=float, default=100.0)
    pb.add_argument("--v-max", type=float, default=0.5)
    pb.add_argument("--overwrite", action="store_true")
    sub.add_parser("summarize")
    args = p.parse_args()
    {"oracle": cmd_oracle, "gates": cmd_gates, "benchmark": cmd_benchmark, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
