"""paper/make_td_tables.py - Aggregate Time-Dependent Heston TD benchmark into runs/td_summary.json and paper/td_tables.md."""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ARMS = ("B0_td", "M1_td", "A1_td", "A2b_td")
SEEDS = (0, 1, 2, 3, 4)
BUCKETS = (
    ("overall", "Overall (875 pts)"),
    ("short_mat", "Short maturity ($\\tau \\le 0.1$, 375 pts)"),
    ("low_var", "Low variance ($v \\le 0.01$, 350 pts)"),
    ("mod_var", "Mod/high variance ($v \\ge 0.04$, 525 pts)"),
    ("short_low_var", "Short & low var ($\\tau \\le 0.1, v \\le 0.01$, 150 pts)"),
    ("short_mod_var", "Short & mod var ($\\tau \\le 0.1, v \\ge 0.04$, 225 pts)"),
)
KEYS = ("c", "cx", "gam", "cxv")


def load_td_runs():
    data = {}
    for a in ARMS:
        recs = []
        for s in SEEDS:
            path = os.path.join(ROOT, "runs", "td", f"{a}_TD_seed{s}.json")
            if os.path.exists(path):
                with open(path) as f:
                    recs.append(json.load(f))
        data[a] = recs
    return data


def _mean_sd(vals):
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def paired_stats(recs_a, recs_b, bucket, key):
    by_s_a = {r["seed"]: r for r in recs_a}
    by_s_b = {r["seed"]: r for r in recs_b}
    common = sorted(set(by_s_a) & set(by_s_b))
    ratios = []
    wins = 0
    for s in common:
        va = by_s_a[s]["metrics"][bucket][key]["rmse"]
        vb = by_s_b[s]["metrics"][bucket][key]["rmse"]
        ratios.append(va / vb)
        if va < vb:
            wins += 1
    return wins, len(common), statistics.median(ratios) if ratios else float("nan"), ratios


def build_td_summary(data):
    summary = {"arms": {}, "paired": {}}
    for a in ARMS:
        recs = data[a]
        if not recs:
            continue
        t_mean, t_sd = _mean_sd([r["train_sec"] for r in recs])
        i_mean, i_sd = _mean_sd([r["infer_ms_875pt"] for r in recs])
        rollbacks = sum(int(r["train_info"]["lbfgs_rolled_back"]) for r in recs)
        arm_entry = {
            "n_seeds": len(recs),
            "train_sec_mean": t_mean,
            "train_sec_std": t_sd,
            "train_sec_median": float(statistics.median(r["train_sec"] for r in recs)),
            "infer_ms_mean": i_mean,
            "infer_ms_std": i_sd,
            "lbfgs_rollbacks": rollbacks,
            "buckets": {},
        }
        for b, _ in BUCKETS:
            arm_entry["buckets"][b] = {}
            for k in KEYS:
                rmses = [r["metrics"][b][k]["rmse"] for r in recs]
                rels = [r["metrics"][b][k]["rel_rmse"] for r in recs]
                maxs = [r["metrics"][b][k]["max_abs"] for r in recs]
                rm, rsd = _mean_sd(rmses)
                lm, lsd = _mean_sd(rels)
                mm, msd = _mean_sd(maxs)
                arm_entry["buckets"][b][k] = {
                    "rmse_mean": rm,
                    "rmse_std": rsd,
                    "rmse_median": float(statistics.median(rmses)),
                    "rel_rmse_mean": lm,
                    "rel_rmse_std": lsd,
                    "rel_rmse_median": float(statistics.median(rels)),
                    "max_abs_mean": mm,
                    "max_abs_std": msd,
                    "max_abs_worst": max(maxs),
                }
        summary["arms"][a] = arm_entry

    for cmp_name, arm_a, arm_b in (
        ("M1_td_vs_B0_td", "M1_td", "B0_td"),
        ("A2b_td_vs_B0_td", "A2b_td", "B0_td"),
        ("M1_td_vs_A1_td", "M1_td", "A1_td"),
        ("A2b_td_vs_M1_td", "A2b_td", "M1_td"),
    ):
        if data[arm_a] and data[arm_b]:
            summary["paired"][cmp_name] = {}
            for b, _ in BUCKETS:
                summary["paired"][cmp_name][b] = {}
                for k in KEYS:
                    w, n, med_r, rs = paired_stats(data[arm_a], data[arm_b], b, k)
                    summary["paired"][cmp_name][b][k] = {
                        "wins": w,
                        "n": n,
                        "median_ratio": med_r,
                        "per_seed_ratios": rs,
                    }
    return summary


def fmt_ms(mean, sd, digits=4):
    return f"`{mean:.{digits}f} ± {sd:.{digits}f}`"


def render_td_markdown(data, summary):
    lines = ["# Time-Dependent Heston Benchmark Tables (`T15` vs. 2D ADI Oracle)\n"]
    lines.append(
        "Parameter curves: $\\kappa(\\tau)=1.5+0.8\\sin(2\\tau)$, $\\theta(\\tau)=0.04+0.02\\tau$, "
        "$\\xi(\\tau)=0.5-0.1\\tau$, $\\rho(\\tau)=-0.7+0.2\\tau$; scored against cached fine "
        "`800x400`, `2700`-step ADI oracle (`runs/ref_grids/TD_ADI.npz`).\n"
    )

    lines.append("## Table TD-1: All Arms Across 6 Buckets (5 Seeds per Arm, Mean ± Std Rel. RMSE)\n")
    lines.append("| Bucket | Arm | Price Rel RMSE | Delta Rel RMSE | Norm. Gamma Rel RMSE (Max Abs) | Vanna Rel RMSE (Max Abs) | Wins vs `B0_td` (Gam / Vanna, Med. Ratio) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for b, b_label in BUCKETS:
        for a in ARMS:
            be = summary["arms"][a]["buckets"][b]
            c_s = fmt_ms(be["c"]["rel_rmse_mean"], be["c"]["rel_rmse_std"])
            cx_s = fmt_ms(be["cx"]["rel_rmse_mean"], be["cx"]["rel_rmse_std"])
            g_s = f"{fmt_ms(be['gam']['rel_rmse_mean'], be['gam']['rel_rmse_std'])} (`{be['gam']['max_abs_mean']:.2f}`)"
            v_s = f"{fmt_ms(be['cxv']['rel_rmse_mean'], be['cxv']['rel_rmse_std'])} (`{be['cxv']['max_abs_mean']:.2f}`)"
            if a in ("M1_td", "A2b_td") and f"{a}_vs_B0_td" in summary["paired"]:
                pg = summary["paired"][f"{a}_vs_B0_td"][b]["gam"]
                pv = summary["paired"][f"{a}_vs_B0_td"][b]["cxv"]
                pair_s = f"`{pg['wins']}/{pg['n']}` (`{pg['median_ratio']:.3f}×`) / `{pv['wins']}/{pv['n']}` (`{pv['median_ratio']:.3f}×`)"
            else:
                pair_s = "—"
            lines.append(f"| {b_label} | `{a}` | {c_s} | {cx_s} | {g_s} | {v_s} | {pair_s} |")

    lines.append("\n## Table TD-2: Paired Attention Gain (`A2b_td` vs. `M1_td`) Across Buckets\n")
    lines.append("| Bucket | Gamma Wins (`A2b_td < M1_td`) | Gamma Med. Ratio | Vanna Wins (`A2b_td < M1_td`) | Vanna Med. Ratio | Price Med. Ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    if "A2b_td_vs_M1_td" in summary["paired"]:
        for b, b_label in BUCKETS:
            pg = summary["paired"]["A2b_td_vs_M1_td"][b]["gam"]
            pv = summary["paired"]["A2b_td_vs_M1_td"][b]["cxv"]
            pc = summary["paired"]["A2b_td_vs_M1_td"][b]["c"]
            lines.append(
                f"| {b_label} | `{pg['wins']}/{pg['n']}` | `{pg['median_ratio']:.3f}×` | "
                f"`{pv['wins']}/{pv['n']}` | `{pv['median_ratio']:.3f}×` | `{pc['median_ratio']:.3f}×` |"
            )

    lines.append("\n## Table TD-3: Per-Seed Short-Maturity (`short_mat`) RMSE Breakdown\n")
    lines.append("| Seed | `B0_td` Gam | `M1_td` Gam | `A2b_td` Gam | `B0_td` Vanna | `M1_td` Vanna | `A2b_td` Vanna |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    by_arm = {a: {r["seed"]: r for r in data[a]} for a in ARMS}
    for s in SEEDS:
        if all(s in by_arm[a] for a in ("B0_td", "M1_td", "A2b_td")):
            bg = by_arm["B0_td"][s]["metrics"]["short_mat"]["gam"]["rmse"]
            mg = by_arm["M1_td"][s]["metrics"]["short_mat"]["gam"]["rmse"]
            ag = by_arm["A2b_td"][s]["metrics"]["short_mat"]["gam"]["rmse"]
            bv = by_arm["B0_td"][s]["metrics"]["short_mat"]["cxv"]["rmse"]
            mv = by_arm["M1_td"][s]["metrics"]["short_mat"]["cxv"]["rmse"]
            av = by_arm["A2b_td"][s]["metrics"]["short_mat"]["cxv"]["rmse"]
            lines.append(f"| {s} | `{bg:.4f}` | `{mg:.4f}` | `{ag:.4f}` | `{bv:.4f}` | `{mv:.4f}` | `{av:.4f}` |")

    lines.append("\n## Table TD-4: Compute Cost & Optimization Telemetry (TD Matrix)\n")
    lines.append("| Arm | Train Time (s) | 875-pt Infer + Greeks (ms) | Mean Adam Loss | Mean Post-L-BFGS Loss | L-BFGS Rollbacks |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for a in ARMS:
        recs = data[a]
        ae = summary["arms"][a]
        adam_m = statistics.mean(r["train_info"]["adam_last_loss"] for r in recs)
        post_m = statistics.mean(r["train_info"]["lbfgs_post_loss"] for r in recs)
        lines.append(
            f"| `{a}` | `{ae['train_sec_mean']:.1f} ± {ae['train_sec_std']:.1f}` | "
            f"`{ae['infer_ms_mean']:.2f} ± {ae['infer_ms_std']:.2f}` | "
            f"`{adam_m:.3e}` | `{post_m:.3e}` | `{ae['lbfgs_rollbacks']}/{ae['n_seeds']}` |"
        )

    return "\n".join(lines) + "\n"


def main():
    data = load_td_runs()
    summary = build_td_summary(data)
    with open(os.path.join(ROOT, "runs", "td_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = render_td_markdown(data, summary)
    with open(os.path.join(ROOT, "paper", "td_tables.md"), "w") as f:
        f.write(md)
    print("Wrote runs/td_summary.json and paper/td_tables.md")


if __name__ == "__main__":
    main()
