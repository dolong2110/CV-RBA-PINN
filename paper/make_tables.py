"""paper/make_tables.py - Aggregate 50-run benchmark metrics into summary JSON & markdown tables."""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ARMS = ("B0", "M1", "A1", "A2a", "A2b")
PARAMS = ("P1", "P2")
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


def load_runs():
    data = {}
    for p in PARAMS:
        data[p] = {}
        for a in ARMS:
            recs = []
            for s in SEEDS:
                path = os.path.join(ROOT, "runs", f"{a}_{p}_seed{s}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        recs.append(json.load(f))
            data[p][a] = recs
    return data


def _mean_sd(vals):
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def paired_stats(recs_a, recs_b, bucket, key):
    """Compare arm A against arm B seed-by-seed (wins: A < B, ratio: A / B)."""
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


def build_summary(data):
    summary = {}
    for p in PARAMS:
        summary[p] = {}
        for a in ARMS:
            recs = data[p][a]
            if not recs:
                continue
            t_mean, t_sd = _mean_sd([r["train_sec"] for r in recs])
            i_mean, i_sd = _mean_sd([r["infer_ms_875pt"] for r in recs])
            rollbacks = sum(int(r["train_info"]["lbfgs_rolled_back"]) for r in recs)
            arm_entry = {
                "n_seeds": len(recs),
                "train_sec_mean": t_mean,
                "train_sec_std": t_sd,
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
                        "rel_rmse_mean": lm,
                        "rel_rmse_std": lsd,
                        "max_abs_mean": mm,
                        "max_abs_std": msd,
                        "max_abs_worst": max(maxs),
                    }
            summary[p][a] = arm_entry

        # Paired summaries
        summary[p]["paired"] = {}
        for cmp_name, arm_a, arm_b in (
            ("M1_vs_B0", "M1", "B0"),
            ("M1_vs_A1", "M1", "A1"),
            ("A2b_vs_A2a", "A2b", "A2a"),
            ("A2a_vs_M1", "A2a", "M1"),
        ):
            if data[p][arm_a] and data[p][arm_b]:
                summary[p]["paired"][cmp_name] = {}
                for b, _ in BUCKETS:
                    summary[p]["paired"][cmp_name][b] = {}
                    for k in KEYS:
                        w, n, med_r, rs = paired_stats(data[p][arm_a], data[p][arm_b], b, k)
                        summary[p]["paired"][cmp_name][b][k] = {
                            "wins": w,
                            "n": n,
                            "median_ratio": med_r,
                            "per_seed_ratios": rs,
                        }
    return summary


def fmt_ms(mean, sd, digits=4):
    return f"`{mean:.{digits}f} ± {sd:.{digits}f}`"


def render_markdown(data, summary):
    lines = ["# CV-RBA-PINN Benchmark Tables (50-Run Matrix)\n"]

    lines.append("## Table 1: Main Comparison (`B0` vs. `M1` across 6 Buckets, 5 Seeds)\n")
    lines.append("| Param | Bucket | Arm | Price Rel RMSE | Delta Rel RMSE | Norm. Gamma Rel RMSE (Max Abs) | Vanna Rel RMSE (Max Abs) | M1 Wins (Gamma / Vanna, Med. Ratio) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in PARAMS:
        for b, b_label in BUCKETS:
            for a in ("B0", "M1"):
                be = summary[p][a]["buckets"][b]
                c_s = fmt_ms(be["c"]["rel_rmse_mean"], be["c"]["rel_rmse_std"])
                cx_s = fmt_ms(be["cx"]["rel_rmse_mean"], be["cx"]["rel_rmse_std"])
                g_s = f"{fmt_ms(be['gam']['rel_rmse_mean'], be['gam']['rel_rmse_std'])} (`{be['gam']['max_abs_mean']:.2f}`)"
                v_s = f"{fmt_ms(be['cxv']['rel_rmse_mean'], be['cxv']['rel_rmse_std'])} (`{be['cxv']['max_abs_mean']:.2f}`)"
                if a == "M1":
                    pg = summary[p]["paired"]["M1_vs_B0"][b]["gam"]
                    pv = summary[p]["paired"]["M1_vs_B0"][b]["cxv"]
                    pair_s = f"`{pg['wins']}/{pg['n']}` (`{pg['median_ratio']:.3f}×`) / `{pv['wins']}/{pv['n']}` (`{pv['median_ratio']:.3f}×`)"
                else:
                    pair_s = "—"
                lines.append(f"| {p} | {b_label} | `{a}` | {c_s} | {cx_s} | {g_s} | {v_s} | {pair_s} |")

    lines.append("\n## Table 2: Controlled Ablations (`M1` vs. `A1` Spot-CV; `A2a` vs. `A2b` Energy-Locked RBA)\n")
    lines.append("| Param | Bucket | Arm | Price Rel RMSE | Delta Rel RMSE | Norm. Gamma Rel RMSE (Max Abs) | Vanna Rel RMSE (Max Abs) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for p in PARAMS:
        for b in ("overall", "short_mat", "low_var", "mod_var", "short_low_var", "short_mod_var"):
            for a in ("M1", "A1", "A2a", "A2b"):
                be = summary[p][a]["buckets"][b]
                lines.append(
                    f"| {p} | `{b}` | `{a}` | "
                    f"{fmt_ms(be['c']['rel_rmse_mean'], be['c']['rel_rmse_std'])} | "
                    f"{fmt_ms(be['cx']['rel_rmse_mean'], be['cx']['rel_rmse_std'])} | "
                    f"{fmt_ms(be['gam']['rel_rmse_mean'], be['gam']['rel_rmse_std'])} (`{be['gam']['max_abs_mean']:.2f}`) | "
                    f"{fmt_ms(be['cxv']['rel_rmse_mean'], be['cxv']['rel_rmse_std'])} (`{be['cxv']['max_abs_mean']:.2f}`) |"
                )

    lines.append("\n## Table 3: Per-Seed Short-Maturity (`short_mat`) RMSE for `B0` vs. `M1`\n")
    lines.append("| Param | Seed | `B0` Gamma RMSE | `M1` Gamma RMSE | Ratio (`M1/B0`) | `B0` Vanna RMSE | `M1` Vanna RMSE | Ratio (`M1/B0`) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in PARAMS:
        b0s = {r["seed"]: r for r in data[p]["B0"]}
        m1s = {r["seed"]: r for r in data[p]["M1"]}
        for s in SEEDS:
            bg = b0s[s]["metrics"]["short_mat"]["gam"]["rmse"]
            mg = m1s[s]["metrics"]["short_mat"]["gam"]["rmse"]
            bv = b0s[s]["metrics"]["short_mat"]["cxv"]["rmse"]
            mv = m1s[s]["metrics"]["short_mat"]["cxv"]["rmse"]
            lines.append(
                f"| {p} | {s} | `{bg:.4f}` | `{mg:.4f}` | `{mg/bg:.4f}` | `{bv:.4f}` | `{mv:.4f}` | `{mv/bv:.4f}` |"
            )

    lines.append("\n## Table 4: Compute Cost and Optimization Telemetry (5 Seeds per Arm)\n")
    lines.append("| Param | Arm | Train Time (s) | 875-pt Infer + Greeks (ms) | Mean Adam Loss | Mean Post-L-BFGS Loss | L-BFGS Rollbacks |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for p in PARAMS:
        for a in ARMS:
            recs = data[p][a]
            ae = summary[p][a]
            adam_m = statistics.mean(r["train_info"]["adam_last_loss"] for r in recs)
            post_m = statistics.mean(r["train_info"]["lbfgs_post_loss"] for r in recs)
            lines.append(
                f"| {p} | `{a}` | `{ae['train_sec_mean']:.1f} ± {ae['train_sec_std']:.1f}` | "
                f"`{ae['infer_ms_mean']:.2f} ± {ae['infer_ms_std']:.2f}` | "
                f"`{adam_m:.3e}` | `{post_m:.3e}` | `{ae['lbfgs_rollbacks']}/{ae['n_seeds']}` |"
            )

    return "\n".join(lines) + "\n"


def main():
    data = load_runs()
    summary = build_summary(data)
    with open(os.path.join(ROOT, "runs", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = render_markdown(data, summary)
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    with open(os.path.join(ROOT, "paper", "tables.md"), "w") as f:
        f.write(md)
    print("Wrote runs/summary.json and paper/tables.md")


if __name__ == "__main__":
    main()
