"""paper/make_tier2_tables_and_figs.py - Aggregate Tier-2 (M2_zv_lw, ablations, Vega/Volga)
and 8D/10D amortized parametric results into tables & figures.
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".mplconfig"))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = (0, 1, 2, 3, 4)
BUCKETS = (
    ("overall", "Overall (875 pts)"),
    ("short_mat", "Short maturity ($\\tau \\le 0.1$)"),
    ("low_var", "Low variance ($v \\le 0.01$)"),
    ("mod_var", "Mod/high variance ($v \\ge 0.04$)"),
    ("short_low_var", "Short & low var"),
    ("short_mod_var", "Short & mod var"),
)


def _mean_sd(vals):
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def load_tier2_group(arm, pname, seeds=SEEDS):
    recs = []
    for s in seeds:
        fp = os.path.join(ROOT, "runs", "tier2", f"{arm}_{pname}_seed{s}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                recs.append(json.load(f))
    return recs


def summarize_recs(recs, keys=("c", "cx", "gam", "cv", "cxv", "cvv")):
    if not recs:
        return None
    out = {
        "n_seeds": len(recs),
        "train_sec_mean": statistics.mean(r["train_sec"] for r in recs),
        "train_sec_std": statistics.stdev(r["train_sec"] for r in recs) if len(recs) > 1 else 0.0,
        "lbfgs_rollbacks": sum(int(r["train_info"]["lbfgs_rolled_back"]) for r in recs),
        "buckets": {},
    }
    for b, _ in BUCKETS:
        out["buckets"][b] = {}
        for k in keys:
            if k not in recs[0]["metrics"][b]:
                continue
            rmses = [r["metrics"][b][k]["rmse"] for r in recs]
            rels = [r["metrics"][b][k]["rel_rmse"] for r in recs]
            maxs = [r["metrics"][b][k]["max_abs"] for r in recs]
            rm, rsd = _mean_sd(rmses)
            lm, lsd = _mean_sd(rels)
            mm, msd = _mean_sd(maxs)
            out["buckets"][b][k] = {
                "rmse_mean": rm, "rmse_std": rsd, "rmse_median": statistics.median(rmses),
                "rel_rmse_mean": lm, "rel_rmse_std": lsd, "rel_rmse_median": statistics.median(rels),
                "max_abs_mean": mm, "max_abs_std": msd,
            }
    return out


def paired_gain(recs_new, recs_old, bucket, key):
    by_new = {r["seed"]: r for r in recs_new}
    by_old = {r["seed"]: r for r in recs_old}
    common = sorted(set(by_new) & set(by_old))
    if not common:
        return None
    wins = 0
    ratios = []
    for s in common:
        vn = by_new[s]["metrics"][bucket][key]["rel_rmse"]
        vo = by_old[s]["metrics"][bucket][key]["rel_rmse"]
        ratios.append(vn / vo)
        if vn < vo:
            wins += 1
    return {"wins": wins, "n": len(common), "median_ratio": statistics.median(ratios)}


def load_const_recs(arm, pname, seeds=SEEDS):
    recs = []
    for s in seeds:
        fp = os.path.join(ROOT, "runs", f"{arm}_{pname}_seed{s}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                recs.append(json.load(f))
    return recs


def load_td_recs(arm, seeds=SEEDS):
    recs = []
    for s in seeds:
        fp = os.path.join(ROOT, "runs", "td", f"{arm}_TD_seed{s}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                recs.append(json.load(f))
    return recs


def build_report():
    vv_path = os.path.join(ROOT, "runs", "vega_volga_summary.json")
    vv = json.load(open(vv_path)) if os.path.exists(vv_path) else {}

    summary = {"const": {}, "td": {}, "ablations_p2_seed0": {}, "param8d": {}, "paired": {}}

    # 1. Constant-param M2_zv_lw vs M1 vs B0
    for pname in ("P1", "P2"):
        m1_recs = load_const_recs("M1", pname)
        b0_recs = load_const_recs("B0", pname)
        m2_recs = load_tier2_group("M2_zv_lw", pname)
        summary["const"][pname] = {
            "B0": summarize_recs(b0_recs, keys=("c", "cx", "gam", "cxv")),
            "M1": summarize_recs(m1_recs, keys=("c", "cx", "gam", "cxv")),
            "M2_zv_lw": summarize_recs(m2_recs),
        }
        if pname in vv:
            summary["const"][pname]["vega_volga_base"] = vv[pname]
        if m2_recs and m1_recs:
            summary["paired"][f"M2_vs_M1_{pname}"] = {
                b: {k: paired_gain(m2_recs, m1_recs, b, k) for k in ("c", "cx", "gam", "cxv")}
                for b, _ in BUCKETS
            }
        if m2_recs and b0_recs:
            summary["paired"][f"M2_vs_B0_{pname}"] = {
                b: {k: paired_gain(m2_recs, b0_recs, b, k) for k in ("c", "cx", "gam", "cxv")}
                for b, _ in BUCKETS
            }

    # 2. TD M2_zv_lw_td vs M1_td vs B0_td
    m1_td = load_td_recs("M1_td")
    b0_td = load_td_recs("B0_td")
    m2_td = load_tier2_group("M2_zv_lw_td", "TD")
    summary["td"] = {
        "B0_td": summarize_recs(b0_td, keys=("c", "cx", "gam", "cxv")),
        "M1_td": summarize_recs(m1_td, keys=("c", "cx", "gam", "cxv")),
        "M2_zv_lw_td": summarize_recs(m2_td, keys=("c", "cx", "gam", "cxv")),
    }
    if m2_td and m1_td:
        summary["paired"]["M2_td_vs_M1_td"] = {
            b: {k: paired_gain(m2_td, m1_td, b, k) for k in ("c", "cx", "gam", "cxv")}
            for b, _ in BUCKETS
        }

    # 3. Single-seed Tier-2 ablations on P2 (10k/500 full budget when available)
    for arm in ("A0_intr", "M1_noz", "M2_zband", "M2_zband_lw", "M2_zv_lw", "M2_zv_feat_lw"):
        fp = os.path.join(ROOT, "runs", "tier2", f"{arm}_P2_seed0.json")
        if os.path.exists(fp):
            with open(fp) as f:
                summary["ablations_p2_seed0"][arm] = json.load(f)

    # 4. 8D/10D Amortized parametric runs
    for arm in ("B0_param", "M_param", "M_param_plus"):
        fp = os.path.join(ROOT, "runs", "param", f"{arm}_seed0.json")
        if os.path.exists(fp):
            with open(fp) as f:
                summary["param8d"][arm] = json.load(f)

    return summary


def fmt_ms(mean, sd):
    return f"`{mean:.4f} ± {sd:.4f}`"


def render_markdown(summary):
    lines = ["# Tier-2 Upgrades, Vega/Volga Evaluation & 8D Amortized Surrogate Tables\n"]

    # Table T2-1: Constant-Param & TD M2_zv_lw vs M1 & B0
    lines.append("## Table T2-1: Stratified-$z$ & Low-$v$ Weighted Solver (`M2_zv_lw`) vs. `M1` & `B0` (5 Seeds)\n")
    lines.append("| Regime | Bucket | Arm | Price Rel RMSE | Norm. Gamma Rel RMSE | Vanna Rel RMSE | Vega Rel RMSE | Volga Rel RMSE |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for pname in ("P1", "P2"):
        vv = summary["const"][pname].get("vega_volga_base", {})
        for b, b_lbl in BUCKETS:
            for arm in ("B0", "M1", "M2_zv_lw"):
                entry = summary["const"][pname].get(arm)
                if not entry:
                    continue
                be = entry["buckets"][b]
                c_s = fmt_ms(be["c"]["rel_rmse_mean"], be["c"]["rel_rmse_std"])
                g_s = fmt_ms(be["gam"]["rel_rmse_mean"], be["gam"]["rel_rmse_std"])
                v_s = fmt_ms(be["cxv"]["rel_rmse_mean"], be["cxv"]["rel_rmse_std"])
                if "cv" in be:
                    veg_s = fmt_ms(be["cv"]["rel_rmse_mean"], be["cv"]["rel_rmse_std"])
                    vol_s = fmt_ms(be["cvv"]["rel_rmse_mean"], be["cvv"]["rel_rmse_std"])
                elif arm in vv and b in vv[arm]:
                    veg_s = fmt_ms(vv[arm][b]["cv"]["rel_rmse_mean"], vv[arm][b]["cv"]["rel_rmse_std"])
                    vol_s = fmt_ms(vv[arm][b]["cvv"]["rel_rmse_mean"], vv[arm][b]["cvv"]["rel_rmse_std"])
                else:
                    veg_s, vol_s = "—", "—"
                lines.append(f"| {pname} | {b_lbl} | `{arm}` | {c_s} | {g_s} | {v_s} | {veg_s} | {vol_s} |")

    # Table T2-2: TD Comparison
    lines.append("\n## Table T2-2: Time-Dependent `TD` Upgrade (`M2_zv_lw_td` vs. `M1_td` vs. `B0_td`, 5 Seeds)\n")
    lines.append("| Bucket | Arm | Price Rel RMSE | Delta Rel RMSE | Norm. Gamma Rel RMSE | Vanna Rel RMSE | Wins vs `M1_td` (Gam / Vanna) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for b, b_lbl in BUCKETS:
        for arm in ("B0_td", "M1_td", "M2_zv_lw_td"):
            entry = summary["td"].get(arm)
            if not entry:
                continue
            be = entry["buckets"][b]
            pair_s = "—"
            if arm == "M2_zv_lw_td" and "M2_td_vs_M1_td" in summary["paired"]:
                pg = summary["paired"]["M2_td_vs_M1_td"][b]["gam"]
                pv = summary["paired"]["M2_td_vs_M1_td"][b]["cxv"]
                pair_s = f"`{pg['wins']}/{pg['n']}` (`{pg['median_ratio']:.3f}×`) / `{pv['wins']}/{pv['n']}` (`{pv['median_ratio']:.3f}×`)"
            lines.append(
                f"| {b_lbl} | `{arm}` | {fmt_ms(be['c']['rel_rmse_mean'], be['c']['rel_rmse_std'])} | "
                f"{fmt_ms(be['cx']['rel_rmse_mean'], be['cx']['rel_rmse_std'])} | "
                f"{fmt_ms(be['gam']['rel_rmse_mean'], be['gam']['rel_rmse_std'])} | "
                f"{fmt_ms(be['cxv']['rel_rmse_mean'], be['cxv']['rel_rmse_std'])} | {pair_s} |"
            )

    # Table T2-3: Full-Budget Controlled Ablations on P2 Seed 0
    if summary["ablations_p2_seed0"]:
        lines.append("\n## Table T2-3: Tier-2 Full-Budget Controlled Ablations (`P2`, Seed 0, `10k Adam + 500 L-BFGS`)\n")
        lines.append("| Variant | Description | Ov Price | Ov Gamma | Ov Vanna | Ov Vega | Ov Volga | Short-Low Gamma | Short-Low Vanna |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        desc = {
            "A0_intr": "Hard payoff intrinsic + `tau*N` (no BS carrier)",
            "M1_noz": "Exp-Var CV without `tanh(0.5*z)` feature",
            "M2_zband": "`M1` + 50% stratified `z`-band sampling (`16k` L-BFGS)",
            "M2_zband_lw": "`M2_zband` + `1/sqrt(v+0.02)` residual weighting",
            "M2_zv_lw": "`M2` + smooth `(v,tau)` scales + `tanh(0.5*z)` + `lowv_weight` (`4x80`)",
            "M2_zv_feat_lw": "`M2_zv_lw` + `tanh(1.5*z)` (`4x96`)",
        }
        for arm, rec in summary["ablations_p2_seed0"].items():
            ov = rec["metrics"]["overall"]
            sl = rec["metrics"]["short_low_var"]
            lines.append(
                f"| `{arm}` | {desc.get(arm, arm)} | `{ov['c']['rel_rmse']:.4f}` | "
                f"`{ov['gam']['rel_rmse']:.4f}` | `{ov['cxv']['rel_rmse']:.4f}` | "
                f"`{ov['cv']['rel_rmse']:.4f}` | `{ov['cvv']['rel_rmse']:.4f}` | "
                f"`{sl['gam']['rel_rmse']:.4f}` | `{sl['cxv']['rel_rmse']:.4f}` |"
            )

    # Table T2-4: 8D/10D Amortized Parametric Surrogate Zero-Shot
    if summary["param8d"]:
        lines.append("\n## Table T2-4: Amortized 8D/10D Parametric Surrogate — Zero-Shot Evaluation Across Unseen Regimes\n")
        lines.append("| Test Regime | Model | Overall Price | Overall Gamma | Overall Vanna | Overall Vega | Overall Volga | `short_low_var` Gamma | 875-pt Infer (ms) |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for pname in ("P1", "P2", "P3_steep", "P4_calm"):
            for arm in ("B0_param", "M_param", "M_param_plus"):
                if arm not in summary["param8d"]:
                    continue
                ev = summary["param8d"][arm]["zero_shot_evals"][pname]
                ov = ev["metrics"]["overall"]
                sl = ev["metrics"]["short_low_var"]
                lines.append(
                    f"| `{pname}` | `{arm}` | `{ov['c']['rel_rmse']:.4f}` | `{ov['gam']['rel_rmse']:.4f}` | "
                    f"`{ov['cxv']['rel_rmse']:.4f}` | `{ov['cv']['rel_rmse']:.4f}` | `{ov['cvv']['rel_rmse']:.4f}` | "
                    f"`{sl['gam']['rel_rmse']:.4f}` | `{ev['infer_ms_875pt']:.2f}` |"
                )

    return "\n".join(lines) + "\n"


def plot_tier2_comparison(summary):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    vv_p2 = summary["const"]["P2"].get("vega_volga_base", {})
    panels = (
        ("P2 Overall (875 pts)", "const_p2", "overall", [("c", "Price"), ("cx", "Delta"), ("gam", "Gamma"), ("cv", "Vega"), ("cxv", "Vanna"), ("cvv", "Volga")], [("B0", "#d62728"), ("M1", "#1f77b4"), ("M2_zv_lw", "#2ca02c")]),
        ("P2 Short & Low-Var Corner", "const_p2", "short_low_var", [("c", "Price"), ("cx", "Delta"), ("gam", "Gamma"), ("cv", "Vega"), ("cxv", "Vanna"), ("cvv", "Volga")], [("B0", "#d62728"), ("M1", "#1f77b4"), ("M2_zv_lw", "#2ca02c")]),
        ("TD Short & Low-Var vs ADI", "td", "short_low_var", [("c", "Price"), ("cx", "Delta"), ("gam", "Gamma"), ("cxv", "Vanna")], [("B0_td", "#d62728"), ("M1_td", "#1f77b4"), ("M2_zv_lw_td", "#2ca02c")]),
    )

    for ax, (title, kind, bucket, mlist, arms) in zip(axes, panels):
        x = np.arange(len(mlist))
        w = 0.25
        for i, (arm, col) in enumerate(arms):
            src = summary["const"]["P2"] if kind == "const_p2" else summary["td"]
            vals = []
            for mk, _ in mlist:
                entry = src[arm]["buckets"][bucket]
                if mk in entry:
                    vals.append(entry[mk]["rel_rmse_mean"])
                elif kind == "const_p2" and arm in vv_p2 and bucket in vv_p2[arm]:
                    vals.append(vv_p2[arm][bucket][mk]["rel_rmse_mean"])
                else:
                    vals.append(np.nan)
            ax.bar(x + (i - 1) * w, vals, width=w, color=col, label=arm, alpha=0.9)
        ax.set_yscale("log")
        ax.set_ylim(2e-4, 5.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in mlist])
        ax.set_ylabel("Mean Rel. RMSE across 5 seeds (log scale)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8.5, frameon=True, loc="upper left", ncol=3)

    fig.suptitle("Tier-2 Upgrade (M2_zv_lw / M2_zv_lw_td) vs. M1 and Tuned Baseline B0 Across 5 Seeds", fontsize=12)
    out = os.path.join(ROOT, "paper", "fig7_tier2_accuracy_comparison.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_param8d(summary):
    if not summary["param8d"]:
        return
    regimes = ("P1", "P2", "P3_steep", "P4_calm")
    arms = [("B0_param", "#d62728"), ("M_param", "#1f77b4"), ("M_param_plus", "#2ca02c")]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    for ax, (mk, title) in zip(axes, (("gam", "Zero-Shot Gamma Rel RMSE"), ("cxv", "Zero-Shot Vanna Rel RMSE"), ("cv", "Zero-Shot Vega Rel RMSE"))):
        x = np.arange(len(regimes))
        w = 0.25
        for i, (arm, col) in enumerate(arms):
            if arm not in summary["param8d"]:
                continue
            vals = [summary["param8d"][arm]["zero_shot_evals"][r]["metrics"]["overall"][mk]["rel_rmse"] for r in regimes]
            ax.bar(x + (i - 1) * w, vals, width=w, color=col, label=arm, alpha=0.9)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 4.5)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, rotation=15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8.0, frameon=True, loc="upper center", ncol=3)
    fig.suptitle("Amortized Parametric Surrogate: Zero-Shot Accuracy Without Retraining Across Regimes", fontsize=12)
    out = os.path.join(ROOT, "paper", "fig8_param8d_zeroshot.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    summary = build_report()
    with open(os.path.join(ROOT, "runs", "tier2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(ROOT, "paper", "tier2_tables.md"), "w") as f:
        f.write(render_markdown(summary))
    print("Wrote runs/tier2_summary.json and paper/tier2_tables.md")
    if summary["const"]["P2"].get("M2_zv_lw"):
        plot_tier2_comparison(summary)
    if summary["param8d"]:
        plot_param8d(summary)


if __name__ == "__main__":
    main()
