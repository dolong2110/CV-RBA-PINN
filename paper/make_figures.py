"""paper/make_figures.py - Generate the three publication figures from Tab 7.6."""
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
from matplotlib.colors import LogNorm
import numpy as np
import torch

import cvpinn
import heston_grid_vec as hgv
import heston_ref as hr


def median_seed(arm, pname="P2"):
    recs = []
    for s in range(5):
        with open(os.path.join(ROOT, "runs", f"{arm}_{pname}_seed{s}.json")) as f:
            recs.append(json.load(f))
    recs.sort(key=lambda r: r["metrics"]["short_mat"]["gam"]["rmse"])
    return recs[len(recs) // 2]["seed"]


def load_price_fn(arm, pname, seed):
    ckpt = torch.load(
        os.path.join(ROOT, "runs", "checkpoints", f"{arm}_{pname}_seed{seed}.pt"),
        map_location="cpu",
        weights_only=True,
    )
    n_in = 3 if arm == "B0" else 4
    net = cvpinn.Mlp(n_in)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    p = cvpinn.Heston(*hr.P_FELLER_BAD if pname == "P2" else hr.P_FELLER_OK)
    return cvpinn.make_price_from_net(arm, net, p, v_max=ckpt["v_max"])


def eval_arm_grid(arm, pname="P2"):
    s = median_seed(arm, pname)
    data = np.load(os.path.join(ROOT, "runs", "ref_grids", f"{pname}.npz"))
    x_grid = data["x_grid"]
    ref = {k: data[k] for k in ("c", "cx", "gam", "cxv")}
    pfn = load_price_fn(arm, pname, s)
    pred = cvpinn.evaluate_on_grid(pfn, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)
    return s, x_grid, ref, pred


def plot_error_surface_pair(ref, pred_b0, pred_m1, key, title_prefix, out_name, tau_indices=(0, 2)):
    taus = hgv.EVAL_TAUS
    vs = hgv.EVAL_VS
    zs = np.array(hgv.EVAL_ZS)
    fig, axes = plt.subplots(len(tau_indices), 2, figsize=(11.5, 4.2 * len(tau_indices)), sharex=True, sharey=True)
    if len(tau_indices) == 1:
        axes = axes[None, :]

    all_errs = []
    for it in tau_indices:
        all_errs.append(np.abs(pred_b0[key][it] - ref[key][it]))
        all_errs.append(np.abs(pred_m1[key][it] - ref[key][it]))
    vmax = max(float(np.max(e)) for e in all_errs)
    vmin = max(vmax * 1e-4, 1e-5)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    im = None
    for row, it in enumerate(tau_indices):
        for col, (arm_lbl, pred) in enumerate((("Tuned Baseline B0", pred_b0), ("CV-PINN M1", pred_m1))):
            ax = axes[row, col]
            err = np.clip(np.abs(pred[key][it] - ref[key][it]), vmin, None)
            im = ax.pcolormesh(zs, np.arange(len(vs)), err, norm=norm, cmap="magma", shading="auto")
            ax.set_yticks(np.arange(len(vs)))
            ax.set_yticklabels([f"{v:g}" for v in vs])
            rmse_slice = float(np.sqrt(np.mean((pred[key][it] - ref[key][it]) ** 2)))
            ax.set_title(f"{arm_lbl} ($\\tau = {taus[it]}$, RMSE = {rmse_slice:.3f})", fontsize=10.5)
            if col == 0:
                ax.set_ylabel("Spot variance $v$", fontsize=10)
            if row == len(tau_indices) - 1:
                ax.set_xlabel("Scaled moneyness $z = x / \\sqrt{w_{\\mathrm{exp}}}$", fontsize=10)

    fig.subplots_adjust(right=0.88, hspace=0.28, wspace=0.14)
    cbar_ax = fig.add_axes([0.90, 0.14, 0.022, 0.72])
    fig.colorbar(im, cax=cbar_ax, label=f"Absolute {title_prefix} Error (log scale)")
    fig.suptitle(f"P2 (Feller-Violated): {title_prefix} Error Surface — B0 vs. M1 (Median Seeds)", fontsize=12, y=0.98)
    out_path = os.path.join(ROOT, "paper", out_name)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_boundary_layer_collapse(seed_m1):
    p_tup = hr.P_FELLER_BAD
    p = cvpinn.Heston(*p_tup)
    ckpt = torch.load(
        os.path.join(ROOT, "runs", "checkpoints", f"M1_P2_seed{seed_m1}.pt"),
        map_location="cpu",
        weights_only=True,
    )
    net = cvpinn.Mlp(4)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    dom = cvpinn.Domain(v_min=0.0, v_max=ckpt["v_max"])
    u_fn = cvpinn.make_correction(net, p, dom, "expected", use_z=True)

    zs = np.linspace(-3.0, 3.0, 61)
    v = 0.04
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for col, tau in zip(colors, (0.08, 0.02, 0.005)):
        w = hgv.w_exp_stable(v, tau, p.kappa, p.theta)
        xs = zs * np.sqrt(w)
        exact_rate = np.array([(hr.heston(float(x), v, tau, p_tup)["c"] - hr.bs(float(x), w)) / tau for x in xs])
        with torch.no_grad():
            xt = torch.tensor(xs, dtype=torch.float64)
            vt = torch.full_like(xt, v)
            tt = torch.full_like(xt, tau)
            pred_rate = (u_fn(xt, vt, tt) / tt).cpu().numpy()
        ax.plot(zs, exact_rate, color=col, lw=2.2, label=f"Exact $\\tau={tau}$")
        ax.plot(zs, pred_rate, color=col, lw=1.8, ls="--", label=f"M1 $\\tau={tau}$")

    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.axvline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("Scaled moneyness $z = x / \\sqrt{w_{\\mathrm{exp}}(v, \\tau)}$", fontsize=10.5)
    ax.set_ylabel("Correction rate $u_{\\mathrm{exp}} / \\tau$", fontsize=10.5)
    ax.set_title("Near-Expiry Boundary-Layer Collapse in Scaled Coordinate $z$ ($P2, v=0.04$)", fontsize=11.5)
    ax.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=True)
    ax.grid(alpha=0.25)
    out_path = os.path.join(ROOT, "paper", "fig3_boundary_layer_collapse.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    s_b0, _, ref, pred_b0 = eval_arm_grid("B0", "P2")
    s_m1, _, _, pred_m1 = eval_arm_grid("M1", "P2")
    print(f"Using median seeds on P2: B0 seed={s_b0}, M1 seed={s_m1}")
    plot_error_surface_pair(
        ref, pred_b0, pred_m1, "gam", "Normalized Gamma",
        "fig1_gamma_error_surface_p2.png", tau_indices=(0, 2),
    )
    plot_error_surface_pair(
        ref, pred_b0, pred_m1, "cxv", "Normalized Vanna",
        "fig2_vanna_error_surface_p2.png", tau_indices=(0, 4),
    )
    plot_boundary_layer_collapse(s_m1)


if __name__ == "__main__":
    main()
