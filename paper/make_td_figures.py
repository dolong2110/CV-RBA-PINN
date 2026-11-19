"""paper/make_td_figures.py - Generate publication figures for the Time-Dependent Heston TD benchmark."""
import json
import os
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
import cvpinn_td as tdmod
import heston_adi_2d as adi
import heston_grid_vec as hgv
import heston_ref as hr


torch.set_default_dtype(torch.float64)


def median_seed(arm):
    recs = []
    for s in range(5):
        fp = os.path.join(ROOT, "runs", "td", f"{arm}_TD_seed{s}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                recs.append(json.load(f))
    recs.sort(key=lambda r: r["metrics"]["short_mat"]["gam"]["rmse"])
    return recs[len(recs) // 2]["seed"]


def load_td_price_fn(arm, seed, td):
    ckpt = torch.load(
        os.path.join(ROOT, "runs", "td", "checkpoints", f"{arm}_TD_seed{seed}.pt"),
        map_location="cpu",
        weights_only=True,
    )
    n_in = 3 if arm == "B0_td" else 4
    v_max = float(ckpt.get("v_max", 0.5))
    net = cvpinn.Mlp(n_in)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    if arm == "B0_td":
        dom = cvpinn.Domain(v_max=v_max)
        return lambda x, v, tau: net(cvpinn.base_features(x, v, tau, dom))
    kind = "spot" if arm == "A1_td" else "expected"
    dom = cvpinn.Domain(v_min=1e-3 if kind == "spot" else 0.0, v_max=v_max)
    u = tdmod.make_correction_td(net, td, dom, kind=kind, use_z=True)
    return tdmod.make_price_td(u, td, kind=kind)


def plot_td_error_surfaces(ref, preds, key, title_prefix, out_name, tau_indices=(0, 2)):
    taus = hgv.EVAL_TAUS
    vs = hgv.EVAL_VS
    zs = np.array(hgv.EVAL_ZS)
    arms_lbl = (("Tuned Baseline B0_td", "B0_td"), ("Spectral CV M1_td", "M1_td"), ("Spectral CV+RBA A2b_td", "A2b_td"))
    fig, axes = plt.subplots(len(tau_indices), len(arms_lbl), figsize=(14.5, 4.1 * len(tau_indices)), sharex=True, sharey=True)

    all_errs = [np.abs(preds[a][key][it] - ref[key][it]) for it in tau_indices for _, a in arms_lbl]
    vmax = max(float(np.max(e)) for e in all_errs)
    vmin = max(vmax * 1e-4, 1e-5)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    im = None
    for row, it in enumerate(tau_indices):
        for col, (lbl, arm) in enumerate(arms_lbl):
            ax = axes[row, col]
            diff = preds[arm][key][it] - ref[key][it]
            err = np.clip(np.abs(diff), vmin, None)
            im = ax.pcolormesh(zs, np.arange(len(vs)), err, norm=norm, cmap="magma", shading="auto")
            ax.set_yticks(np.arange(len(vs)))
            ax.set_yticklabels([f"{v:g}" for v in vs])
            rmse_slice = float(np.sqrt(np.mean(diff ** 2)))
            ax.set_title(f"{lbl} ($\\tau = {taus[it]}$, RMSE = {rmse_slice:.3f})", fontsize=10)
            if col == 0:
                ax.set_ylabel("Spot variance $v$", fontsize=10)
            if row == len(tau_indices) - 1:
                ax.set_xlabel("Scaled moneyness $z = x / \\sqrt{w_{\\mathrm{td}}}$", fontsize=10)

    fig.subplots_adjust(right=0.90, hspace=0.28, wspace=0.12)
    cbar_ax = fig.add_axes([0.92, 0.14, 0.018, 0.72])
    fig.colorbar(im, cax=cbar_ax, label=f"Absolute {title_prefix} Error (log scale)")
    fig.suptitle(f"Time-Dependent Heston (Check T15): {title_prefix} Error vs. ADI Oracle (Median Seeds)", fontsize=12, y=0.98)
    out_path = os.path.join(ROOT, "paper", out_name)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_td_boundary_collapse(td, seed_a2b):
    ckpt = torch.load(
        os.path.join(ROOT, "runs", "td", "checkpoints", f"A2b_td_TD_seed{seed_a2b}.pt"),
        map_location="cpu",
        weights_only=True,
    )
    net = cvpinn.Mlp(4)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    dom = cvpinn.Domain(v_min=0.0, v_max=float(ckpt.get("v_max", 0.5)))
    u_fn = tdmod.make_correction_td(net, td, dom, "expected", use_z=True)

    # Solve ADI at fine resolution on exact slice v=0.04 for taus=(0.005, 0.02, 0.08)
    slice_taus = (0.005, 0.02, 0.08)
    x_grid, v_grid, snaps = adi.solve_adi_heston(
        td.params_at, slice_taus, nx=600, nv=300, n_steps=1600, rannacher_steps=6,
    )
    from scipy.interpolate import RectBivariateSpline
    zs = np.linspace(-3.0, 3.0, 61)
    v = 0.04
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for col, tau in zip(colors, slice_taus):
        w = float(tdmod.w_td_np(v, tau, td))
        xs = zs * np.sqrt(w)
        spl = RectBivariateSpline(v_grid, x_grid, snaps[float(tau)], kx=3, ky=3)
        c_adi = spl(np.full_like(xs, v), xs, dx=0, dy=0, grid=False)
        bs_vals = np.array([hr.bs(float(xv), w) for xv in xs])
        adi_rate = (c_adi - bs_vals) / tau
        with torch.no_grad():
            xt = torch.tensor(xs, dtype=torch.float64)
            vt = torch.full_like(xt, v)
            tt = torch.full_like(xt, tau)
            pred_rate = (u_fn(xt, vt, tt) / tt).cpu().numpy()
        ax.plot(zs, adi_rate, color=col, lw=2.2, label=f"ADI Oracle $\\tau={tau}$")
        ax.plot(zs, pred_rate, color=col, lw=1.8, ls="--", label=f"A2b_td $\\tau={tau}$")

    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.axvline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("Scaled moneyness $z = x / \\sqrt{w_{\\mathrm{td}}(v, \\tau)}$", fontsize=10.5)
    ax.set_ylabel("Spectral correction rate $u_{\\mathrm{td}} / \\tau$", fontsize=10.5)
    ax.set_title("Time-Dependent Near-Expiry Collapse in Spectral Coordinate $z$ ($v=0.04$)", fontsize=11.5)
    ax.legend(loc="upper left", ncol=2, fontsize=8.5, frameon=True)
    ax.grid(alpha=0.25)
    out_path = os.path.join(ROOT, "paper", "fig6_td_boundary_layer_collapse.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    td = tdmod.default_td_heston()
    data = np.load(os.path.join(ROOT, "runs", "ref_grids", "TD_ADI.npz"))
    x_grid = data["x_grid"]
    ref = {k: data[k] for k in ("c", "cx", "gam", "cxv")}
    preds = {}
    med_seeds = {}
    for arm in ("B0_td", "M1_td", "A2b_td"):
        s = median_seed(arm)
        med_seeds[arm] = s
        pfn = load_td_price_fn(arm, s, td)
        preds[arm] = cvpinn.evaluate_on_grid(pfn, x_grid, hgv.EVAL_TAUS, hgv.EVAL_VS)
    print(f"Median TD seeds: {med_seeds}")
    plot_td_error_surfaces(ref, preds, "gam", "Normalized Gamma", "fig4_td_gamma_error_surface.png", tau_indices=(0, 2))
    plot_td_error_surfaces(ref, preds, "cxv", "Normalized Vanna", "fig5_td_vanna_error_surface.png", tau_indices=(0, 4))
    plot_td_boundary_collapse(td, med_seeds["A2b_td"])


if __name__ == "__main__":
    main()
