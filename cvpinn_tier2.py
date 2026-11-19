"""cvpinn_tier2.py - Tier-2 extensions: Stratified z-band collocation, multi-scale smooth
boundary-layer features, gPINN low-variance residual gradients, and missing ablations
(intrinsic-value baseline A0_intr, no-z ablation M1_noz, and full 6-Greek Vega/Volga scoring).
"""
import math
import time
import numpy as np
import torch
from torch import nn

import cvpinn
import heston_grid_vec as hgv


def sample_stratified_z(n_int, n_strip, p_or_td, dom, gen, z_frac=0.5, z_max=4.0,
                        v_power=2.0, t_power=2.0, is_td=False, device="cpu"):
    """Samples interior + v_min strip with a fraction z_frac in scaled band |z| <= z_max."""
    def _draw_vt(n, vp, tp):
        # Mix 50% quadratic (dom coverage) with 50% higher power (corner density) if vp > 2.0
        if vp > 2.0:
            n_hi = n // 2
            n_lo = n - n_hi
            u_v = torch.cat([torch.rand(n_lo, generator=gen) ** 2.0, torch.rand(n_hi, generator=gen) ** vp])
            u_t = torch.cat([torch.rand(n_lo, generator=gen) ** 2.0, torch.rand(n_hi, generator=gen) ** tp])
            perm = torch.randperm(n, generator=gen)
            u_v, u_t = u_v[perm], u_t[perm]
        else:
            u_v = torch.rand(n, generator=gen) ** vp
            u_t = torch.rand(n, generator=gen) ** tp
        v_out = dom.v_min + (dom.v_max - dom.v_min) * u_v
        t_out = dom.tau_min + (dom.tau_max - dom.tau_min) * u_t
        return v_out, t_out

    def _draw_x(v_t, tau_t, n):
        n_z = int(round(n * z_frac))
        n_u = n - n_z
        x_u = dom.x_max * (2.0 * torch.rand(n_u, generator=gen) - 1.0)
        if n_z > 0:
            if is_td:
                import cvpinn_td as tdmod
                w_sub = tdmod.total_variance_td(v_t[:n_z], tau_t[:n_z], p_or_td, "expected")
            else:
                w_sub = cvpinn.total_variance(v_t[:n_z], tau_t[:n_z], p_or_td, "expected")
            z = z_max * (2.0 * torch.rand(n_z, generator=gen) - 1.0)
            x_z = torch.clamp(z * torch.sqrt(torch.clamp(w_sub, min=1e-30)), -dom.x_max, dom.x_max)
            x_all = torch.cat([x_z, x_u])
        else:
            x_all = x_u
        return x_all

    v, tau = _draw_vt(n_int, v_power, t_power)
    x = _draw_x(v, tau, n_int)
    if n_strip > 0:
        # Concentrate 50% of strip points in |z(0, tau)| <= z_max when z_frac > 0
        _, ts = _draw_vt(n_strip, v_power, t_power)
        vs = torch.full((n_strip,), dom.v_min)
        xs = _draw_x(vs, ts, n_strip)
        x = torch.cat([x, xs])
        v = torch.cat([v, vs])
        tau = torch.cat([tau, ts])
    return x.to(device), v.to(device), tau.to(device)


def smooth_features(x, v, tau, p_or_td, dom, kind="expected", use_z=True,
                    extra_scales=False, z2=False, gauss_z=False, is_td=False):
    feats = cvpinn.base_features(x, v, tau, dom)
    if extra_scales:
        # C^inf smooth magnified boundary-layer coordinates for small v and small tau
        feats.append(torch.tanh(v / 0.02))
        feats.append(torch.tanh(tau / 0.05))
    if use_z:
        if is_td:
            import cvpinn_td as tdmod
            w_safe = torch.clamp(tdmod.total_variance_td(v, tau, p_or_td, kind), min=1e-30)
        else:
            w_safe = torch.clamp(cvpinn.total_variance(v, tau, p_or_td, kind), min=1e-30)
        z = x / torch.sqrt(w_safe)
        feats.append(torch.tanh(0.5 * z))
        if gauss_z:
            feats.append(torch.exp(-0.25 * z * z))
        elif z2:
            feats.append(torch.tanh(1.5 * z))
    return feats


class ModifiedMlp(nn.Module):
    """Wang, Sankaran & Perdikaris (2023) input-gated MLP mitigating multi-scale spectral bias."""

    def __init__(self, n_in, width=64, depth=4):
        super().__init__()
        self.enc_u = nn.Linear(n_in, width)
        self.enc_v = nn.Linear(n_in, width)
        self.in_layer = nn.Linear(n_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out_layer = nn.Linear(width, 1)

    def forward(self, feats):
        x_in = torch.stack(feats, dim=-1)
        u = torch.tanh(self.enc_u(x_in))
        v = torch.tanh(self.enc_v(x_in))
        h = torch.tanh(self.in_layer(x_in))
        for lin in self.hidden:
            z = torch.tanh(lin(h))
            h = (1.0 - z) * u + z * v
        return self.out_layer(h).squeeze(-1)


def make_correction_tier2(net, p_or_td, dom, kind="expected", use_z=True,
                          extra_scales=False, z2=False, gauss_z=False, is_td=False):
    def u(x, v, tau):
        feats = smooth_features(
            x, v, tau, p_or_td, dom, kind=kind, use_z=use_z,
            extra_scales=extra_scales, z2=z2, gauss_z=gauss_z, is_td=is_td,
        )
        return tau * net(feats)
    return u


def build_net_from_cfg(cfg):
    n_in = (
        3
        + (1 if cfg.get("use_z", True) else 0)
        + (2 if cfg.get("extra_scales", False) else 0)
        + (1 if (cfg.get("use_z", True) and (cfg.get("z2", False) or cfg.get("gauss_z", False))) else 0)
    )
    cls = ModifiedMlp if cfg.get("modified", False) else cvpinn.Mlp
    return cls(n_in, width=cfg["width"], depth=cfg["depth"])


def _make_price_from_net_t2(cfg, net, p, v_max=0.5):
    dom = cvpinn.Domain(v_max=v_max)
    if cfg.get("intrinsic"):
        payoff = lambda x: torch.relu(torch.exp(x) - 1.0)
        return lambda x, v, tau: payoff(x) + tau * net(cvpinn.base_features(x, v, tau, dom))
    u = make_correction_tier2(
        net, p, dom, kind="expected",
        use_z=cfg.get("use_z", True),
        extra_scales=cfg.get("extra_scales", False),
        z2=cfg.get("z2", cfg.get("extra_scales", False) and not cfg.get("no_z2", False)),
        gauss_z=cfg.get("gauss_z", False),
        is_td=False,
    )
    return cvpinn.make_price(u, p, "expected")


def train_tier2(arm_name, p, seed, adam_steps=10000, lbfgs_iter=500, v_max=0.5,
                device="cpu", return_info=False):
    """Unified trainer for Tier-2 variants and missing ablations on constant-param Heston."""
    configs = {
        "M1_noz": {"use_z": False, "extra_scales": False, "z_frac": 0.0, "vp": 2.0, "tp": 2.0, "width": 64, "depth": 4, "n_lbfgs": 8192},
        "A0_intr": {"intrinsic": True, "width": 64, "depth": 4, "n_lbfgs": 8192},
        "M2_zband": {"use_z": True, "extra_scales": False, "z_frac": 0.5, "vp": 2.0, "tp": 2.0, "width": 64, "depth": 4, "n_lbfgs": 16384},
        "M2_zband_lw": {"use_z": True, "extra_scales": False, "z_frac": 0.55, "vp": 2.5, "tp": 2.2, "width": 64, "depth": 4, "n_lbfgs": 16384, "lowv_weight": True},
        "M2_zv_lw": {"use_z": True, "extra_scales": True, "no_z2": True, "z2": False, "z_frac": 0.55, "vp": 2.5, "tp": 2.2, "width": 80, "depth": 4, "n_lbfgs": 16384, "lowv_weight": True},
        "M2_zv_feat_lw": {"use_z": True, "extra_scales": True, "z2": True, "z_frac": 0.55, "vp": 2.5, "tp": 2.2, "width": 96, "depth": 4, "n_lbfgs": 16384, "lowv_weight": True},
        "M2_zv": {"use_z": True, "extra_scales": False, "z_frac": 0.65, "vp": 4.0, "tp": 3.0, "width": 64, "depth": 4, "n_lbfgs": 16384},
        "M2_zv_feat": {"use_z": True, "extra_scales": True, "z2": True, "z_frac": 0.65, "vp": 4.0, "tp": 3.0, "width": 96, "depth": 4, "n_lbfgs": 16384},
        "M2_zv_gpinn": {"use_z": True, "extra_scales": True, "z2": True, "z_frac": 0.65, "vp": 4.0, "tp": 3.0, "width": 96, "depth": 4, "n_lbfgs": 16384, "gpinn": True},
        "M2_zband_mod": {"use_z": True, "extra_scales": False, "modified": True, "z_frac": 0.5, "vp": 2.0, "tp": 2.0, "width": 64, "depth": 4, "n_lbfgs": 8192},
        "M2_zv_gauss_mod": {"use_z": True, "extra_scales": True, "gauss_z": True, "modified": True, "z_frac": 0.6, "vp": 3.0, "tp": 2.5, "width": 80, "depth": 4, "n_lbfgs": 12288},
        "M2_zv_wmod": {"use_z": True, "extra_scales": True, "gauss_z": True, "modified": True, "z_frac": 0.6, "vp": 3.0, "tp": 2.5, "width": 80, "depth": 4, "n_lbfgs": 12288, "lowv_weight": True},
    }
    if arm_name not in configs:
        raise ValueError(f"Unknown Tier-2 arm {arm_name!r}")
    cfg = configs[arm_name]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dom = cvpinn.Domain(v_max=v_max)

    if cfg.get("intrinsic"):
        net = cvpinn.Mlp(3, width=cfg["width"], depth=cfg["depth"]).to(device)
        payoff = lambda x: torch.relu(torch.exp(x) - 1.0)
        c_fn = lambda x, v, tau: payoff(x) + tau * net(cvpinn.base_features(x, v, tau, dom))
        n_col = 2048
        n_strip = n_col // 16
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=1e-4)
        t0 = time.perf_counter()
        last_loss = float("nan")
        for _ in range(adam_steps):
            x, v, tau = cvpinn.sample_with_vmin(n_col, n_strip, dom, gen, device=device)
            xe, ve, te = cvpinn.sample_edges(n_col // 8, dom, gen, device=device)
            loss = (cvpinn.operator(c_fn, x, v, tau, p) ** 2).mean()
            loss = loss + 1.0 * ((c_fn(xe, ve, te) - payoff(xe)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            last_loss = float(loss.detach().item())
        adam_sec = time.perf_counter() - t0
        lx, lv, lt = cvpinn.sample_with_vmin(8192, 512, dom, gen, device=device)
        lxe, lve, lte = cvpinn.sample_edges(1024, dom, gen, device=device)
        pay_edge = payoff(lxe).detach()
        def lbfgs_loss():
            return (cvpinn.operator(c_fn, lx, lv, lt, p) ** 2).mean() + ((c_fn(lxe, lve, lte) - pay_edge) ** 2).mean()
        linfo = cvpinn._run_lbfgs_safe(net, lbfgs_loss, lbfgs_iter)
        info = {"arm": arm_name, "config": cfg, "adam_sec": round(adam_sec, 4), "adam_last_loss": last_loss, **{k: (round(v, 4) if k.endswith("_sec") else v) for k, v in linfo.items()}}
        return (c_fn, net, info) if return_info else c_fn

    net = build_net_from_cfg(cfg).to(device)
    u = make_correction_tier2(
        net, p, dom, kind="expected",
        use_z=cfg["use_z"], extra_scales=cfg["extra_scales"],
        z2=cfg.get("z2", False), gauss_z=cfg.get("gauss_z", False), is_td=False,
    )
    n_col = 2048
    n_strip = n_col // 16
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=1e-4)

    t0 = time.perf_counter()
    last_loss = float("nan")
    use_gpinn = cfg.get("gpinn", False)
    for _ in range(adam_steps):
        if cfg["z_frac"] > 0.0:
            x, v, tau = sample_stratified_z(
                n_col, n_strip, p, dom, gen, z_frac=cfg["z_frac"], z_max=4.0,
                v_power=cfg["vp"], t_power=cfg["tp"], is_td=False, device=device,
            )
        else:
            x, v, tau = cvpinn.sample_with_vmin(n_col, n_strip, dom, gen, device=device)
        xe, ve, te = cvpinn.sample_edges(n_col // 8, dom, gen, device=device)

        if use_gpinn:
            # Sobolev gradient of residual scaled smoothly by sqrt(w_exp) and 0.02 in v
            xg = x.detach().clone().requires_grad_(True)
            vg = v.detach().clone().requires_grad_(True)
            tg = tau.detach().clone().requires_grad_(True)
            f_val = u(xg, vg, tg)
            ones = torch.ones_like(f_val)
            fx, fv, ft = torch.autograd.grad(f_val, (xg, vg, tg), ones, create_graph=True)
            fxx, fxv = torch.autograd.grad(fx, (xg, vg), torch.ones_like(fx), create_graph=True)
            fvv = cvpinn.grad(fv, vg)
            l_u = (ft - 0.5 * vg * (fxx - fx) - p.rho * p.xi * vg * fxv
                   - 0.5 * (p.xi ** 2) * vg * fvv - p.kappa * (p.theta - vg) * fv)
            r = l_u - cvpinn.source(xg, vg, tg, p, "expected")
            # Subsample lowest-v quarter (v <= 0.04) for gPINN penalty to preserve speed
            mask_sub = (vg <= 0.05) & (tg <= 0.5)
            n_m = torch.clamp(mask_sub.sum().to(xg.dtype), min=1.0)
            rx, rv = torch.autograd.grad(r[mask_sub].sum(), (xg, vg), create_graph=True, allow_unused=True)
            rx = torch.zeros_like(xg) if rx is None else rx
            rv = torch.zeros_like(vg) if rv is None else rv
            w_sub = torch.sqrt(torch.clamp(cvpinn.total_variance(vg, tg, p, "expected"), min=1e-12))
            gp = 0.05 * (((w_sub * rx) ** 2 + (0.02 * rv) ** 2) * mask_sub.to(xg.dtype)).sum() / n_m
            loss = (r ** 2).mean() + gp + (u(xe, ve, te) ** 2).mean()
        else:
            r = cvpinn.operator(u, x, v, tau, p) - cvpinn.source(x, v, tau, p, "expected")
            if cfg.get("lowv_weight", False):
                # Smoothly boost low-variance residuals where diffusion v is O(10^-2)
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
    if cfg["z_frac"] > 0.0:
        lx, lv, lt = sample_stratified_z(
            n_lbfgs, n_lbfgs // 16, p, dom, gen, z_frac=cfg["z_frac"], z_max=4.0,
            v_power=cfg["vp"], t_power=cfg["tp"], is_td=False, device=device,
        )
    else:
        lx, lv, lt = cvpinn.sample_with_vmin(n_lbfgs, n_lbfgs // 16, dom, gen, device=device)
    lxe, lve, lte = cvpinn.sample_edges(1024, dom, gen, device=device)
    src_cached = cvpinn.source(lx, lv, lt, p, "expected").detach()
    lw = None
    if cfg.get("lowv_weight", False):
        lw_raw = 1.0 / torch.sqrt(lv + 0.02)
        lw = (lw_raw / torch.sqrt((lw_raw ** 2).mean() + 1e-30)).detach()

    def lbfgs_loss():
        r_l = cvpinn.operator(u, lx, lv, lt, p) - src_cached
        lp = ((lw * r_l) ** 2).mean() if lw is not None else (r_l ** 2).mean()
        return lp + (u(lxe, lve, lte) ** 2).mean()

    linfo = cvpinn._run_lbfgs_safe(net, lbfgs_loss, lbfgs_iter)
    price_fn = cvpinn.make_price(u, p, "expected")
    info = {
        "arm": arm_name, "config": cfg, "adam_steps": adam_steps, "adam_sec": round(adam_sec, 4),
        "adam_last_loss": last_loss, **{k: (round(v, 4) if k.endswith("_sec") else v) for k, v in linfo.items()},
    }
    return (price_fn, net, info) if return_info else price_fn


def evaluate_all_greeks(price_fn, x_grid_np, taus=hgv.EVAL_TAUS, vs=hgv.EVAL_VS, device="cpu"):
    """Evaluates c, cx, gam, cv (Vega), cxv (Vanna), and cvv (Volga) by PyTorch autodiff."""
    shape = x_grid_np.shape
    t_arr = np.broadcast_to(np.asarray(taus, dtype=np.float64)[:, None, None], shape).ravel()
    v_arr = np.broadcast_to(np.asarray(vs, dtype=np.float64)[None, :, None], shape).ravel()
    x_arr = np.asarray(x_grid_np, dtype=np.float64).ravel()

    x = torch.tensor(x_arr, device=device, requires_grad=True)
    v = torch.tensor(v_arr, device=device, requires_grad=True)
    tau = torch.tensor(t_arr, device=device, requires_grad=True)

    c = price_fn(x, v, tau)
    c_x, c_v = torch.autograd.grad(
        c, (x, v), torch.ones_like(c), create_graph=True, retain_graph=True, allow_unused=True
    )
    c_x = torch.zeros_like(x) if c_x is None else c_x
    c_v = torch.zeros_like(v) if c_v is None else c_v
    if c_x.requires_grad:
        c_xx, c_xv = torch.autograd.grad(
            c_x, (x, v), torch.ones_like(c_x), create_graph=False, retain_graph=True, allow_unused=True
        )
        c_xx = torch.zeros_like(x) if c_xx is None else c_xx
        c_xv = torch.zeros_like(v) if c_xv is None else c_xv
    else:
        c_xx, c_xv = torch.zeros_like(x), torch.zeros_like(v)
    if c_v.requires_grad:
        c_vv = torch.autograd.grad(c_v, v, torch.ones_like(c_v), create_graph=False, allow_unused=True)[0]
        c_vv = torch.zeros_like(v) if c_vv is None else c_vv
    else:
        c_vv = torch.zeros_like(v)

    gam = c_xx - c_x
    return {
        k: t.detach().cpu().numpy().reshape(shape)
        for k, t in (("c", c), ("cx", c_x), ("gam", gam), ("cv", c_v), ("cxv", c_xv), ("cvv", c_vv))
    }


def score_all_six_buckets(pred_dict, ref_dict):
    """Scores all 6 buckets for c, cx, gam, cv, cxv, cvv."""
    buckets = {
        "overall": (slice(None), slice(None)),
        "short_mat": (slice(0, 3), slice(None)),
        "low_var": (slice(None), slice(0, 2)),
        "mod_var": (slice(None), slice(2, 5)),
        "short_low_var": (slice(0, 3), slice(0, 2)),
        "short_mod_var": (slice(0, 3), slice(2, 5)),
    }
    report = {}
    for b_name, (st, sv) in buckets.items():
        report[b_name] = {}
        for key in ("c", "cx", "gam", "cv", "cxv", "cvv"):
            if key not in pred_dict or key not in ref_dict:
                continue
            err = pred_dict[key][st, sv, :] - ref_dict[key][st, sv, :]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            ref_rms = float(np.sqrt(np.mean(ref_dict[key][st, sv, :] ** 2)))
            report[b_name][key] = {
                "rmse": rmse,
                "rel_rmse": rmse / max(ref_rms, 1e-30),
                "max_abs": float(np.max(np.abs(err))),
            }
    return report
