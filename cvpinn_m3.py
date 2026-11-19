"""cvpinn_m3.py - Championship Tier-3 CV-PINN: Stratified-z + Low-v Weighting +
Multi-Round Residual-Adaptive Refinement (RAR) Strong-Wolfe L-BFGS + Free No-Arbitrage
Convexity/Floor Guard + Full Training Trajectory & Admissibility Telemetry.

Supports constant-parameter Heston (P1/P2), smooth time-dependent Heston (TD),
and 8D/10D amortized parametric Heston (ParamBox).
"""
import math
import time
import numpy as np
import torch
from torch import nn

import cvpinn
import cvpinn_td as tdmod
import cvpinn_tier2 as t2
import heston_grid_vec as hgv


class FlexMlp(nn.Module):
    """Configurable MLP supporting Tanh or SiLU (Swish) smooth activations."""

    def __init__(self, n_in, width=112, depth=4, act="tanh"):
        super().__init__()
        act_cls = nn.SiLU if act == "silu" else nn.Tanh
        layers, n = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(n, width), act_cls()]
            n = width
        layers.append(nn.Linear(n, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feats):
        return self.net(torch.stack(feats, dim=-1)).squeeze(-1)


def count_parameters(net):
    return int(sum(p.numel() for p in net.parameters()))


def operator_with_gam_and_c(u_fn, x, v, tau, p_or_td, is_td=False):
    """Computes L[u], normalized price c = b + u, and normalized Gamma = b_gam + u_xx - u_x in one pass."""
    x, v, tau = (t.detach().clone().requires_grad_(True) for t in (x, v, tau))
    u_val = u_fn(x, v, tau)
    if not u_val.requires_grad:
        z = torch.zeros_like(u_val)
        return z, z, z
    ones = torch.ones_like(u_val)
    u_x, u_v, u_tau = torch.autograd.grad(
        u_val, (x, v, tau), ones, create_graph=True, allow_unused=True
    )
    u_x = torch.zeros_like(x) if u_x is None else u_x
    u_v = torch.zeros_like(v) if u_v is None else u_v
    u_tau = torch.zeros_like(tau) if u_tau is None else u_tau
    if u_x.requires_grad:
        u_xx, u_xv = torch.autograd.grad(
            u_x, (x, v), torch.ones_like(u_x), create_graph=True, allow_unused=True
        )
        u_xx = torch.zeros_like(x) if u_xx is None else u_xx
        u_xv = torch.zeros_like(v) if u_xv is None else u_xv
    else:
        u_xx, u_xv = torch.zeros_like(x), torch.zeros_like(v)
    u_vv = cvpinn.grad(u_v, v)

    if is_td:
        kap = p_or_td.kappa_t(tau)
        tht = p_or_td.theta_t(tau)
        xi = p_or_td.xi_t(tau)
        rho = p_or_td.rho_t(tau)
        w_raw = tdmod.total_variance_td(v, tau, p_or_td, "expected")
    else:
        kap, tht, xi, rho = p_or_td.kappa, p_or_td.theta, p_or_td.xi, p_or_td.rho
        w_raw = cvpinn.total_variance(v, tau, p_or_td, "expected")

    l_u = (
        u_tau
        - 0.5 * v * (u_xx - u_x)
        - rho * xi * v * u_xv
        - 0.5 * (xi ** 2) * v * u_vv
        - kap * (tht - v) * u_v
    )

    with torch.no_grad():
        w_safe = torch.clamp(w_raw, min=1e-30)
        s = torch.sqrt(w_safe)
        d2 = x / s - 0.5 * s
        n_d2 = torch.where(torch.abs(d2) <= 38.0, cvpinn.npdf(d2), torch.zeros_like(d2))
        b_gam = n_d2 / s
        b_val = cvpinn.bs_call(x, w_raw)

    c_val = b_val + u_val
    gam_val = b_gam + (u_xx - u_x)
    return l_u, c_val, gam_val


def audit_arbitrage_and_percentiles(pred, ref, x_grid_np):
    """Comprehensive financial admissibility & error percentile metrics for final reporting."""
    payoff = np.maximum(np.exp(x_grid_np) - 1.0, 0.0)
    ex = np.exp(x_grid_np)
    gam = pred["gam"]
    c = pred["c"]
    cx = pred["cx"]

    arb = {
        "neg_gam_count": int((gam < 0.0).sum()),
        "neg_gam_core_count": int((gam[:, :, 4:21] < 0.0).sum()),  # |z| <= 2
        "min_gam": float(gam.min()),
        "floor_viol_count": int((c < payoff - 1e-6).sum()),
        "min_c_minus_payoff": float((c - payoff).min()),
        "delta_oob_count": int(((cx < -1e-6) | (cx > ex + 1e-6)).sum()),
        "put_parity_max_err": float(np.max(np.abs((c - (ex - 1.0)) - (c - ex + 1.0)))),
    }

    dist = {}
    for k in ("c", "cx", "gam", "cv", "cxv", "cvv"):
        if k not in pred or k not in ref:
            continue
        ae = np.abs(pred[k] - ref[k])
        dist[k] = {
            "mae": float(np.mean(ae)),
            "p95_abs": float(np.percentile(ae, 95)),
            "p99_abs": float(np.percentile(ae, 99)),
            "max_abs": float(np.max(ae)),
        }
    return arb, dist


def _run_lbfgs_logged(net, loss_fn, max_iter, history_size=100, log_every=100):
    """Strong-Wolfe L-BFGS with safety rollback and periodic loss trajectory logging."""
    best_state = cvpinn._snapshot_state(net)
    with torch.enable_grad():
        pre_loss = float(loss_fn().detach().item())
    best_loss = pre_loss if math.isfinite(pre_loss) else float("inf")
    eval_trace = [(0, pre_loss)]
    info = {
        "lbfgs_pre_loss": pre_loss,
        "lbfgs_post_loss": pre_loss,
        "lbfgs_best_loss": best_loss,
        "lbfgs_evals": 0,
        "lbfgs_iters": 0,
        "lbfgs_rolled_back": False,
        "lbfgs_sec": 0.0,
    }
    if max_iter <= 0:
        info["trace"] = eval_trace
        return info

    t0 = time.perf_counter()
    lbfgs = torch.optim.LBFGS(
        net.parameters(), lr=1.0, max_iter=max_iter,
        history_size=history_size, tolerance_grad=1e-13, tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure():
        nonlocal best_loss, best_state
        lbfgs.zero_grad()
        tot = loss_fn()
        val = float(tot.detach().item())
        info["lbfgs_evals"] += 1
        if math.isfinite(val):
            if val < best_loss and all(torch.isfinite(pr).all().item() for pr in net.parameters()):
                best_loss = val
                best_state = cvpinn._snapshot_state(net)
        if info["lbfgs_evals"] % log_every == 0:
            eval_trace.append((info["lbfgs_evals"], val if math.isfinite(val) else float("nan")))
        tot.backward()
        return tot

    lbfgs.step(closure)
    info["lbfgs_sec"] = round(time.perf_counter() - t0, 4)
    st = lbfgs.state.get(next(iter(net.parameters())), {})
    info["lbfgs_iters"] = int(st.get("n_iter", 0))

    params_finite = all(torch.isfinite(pr).all().item() for pr in net.parameters())
    if params_finite:
        with torch.enable_grad():
            post_loss = float(loss_fn().detach().item())
    else:
        post_loss = float("nan")
    if (not math.isfinite(post_loss)) or post_loss > best_loss:
        net.load_state_dict(best_state)
        info["lbfgs_rolled_back"] = True
        post_loss = best_loss

    eval_trace.append((info["lbfgs_evals"], post_loss))
    info["lbfgs_post_loss"] = post_loss
    info["lbfgs_best_loss"] = best_loss
    info["trace"] = eval_trace
    return info


def _select_rar_points(u_fn, p_or_td, dom, gen, cfg, is_td=False, device="cpu"):
    """Draws n_pool candidate points, scores weighted PDE residual, returns top n_rar hotspots."""
    n_pool = cfg.get("rar_pool", 49152)
    n_rar = cfg.get("rar_add", 4096)
    if n_rar <= 0:
        return None
    xp, vp, tp = t2.sample_stratified_z(
        n_pool, n_pool // 16, p_or_td, dom, gen,
        z_frac=cfg["z_frac"], z_max=cfg["z_max"],
        v_power=cfg["vp"], t_power=cfg["tp"], is_td=is_td, device=device,
    )
    # Chunked evaluation to keep peak RAM modest
    scores = []
    chunk = 8192
    for i0 in range(0, len(xp), chunk):
        xc, vc, tc = xp[i0:i0 + chunk], vp[i0:i0 + chunk], tp[i0:i0 + chunk]
        if is_td:
            rc = tdmod.operator_td(u_fn, xc, vc, tc, p_or_td) - tdmod.source_td(xc, vc, tc, p_or_td, "expected")
        else:
            rc = cvpinn.operator(u_fn, xc, vc, tc, p_or_td) - cvpinn.source(xc, vc, tc, p_or_td, "expected")
        with torch.no_grad():
            w = 1.0 / torch.sqrt(vc + 0.02) if cfg.get("lowv_weight", True) else torch.ones_like(vc)
            scores.append((w * rc.detach()).abs())
    all_sc = torch.cat(scores)
    top_idx = torch.topk(all_sc, k=min(n_rar, len(all_sc))).indices
    return xp[top_idx].detach(), vp[top_idx].detach(), tp[top_idx].detach()


M3_CONFIGS = {
    "M3_rar_80": {
        "width": 80, "depth": 4, "act": "tanh", "extra_scales": True, "z2": False,
        "z_frac": 0.60, "z_max": 5.0, "vp": 2.5, "tp": 2.2, "lowv_weight": True,
        "adam_steps": 4000, "lbfgs_rounds": 3, "lbfgs_per_round": 1200, "history_size": 100,
        "n_lbfgs_base": 16384, "rar_pool": 49152, "rar_add": 4096,
        "arb_gam_w": 0.0, "arb_flr_w": 0.0,
    },
    "M3_rar_112": {
        "width": 112, "depth": 4, "act": "tanh", "extra_scales": True, "z2": False,
        "z_frac": 0.60, "z_max": 5.0, "vp": 2.5, "tp": 2.2, "lowv_weight": True,
        "adam_steps": 4000, "lbfgs_rounds": 3, "lbfgs_per_round": 1200, "history_size": 100,
        "n_lbfgs_base": 16384, "rar_pool": 49152, "rar_add": 4096,
        "arb_gam_w": 0.0, "arb_flr_w": 0.0,
    },
    "M3_rar_112_arb": {
        "width": 112, "depth": 4, "act": "tanh", "extra_scales": True, "z2": False,
        "z_frac": 0.60, "z_max": 5.0, "vp": 2.5, "tp": 2.2, "lowv_weight": True,
        "adam_steps": 4000, "lbfgs_rounds": 3, "lbfgs_per_round": 1200, "history_size": 100,
        "n_lbfgs_base": 16384, "rar_pool": 49152, "rar_add": 4096,
        "arb_gam_w": 1e-4, "arb_flr_w": 10.0,
    },
    "M3_rar_112_silu": {
        "width": 112, "depth": 4, "act": "silu", "extra_scales": True, "z2": False,
        "z_frac": 0.60, "z_max": 5.0, "vp": 2.5, "tp": 2.2, "lowv_weight": True,
        "adam_steps": 4000, "lbfgs_rounds": 3, "lbfgs_per_round": 1200, "history_size": 100,
        "n_lbfgs_base": 16384, "rar_pool": 49152, "rar_add": 4096,
        "arb_gam_w": 1e-4, "arb_flr_w": 10.0,
    },
}


def build_m3_net(cfg):
    n_in = 4 + (2 if cfg.get("extra_scales", True) else 0) + (1 if cfg.get("z2", False) else 0)
    return FlexMlp(n_in, width=cfg["width"], depth=cfg["depth"], act=cfg.get("act", "tanh"))


def make_m3_price_from_net(cfg, net, p_or_td, v_max=0.5, is_td=False):
    dom = cvpinn.Domain(v_max=v_max)
    u = t2.make_correction_tier2(
        net, p_or_td, dom, kind="expected", use_z=True,
        extra_scales=cfg.get("extra_scales", True), z2=cfg.get("z2", False), is_td=is_td,
    )
    return tdmod.make_price_td(u, p_or_td, kind="expected") if is_td else cvpinn.make_price(u, p_or_td, "expected")


def train_m3(arm_name, p_or_td, seed, x_grid_eval=None, ref_eval=None,
             is_td=False, v_max=0.5, device="cpu", cfg_override=None):
    """Trains an M3 solver with Adam warmup + Multi-Round RAR-L-BFGS and logs full trajectories."""
    cfg = dict(M3_CONFIGS[arm_name])
    if cfg_override:
        cfg.update(cfg_override)

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dom = cvpinn.Domain(v_max=v_max)
    net = build_m3_net(cfg).to(device)
    u = t2.make_correction_tier2(
        net, p_or_td, dom, kind="expected", use_z=True,
        extra_scales=cfg["extra_scales"], z2=cfg["z2"], is_td=is_td,
    )
    price_fn = tdmod.make_price_td(u, p_or_td, kind="expected") if is_td else cvpinn.make_price(u, p_or_td, "expected")

    n_col = 2048
    n_strip = n_col // 16
    adam_steps = cfg["adam_steps"]
    arb_gw = cfg["arb_gam_w"]
    arb_fw = cfg["arb_flr_w"]
    use_arb = (arb_gw > 0.0) or (arb_fw > 0.0)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=1e-4)

    adam_trace = []
    stage_snapshots = []

    def _snapshot(stage_name, elapsed_sec, loss_val):
        if x_grid_eval is None or ref_eval is None:
            return
        pred = t2.evaluate_all_greeks(price_fn, x_grid_eval, hgv.EVAL_TAUS, hgv.EVAL_VS, device=device)
        sc = t2.score_all_six_buckets(pred, ref_eval)
        arb, _ = audit_arbitrage_and_percentiles(pred, ref_eval, x_grid_eval)
        ov, sl = sc["overall"], sc["short_low_var"]
        snap = {
            "stage": stage_name,
            "elapsed_sec": round(elapsed_sec, 2),
            "loss": loss_val,
            "ov_c": ov["c"]["rel_rmse"],
            "ov_gam": ov["gam"]["rel_rmse"],
            "ov_cxv": ov["cxv"]["rel_rmse"],
            "slv_gam": sl["gam"]["rel_rmse"],
            "slv_cxv": sl["cxv"]["rel_rmse"],
            "neg_gam_count": arb["neg_gam_count"],
            "min_gam": arb["min_gam"],
            "floor_viol_count": arb["floor_viol_count"],
        }
        if "cv" in ov:
            snap["ov_cv"] = ov["cv"]["rel_rmse"]
            snap["ov_cvv"] = ov["cvv"]["rel_rmse"]
        stage_snapshots.append(snap)

    t_total0 = time.perf_counter()
    last_loss = float("nan")
    for step in range(1, adam_steps + 1):
        x, v, tau = t2.sample_stratified_z(
            n_col, n_strip, p_or_td, dom, gen,
            z_frac=cfg["z_frac"], z_max=cfg["z_max"],
            v_power=cfg["vp"], t_power=cfg["tp"], is_td=is_td, device=device,
        )
        xe, ve, te = cvpinn.sample_edges(n_col // 8, dom, gen, device=device)
        if use_arb:
            l_u, c_col, gam_col = operator_with_gam_and_c(u, x, v, tau, p_or_td, is_td=is_td)
            src = tdmod.source_td(x, v, tau, p_or_td, "expected") if is_td else cvpinn.source(x, v, tau, p_or_td, "expected")
            r = l_u - src
        else:
            if is_td:
                r = tdmod.operator_td(u, x, v, tau, p_or_td) - tdmod.source_td(x, v, tau, p_or_td, "expected")
            else:
                r = cvpinn.operator(u, x, v, tau, p_or_td) - cvpinn.source(x, v, tau, p_or_td, "expected")

        if cfg["lowv_weight"]:
            with torch.no_grad():
                rw = 1.0 / torch.sqrt(v + 0.02)
                rw = rw / torch.sqrt((rw ** 2).mean() + 1e-30)
            loss_pde = ((rw * r) ** 2).mean()
        else:
            loss_pde = (r ** 2).mean()

        loss = loss_pde + (u(xe, ve, te) ** 2).mean()
        if use_arb:
            pay_col = torch.relu(torch.exp(x) - 1.0)
            loss = loss + arb_gw * (torch.relu(-gam_col) ** 2).mean() + arb_fw * (torch.relu(pay_col - c_col) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        last_loss = float(loss.detach().item())
        if step == 1 or step % 250 == 0 or step == adam_steps:
            adam_trace.append((step, last_loss))

    adam_sec = time.perf_counter() - t_total0
    _snapshot("post_adam", adam_sec, last_loss)

    # Multi-round refreshed RAR + Strong-Wolfe L-BFGS
    round_infos = []
    rar_pts = None
    for rd in range(1, cfg["lbfgs_rounds"] + 1):
        lx, lv, lt = t2.sample_stratified_z(
            cfg["n_lbfgs_base"], cfg["n_lbfgs_base"] // 16, p_or_td, dom, gen,
            z_frac=cfg["z_frac"], z_max=cfg["z_max"],
            v_power=cfg["vp"], t_power=cfg["tp"], is_td=is_td, device=device,
        )
        if rd >= 2 and cfg["rar_add"] > 0:
            rar_pts = _select_rar_points(u, p_or_td, dom, gen, cfg, is_td=is_td, device=device)
        if rar_pts is not None:
            lx = torch.cat([lx, rar_pts[0]])
            lv = torch.cat([lv, rar_pts[1]])
            lt = torch.cat([lt, rar_pts[2]])

        lxe, lve, lte = cvpinn.sample_edges(1024, dom, gen, device=device)
        src_cached = (
            tdmod.source_td(lx, lv, lt, p_or_td, "expected")
            if is_td
            else cvpinn.source(lx, lv, lt, p_or_td, "expected")
        ).detach()
        pay_cached = torch.relu(torch.exp(lx) - 1.0).detach() if use_arb else None
        lw = None
        if cfg["lowv_weight"]:
            lw_raw = 1.0 / torch.sqrt(lv + 0.02)
            lw = (lw_raw / torch.sqrt((lw_raw ** 2).mean() + 1e-30)).detach()

        def lbfgs_loss():
            if use_arb:
                l_u, c_col, gam_col = operator_with_gam_and_c(u, lx, lv, lt, p_or_td, is_td=is_td)
                r_l = l_u - src_cached
            else:
                l_u = tdmod.operator_td(u, lx, lv, lt, p_or_td) if is_td else cvpinn.operator(u, lx, lv, lt, p_or_td)
                r_l = l_u - src_cached
            lp = ((lw * r_l) ** 2).mean() if lw is not None else (r_l ** 2).mean()
            tot = lp + (u(lxe, lve, lte) ** 2).mean()
            if use_arb:
                tot = tot + arb_gw * (torch.relu(-gam_col) ** 2).mean() + arb_fw * (torch.relu(pay_cached - c_col) ** 2).mean()
            return tot

        rinfo = _run_lbfgs_logged(net, lbfgs_loss, cfg["lbfgs_per_round"], history_size=cfg["history_size"], log_every=100)
        rinfo["round"] = rd
        rinfo["n_points"] = int(len(lx))
        round_infos.append(rinfo)
        _snapshot(f"lbfgs_rd{rd}", time.perf_counter() - t_total0, rinfo["lbfgs_post_loss"])

    info = {
        "arm": arm_name,
        "config": cfg,
        "n_params": count_parameters(net),
        "adam_sec": round(adam_sec, 4),
        "adam_last_loss": last_loss,
        "adam_trace": adam_trace,
        "lbfgs_rounds": round_infos,
        "lbfgs_post_loss": round_infos[-1]["lbfgs_post_loss"] if round_infos else last_loss,
        "lbfgs_rolled_back": any(r["lbfgs_rolled_back"] for r in round_infos),
        "stage_snapshots": stage_snapshots,
    }
    return price_fn, net, info
