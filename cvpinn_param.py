"""cvpinn_param.py - Tier-1.2 Amortized 8D Parametric Heston CV-PINN across (x, v, tau, kappa, theta, xi, rho).

Promotes (kappa, theta, xi, rho) to network inputs alongside (x, v, tau, tanh(0.5*z))
while retaining the exact parameter-explicit expected-variance Black-Scholes carrier
b(x, w_exp(v, tau; kappa, theta)) so the network never has to re-learn the terminal kink
or mean-reversion drift across parameter space.
"""
import dataclasses
import math
import time
import numpy as np
import torch

import cvpinn
import heston_grid_vec as hgv


@dataclasses.dataclass
class ParamBox:
    kappa_min: float = 0.5
    kappa_max: float = 4.0
    theta_min: float = 0.015
    theta_max: float = 0.12
    xi_min: float = 0.1
    xi_max: float = 0.8
    rho_min: float = -0.9
    rho_max: float = -0.1


def wide_spx_box():
    """Expanded parameter box covering high-vol-of-vol stressed SPX/SPXW regimes (xi <= 1.4)."""
    return ParamBox(
        kappa_min=0.3, kappa_max=8.0,
        theta_min=0.005, theta_max=0.20,
        xi_min=0.05, xi_max=1.40,
        rho_min=-0.95, rho_max=-0.05,
    )


def beta_param(tau, kap):
    return -torch.expm1(-kap * tau) / kap


def total_variance_param(v, tau, kap, tht, kind="expected"):
    if kind == "spot":
        return v * tau
    kt = kap * tau
    b = beta_param(tau, kap)
    tau_minus_b = torch.where(
        kt < 1e-3,
        tau * (0.5 * kt - (kt ** 2) / 6.0 + (kt ** 3) / 24.0 - (kt ** 4) / 120.0),
        (kt + torch.expm1(-kt)) / kap,
    )
    return tht * tau_minus_b + v * b


def source_param(x, v, tau, kap, tht, xi, rho, kind="expected"):
    w_safe = torch.clamp(total_variance_param(v, tau, kap, tht, kind), min=1e-30)
    s = torch.sqrt(w_safe)
    d1 = x / s + 0.5 * s
    d2 = d1 - s
    n_d2 = torch.where(torch.abs(d2) <= 38.0, cvpinn.npdf(d2), torch.zeros_like(d2))
    b_w = n_d2 / (2.0 * s)
    b_xw = -d2 * n_d2 / (2.0 * w_safe)
    b_ww = n_d2 * (d1 * d2 - 1.0) / (4.0 * w_safe * s)
    if kind == "spot":
        return (
            rho * xi * v * tau * b_xw
            + 0.5 * (xi ** 2) * v * (tau ** 2) * b_ww
            + kap * (tht - v) * tau * b_w
        )
    b = beta_param(tau, kap)
    return rho * xi * v * b * b_xw + 0.5 * (xi ** 2) * v * (b ** 2) * b_ww


def operator_param(fn, x, v, tau, kap, tht, xi, rho):
    """PDE operator L wrt (x, v, tau) conditioned on tensorized Heston parameters."""
    x, v, tau = (t.detach().clone().requires_grad_(True) for t in (x, v, tau))
    kap, tht, xi, rho = (t.detach() for t in (kap, tht, xi, rho))
    f = fn(x, v, tau, kap, tht, xi, rho)
    if not f.requires_grad:
        return torch.zeros_like(f)
    ones = torch.ones_like(f)
    f_x, f_v, f_tau = torch.autograd.grad(
        f, (x, v, tau), ones, create_graph=True, allow_unused=True
    )
    f_x = torch.zeros_like(x) if f_x is None else f_x
    f_v = torch.zeros_like(v) if f_v is None else f_v
    f_tau = torch.zeros_like(tau) if f_tau is None else f_tau
    if f_x.requires_grad:
        f_xx, f_xv = torch.autograd.grad(
            f_x, (x, v), torch.ones_like(f_x), create_graph=True, allow_unused=True
        )
        f_xx = torch.zeros_like(x) if f_xx is None else f_xx
        f_xv = torch.zeros_like(v) if f_xv is None else f_xv
    else:
        f_xx, f_xv = torch.zeros_like(x), torch.zeros_like(v)
    f_vv = cvpinn.grad(f_v, v)
    return (
        f_tau
        - 0.5 * v * (f_xx - f_x)
        - rho * xi * v * f_xv
        - 0.5 * (xi ** 2) * v * f_vv
        - kap * (tht - v) * f_v
    )


def param_features(x, v, tau, kap, tht, xi, rho, dom: cvpinn.Domain, box: ParamBox):
    return [
        x / dom.x_max,
        2.0 * v / dom.v_max - 1.0,
        2.0 * tau / dom.tau_max - 1.0,
        2.0 * (kap - box.kappa_min) / (box.kappa_max - box.kappa_min) - 1.0,
        2.0 * (tht - box.theta_min) / (box.theta_max - box.theta_min) - 1.0,
        xi / box.xi_max,
        2.0 * (rho - box.rho_min) / (box.rho_max - box.rho_min) - 1.0,
    ]


def make_correction_param(net, dom: cvpinn.Domain, box: ParamBox, kind="expected",
                          use_z=True, extra_scales=False):
    def u(x, v, tau, kap, tht, xi, rho):
        feats = param_features(x, v, tau, kap, tht, xi, rho, dom, box)
        if extra_scales:
            feats.append(torch.tanh(v / 0.02))
            feats.append(torch.tanh(tau / 0.05))
        if use_z:
            w_safe = torch.clamp(total_variance_param(v, tau, kap, tht, kind), min=1e-30)
            z = x / torch.sqrt(w_safe)
            feats.append(torch.tanh(0.5 * z))
        return tau * net(feats)
    return u


def make_price_param(u, kind="expected"):
    return lambda x, v, tau, kap, tht, xi, rho: (
        cvpinn.bs_call(x, total_variance_param(v, tau, kap, tht, kind))
        + u(x, v, tau, kap, tht, xi, rho)
    )


def _sample_box(n, box: ParamBox, gen, device="cpu"):
    kap = box.kappa_min + (box.kappa_max - box.kappa_min) * torch.rand(n, generator=gen)
    tht = box.theta_min + (box.theta_max - box.theta_min) * torch.rand(n, generator=gen)
    xi = box.xi_min + (box.xi_max - box.xi_min) * torch.rand(n, generator=gen)
    rho = box.rho_min + (box.rho_max - box.rho_min) * torch.rand(n, generator=gen)
    return kap.to(device), tht.to(device), xi.to(device), rho.to(device)


def sample_param_collocation(n_int, n_strip, dom: cvpinn.Domain, box: ParamBox, gen,
                             z_frac=0.5, z_max=4.0, v_power=3.0, t_power=2.5, device="cpu"):
    n_tot = n_int + n_strip
    kap, tht, xi, rho = _sample_box(n_tot, box, gen, device=device)

    u_v = torch.rand(n_int, generator=gen) ** v_power
    u_t = torch.rand(n_int, generator=gen) ** t_power
    v_int = dom.v_min + (dom.v_max - dom.v_min) * u_v
    t_int = dom.tau_min + (dom.tau_max - dom.tau_min) * u_t

    n_z = int(round(n_int * z_frac))
    n_u = n_int - n_z
    x_u = dom.x_max * (2.0 * torch.rand(n_u, generator=gen) - 1.0)
    if n_z > 0:
        w_sub = total_variance_param(
            v_int[:n_z].to(device), t_int[:n_z].to(device), kap[:n_z], tht[:n_z], "expected"
        ).cpu()
        z = z_max * (2.0 * torch.rand(n_z, generator=gen) - 1.0)
        x_z = torch.clamp(z * torch.sqrt(torch.clamp(w_sub, min=1e-30)), -dom.x_max, dom.x_max)
        x_int = torch.cat([x_z, x_u])
    else:
        x_int = x_u

    if n_strip > 0:
        ts = dom.tau_min + (dom.tau_max - dom.tau_min) * (torch.rand(n_strip, generator=gen) ** t_power)
        vs = torch.full((n_strip,), dom.v_min)
        n_sz = int(round(n_strip * z_frac))
        n_su = n_strip - n_sz
        xs_u = dom.x_max * (2.0 * torch.rand(n_su, generator=gen) - 1.0)
        if n_sz > 0:
            ws_sub = total_variance_param(
                vs[:n_sz].to(device), ts[:n_sz].to(device),
                kap[n_int:n_int + n_sz], tht[n_int:n_int + n_sz], "expected",
            ).cpu()
            zs = z_max * (2.0 * torch.rand(n_sz, generator=gen) - 1.0)
            xs_z = torch.clamp(zs * torch.sqrt(torch.clamp(ws_sub, min=1e-30)), -dom.x_max, dom.x_max)
            xs = torch.cat([xs_z, xs_u])
        else:
            xs = xs_u
        x_all = torch.cat([x_int, xs]).to(device)
        v_all = torch.cat([v_int, vs]).to(device)
        t_all = torch.cat([t_int, ts]).to(device)
    else:
        x_all, v_all, t_all = x_int.to(device), v_int.to(device), t_int.to(device)
    return x_all, v_all, t_all, kap, tht, xi, rho


def sample_param_edges(n_edge, dom: cvpinn.Domain, box: ParamBox, gen, device="cpu"):
    x_e, v_e, t_e = cvpinn.sample_edges(n_edge, dom, gen, device=device)
    kap, tht, xi, rho = _sample_box(n_edge, box, gen, device=device)
    return x_e, v_e, t_e, kap, tht, xi, rho


def gate2_param(dom: cvpinn.Domain, box: ParamBox, kind="expected", n=4096, seed=0, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    x, v, tau, kap, tht, xi, rho = sample_param_collocation(n, n // 16, dom, box, gen, device=device)
    lhs = operator_param(
        lambda a, b, c, kp, th, _x, _r: cvpinn.bs_call(a, total_variance_param(b, c, kp, th, kind)),
        x, v, tau, kap, tht, xi, rho,
    )
    return float((lhs + source_param(x, v, tau, kap, tht, xi, rho, kind)).abs().max().item())


def gate3_param(dom: cvpinn.Domain, box: ParamBox, n=4096, seed=0, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    x, v, tau, kap, tht, _xi, rho = sample_param_collocation(n, n // 16, dom, box, gen, device=device)
    xi0 = torch.zeros_like(kap)
    zero_u = lambda a, b, c, kp, th, x_v, rh: torch.zeros_like(a)
    zero_net = cvpinn.Mlp(8, width=64, depth=4).to(device)
    with torch.no_grad():
        zero_net.net[-1].weight.zero_()
        zero_net.net[-1].bias.zero_()
    zero_corr = make_correction_param(zero_net, dom, box, kind="expected", use_z=True)
    src_max = source_param(x, v, tau, kap, tht, xi0, rho, "expected").abs().max().item()
    res_max = operator_param(zero_u, x, v, tau, kap, tht, xi0, rho).abs().max().item()
    net_max = operator_param(zero_corr, x, v, tau, kap, tht, xi0, rho).abs().max().item()
    return float(max(src_max, res_max, net_max))


def bind_params(price_fn_param, p: cvpinn.Heston, device="cpu"):
    """Fixes (kappa, theta, xi, rho) to return a standard price_fn(x, v, tau) for grid scoring."""
    def price_fn(x, v, tau):
        kap = torch.full_like(x, float(p.kappa))
        tht = torch.full_like(x, float(p.theta))
        xi = torch.full_like(x, float(p.xi))
        rho = torch.full_like(x, float(p.rho))
        return price_fn_param(x, v, tau, kap, tht, xi, rho)
    return price_fn


def train_param_model(arm="M_param", seed=0, adam_steps=12000, lbfgs_iter=500, n_col=2048,
                      width=128, depth=4, v_max=0.5, term_weight=100.0, device="cpu"):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dom = cvpinn.Domain(v_max=v_max)
    box = ParamBox()
    n_strip = n_col // 16

    if arm == "B0_param":
        net = cvpinn.Mlp(7, width=width, depth=depth).to(device)
        c_fn = lambda x, v, tau, kp, th, xi, rh: net(param_features(x, v, tau, kp, th, xi, rh, dom, box))
        payoff = lambda x: torch.relu(torch.exp(x) - 1.0)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=1e-4)
        t0 = time.perf_counter()
        last_loss = float("nan")
        for _ in range(adam_steps):
            x, v, tau, kp, th, xi, rh = sample_param_collocation(n_col, n_strip, dom, box, gen, z_frac=0.0, v_power=2.0, t_power=2.0, device=device)
            x0, v0, _, kp0, th0, xi0, rh0 = sample_param_collocation(n_col // 4, 0, dom, box, gen, z_frac=0.0, v_power=2.0, t_power=2.0, device=device)
            xe, ve, te, kpe, the, xie, rhe = sample_param_edges(n_col // 8, dom, box, gen, device=device)
            lp = (operator_param(c_fn, x, v, tau, kp, th, xi, rh) ** 2).mean()
            lt = ((c_fn(x0, v0, torch.zeros_like(x0), kp0, th0, xi0, rh0) - payoff(x0)) ** 2).mean()
            le = ((c_fn(xe, ve, te, kpe, the, xie, rhe) - payoff(xe)) ** 2).mean()
            loss = lp + term_weight * lt + le
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            last_loss = float(loss.detach().item())
        adam_sec = time.perf_counter() - t0
        lx, lv, lt_t, lkp, lth, lxi, lrh = sample_param_collocation(8192, 512, dom, box, gen, z_frac=0.0, v_power=2.0, t_power=2.0, device=device)
        lx0, lv0, _, lkp0, lth0, lxi0, lrh0 = sample_param_collocation(2048, 0, dom, box, gen, z_frac=0.0, v_power=2.0, t_power=2.0, device=device)
        lt0 = torch.zeros_like(lx0)
        pay0 = payoff(lx0).detach()
        lxe, lve, lte, lkpe, lthe, lxie, lrhe = sample_param_edges(1024, dom, box, gen, device=device)
        pay_e = payoff(lxe).detach()
        def lbfgs_loss():
            return (
                (operator_param(c_fn, lx, lv, lt_t, lkp, lth, lxi, lrh) ** 2).mean()
                + term_weight * ((c_fn(lx0, lv0, lt0, lkp0, lth0, lxi0, lrh0) - pay0) ** 2).mean()
                + ((c_fn(lxe, lve, lte, lkpe, lthe, lxie, lrhe) - pay_e) ** 2).mean()
            )
        linfo = cvpinn._run_lbfgs_safe(net, lbfgs_loss, lbfgs_iter)
        info = {"arm": arm, "adam_sec": round(adam_sec, 4), "adam_last_loss": last_loss, **{k: (round(v, 4) if k.endswith("_sec") else v) for k, v in linfo.items()}}
        return c_fn, net, info

    extra = arm == "M_param_plus"
    n_in = 8 + (2 if extra else 0)
    net = cvpinn.Mlp(n_in, width=width, depth=depth).to(device)
    u = make_correction_param(net, dom, box, kind="expected", use_z=True, extra_scales=extra)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=1e-4)
    z_frac = 0.55 if extra else 0.5
    v_pwr = 2.5 if extra else 2.5
    t_pwr = 2.2 if extra else 2.2
    t0 = time.perf_counter()
    last_loss = float("nan")
    for _ in range(adam_steps):
        x, v, tau, kp, th, xi, rh = sample_param_collocation(
            n_col, n_strip, dom, box, gen, z_frac=z_frac, z_max=4.0, v_power=v_pwr, t_power=t_pwr, device=device
        )
        xe, ve, te, kpe, the, xie, rhe = sample_param_edges(n_col // 8, dom, box, gen, device=device)
        r = operator_param(u, x, v, tau, kp, th, xi, rh) - source_param(x, v, tau, kp, th, xi, rh, "expected")
        if extra:
            rw = 1.0 / torch.sqrt(v + 0.02)
            rw = rw / torch.sqrt((rw ** 2).mean() + 1e-30)
            loss = ((rw * r) ** 2).mean() + (u(xe, ve, te, kpe, the, xie, rhe) ** 2).mean()
        else:
            loss = (r ** 2).mean() + (u(xe, ve, te, kpe, the, xie, rhe) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        last_loss = float(loss.detach().item())
    adam_sec = time.perf_counter() - t0
    n_lb = 12288 if extra else 8192
    lx, lv, lt_t, lkp, lth, lxi, lrh = sample_param_collocation(
        n_lb, n_lb // 16, dom, box, gen, z_frac=z_frac, z_max=4.0, v_power=v_pwr, t_power=t_pwr, device=device
    )
    lxe, lve, lte, lkpe, lthe, lxie, lrhe = sample_param_edges(1024, dom, box, gen, device=device)
    src_cached = source_param(lx, lv, lt_t, lkp, lth, lxi, lrh, "expected").detach()
    lw = None
    if extra:
        lw_raw = 1.0 / torch.sqrt(lv + 0.02)
        lw = (lw_raw / torch.sqrt((lw_raw ** 2).mean() + 1e-30)).detach()
    def lbfgs_loss():
        r_l = operator_param(u, lx, lv, lt_t, lkp, lth, lxi, lrh) - src_cached
        lp = ((lw * r_l) ** 2).mean() if lw is not None else (r_l ** 2).mean()
        return lp + (u(lxe, lve, lte, lkpe, lthe, lxie, lrhe) ** 2).mean()
    linfo = cvpinn._run_lbfgs_safe(net, lbfgs_loss, lbfgs_iter)
    price_fn = make_price_param(u, "expected")
    info = {"arm": arm, "adam_sec": round(adam_sec, 4), "adam_last_loss": last_loss, **{k: (round(v, 4) if k.endswith("_sec") else v) for k, v in linfo.items()}}
    return price_fn, net, info
