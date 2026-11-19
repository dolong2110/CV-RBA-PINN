"""cvpinn_td.py - Tier-1 Time-Dependent Expected-Variance CV-RBA-PINN (Check T15 extension).

Extends hardened cvpinn.py to smooth time-dependent Heston curves (kappa(tau), theta(tau),
xi(tau), rho(tau)) where closed-form characteristic functions do not exist:
  w_td(v, tau) = alpha(tau) + v * beta(tau),
  beta'(tau) = 1 - kappa(tau)*beta(tau),  beta(0) = 0,
  alpha'(tau) = kappa(tau)*theta(tau)*beta(tau),  alpha(0) = 0.

Solves beta(tau) = tau * beta_tilde(tau) and alpha(tau) = tau^2 * alpha_tilde(tau) on [0, tau_max]
by 65-node Chebyshev spectral collocation (preserving ~15-16 relative digits down to tau=1e-4)
and evaluates via C^inf Clenshaw recurrence in PyTorch float64.
"""
import dataclasses
import math
import time
from typing import Callable
import numpy as np
import torch

import cvpinn
import heston_grid_vec as hgv


@dataclasses.dataclass
class TimeDepHeston:
    kappa_np: Callable
    theta_np: Callable
    xi_np: Callable
    rho_np: Callable
    kappa_t: Callable
    theta_t: Callable
    xi_t: Callable
    rho_t: Callable
    tau_max: float = 2.0
    n_cheb: int = 64

    def __post_init__(self):
        self.beta_tilde_coeffs, self.alpha_tilde_coeffs = solve_ab_chebyshev(
            self.kappa_np, self.theta_np, self.tau_max, self.n_cheb
        )

    def params_at(self, tau):
        """Returns (kappa, theta, xi, rho) as floats for scalar tau (compatible with ADI)."""
        t = float(tau)
        return (
            float(self.kappa_np(t)),
            float(self.theta_np(t)),
            float(self.xi_np(t)),
            float(self.rho_np(t)),
        )


def default_td_heston():
    """Smooth time-dependent curve verified in Check T15 (starts at P2 at tau=0)."""
    return TimeDepHeston(
        kappa_np=lambda t: 1.5 + 0.8 * np.sin(2.0 * t),
        theta_np=lambda t: 0.04 + 0.02 * t,
        xi_np=lambda t: 0.5 - 0.1 * t,
        rho_np=lambda t: -0.7 + 0.2 * t,
        kappa_t=lambda t: 1.5 + 0.8 * torch.sin(2.0 * t),
        theta_t=lambda t: 0.04 + 0.02 * t,
        xi_t=lambda t: 0.5 - 0.1 * t,
        rho_t=lambda t: -0.7 + 0.2 * t,
    )


def _cheb_diff_matrix(n):
    """Standard (n+1)x(n+1) Chebyshev-Lobatto differentiation matrix on x_k = cos(pi*k/n)."""
    x = np.cos(math.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1, dtype=np.float64)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1))
    dx = x[:, None] - x[None, :]
    d = np.outer(c, 1.0 / c) / (dx + np.eye(n + 1))
    d = d - np.diag(np.sum(d, axis=1))
    return x, d


def _values_to_cheb_coeffs(vals_desc):
    """Converts values at x_k = cos(pi*k/n) (k=0..n, from +1 down to -1) to T_m coeffs."""
    n = len(vals_desc) - 1
    theta = math.pi * np.arange(n + 1) / n
    # Discrete cosine transform type I (exact interpolation at Lobatto nodes)
    coeffs = np.zeros(n + 1, dtype=np.float64)
    w = np.ones(n + 1, dtype=np.float64)
    w[0] = 0.5
    w[-1] = 0.5
    for m in range(n + 1):
        coeffs[m] = (2.0 / n) * np.sum(w * vals_desc * np.cos(m * theta))
    coeffs[0] *= 0.5
    coeffs[-1] *= 0.5
    return tuple(float(v) for v in coeffs)


def solve_ab_chebyshev(kappa_fn, theta_fn, tau_max=2.0, n=64):
    """Solves scaled ODEs for beta_tilde = beta/tau and alpha_tilde = alpha/tau^2."""
    # Map s in [-1, 1] -> tau = 0.5 * tau_max * (1 + s), so s=-1 (index n) is tau=0
    s, d_s = _cheb_diff_matrix(n)
    taus = 0.5 * tau_max * (1.0 + s)
    d_tau = (2.0 / tau_max) * d_s

    kaps = np.array([float(kappa_fn(t)) for t in taus], dtype=np.float64)
    thts = np.array([float(theta_fn(t)) for t in taus], dtype=np.float64)

    # 1. tau * bt' + (1 + kappa*tau) * bt = 1, bt(0) = 1 (at index n where s=-1, tau=0)
    mat_b = taus[:, None] * d_tau + np.diag(1.0 + kaps * taus)
    rhs_b = np.ones(n + 1, dtype=np.float64)
    mat_b[-1, :] = 0.0
    mat_b[-1, -1] = 1.0
    rhs_b[-1] = 1.0
    bt_vals = np.linalg.solve(mat_b, rhs_b)

    # 2. tau * at' + 2 * at = kappa * theta * bt, at(0) = 0.5 * kappa(0) * theta(0)
    mat_a = taus[:, None] * d_tau + 2.0 * np.eye(n + 1)
    rhs_a = kaps * thts * bt_vals
    mat_a[-1, :] = 0.0
    mat_a[-1, -1] = 1.0
    rhs_a[-1] = 0.5 * float(kappa_fn(0.0)) * float(theta_fn(0.0))
    at_vals = np.linalg.solve(mat_a, rhs_a)

    return _values_to_cheb_coeffs(bt_vals), _values_to_cheb_coeffs(at_vals)


def _cheb_eval_np(coeffs, tau, tau_max):
    u = 2.0 * np.asarray(tau, dtype=np.float64) / tau_max - 1.0
    b1 = np.zeros_like(u)
    b2 = np.zeros_like(u)
    two_u = 2.0 * u
    for c in reversed(coeffs[1:]):
        b0 = two_u * b1 - b2 + c
        b2 = b1
        b1 = b0
    return u * b1 - b2 + coeffs[0]


def _cheb_eval_torch(coeffs, tau, tau_max):
    u = 2.0 * tau / tau_max - 1.0
    b1 = torch.zeros_like(u)
    b2 = torch.zeros_like(u)
    two_u = 2.0 * u
    for c in reversed(coeffs[1:]):
        b0 = two_u * b1 - b2 + c
        b2 = b1
        b1 = b0
    return u * b1 - b2 + coeffs[0]


def alpha_beta_np(tau, td: TimeDepHeston):
    t = np.asarray(tau, dtype=np.float64)
    beta = t * _cheb_eval_np(td.beta_tilde_coeffs, t, td.tau_max)
    alpha = (t * t) * _cheb_eval_np(td.alpha_tilde_coeffs, t, td.tau_max)
    return alpha, beta


def w_td_np(v, tau, td: TimeDepHeston):
    alpha, beta = alpha_beta_np(tau, td)
    return alpha + np.asarray(v, dtype=np.float64) * beta


def beta_td_torch(tau, td: TimeDepHeston):
    return tau * _cheb_eval_torch(td.beta_tilde_coeffs, tau, td.tau_max)


def alpha_td_torch(tau, td: TimeDepHeston):
    return (tau * tau) * _cheb_eval_torch(td.alpha_tilde_coeffs, tau, td.tau_max)


def total_variance_td(v, tau, td: TimeDepHeston, kind="expected"):
    if kind == "spot":
        return v * tau
    return alpha_td_torch(tau, td) + v * beta_td_torch(tau, td)


def source_td(x, v, tau, td: TimeDepHeston, kind="expected"):
    """Closed-form time-dependent PDE source F_td = -L_td[b]."""
    w_safe = torch.clamp(total_variance_td(v, tau, td, kind), min=1e-30)
    s = torch.sqrt(w_safe)
    d1 = x / s + 0.5 * s
    d2 = d1 - s
    n_d2 = torch.where(torch.abs(d2) <= 38.0, cvpinn.npdf(d2), torch.zeros_like(d2))
    b_w = n_d2 / (2.0 * s)
    b_xw = -d2 * n_d2 / (2.0 * w_safe)
    b_ww = n_d2 * (d1 * d2 - 1.0) / (4.0 * w_safe * s)
    kap = td.kappa_t(tau)
    tht = td.theta_t(tau)
    xi = td.xi_t(tau)
    rho = td.rho_t(tau)
    if kind == "spot":
        return (rho * xi * v * tau * b_xw
                + 0.5 * xi ** 2 * v * tau ** 2 * b_ww
                + kap * (tht - v) * tau * b_w)
    b = beta_td_torch(tau, td)
    return rho * xi * v * b * b_xw + 0.5 * xi ** 2 * v * b ** 2 * b_ww


def operator_td(fn, x, v, tau, td: TimeDepHeston):
    """Time-dependent normalized Heston operator L_td[fn] by automatic differentiation."""
    x, v, tau = (t.detach().clone().requires_grad_(True) for t in (x, v, tau))
    f = fn(x, v, tau)
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
        f_xx = torch.zeros_like(x)
        f_xv = torch.zeros_like(v)
    f_vv = cvpinn.grad(f_v, v)
    kap = td.kappa_t(tau)
    tht = td.theta_t(tau)
    xi = td.xi_t(tau)
    rho = td.rho_t(tau)
    return (f_tau - 0.5 * v * (f_xx - f_x) - rho * xi * v * f_xv
            - 0.5 * (xi ** 2) * v * f_vv - kap * (tht - v) * f_v)


def make_correction_td(net, td: TimeDepHeston, dom: cvpinn.Domain, kind="expected", use_z=True):
    def u(x, v, tau):
        feats = cvpinn.base_features(x, v, tau, dom)
        if use_z:
            w_safe = torch.clamp(total_variance_td(v, tau, td, kind), min=1e-30)
            feats.append(torch.tanh(0.5 * x / torch.sqrt(w_safe)))
        return tau * net(feats)
    return u


def make_price_td(u, td: TimeDepHeston, kind="expected"):
    return lambda x, v, tau: cvpinn.bs_call(x, total_variance_td(v, tau, td, kind)) + u(x, v, tau)


def train_cv_td(net, td: TimeDepHeston, dom: cvpinn.Domain, kind="expected",
                adam_steps=10000, lbfgs_iter=500, n_col=2048, lr=1e-3,
                fixed_points=False, attention=False, energy_lock=True,
                seed=0, far_weight=1.0, gamma=0.999, eta=0.01, device="cpu"):
    if attention and not fixed_points:
        raise ValueError("Residual-based attention requires fixed_points=True")
    net.to(device)
    gen = torch.Generator().manual_seed(seed)
    u = make_correction_td(net, td, dom, kind, use_z=True)
    n_strip = n_col // 16
    n_tot = n_col + n_strip

    fixed_int = cvpinn.sample_with_vmin(n_col, n_strip, dom, gen, device=device)
    fixed_edg = cvpinn.sample_edges(n_col // 8, dom, gen, device=device)
    lam = torch.zeros(n_tot, device=device)
    w_eff = torch.ones(n_tot, device=device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=lr * 0.1)
    t_adam0 = time.perf_counter()
    adam_last_loss = float("nan")
    for _ in range(adam_steps):
        x, v, tau = fixed_int if fixed_points else cvpinn.sample_with_vmin(n_col, n_strip, dom, gen, device=device)
        xe, ve, te = fixed_edg if fixed_points else cvpinn.sample_edges(n_col // 8, dom, gen, device=device)
        r = operator_td(u, x, v, tau, td) - source_td(x, v, tau, td, kind)
        if attention:
            with torch.no_grad():
                lam = gamma * lam + eta * r.abs() / torch.clamp(r.abs().max(), min=1e-30)
                if energy_lock:
                    r2_mean = (r ** 2).mean()
                    lam_r2_mean = ((lam * r) ** 2).mean()
                    w_eff = lam * torch.sqrt(r2_mean / (lam_r2_mean + 1e-30))
                else:
                    w_eff = lam
            loss_pde = ((w_eff * r) ** 2).mean()
        else:
            loss_pde = (r ** 2).mean()
        loss = loss_pde + far_weight * (u(xe, ve, te) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        adam_last_loss = float(loss.detach().item())
    adam_sec = time.perf_counter() - t_adam0

    if fixed_points:
        lx, lv, lt = fixed_int
        lxe, lve, lte = fixed_edg
        lw = w_eff.detach() if attention else None
    else:
        lx, lv, lt = cvpinn.sample_with_vmin(8192, 512, dom, gen, device=device)
        lxe, lve, lte = cvpinn.sample_edges(1024, dom, gen, device=device)
        lw = None
    src_cached = source_td(lx, lv, lt, td, kind).detach()

    def lbfgs_loss():
        r_l = operator_td(u, lx, lv, lt, td) - src_cached
        lp = ((lw * r_l) ** 2).mean() if lw is not None else (r_l ** 2).mean()
        return lp + far_weight * (u(lxe, lve, lte) ** 2).mean()

    lbfgs_info = cvpinn._run_lbfgs_safe(net, lbfgs_loss, lbfgs_iter)
    info = {
        "adam_steps": adam_steps, "adam_sec": round(adam_sec, 4),
        "adam_last_loss": adam_last_loss, "far_weight": far_weight,
        "fixed_points": fixed_points, "attention": attention, "energy_lock": energy_lock,
        **{k: (round(v, 4) if k.endswith("_sec") else v) for k, v in lbfgs_info.items()},
    }
    return u, info


def train_baseline_td(net, td: TimeDepHeston, dom: cvpinn.Domain,
                      adam_steps=10000, lbfgs_iter=500, n_col=2048,
                      lr=1e-3, seed=0, term_weight=100.0, far_weight=1.0, device="cpu"):
    net.to(device)
    gen = torch.Generator().manual_seed(seed)
    c = lambda x, v, tau: net(cvpinn.base_features(x, v, tau, dom))
    payoff = lambda x: torch.relu(torch.exp(x) - 1.0)
    n_strip = n_col // 16
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=lr * 0.1)
    t_adam0 = time.perf_counter()
    adam_last_loss = float("nan")
    for _ in range(adam_steps):
        x, v, tau = cvpinn.sample_with_vmin(n_col, n_strip, dom, gen, device=device)
        x0, v0, _ = cvpinn.sample_with_vmin(n_col // 4, 0, dom, gen, device=device)
        xe, ve, te = cvpinn.sample_edges(n_col // 8, dom, gen, device=device)
        loss = (operator_td(c, x, v, tau, td) ** 2).mean()
        loss = loss + term_weight * ((c(x0, v0, torch.zeros_like(x0)) - payoff(x0)) ** 2).mean()
        loss = loss + far_weight * ((c(xe, ve, te) - payoff(xe)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        adam_last_loss = float(loss.detach().item())
    adam_sec = time.perf_counter() - t_adam0

    lx, lv, lt = cvpinn.sample_with_vmin(8192, 512, dom, gen, device=device)
    lx0, lv0, _ = cvpinn.sample_with_vmin(2048, 0, dom, gen, device=device)
    lt0 = torch.zeros_like(lx0)
    pay0 = payoff(lx0).detach()
    lxe, lve, lte = cvpinn.sample_edges(1024, dom, gen, device=device)
    pay_edge = payoff(lxe).detach()

    def lbfgs_loss():
        lp = (operator_td(c, lx, lv, lt, td) ** 2).mean()
        lt_loss = ((c(lx0, lv0, lt0) - pay0) ** 2).mean()
        le_loss = ((c(lxe, lve, lte) - pay_edge) ** 2).mean()
        return lp + term_weight * lt_loss + far_weight * le_loss

    lbfgs_info = cvpinn._run_lbfgs_safe(net, lbfgs_loss, lbfgs_iter)
    info = {
        "adam_steps": adam_steps, "adam_sec": round(adam_sec, 4),
        "adam_last_loss": adam_last_loss, "term_weight": term_weight, "far_weight": far_weight,
        **{k: (round(v, 4) if k.endswith("_sec") else v) for k, v in lbfgs_info.items()},
    }
    return c, info


def build_and_train_td(arm, td: TimeDepHeston, seed, adam_steps=10000, lbfgs_iter=500,
                       term_weight=100.0, v_max=0.5, device="cpu"):
    torch.manual_seed(seed)
    if arm == "B0_td":
        dom = cvpinn.Domain(v_max=v_max)
        net = cvpinn.Mlp(3)
        c, info = train_baseline_td(
            net, td, dom, adam_steps=adam_steps, lbfgs_iter=lbfgs_iter,
            seed=seed, term_weight=term_weight, device=device,
        )
        return c, net, info
    kind = "spot" if arm == "A1_td" else "expected"
    dom = cvpinn.Domain(v_min=1e-3 if kind == "spot" else 0.0, v_max=v_max)
    net = cvpinn.Mlp(4)
    u, info = train_cv_td(
        net, td, dom, kind=kind, adam_steps=adam_steps, lbfgs_iter=lbfgs_iter,
        seed=seed, fixed_points=arm in ("A2a_td", "A2b_td"), attention=arm == "A2b_td",
        device=device,
    )
    return make_price_td(u, td, kind=kind), net, info


def gate2_td(td: TimeDepHeston, dom: cvpinn.Domain, kind="expected", n=4096, seed=0, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    x, v, tau = cvpinn.sample_with_vmin(n, n // 16, dom, gen, device=device)
    lhs = operator_td(lambda a, b, c: cvpinn.bs_call(a, total_variance_td(b, c, td, kind)), x, v, tau, td)
    return float((lhs + source_td(x, v, tau, td, kind)).abs().max().item())


def gate3_td(td: TimeDepHeston, dom: cvpinn.Domain, n=4096, seed=0, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    x, v, tau = cvpinn.sample_with_vmin(n, n // 16, dom, gen, device=device)
    td0 = TimeDepHeston(
        kappa_np=td.kappa_np, theta_np=td.theta_np,
        xi_np=lambda t: 0.0 * t, rho_np=td.rho_np,
        kappa_t=td.kappa_t, theta_t=td.theta_t,
        xi_t=lambda t: torch.zeros_like(t), rho_t=td.rho_t,
        tau_max=td.tau_max, n_cheb=td.n_cheb,
    )
    zero_u = lambda a, b, c: torch.zeros_like(a)
    zero_net = cvpinn.Mlp(4).to(device)
    with torch.no_grad():
        zero_net.net[-1].weight.zero_()
        zero_net.net[-1].bias.zero_()
    zero_corr = make_correction_td(zero_net, td0, dom, kind="expected", use_z=True)
    src_max = source_td(x, v, tau, td0, "expected").abs().max().item()
    res_max = operator_td(zero_u, x, v, tau, td0).abs().max().item()
    net_max = operator_td(zero_corr, x, v, tau, td0).abs().max().item()
    return float(max(src_max, res_max, net_max))


def eval_grid_td(td: TimeDepHeston, taus=hgv.EVAL_TAUS, vs=hgv.EVAL_VS, zs=hgv.EVAL_ZS):
    """Builds canonical (len(taus), len(vs), len(zs)) x-grid via x = z * sqrt(w_td(v, tau))."""
    taus_arr = np.asarray(taus, dtype=np.float64)[:, None, None]
    vs_arr = np.asarray(vs, dtype=np.float64)[None, :, None]
    zs_arr = np.asarray(zs, dtype=np.float64)[None, None, :]
    w = np.maximum(w_td_np(vs_arr, taus_arr, td), 1e-30)
    return zs_arr * np.sqrt(w)

