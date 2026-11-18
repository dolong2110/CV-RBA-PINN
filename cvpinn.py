"""cvpinn.py - Hardened control-variate PINN for the normalized Heston PDE.
DRAFT: Syntax-checked only (PyTorch was not installed in the audit environment).
Run gate2() and gate3() first; both must pass before any training run.

Resolves audit items D9, D13, D14, D15, and D16:
  - D9:  x_max = 2.5; N_col/16 boundary collocation strip at v = v_min; free v = v_max
  - D13: Energy-locked RBA (lam^(0) = 0, E[(w_eff * r)^2] == E[r^2] at every step)
  - D14: Cancellation-free beta(tau) and 4-term Taylor series for (tau - beta)
  - D15: Pre-branch w_safe clamp (no 0/0 autograd NaN) + |d2| <= 38 underflow guard
  - D16: Two-stage Cosine-Adam + frozen-batch Strong-Wolfe L-BFGS for ALL arms (incl. B0)
"""
import dataclasses
import math
import time
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
SQRT2PI = math.sqrt(2.0 * math.pi)


@dataclasses.dataclass
class Heston:
    kappa: float
    theta: float
    xi: float
    rho: float


@dataclasses.dataclass
class Domain:
    x_max: float = 2.5       # D9: widened from 1.5 (cuts lateral mismatch 14.2x on P2)
    v_min: float = 0.0       # 0.0 for B0/M1/A2a/A2b; 1e-3 for spot-variance arm A1
    v_max: float = 0.5       # Default 0.5; check sensitivity at v_max = 0.75 on Day 4
    tau_min: float = 1e-4
    tau_max: float = 2.0


def npdf(d):
    return torch.exp(-0.5 * d * d) / SQRT2PI


def ncdf(d):
    return 0.5 * torch.erfc(-d / math.sqrt(2.0))


def beta(tau, p):
    """D14 fix: -expm1(-kappa*tau)/kappa avoids small-tau subtractive cancellation."""
    return -torch.expm1(-p.kappa * tau) / p.kappa


def total_variance(v, tau, p, kind):
    """D14 fix: 4-term alternating Taylor series for (tau - beta) when kappa*tau < 1e-3."""
    if kind == "spot":
        return v * tau
    kt = p.kappa * tau
    b = beta(tau, p)
    tau_minus_b = torch.where(
        kt < 1e-3,
        tau * (0.5 * kt - (kt ** 2) / 6.0 + (kt ** 3) / 24.0 - (kt ** 4) / 120.0),
        (kt + torch.expm1(-kt)) / p.kappa,
    )
    return p.theta * tau_minus_b + v * b


def bs_call(x, w):
    """D15 fix: clamp w BEFORE sqrt so torch.where backward never sees 0/0 NaNs."""
    w_safe = torch.clamp(w, min=1e-30)
    s = torch.sqrt(w_safe)
    d1 = x / s + 0.5 * s
    val = torch.exp(x) * ncdf(d1) - ncdf(d1 - s)
    return torch.where(w <= 1e-30, torch.relu(torch.exp(x) - 1.0), val)


def source(x, v, tau, p, kind):
    """Closed-form F = -L[b] with D15 safe w clamp and |d2| <= 38 Gaussian guard."""
    w_safe = torch.clamp(total_variance(v, tau, p, kind), min=1e-30)
    s = torch.sqrt(w_safe)
    d1 = x / s + 0.5 * s
    d2 = d1 - s
    n_d2 = torch.where(torch.abs(d2) <= 38.0, npdf(d2), torch.zeros_like(d2))
    b_w = n_d2 / (2.0 * s)
    b_xw = -d2 * n_d2 / (2.0 * w_safe)
    b_ww = n_d2 * (d1 * d2 - 1.0) / (4.0 * w_safe * s)
    if kind == "spot":
        return (p.rho * p.xi * v * tau * b_xw
                + 0.5 * p.xi ** 2 * v * tau ** 2 * b_ww
                + p.kappa * (p.theta - v) * tau * b_w)
    b = beta(tau, p)
    return p.rho * p.xi * v * b * b_xw + 0.5 * p.xi ** 2 * v * b ** 2 * b_ww


class Mlp(nn.Module):
    def __init__(self, n_in, width=64, depth=4):
        super().__init__()
        layers, n = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.Tanh()]
            n = width
        layers.append(nn.Linear(n, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feats):
        return self.net(torch.stack(feats, dim=-1)).squeeze(-1)


def base_features(x, v, tau, dom):
    return [x / dom.x_max, 2.0 * v / dom.v_max - 1.0, 2.0 * tau / dom.tau_max - 1.0]


def make_correction(net, p, dom, kind, use_z=True):
    """Returns u(x, v, tau) = tau * net(...), vanishing identically at tau = 0."""
    def u(x, v, tau):
        feats = base_features(x, v, tau, dom)
        if use_z:
            w_safe = torch.clamp(total_variance(v, tau, p, kind), min=1e-30)
            feats.append(torch.tanh(0.5 * x / torch.sqrt(w_safe)))
        return tau * net(feats)
    return u


def make_price(u, p, kind):
    return lambda x, v, tau: bs_call(x, total_variance(v, tau, p, kind)) + u(x, v, tau)


def grad(y, x):
    if not y.requires_grad:
        return torch.zeros_like(x)
    g = torch.autograd.grad(
        y, x, torch.ones_like(y), create_graph=True, allow_unused=True
    )[0]
    return torch.zeros_like(x) if g is None else g


def operator(fn, x, v, tau, p):
    """L[fn] evaluated by automatic differentiation (reduces to fn_tau - kappa*theta*fn_v at v=0)."""
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
    f_vv = grad(f_v, v)
    return (f_tau - 0.5 * v * (f_xx - f_x) - p.rho * p.xi * v * f_xv
            - 0.5 * p.xi ** 2 * v * f_vv - p.kappa * (p.theta - v) * f_v)


def sample_with_vmin(n_int, n_strip, dom, gen, device="cpu"):
    """Samples n_int interior points + n_strip boundary points at v = dom.v_min (D9)."""
    x = dom.x_max * (2.0 * torch.rand(n_int, generator=gen) - 1.0)
    v = dom.v_min + (dom.v_max - dom.v_min) * torch.rand(n_int, generator=gen) ** 2
    tau = dom.tau_min + (dom.tau_max - dom.tau_min) * torch.rand(n_int, generator=gen) ** 2
    if n_strip > 0:
        xs = dom.x_max * (2.0 * torch.rand(n_strip, generator=gen) - 1.0)
        vs = torch.full((n_strip,), dom.v_min)
        ts = dom.tau_min + (dom.tau_max - dom.tau_min) * torch.rand(n_strip, generator=gen) ** 2
        x, v, tau = torch.cat([x, xs]), torch.cat([v, vs]), torch.cat([tau, ts])
    return x.to(device), v.to(device), tau.to(device)


def sample_edges(n_edge, dom, gen, device="cpu"):
    """Samples lateral edge coordinates x = +-x_max."""
    _, v, tau = sample_with_vmin(n_edge, 0, dom, gen, device="cpu")
    side = 2.0 * (torch.rand(n_edge, generator=gen) < 0.5).to(torch.get_default_dtype()) - 1.0
    x = dom.x_max * side
    return x.to(device), v.to(device), tau.to(device)


def _snapshot_state(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _run_lbfgs_safe(net, loss_fn, lbfgs_iter):
    """Runs Strong-Wolfe L-BFGS on a frozen batch and rolls back if non-finite or worse."""
    best_state = _snapshot_state(net)
    with torch.enable_grad():
        pre_loss_t = loss_fn()
    pre_loss = float(pre_loss_t.detach().item())
    best_loss = pre_loss if math.isfinite(pre_loss) else float("inf")
    info = {
        "lbfgs_pre_loss": pre_loss,
        "lbfgs_post_loss": pre_loss,
        "lbfgs_best_loss": best_loss,
        "lbfgs_evals": 0,
        "lbfgs_iters": 0,
        "lbfgs_rolled_back": False,
        "lbfgs_sec": 0.0,
    }
    if lbfgs_iter <= 0:
        return info

    t0 = time.perf_counter()
    lbfgs = torch.optim.LBFGS(
        net.parameters(), lr=1.0, max_iter=lbfgs_iter,
        history_size=50, tolerance_grad=1e-11, tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        nonlocal best_loss, best_state
        lbfgs.zero_grad()
        tot = loss_fn()
        tot_val = float(tot.detach().item())
        info["lbfgs_evals"] += 1
        if math.isfinite(tot_val):
            params_ok = all(torch.isfinite(param).all().item() for param in net.parameters())
            if params_ok and tot_val < best_loss:
                best_loss = tot_val
                best_state = _snapshot_state(net)
        tot.backward()
        return tot

    lbfgs.step(closure)
    info["lbfgs_sec"] = time.perf_counter() - t0
    state_dict = lbfgs.state.get(next(iter(net.parameters())), {})
    info["lbfgs_iters"] = int(state_dict.get("n_iter", 0))

    params_finite = all(torch.isfinite(param).all().item() for param in net.parameters())
    if params_finite:
        with torch.enable_grad():
            post_loss = float(loss_fn().detach().item())
    else:
        post_loss = float("nan")
    if (not math.isfinite(post_loss)) or post_loss > best_loss:
        net.load_state_dict(best_state)
        info["lbfgs_rolled_back"] = True
        post_loss = best_loss

    info["lbfgs_post_loss"] = post_loss
    info["lbfgs_best_loss"] = best_loss
    return info


def train_cv_two_stage(net, p, dom, kind, adam_steps=10000, lbfgs_iter=500, n_col=2048,
                       lr=1e-3, fixed_points=False, attention=False, energy_lock=True,
                       seed=0, far_weight=1.0, gamma=0.999, eta=0.01, use_z=True,
                       device="cpu", return_info=False):
    """Trains M1, A1, A2a, or A2b with Cosine-Adam (Stage 1) + Strong-Wolfe L-BFGS (Stage 2)."""
    if attention and not fixed_points:
        raise ValueError("Residual-based attention requires fixed_points=True")
    net.to(device)
    gen = torch.Generator().manual_seed(seed)
    u = make_correction(net, p, dom, kind, use_z)
    n_strip = n_col // 16
    n_tot = n_col + n_strip

    fixed_int = sample_with_vmin(n_col, n_strip, dom, gen, device=device)
    fixed_edg = sample_edges(n_col // 8, dom, gen, device=device)
    lam = torch.zeros(n_tot, device=device)
    w_eff = torch.ones(n_tot, device=device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=lr * 0.1)

    t_adam0 = time.perf_counter()
    adam_last_loss = float("nan")
    for _ in range(adam_steps):
        x, v, tau = fixed_int if fixed_points else sample_with_vmin(n_col, n_strip, dom, gen, device=device)
        xe, ve, te = fixed_edg if fixed_points else sample_edges(n_col // 8, dom, gen, device=device)
        r = operator(u, x, v, tau, p) - source(x, v, tau, p, kind)
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
        lx, lv, lt = sample_with_vmin(8192, 512, dom, gen, device=device)
        lxe, lve, lte = sample_edges(1024, dom, gen, device=device)
        lw = None

    src_cached = source(lx, lv, lt, p, kind).detach()

    def lbfgs_loss():
        r_l = operator(u, lx, lv, lt, p) - src_cached
        lp = ((lw * r_l) ** 2).mean() if lw is not None else (r_l ** 2).mean()
        return lp + far_weight * (u(lxe, lve, lte) ** 2).mean()

    lbfgs_info = _run_lbfgs_safe(net, lbfgs_loss, lbfgs_iter)
    if return_info:
        info = {
            "adam_steps": adam_steps,
            "adam_sec": round(adam_sec, 4),
            "adam_last_loss": adam_last_loss,
            "far_weight": far_weight,
            "fixed_points": fixed_points,
            "attention": attention,
            "energy_lock": energy_lock,
            **{k: (round(v, 4) if k.endswith("_sec") else v) for k, v in lbfgs_info.items()},
        }
        return u, info
    return u


def train_baseline_two_stage(net, p, dom, adam_steps=10000, lbfgs_iter=500, n_col=2048,
                             lr=1e-3, seed=0, term_weight=10.0, far_weight=1.0, device="cpu",
                             return_info=False):
    """Arm B0 with full optimization parity: v=0 strip, Cosine-Adam, and frozen-batch L-BFGS."""
    net.to(device)
    gen = torch.Generator().manual_seed(seed)
    c = lambda x, v, tau: net(base_features(x, v, tau, dom))
    payoff = lambda x: torch.relu(torch.exp(x) - 1.0)
    n_strip = n_col // 16

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(adam_steps, 1), eta_min=lr * 0.1)

    t_adam0 = time.perf_counter()
    adam_last_loss = float("nan")
    for _ in range(adam_steps):
        x, v, tau = sample_with_vmin(n_col, n_strip, dom, gen, device=device)
        x0, v0, _ = sample_with_vmin(n_col // 4, 0, dom, gen, device=device)
        xe, ve, te = sample_edges(n_col // 8, dom, gen, device=device)
        loss = (operator(c, x, v, tau, p) ** 2).mean()
        loss = loss + term_weight * ((c(x0, v0, torch.zeros_like(x0)) - payoff(x0)) ** 2).mean()
        loss = loss + far_weight * ((c(xe, ve, te) - payoff(xe)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        adam_last_loss = float(loss.detach().item())
    adam_sec = time.perf_counter() - t_adam0

    lx, lv, lt = sample_with_vmin(8192, 512, dom, gen, device=device)
    lx0, lv0, _ = sample_with_vmin(2048, 0, dom, gen, device=device)
    lt0 = torch.zeros_like(lx0)
    pay0 = payoff(lx0).detach()
    lxe, lve, lte = sample_edges(1024, dom, gen, device=device)
    pay_edge = payoff(lxe).detach()

    def lbfgs_loss():
        lp = (operator(c, lx, lv, lt, p) ** 2).mean()
        lt_loss = ((c(lx0, lv0, lt0) - pay0) ** 2).mean()
        le_loss = ((c(lxe, lve, lte) - pay_edge) ** 2).mean()
        return lp + term_weight * lt_loss + far_weight * le_loss

    lbfgs_info = _run_lbfgs_safe(net, lbfgs_loss, lbfgs_iter)
    if return_info:
        info = {
            "adam_steps": adam_steps,
            "adam_sec": round(adam_sec, 4),
            "adam_last_loss": adam_last_loss,
            "term_weight": term_weight,
            "far_weight": far_weight,
            **{k: (round(v, 4) if k.endswith("_sec") else v) for k, v in lbfgs_info.items()},
        }
        return c, info
    return c


def build_and_train(arm, p, seed, adam_steps=10000, lbfgs_iter=500, term_weight=10.0,
                    v_max=0.5, device="cpu", return_info=False):
    """Trains one arm ("B0", "M1", "A1", "A2a", "A2b") and returns price_fn(x, v, tau)."""
    torch.manual_seed(seed)
    if arm == "B0":
        dom = Domain(v_max=v_max)
        net = Mlp(3)
        c, info = train_baseline_two_stage(
            net, p, dom, adam_steps=adam_steps, lbfgs_iter=lbfgs_iter,
            seed=seed, term_weight=term_weight, device=device, return_info=True,
        )
        return (c, net, info) if return_info else c
    kind = "spot" if arm == "A1" else "expected"
    dom = Domain(v_min=1e-3 if kind == "spot" else 0.0, v_max=v_max)
    net = Mlp(4)
    u, info = train_cv_two_stage(
        net, p, dom, kind, adam_steps=adam_steps, lbfgs_iter=lbfgs_iter,
        seed=seed, fixed_points=arm in ("A2a", "A2b"), attention=arm == "A2b",
        device=device, return_info=True,
    )
    price_fn = make_price(u, p, kind)
    return (price_fn, net, info) if return_info else price_fn


def make_price_from_net(arm, net, p, v_max=0.5):
    """Reconstructs price_fn(x, v, tau) from a loaded Mlp checkpoint."""
    if arm == "B0":
        dom = Domain(v_max=v_max)
        return lambda x, v, tau: net(base_features(x, v, tau, dom))
    kind = "spot" if arm == "A1" else "expected"
    dom = Domain(v_min=1e-3 if kind == "spot" else 0.0, v_max=v_max)
    u = make_correction(net, p, dom, kind, use_z=True)
    return make_price(u, p, kind)


def greeks(price_fn, x, v, tau):
    """Returns detached c, c_x, normalized Gamma (c_xx - c_x), and Vanna c_xv."""
    x, v, tau = (t.detach().clone().requires_grad_(True) for t in (x, v, tau))
    c = price_fn(x, v, tau)
    c_x = grad(c, x)
    if c_x.requires_grad:
        c_xx, c_xv = torch.autograd.grad(
            c_x, (x, v), torch.ones_like(c_x), create_graph=False, allow_unused=True
        )
        c_xx = torch.zeros_like(x) if c_xx is None else c_xx
        c_xv = torch.zeros_like(v) if c_xv is None else c_xv
    else:
        c_xx = torch.zeros_like(x)
        c_xv = torch.zeros_like(v)
    gam = c_xx - c_x
    return c.detach(), c_x.detach(), gam.detach(), c_xv.detach()


def evaluate_on_grid(price_fn, x_grid_np, taus, vs, device="cpu"):
    """Evaluates a trained PyTorch price_fn on the 875-pt grid and returns NumPy arrays."""
    shape = x_grid_np.shape
    t_arr = np.broadcast_to(np.asarray(taus, dtype=np.float64)[:, None, None], shape).ravel()
    v_arr = np.broadcast_to(np.asarray(vs, dtype=np.float64)[None, :, None], shape).ravel()
    x_arr = np.asarray(x_grid_np, dtype=np.float64).ravel()
    c, cx, gam, cxv = greeks(
        price_fn,
        torch.tensor(x_arr, device=device),
        torch.tensor(v_arr, device=device),
        torch.tensor(t_arr, device=device),
    )
    return {
        "c": c.cpu().numpy().reshape(shape),
        "cx": cx.cpu().numpy().reshape(shape),
        "gam": gam.cpu().numpy().reshape(shape),
        "cxv": cxv.cpu().numpy().reshape(shape),
    }


def gate2(p, dom, kind, n=4096, seed=0, device="cpu"):
    """Gate G2: operator identity max |L[b] + F| by PyTorch automatic differentiation."""
    gen = torch.Generator().manual_seed(seed)
    x, v, tau = sample_with_vmin(n, n // 16, dom, gen, device=device)
    lhs = operator(lambda a, b, c: bs_call(a, total_variance(b, c, p, kind)), x, v, tau, p)
    return (lhs + source(x, v, tau, p, kind)).abs().max().item()


def gate3(p, dom, n=4096, seed=0, device="cpu"):
    """Gate G3: expected-variance source and zero-network residual vanish identically at xi = 0."""
    gen = torch.Generator().manual_seed(seed)
    x, v, tau = sample_with_vmin(n, n // 16, dom, gen, device=device)
    p0 = Heston(p.kappa, p.theta, 0.0, p.rho)
    zero_u = lambda a, b, c: torch.zeros_like(a)
    zero_net = Mlp(4).to(device)
    with torch.no_grad():
        zero_net.net[-1].weight.zero_()
        zero_net.net[-1].bias.zero_()
    zero_corr = make_correction(zero_net, p0, dom, "expected", use_z=True)
    src_max = source(x, v, tau, p0, "expected").abs().max().item()
    res_max = operator(zero_u, x, v, tau, p0).abs().max().item()
    net_max = operator(zero_corr, x, v, tau, p0).abs().max().item()
    return max(src_max, res_max, net_max)


if __name__ == "__main__":
    P2 = Heston(1.5, 0.04, 0.5, -0.7)
    print("gate2 expected:", gate2(P2, Domain(), "expected"))
    print("gate2 spot    :", gate2(P2, Domain(v_min=1e-3), "spot"))
    print("gate3         :", gate3(P2, Domain()))
