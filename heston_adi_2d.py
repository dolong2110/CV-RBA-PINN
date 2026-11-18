"""heston_adi_2d.py - 2D Hundsdorfer-Verwer ADI finite-difference solver for time-dependent Heston.

Solves the normalized forward-log-moneyness Heston PDE on [-x_max, x_max] x [0, v_max] x [0, T]:
  c_tau = 0.5*v*(c_xx - c_x) + rho(tau)*xi(tau)*v*c_xv
          + 0.5*xi(tau)^2*v*c_vv + kappa(tau)*(theta(tau) - v)*c_v,
with Ekstrom-Tysk transport c_tau = kappa(tau)*theta(tau)*c_v at v = 0 (valid whether Feller holds
or fails), Rannacher startup + Hundsdorfer-Verwer splitting (In 't Hout & Foulon 2010),
non-uniform sinh grids clustered at x = 0 and v = 0, and cubic spline Greek evaluation.
"""
import math
import time
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import solve_banded

import heston_grid_vec as hgv


def make_sinh_grid(lo, hi, center, c_scale, n):
    """Generates n+1 non-uniform points on [lo, hi] clustered near center."""
    u_lo = math.asinh((lo - center) / c_scale)
    u_hi = math.asinh((hi - center) / c_scale)
    u = np.linspace(u_lo, u_hi, n + 1)
    g = center + c_scale * np.sinh(u)
    g[0] = lo
    g[-1] = hi
    return g


def fd1_nonuniform_coeffs(g):
    """Returns (N+1, 3) 2nd-order weights for f'(g_i): interior central, 3-pt one-sided at edges."""
    n = len(g) - 1
    w = np.zeros((n + 1, 3), dtype=np.float64)
    # Interior central 3-point
    hm = g[1:-1] - g[:-2]
    hp = g[2:] - g[1:-1]
    w[1:-1, 0] = -hp / (hm * (hm + hp))
    w[1:-1, 1] = (hp - hm) / (hm * hp)
    w[1:-1, 2] = hm / (hp * (hm + hp))
    # Left boundary (0, 1, 2)
    h1, h2 = g[1] - g[0], g[2] - g[1]
    w[0, 0] = -(2.0 * h1 + h2) / (h1 * (h1 + h2))
    w[0, 1] = (h1 + h2) / (h1 * h2)
    w[0, 2] = -h1 / (h2 * (h1 + h2))
    # Right boundary (n-2, n-1, n)
    hm1, hm2 = g[-1] - g[-2], g[-2] - g[-3]
    w[-1, 0] = hm1 / (hm2 * (hm1 + hm2))
    w[-1, 1] = -(hm1 + hm2) / (hm1 * hm2)
    w[-1, 2] = (2.0 * hm1 + hm2) / (hm1 * (hm1 + hm2))
    return w


def fd2_nonuniform_coeffs(g):
    """Returns (N+1, 3) 2nd-order weights for f''(g_i) on interior points (0 at edges)."""
    n = len(g) - 1
    w = np.zeros((n + 1, 3), dtype=np.float64)
    hm = g[1:-1] - g[:-2]
    hp = g[2:] - g[1:-1]
    w[1:-1, 0] = 2.0 / (hm * (hm + hp))
    w[1:-1, 1] = -2.0 / (hm * hp)
    w[1:-1, 2] = 2.0 / (hp * (hm + hp))
    return w


def apply_fd1_x(U, wx):
    """Applies 1D derivative along axis 1 (interior cols 1..Nx-1 only)."""
    out = np.zeros_like(U)
    out[:, 1:-1] = (
        wx[1:-1, 0][None, :] * U[:, :-2]
        + wx[1:-1, 1][None, :] * U[:, 1:-1]
        + wx[1:-1, 2][None, :] * U[:, 2:]
    )
    return out


def apply_fd2_x(U, wxx):
    out = np.zeros_like(U)
    out[:, 1:-1] = (
        wxx[1:-1, 0][None, :] * U[:, :-2]
        + wxx[1:-1, 1][None, :] * U[:, 1:-1]
        + wxx[1:-1, 2][None, :] * U[:, 2:]
    )
    return out


def apply_fd1_v(U, wv):
    """Applies 1D derivative along axis 0 (including 3-pt one-sided j=0 and j=Nv)."""
    out = np.zeros_like(U)
    out[1:-1] = (
        wv[1:-1, 0][:, None] * U[:-2]
        + wv[1:-1, 1][:, None] * U[1:-1]
        + wv[1:-1, 2][:, None] * U[2:]
    )
    out[0] = wv[0, 0] * U[0] + wv[0, 1] * U[1] + wv[0, 2] * U[2]
    out[-1] = wv[-1, 0] * U[-3] + wv[-1, 1] * U[-2] + wv[-1, 2] * U[-1]
    return out


def apply_fd2_v(U, wvv):
    out = np.zeros_like(U)
    out[1:-1] = (
        wvv[1:-1, 0][:, None] * U[:-2]
        + wvv[1:-1, 1][:, None] * U[1:-1]
        + wvv[1:-1, 2][:, None] * U[2:]
    )
    return out


def _solve_tridiag_batch(a, b, c, rhs):
    """Batched Thomas solve for M independent systems of size N: rhs shape (M, N)."""
    m, n = rhs.shape
    c_p = np.empty((m, n), dtype=np.float64)
    d_p = np.empty((m, n), dtype=np.float64)
    c_p[:, 0] = c[:, 0] / b[:, 0]
    d_p[:, 0] = rhs[:, 0] / b[:, 0]
    for i in range(1, n):
        denom = b[:, i] - a[:, i] * c_p[:, i - 1]
        if i < n - 1:
            c_p[:, i] = c[:, i] / denom
        d_p[:, i] = (rhs[:, i] - a[:, i] * d_p[:, i - 1]) / denom
    x = np.empty((m, n), dtype=np.float64)
    x[:, -1] = d_p[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = d_p[:, i] - c_p[:, i] * x[:, i + 1]
    return x


def _cell_averaged_payoff(x):
    """Cell-averaged terminal payoff max(exp(x) - 1, 0) on non-uniform grid x."""
    u = np.maximum(np.exp(x) - 1.0, 0.0)
    # Replace the single kink node x_i nearest 0 with its exact cell integral over [x_L, x_R]
    i0 = int(np.argmin(np.abs(x)))
    if 0 < i0 < len(x) - 1:
        x_l = 0.5 * (x[i0 - 1] + x[i0])
        x_r = 0.5 * (x[i0] + x[i0 + 1])
        # Integral of max(exp(s) - 1, 0) from x_l to x_r divided by (x_r - x_l)
        a = max(x_l, 0.0)
        b = max(x_r, 0.0)
        if b > a:
            u[i0] = ((math.exp(b) - math.exp(a)) - (b - a)) / (x_r - x_l)
        else:
            u[i0] = 0.0
    return u


def _build_v_banded(v, wv, wvv, kappa, theta, xi, factor):
    """Builds (I - factor * A2) in LAPACK banded format (l=2, u=2) over all Nv+1 rows."""
    nv = len(v)
    # Row coeffs on [j-2, j-1, j, j+1, j+2]
    ab = np.zeros((5, nv), dtype=np.float64)
    # Diagonal entries (row 2 in ab)
    ab[2, :] = 1.0
    # Interior 1..nv-2: A2 = 0.5*xi^2*v*Dvv + kappa*(theta - v)*Dv
    diff = 0.5 * xi * xi * v[1:-1]
    drift = kappa * (theta - v[1:-1])
    c_m = diff * wvv[1:-1, 0] + drift * wv[1:-1, 0]
    c_0 = diff * wvv[1:-1, 1] + drift * wv[1:-1, 1]
    c_p = diff * wvv[1:-1, 2] + drift * wv[1:-1, 2]
    idx = np.arange(1, nv - 1)
    ab[3, idx - 1] = -factor * c_m   # subdiagonal (row 2 + 1) -> stored at col j-1
    ab[2, idx] = 1.0 - factor * c_0  # diagonal
    ab[1, idx + 1] = -factor * c_p   # superdiagonal (row 2 - 1) -> stored at col j+1

    # j = 0 (Ekstrom-Tysk: kappa*theta*Dv on [0, 1, 2])
    kt = kappa * theta
    ab[2, 0] = 1.0 - factor * kt * wv[0, 0]
    ab[1, 1] = -factor * kt * wv[0, 1]
    ab[0, 2] = -factor * kt * wv[0, 2]

    # j = nv-1 (top boundary v_max: kappa*(theta - v_max)*Dv on [nv-3, nv-2, nv-1], cvv=0)
    dr_top = kappa * (theta - v[-1])
    ab[4, nv - 3] = -factor * dr_top * wv[-1, 0]
    ab[3, nv - 2] = -factor * dr_top * wv[-1, 1]
    ab[2, nv - 1] = 1.0 - factor * dr_top * wv[-1, 2]
    return ab


def solve_adi_heston(param_fn, snapshot_taus, nx=400, nv=200, n_steps=800,
                     x_max=4.0, v_max=1.0, cx=0.08, cv=0.02, rannacher_steps=2):
    """Solves normalized Heston PDE up to max(snapshot_taus) using Hundsdorfer-Verwer ADI.

    Args:
      param_fn: callable(tau) -> (kappa, theta, xi, rho)
      snapshot_taus: sorted sequence of maturities > 0 to snapshot (must align with dt grid)
    Returns:
      x (nx+1,), v (nv+1,), snapshots dict {tau: U(nv+1, nx+1)}
    """
    t_max = float(max(snapshot_taus))
    dt = t_max / n_steps
    # Verify every snapshot hits an exact step
    snap_steps = {}
    for t_s in snapshot_taus:
        k_step = int(round(float(t_s) / dt))
        if abs(k_step * dt - float(t_s)) > 1e-11:
            raise ValueError(f"Snapshot tau={t_s} not aligned with dt={dt} (n_steps={n_steps})")
        snap_steps[k_step] = float(t_s)

    x = make_sinh_grid(-x_max, x_max, 0.0, cx, nx)
    v = make_sinh_grid(0.0, v_max, 0.0, cv, nv)
    wx = fd1_nonuniform_coeffs(x)
    wxx = fd2_nonuniform_coeffs(x)
    wv = fd1_nonuniform_coeffs(v)
    wvv = fd2_nonuniform_coeffs(v)

    # Precompute tridiagonal entries for A1 on interior x (cols 1..nx-1) across all v
    # A1[j, i] = 0.5*v[j]*(wxx[i] - wx[i])
    lx_a = 0.5 * v[:, None] * (wxx[1:-1, 0] - wx[1:-1, 0])[None, :]
    lx_b = 0.5 * v[:, None] * (wxx[1:-1, 1] - wx[1:-1, 1])[None, :]
    lx_c = 0.5 * v[:, None] * (wxx[1:-1, 2] - wx[1:-1, 2])[None, :]

    def apply_a1(U):
        return 0.5 * v[:, None] * (apply_fd2_x(U, wxx) - apply_fd1_x(U, wx))

    def apply_a2(U, kappa, theta, xi):
        d1v = apply_fd1_v(U, wv)
        d2v = apply_fd2_v(U, wvv)
        out = 0.5 * xi * xi * v[:, None] * d2v + kappa * (theta - v[:, None]) * d1v
        out[:, 0] = 0.0
        out[:, -1] = 0.0
        return out

    def apply_a0(U, xi, rho):
        d1x = apply_fd1_x(U, wx)
        dxv = apply_fd1_v(d1x, wv)
        out = rho * xi * v[:, None] * dxv
        out[0, :] = 0.0
        out[:, 0] = 0.0
        out[:, -1] = 0.0
        return out

    U = np.broadcast_to(_cell_averaged_payoff(x)[None, :], (nv + 1, nx + 1)).copy()
    theta_hv = 0.5 + math.sqrt(3.0) / 6.0
    snapshots = {}

    def step_hv(U_cur, t0, h, th):
        kap0, tht0, xi0, rho0 = param_fn(t0)
        kap1, tht1, xi1, rho1 = param_fn(t0 + h)
        a1_u = apply_a1(U_cur)
        a2_u0 = apply_a2(U_cur, kap0, tht0, xi0)
        a0_u0 = apply_a0(U_cur, xi0, rho0)
        a_u0 = a0_u0 + a1_u + a2_u0

        # Predictor
        Y0 = U_cur + h * a_u0
        # Delta solve in x: (I - th*h*A1)(Y1_int - U_int) = Y0_int - U_int
        fac = th * h
        tri_a = -fac * lx_a
        tri_b = 1.0 - fac * lx_b
        tri_c = -fac * lx_c
        dY1_int = _solve_tridiag_batch(tri_a, tri_b, tri_c, (Y0 - U_cur)[:, 1:-1])
        Y1 = U_cur.copy()
        Y1[:, 1:-1] += dY1_int

        # Implicit v solve: (I - th*h*A2(t1)) Y2_int = Y1_int - th*h*A2(t0) U_int
        ab_v1 = _build_v_banded(v, wv, wvv, kap1, tht1, xi1, fac)
        rhs_v = (Y1 - fac * a2_u0)[:, 1:-1]
        Y2 = U_cur.copy()
        Y2[:, 1:-1] = solve_banded((2, 2), ab_v1, rhs_v)

        if th >= 0.999999:  # Pure Douglas implicit step (Rannacher startup)
            return Y2

        # Corrector
        a_y2 = apply_a0(Y2, xi1, rho1) + apply_a1(Y2) + apply_a2(Y2, kap1, tht1, xi1)
        Y0_t = Y0 + 0.5 * h * (a_y2 - a_u0)
        dY1_t_int = _solve_tridiag_batch(tri_a, tri_b, tri_c, (Y0_t - Y2)[:, 1:-1])
        Y1_t = Y2.copy()
        Y1_t[:, 1:-1] += dY1_t_int

        a2_y2 = apply_a2(Y2, kap1, tht1, xi1)
        rhs_v2 = (Y1_t - fac * a2_y2)[:, 1:-1]
        U_next = U_cur.copy()
        U_next[:, 1:-1] = solve_banded((2, 2), ab_v1, rhs_v2)
        return U_next

    for step_idx in range(1, n_steps + 1):
        t0 = (step_idx - 1) * dt
        if step_idx <= rannacher_steps:
            # Split into 2 half-steps of implicit Douglas (theta=1) to damp initial kink
            U = step_hv(U, t0, 0.5 * dt, 1.0)
            U = step_hv(U, t0 + 0.5 * dt, 0.5 * dt, 1.0)
        else:
            U = step_hv(U, t0, dt, theta_hv)
        if step_idx in snap_steps:
            snapshots[snap_steps[step_idx]] = U.copy()

    return x, v, snapshots


def eval_adi_on_grid(param_fn, x_eval_grid, taus=hgv.EVAL_TAUS, vs=hgv.EVAL_VS,
                     nx=400, nv=200, n_steps=800):
    """Solves PDE by ADI and evaluates c, cx, gam (c_xx - c_x), cxv on 3D (tau, v, z) grid."""
    t0 = time.perf_counter()
    x_grid, v_grid, snaps = solve_adi_heston(
        param_fn, taus, nx=nx, nv=nv, n_steps=n_steps
    )
    out = {k: np.zeros(x_eval_grid.shape, dtype=np.float64) for k in ("c", "cx", "gam", "cxv")}
    for it, tau in enumerate(taus):
        U = snaps[float(tau)]
        # Degree-3 bivariate spline gives smooth C^2 interpolation & exact analytic partial derivatives
        spl = RectBivariateSpline(v_grid, x_grid, U, kx=3, ky=3)
        for iv, vv in enumerate(vs):
            xs_row = x_eval_grid[it, iv, :]
            vv_arr = np.full_like(xs_row, float(vv))
            c_val = spl(vv_arr, xs_row, dx=0, dy=0, grid=False)
            cx_val = spl(vv_arr, xs_row, dx=0, dy=1, grid=False)
            cxx_val = spl(vv_arr, xs_row, dx=0, dy=2, grid=False)
            cxv_val = spl(vv_arr, xs_row, dx=1, dy=1, grid=False)
            out["c"][it, iv, :] = c_val
            out["cx"][it, iv, :] = cx_val
            out["gam"][it, iv, :] = cxx_val - cx_val
            out["cxv"][it, iv, :] = cxv_val
    return out, time.perf_counter() - t0
