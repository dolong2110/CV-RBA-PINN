"""heston_grid_vec.py - Vectorized Lewis-Albrecher Fourier grid evaluator & 6-bucket scorer.
DRAFT: Syntax-checked only in this environment. On Day 1, run verify_against_scalar()
against heston_ref.py on P1 and P2 (require max abs diff <= 1e-11 across all 6 outputs).
"""
import math
import numpy as np

GLX, GLW = np.polynomial.legendre.leggauss(16)
EVAL_TAUS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
EVAL_VS = (0.005, 0.01, 0.04, 0.09, 0.16)
EVAL_ZS = tuple(np.linspace(-3.0, 3.0, 25).tolist())


def w_exp_stable(v, tau, kappa, theta):
    """Cancellation-free expected integrated variance w_exp(v, tau) (audit item D14)."""
    kt = kappa * tau
    b = -np.expm1(-kt) / kappa
    tau_minus_b = np.where(
        kt < 1e-3,
        tau * (0.5 * kt - (kt ** 2) / 6.0 + (kt ** 3) / 24.0 - (kt ** 4) / 120.0),
        (kt + np.expm1(-kt)) / kappa,
    )
    return theta * tau_minus_b + np.asarray(v, dtype=np.float64) * b


def heston_eval_grid(p, taus=EVAL_TAUS, vs=EVAL_VS, zs=EVAL_ZS, tol=1e-16, max_u=4e5):
    """Computes c, cx, gam, cv, cxv, cvv on the (len(taus), len(vs), len(zs)) evaluation grid."""
    kappa, theta, xi, rho = p
    n_t, n_v, n_z = len(taus), len(vs), len(zs)
    vs_arr = np.asarray(vs, dtype=np.float64)
    zs_arr = np.asarray(zs, dtype=np.float64)
    out = {k: np.zeros((n_t, n_v, n_z), dtype=np.float64)
           for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
    x_grid = np.zeros((n_t, n_v, n_z), dtype=np.float64)
    v_min = float(np.min(vs_arr))

    for it, tau in enumerate(taus):
        w_vec = w_exp_stable(vs_arr, tau, kappa, theta)
        x_mat = np.sqrt(w_vec)[:, None] * zs_arr[None, :]
        x_grid[it] = x_mat
        wide = min(2.0, math.pi / (2.0 * max(float(np.max(np.abs(x_mat))), 1e-3)))
        p_near = min(0.5, wide)
        n_near = int(math.ceil(4.0 / p_near))

        sums = np.zeros((6, n_v, n_z), dtype=np.float64)
        u_cursor, first = 0.0, True
        while u_cursor < max_u:
            if first:
                first = False
                starts = np.arange(n_near) * p_near
                widths = np.full(n_near, p_near)
                u_cursor = n_near * p_near
            else:
                starts = u_cursor + np.arange(64) * wide
                widths = np.full(64, wide)
                u_cursor += 64 * wide

            u_nodes = (starts[:, None] + 0.5 * widths[:, None] * (GLX[None, :] + 1.0)).ravel()
            w_nodes = (0.5 * widths[:, None] * GLW[None, :]).ravel()

            z_cf = u_nodes - 0.5j
            a = kappa - 1j * rho * xi * z_cf
            d = np.sqrt(a * a + (xi * xi) * (z_cf * z_cf + 1j * z_cf))
            g = (a - d) / (a + d)
            e = np.exp(-d * tau)
            big_b = (a - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e)
            big_a = (kappa * theta / (xi * xi)) * (
                (a - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g))
            )

            env = np.abs(np.exp(big_a + big_b * v_min)) * (1.0 + np.abs(big_b)) ** 2
            phi = (np.exp(big_a[:, None] + big_b[:, None] * vs_arr[None, :])[:, :, None]
                   * np.exp(1j * u_nodes[:, None, None] * x_mat[None, :, :]))
            den = (u_nodes * u_nodes + 0.25)[:, None, None]
            f = (0.5 + 1j * u_nodes)[:, None, None]
            b_arr, w_arr = big_b[:, None, None], w_nodes[:, None, None]

            sums[0] += np.sum(w_arr * (phi / den).real, axis=0)
            sums[1] += np.sum(w_arr * (f * phi / den).real, axis=0)
            sums[2] += np.sum(w_arr * phi.real, axis=0)
            sums[3] += np.sum(w_arr * (b_arr * phi / den).real, axis=0)
            sums[4] += np.sum(w_arr * (f * b_arr * phi / den).real, axis=0)
            sums[5] += np.sum(w_arr * (b_arr * b_arr * phi / den).real, axis=0)

            if len(starts) >= 3 and np.max(env[-48:]) < tol:
                break

        pref, ex = np.exp(0.5 * x_mat) / math.pi, np.exp(x_mat)
        out["c"][it] = ex - pref * sums[0]
        out["cx"][it] = ex - pref * sums[1]
        out["gam"][it] = pref * sums[2]
        out["cv"][it] = -pref * sums[3]
        out["cxv"][it] = -pref * sums[4]
        out["cvv"][it] = -pref * sums[5]
    return x_grid, out


def score_buckets(pred_dict, ref_dict):
    """Computes RMSE, Relative RMSE, and Max Abs Error across all 6 Tab 4.7 buckets."""
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
        for key in ("c", "cx", "gam", "cxv"):
            err = pred_dict[key][st, sv, :] - ref_dict[key][st, sv, :]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            ref_rms = float(np.sqrt(np.mean(ref_dict[key][st, sv, :] ** 2)))
            max_abs = float(np.max(np.abs(err)))
            report[b_name][key] = {
                "rmse": rmse,
                "rel_rmse": rmse / ref_rms,
                "max_abs": max_abs,
            }
    return report


def verify_against_scalar(p, taus=EVAL_TAUS, vs=EVAL_VS, zs=EVAL_ZS, tol=1e-11):
    """Cross-checks vectorized heston_eval_grid against scalar heston_ref.heston."""
    import time
    import heston_ref

    t0 = time.perf_counter()
    x_grid, vec = heston_eval_grid(p, taus=taus, vs=vs, zs=zs)
    t_vec = time.perf_counter() - t0

    t1 = time.perf_counter()
    max_diff = {k: 0.0 for k in ("c", "cx", "gam", "cv", "cxv", "cvv")}
    for it, tau in enumerate(taus):
        for iv, v in enumerate(vs):
            for iz in range(len(zs)):
                ref = heston_ref.heston(float(x_grid[it, iv, iz]), float(v), float(tau), p)
                for k in max_diff:
                    max_diff[k] = float(max(max_diff[k], abs(float(vec[k][it, iv, iz]) - ref[k])))
    t_sca = time.perf_counter() - t1
    passed = bool(all(float(val) <= tol for val in max_diff.values()))
    return {
        "t_vec_sec": round(t_vec, 4),
        "t_scalar_sec": round(t_sca, 4),
        "speedup": round(t_sca / max(t_vec, 1e-12), 2),
        "max_abs_diff": max_diff,
        "passed": passed,
        "x_grid": x_grid,
        "grid": vec,
    }
