"""verify_expansion.py - Pure-Python verification suite for checks T10 to T15.

Standard library + heston_ref.py only (as audited on 17 November 2026):
  T10: Normalized Vanna decomposition (Tables 2.6a and 2.6b, dz = 0.05 and 0.25)
  T11: Exact 875-point Fourier RMS denominators across 6 buckets (Table 4.7, P1 and P2)
  T12: Degenerate v = 0 transport PDE residual (u_tau - kappa*theta*u_v) & v_max edge checks
  T13: Lateral truncation mismatch max|u_exp(+-x_max, v, tau)| for x_max in {1.5, 2.0, 2.5, 3.0}
  T14: Exact non-central chi-square CIR exceedance P(v_tau > v_max | v_0) + MC cross-check
  T15: 60-digit Decimal (tau - beta) cancellation, analytical RBA S(k), and time-dependent Heston RK4
"""
from decimal import Decimal, getcontext
import math
import random
import time

import heston_ref as hr


def w_exp_stable(v, tau, p):
    kappa, theta = p[0], p[1]
    kt = kappa * tau
    b = -math.expm1(-kt) / kappa
    if kt < 1e-3:
        tau_minus_b = tau * (0.5 * kt - kt ** 2 / 6.0 + kt ** 3 / 24.0 - kt ** 4 / 120.0)
    else:
        tau_minus_b = (kt + math.expm1(-kt)) / kappa
    return theta * tau_minus_b + v * b


def b_xv_exp(x, v, tau, p):
    w = w_exp_stable(v, tau, p)
    b = -math.expm1(-p[0] * tau) / p[0]
    return b * hr.bs_xw(x, w)


def b_xv_spot(x, v, tau, p):
    return tau * hr.bs_xw(x, v * tau)


def check_t10_vanna(p=hr.P_FELLER_BAD):
    print("T10 Vanna decomposition (Tables 2.6a-b) on P2, dz=0.05 on |z|<=3")
    zs = [-3.0 + 0.05 * i for i in range(121)]
    print("  Table 2.6a (v = 0.04):")
    for tau in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00):
        we = w_exp_stable(0.04, tau, p)
        s = math.sqrt(we)
        max_c, max_ue, max_us = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
        atm_vals = None
        for z in zs:
            x = z * s
            cxv = hr.heston(x, 0.04, tau, p)["cxv"]
            bxe = b_xv_exp(x, 0.04, tau, p)
            bxs = b_xv_spot(x, 0.04, tau, p)
            ue, us = cxv - bxe, cxv - bxs
            if abs(cxv) > max_c[0]:
                max_c = (abs(cxv), z)
            if abs(ue) > max_ue[0]:
                max_ue = (abs(ue), z)
            if abs(us) > max_us[0]:
                max_us = (abs(us), z)
            if abs(z) < 1e-12:
                atm_vals = (cxv, bxe, ue)
        print(
            "    tau=%4.2f  max|c_xv|=%.3f(%+.2f)  max|u_exp|=%.3f(%+.2f)  "
            "max|u_spot|=%.3f(%+.2f)  ATM: c=%.3f b=%.3f u=%.3f"
            % (tau, max_c[0], max_c[1], max_ue[0], max_ue[1], max_us[0], max_us[1],
               atm_vals[0], atm_vals[1], atm_vals[2])
        )

    print("  Table 2.6b (selected v, tau regimes):")
    regimes = [
        (0.0001, 0.02), (0.001, 0.02), (0.005, 0.02), (0.005, 0.10), (0.005, 0.50),
        (0.01, 0.02), (0.01, 0.10), (0.01, 0.50), (0.04, 0.02), (0.04, 0.10),
        (0.09, 0.02), (0.09, 0.10), (0.09, 0.50),
    ]
    for v, tau in regimes:
        we = w_exp_stable(v, tau, p)
        s = math.sqrt(we)
        mc = mus = mue = mb = 0.0
        for z in zs:
            x = z * s
            cxv = hr.heston(x, v, tau, p)["cxv"]
            bxe = b_xv_exp(x, v, tau, p)
            bxs = b_xv_spot(x, v, tau, p)
            mc = max(mc, abs(cxv))
            mus = max(mus, abs(cxv - bxs))
            mue = max(mue, abs(cxv - bxe))
            mb = max(mb, abs(bxe))
        print(
            "    v=%-6g tau=%4.2f  max|c_xv|=%7.3f  max|u_spot|=%8.3f  "
            "max|u_exp|=%7.3f (%.1f%%)  max|b_exp|=%7.3f"
            % (v, tau, mc, mus, mue, 100.0 * mue / mc, mb)
        )


def check_t11_rms():
    print("T11 Exact 875-point Fourier RMS denominators (Table 4.7)")
    taus = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    vs = (0.005, 0.01, 0.04, 0.09, 0.16)
    zs = [-3.0 + 0.25 * i for i in range(25)]
    buckets = {
        "overall": (set(range(7)), set(range(5))),
        "short_mat": ({0, 1, 2}, set(range(5))),
        "low_var": (set(range(7)), {0, 1}),
        "mod_var": (set(range(7)), {2, 3, 4}),
        "short_low_var": ({0, 1, 2}, {0, 1}),
        "short_mod_var": ({0, 1, 2}, {2, 3, 4}),
    }
    for pname, p in (("P1", hr.P_FELLER_OK), ("P2", hr.P_FELLER_BAD)):
        grid = {}
        max_abs_x = 0.0
        for it, tau in enumerate(taus):
            for iv, v in enumerate(vs):
                s = math.sqrt(w_exp_stable(v, tau, p))
                for iz, z in enumerate(zs):
                    x = z * s
                    max_abs_x = max(max_abs_x, abs(x))
                    grid[(it, iv, iz)] = hr.heston(x, v, tau, p)
        print(f"  {pname} max|x| = {max_abs_x:.4f}")
        for bname, (st, sv) in buckets.items():
            pts = [grid[(it, iv, iz)] for it in st for iv in sv for iz in range(25)]
            n = len(pts)
            rc = math.sqrt(sum(q["c"] ** 2 for q in pts) / n)
            rcx = math.sqrt(sum(q["cx"] ** 2 for q in pts) / n)
            rg = math.sqrt(sum(q["gam"] ** 2 for q in pts) / n)
            rcxv = math.sqrt(sum(q["cxv"] ** 2 for q in pts) / n)
            print(
                f"    {bname:14s} (n={n:3d}): c={rc:.5f}  cx={rcx:.5f}  "
                f"gam={rg:.4f}  cxv={rcxv:.4f}"
            )


def check_t12_boundaries(p=hr.P_FELLER_BAD):
    print("T12 Degenerate v=0 PDE residual & v=v_max edge magnitudes on P2")
    kappa, theta, _, _ = p
    zs = [-3.0 + 0.25 * i for i in range(25)]
    dt = 1e-5
    max_res = 0.0
    for tau in (0.02, 0.10, 0.50, 2.00):
        s = math.sqrt(w_exp_stable(0.0, tau, p))
        max_u = 0.0
        max_uv = 0.0
        for z in zs:
            x = z * s
            hp = hr.heston(x, 0.0, tau + dt, p)
            hm = hr.heston(x, 0.0, tau - dt, p)
            h0 = hr.heston(x, 0.0, tau, p)
            up = hp["c"] - hr.bs(x, w_exp_stable(0.0, tau + dt, p))
            um = hm["c"] - hr.bs(x, w_exp_stable(0.0, tau - dt, p))
            u0 = h0["c"] - hr.bs(x, w_exp_stable(0.0, tau, p))
            u_tau = (up - um) / (2.0 * dt)
            b = -math.expm1(-kappa * tau) / kappa
            u_v = h0["cv"] - b * hr.bs_w(x, w_exp_stable(0.0, tau, p))
            max_res = max(max_res, abs(u_tau - kappa * theta * u_v))
            max_u = max(max_u, abs(u0))
            max_uv = max(max_uv, abs(u_v))
        print(f"  v=0, tau={tau:.2f}: max|u|={max_u:.3e}, max|u_v|={max_uv:.4f}")
    print(f"  v=0 overall max|u_tau - kappa*theta*u_v| = {max_res:.3e}")

    for vmax in (0.5, 0.75):
        xs = [-1.5 + 0.1 * i for i in range(31)]
        tau = 2.0
        we = w_exp_stable(vmax, tau, p)
        b = -math.expm1(-kappa * tau) / kappa
        mu = muv = 0.0
        for x in xs:
            h = hr.heston(x, vmax, tau, p)
            mu = max(mu, abs(h["c"] - hr.bs(x, we)))
            muv = max(muv, abs(h["cv"] - b * hr.bs_w(x, we)))
        print(f"  v={vmax:.2f}, tau=2.0, |x|<=1.5: max|u|={mu:.3e}, max|u_v|={muv:.3e}")


def check_t13_truncation():
    print("T13 Lateral truncation mismatch max|u_exp(+-x_max, v, tau)|, tau in [0.1, 2.0]")
    taus = (0.1, 0.25, 0.5, 1.0, 2.0)
    vs_eval = (0.005, 0.01, 0.04, 0.09, 0.16)
    vs_box = (0.005, 0.01, 0.04, 0.09, 0.16, 0.30, 0.50)
    for pname, p in (("P1", hr.P_FELLER_OK), ("P2", hr.P_FELLER_BAD)):
        print(f"  {pname}:")
        for xm in (1.5, 2.0, 2.5, 3.0):
            me = mb = 0.0
            for tau in taus:
                for v in vs_box:
                    we = w_exp_stable(v, tau, p)
                    for x in (-xm, xm):
                        u = abs(hr.heston(x, v, tau, p)["c"] - hr.bs(x, we))
                        mb = max(mb, u)
                        if v <= 0.16:
                            me = max(me, u)
            print(f"    x_max={xm:.1f}: eval(v<=0.16)={me:.3e}, box(v<=0.50)={mb:.3e}")


def cir_exceedance_prob(v0, tau, vmax, p, terms=80):
    """Exact P(v_tau > vmax | v_0) via Poisson-weighted central chi-square tail."""
    kappa, theta, xi, _ = p
    e = math.exp(-kappa * tau)
    c = 2.0 * kappa / (xi * xi * (1.0 - e))
    df = 4.0 * kappa * theta / (xi * xi)
    lam = 2.0 * c * v0 * e
    x = 2.0 * c * vmax
    half_x = 0.5 * x
    log_half_x = math.log(half_x)

    def upper_reg_gamma(a, z):
        if z <= 0.0:
            return 1.0
        log_pref = -z + a * math.log(z) - math.lgamma(a)
        if z < a + 1.0:
            # Series for lower regularized gamma P(a, z), then 1 - P
            ap = a
            term = 1.0 / a
            s = term
            for _ in range(1000):
                ap += 1.0
                term *= z / ap
                s += term
                if abs(term) < abs(s) * 1e-16:
                    break
            p_lower = s * math.exp(log_pref) if log_pref > -740.0 else 0.0
            return max(0.0, 1.0 - p_lower)
        # Modified Lentz continued fraction for upper Q(a, z)
        tiny = 1e-300
        b = z + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b if abs(b) > tiny else 1.0 / tiny
        h = d
        for i in range(1, 1000):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        if log_pref < -740.0:
            return 0.0
        return max(0.0, math.exp(log_pref) * h)

    # Start Poisson summation around mean k0 = int(0.5 * lam)
    mu = 0.5 * lam
    w0 = math.exp(-mu)
    prob = 0.0
    w = w0
    for j in range(terms):
        if j > 0:
            w *= mu / j
        if w < 1e-18 and j > mu + 10:
            break
        prob += w * upper_reg_gamma(0.5 * df + j, half_x)
    return prob


def check_t14_cir():
    print("T14 Exact CIR non-central chi-square exceedance over tau in [0.02, 2.0]")
    taus = [0.02 + 0.01 * i for i in range(199)]
    for pname, p, v0s in (("P2", hr.P_FELLER_BAD, (0.16, 0.09)), ("P1", hr.P_FELLER_OK, (0.16,))):
        for v0 in v0s:
            row = []
            for vmax in (0.50, 0.75, 1.00):
                best = max((cir_exceedance_prob(v0, tau, vmax, p), tau) for tau in taus)
                row.append(f"vmax={vmax:.2f}: {best[0]:.3e} (tau={best[1]:.2f})")
            print(f"  {pname} v0={v0:.2f} -> " + " | ".join(row))

    # Poisson-Gamma Monte Carlo cross-check at P2, v0=0.16, tau=0.45, vmax=0.50 (200k paths for fast check)
    p = hr.P_FELLER_BAD
    kappa, theta, xi, _ = p
    tau, v0, vmax = 0.45, 0.16, 0.50
    e = math.exp(-kappa * tau)
    c = 2.0 * kappa / (xi * xi * (1.0 - e))
    df = 4.0 * kappa * theta / (xi * xi)
    mu = c * v0 * e
    rng = random.Random(42)
    n_paths = 200000
    hits = 0
    for _ in range(n_paths):
        # Knuth/Ahrens-Dieter Poisson not needed since mu is small (~1.67)
        l_exp = math.exp(-mu)
        k_pois, prod = 0, 1.0
        while True:
            k_pois += 1
            prod *= rng.random()
            if prod <= l_exp:
                break
        k_pois -= 1
        chi2 = 2.0 * rng.gammavariate(0.5 * df + k_pois, 1.0)
        if chi2 / (2.0 * c) > vmax:
            hits += 1
    p_exact = cir_exceedance_prob(v0, tau, vmax, p)
    p_mc = hits / n_paths
    se = math.sqrt(p_mc * (1.0 - p_mc) / n_paths)
    print(f"  CIR check P2 (v0=0.16, tau=0.45, vmax=0.5): exact={p_exact:.4e}, MC({n_paths})={p_mc:.4e} +- {se:.2e}")


def check_t15_conditioning_and_td():
    print("T15 60-digit Decimal (tau - beta), RBA S(k), and time-dependent Heston RK4")
    getcontext().prec = 60
    kappa = 1.5
    kd = Decimal("1.5")
    for tau in (1e-4, 1e-6):
        td = Decimal(str(tau))
        exact = td - (Decimal(1) - (-kd * td).exp()) / kd
        naive = tau - (1.0 - math.exp(-kappa * tau)) / kappa
        kt = kappa * tau
        taylor = tau * (0.5 * kt - kt ** 2 / 6.0 + kt ** 3 / 24.0 - kt ** 4 / 120.0)
        rel_naive = abs(Decimal(str(naive)) - exact) / exact
        rel_tay = abs(Decimal(str(taylor)) - exact) / exact
        print(f"  tau={tau:.0e}: rel_err(naive)={float(rel_naive):.2e}, rel_err(Taylor4)={float(rel_tay):.2e}")

    gamma, eta = 0.999, 0.01
    for k in (1, 100, 1000, 10000):
        sk = (eta * (1.0 - gamma ** k) / (1.0 - gamma)) ** 2
        print(f"  RBA S({k:5d}) = {sk:.4e}")

    # Smooth time-dependent Heston operator identity check
    def kap_fn(t): return 1.5 + 0.8 * math.sin(2.0 * t)
    def tht_fn(t): return 0.04 + 0.02 * t
    def xi_fn(t): return 0.5 - 0.1 * t
    def rho_fn(t): return -0.7 + 0.2 * t

    # Integrate alpha'(t) = kappa(t)*theta(t)*beta(t), beta'(t) = 1 - kappa(t)*beta(t) by 4000-step RK4 up to T=1.8
    n_steps = 4000
    t_max = 1.8
    h = t_max / n_steps
    ts_grid = [i * h for i in range(n_steps + 1)]
    alphas = [0.0] * (n_steps + 1)
    betas = [0.0] * (n_steps + 1)
    a, b = 0.0, 0.0
    for i in range(n_steps):
        t = ts_grid[i]
        def rhs(tt, bb):
            kk, th = kap_fn(tt), tht_fn(tt)
            return kk * th * bb, 1.0 - kk * bb
        k1a, k1b = rhs(t, b)
        k2a, k2b = rhs(t + 0.5 * h, b + 0.5 * h * k1b)
        k3a, k3b = rhs(t + 0.5 * h, b + 0.5 * h * k2b)
        k4a, k4b = rhs(t + h, b + h * k3b)
        a += (h / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
        b += (h / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
        alphas[i + 1], betas[i + 1] = a, b

    def ab_interp(t):
        idx = t / h
        i0 = min(int(idx), n_steps - 1)
        u = idx - i0
        t0_i, t1_i = ts_grid[i0], ts_grid[i0 + 1]
        a0, a1 = alphas[i0], alphas[i0 + 1]
        b0, b1 = betas[i0], betas[i0 + 1]
        da0 = kap_fn(t0_i) * tht_fn(t0_i) * b0
        da1 = kap_fn(t1_i) * tht_fn(t1_i) * b1
        db0 = 1.0 - kap_fn(t0_i) * b0
        db1 = 1.0 - kap_fn(t1_i) * b1
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        aa = h00 * a0 + h10 * h * da0 + h01 * a1 + h11 * h * da1
        bb = h00 * b0 + h10 * h * db0 + h01 * b1 + h11 * h * db1
        return aa, bb

    worst, scale = 0.0, 0.0
    dx, dv, dt = 1e-4, 1e-5, 1e-5
    for tau in (0.1, 0.5, 1.0, 1.7):
        kk, th, xx, rr = kap_fn(tau), tht_fn(tau), xi_fn(tau), rho_fn(tau)
        _, b_cur = ab_interp(tau)
        for v in (0.02, 0.06):
            a_cur, _ = ab_interp(tau)
            w_cur = a_cur + v * b_cur
            for x in (-0.1, 0.0, 0.05):
                def bf(xq, vq, tq):
                    aa, bb = ab_interp(tq)
                    return hr.bs(xq, aa + vq * bb)
                b0 = bf(x, v, tau)
                bt = (bf(x, v, tau + dt) - bf(x, v, tau - dt)) / (2 * dt)
                bx = (bf(x + dx, v, tau) - bf(x - dx, v, tau)) / (2 * dx)
                bxx = (bf(x + dx, v, tau) - 2 * b0 + bf(x - dx, v, tau)) / (dx * dx)
                bv = (bf(x, v + dv, tau) - bf(x, v - dv, tau)) / (2 * dv)
                bvv = (bf(x, v + dv, tau) - 2 * b0 + bf(x, v - dv, tau)) / (dv * dv)
                bxv = (bf(x + dx, v + dv, tau) - bf(x + dx, v - dv, tau)
                       - bf(x - dx, v + dv, tau) + bf(x - dx, v - dv, tau)) / (4 * dx * dv)
                lhs = (bt - 0.5 * v * (bxx - bx) - rr * xx * v * bxv
                       - 0.5 * xx * xx * v * bvv - kk * (th - v) * bv)
                f_td = rr * xx * v * b_cur * hr.bs_xw(x, w_cur) + 0.5 * xx * xx * v * b_cur * b_cur * hr.bs_ww(x, w_cur)
                worst = max(worst, abs(lhs + f_td))
                scale = max(scale, abs(f_td))
    print(f"  Time-dependent Heston max|L[b_td] + F_td| = {worst:.2e} (max|F_td| = {scale:.2e})")


def main():
    t0 = time.time()
    check_t10_vanna()
    check_t11_rms()
    check_t12_boundaries()
    check_t13_truncation()
    check_t14_cir()
    check_t15_conditioning_and_td()
    print(f"All T10-T15 checks finished in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
