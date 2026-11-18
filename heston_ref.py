"""Pure-Python Heston reference and checks of the CV-PINN formulas.

Standard library only. Normalized forward units:
  x = ln(F/K), F = S*exp((r-q)*tau), C = K*exp(-r*tau)*c(x, v, tau).
The normalized Heston call c solves
  c_tau = 0.5*v*(c_xx - c_x) + rho*xi*v*c_xv + 0.5*xi^2*v*c_vv

          + kappa*(theta - v)*c_v,      c(x, v, 0) = max(exp(x) - 1, 0).
"""

import cmath
import math
import time

SQRT2PI = math.sqrt(2.0 * math.pi)

def npdf(d):
    return math.exp(-0.5 * d * d) / SQRT2PI

def ncdf(d):
    return 0.5 * math.erfc(-d / math.sqrt(2.0))

# ---------- normalized Black-Scholes in total variance w ----------

def bs(x, w):
    if w <= 0.0:
        return max(math.exp(x) - 1.0, 0.0)
    s = math.sqrt(w)
    d1 = x / s + 0.5 * s
    return math.exp(x) * ncdf(d1) - ncdf(d1 - s)

def bs_gam(x, w):
    """b_xx - b_x (normalized Gamma)."""
    s = math.sqrt(w)
    return npdf(x / s - 0.5 * s) / s

def bs_w(x, w):
    s = math.sqrt(w)
    return npdf(x / s - 0.5 * s) / (2.0 * s)

def bs_xw(x, w):
    s = math.sqrt(w)
    d2 = x / s - 0.5 * s
    return -d2 * npdf(d2) / (2.0 * w)

def bs_ww(x, w):
    s = math.sqrt(w)
    d1 = x / s + 0.5 * s
    d2 = d1 - s
    return npdf(d2) * (d1 * d2 - 1.0) / (4.0 * w * s)

def beta(tau, kappa):
    return (1.0 - math.exp(-kappa * tau)) / kappa

def w_spot(v, tau, p):
    return v * tau

def w_exp(v, tau, p):
    kappa, theta = p[0], p[1]
    b = beta(tau, kappa)
    return theta * (tau - b) + v * b

def source_spot(x, v, tau, p):
    """Right-hand side F for the correction with b(x, v*tau)."""
    kappa, theta, xi, rho = p
    w = v * tau
    return (rho * xi * v * tau * bs_xw(x, w)

            + 0.5 * xi * xi * v * tau * tau * bs_ww(x, w)
            + kappa * (theta - v) * tau * bs_w(x, w))

def source_exp(x, v, tau, p):
    """Right-hand side F for the correction with b(x, w_exp)."""
    kappa, theta, xi, rho = p
    w = w_exp(v, tau, p)
    b = beta(tau, kappa)
    return rho * xi * v * b * bs_xw(x, w) + 0.5 * xi * xi * v * b * b * bs_ww(x, w)

# ---------- Heston by Fourier inversion (Lewis form, little-trap CF) ----------

def gauss_legendre(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        pp = 1.0
        for _ in range(100):
            p1, p2 = 1.0, 0.0
            for j in range(1, n + 1):
                p3, p2 = p2, p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            dz = p1 / pp
            z -= dz
            if abs(dz) < 1e-15:
                break
        xs.append(z)
        ws.append(2.0 / ((1.0 - z * z) * pp * pp))
    return xs, ws

GLX, GLW = gauss_legendre(16)

def char_ab(z, tau, p):
    """E[exp(i z X)] = exp(A + B v) for X = ln(F_T / F_0)."""
    kappa, theta, xi, rho = p
    a = kappa - 1j * rho * xi * z
    d = cmath.sqrt(a * a + xi * xi * (z * z + 1j * z))
    g = (a - d) / (a + d)
    e = cmath.exp(-d * tau)
    big_b = (a - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e)
    big_a = kappa * theta / (xi * xi) * ((a - d) * tau - 2.0 * cmath.log((1.0 - g * e) / (1.0 - g)))
    return big_a, big_b

def heston(x, v, tau, p, tol=1e-17, max_u=4e5):
    """Returns c, c_x, c_xx - c_x, c_v, c_xv, c_vv in one integration pass."""
    wide = min(2.0, math.pi / (2.0 * max(abs(x), 1e-3)))
    sums = [0.0] * 6
    u0, quiet = 0.0, 0
    while u0 < max_u:
        # Narrow panels near u = 0, where 1/(u^2 + 1/4) varies fastest.
        panel = min(0.5, wide) if u0 < 4.0 else wide
        biggest = 0.0
        for xk, wk in zip(GLX, GLW):
            u = u0 + 0.5 * panel * (xk + 1.0)
            big_a, big_b = char_ab(complex(u, -0.5), tau, p)
            phi = cmath.exp(big_a + big_b * v + 1j * u * x)
            den = u * u + 0.25
            f = complex(0.5, u)
            wgt = 0.5 * panel * wk
            sums[0] += wgt * (phi / den).real
            sums[1] += wgt * (f * phi / den).real
            sums[2] += wgt * phi.real
            sums[3] += wgt * (big_b * phi / den).real
            sums[4] += wgt * (f * big_b * phi / den).real
            sums[5] += wgt * (big_b * big_b * phi / den).real
            biggest = max(biggest, abs(phi) * (1.0 + abs(big_b)) ** 2)
        u0 += panel
        quiet = quiet + 1 if biggest < tol else 0
        if quiet >= 3:
            break
    pref = math.exp(0.5 * x) / math.pi
    ex = math.exp(x)
    return {"c": ex - pref * sums[0], "cx": ex - pref * sums[1],
            "gam": pref * sums[2], "cv": -pref * sums[3],
            "cxv": -pref * sums[4], "cvv": -pref * sums[5], "umax": u0}

# ---------- checks ----------

P_FELLER_BAD = (1.5, 0.04, 0.5, -0.7)    # 2*kappa*theta = 0.12 < xi^2 = 0.25
P_FELLER_OK = (2.0, 0.04, 0.3, -0.5)     # 2*kappa*theta = 0.16 > xi^2 = 0.09

def check_limit():
    print("T1  xi -> 0: Heston tends to Black-Scholes at expected variance")
    for xi in (0.04, 0.02, 0.01):
        p = (1.5, 0.04, xi, -0.7)
        worst = 0.0
        for x in (-0.1, 0.0, 0.1):
            for v, tau in ((0.02, 0.25), (0.09, 1.0)):
                worst = max(worst, abs(heston(x, v, tau, p)["c"] - bs(x, w_exp(v, tau, p))))
        print("    xi=%.2f  max|c - b| = %.3e" % (xi, worst))

def check_greeks(p):
    print("T2  analytic Greeks vs central differences, params", p)
    worst = [0.0] * 5
    for x, v, tau in ((-0.15, 0.03, 0.3), (0.0, 0.04, 0.05), (0.1, 0.08, 1.0)):
        h = heston(x, v, tau, p)
        dx, dv = 2e-4, 1e-4
        cxp, cxm = heston(x + dx, v, tau, p), heston(x - dx, v, tau, p)
        cvp, cvm = heston(x, v + dv, tau, p), heston(x, v - dv, tau, p)
        fd_cx = (cxp["c"] - cxm["c"]) / (2 * dx)
        fd_gam = (cxp["cx"] - cxm["cx"]) / (2 * dx) - h["cx"]
        fd_cv = (cvp["c"] - cvm["c"]) / (2 * dv)
        fd_cvv = (cvp["cv"] - cvm["cv"]) / (2 * dv)
        fd_cxv = (cvp["cx"] - cvm["cx"]) / (2 * dv)
        errs = [abs(fd_cx - h["cx"]), abs(fd_gam - h["gam"]), abs(fd_cv - h["cv"]),
                abs(fd_cxv - h["cxv"]), abs(fd_cvv - h["cvv"])]
        worst = [max(a, b) for a, b in zip(worst, errs)]
    print("    max abs diff  c_x %.1e  gamma %.1e  c_v %.1e  c_xv %.1e  c_vv %.1e" % tuple(worst))

def check_pde(p):
    print("T3  Fourier price satisfies the normalized Heston PDE, params", p)
    worst, scale = 0.0, 0.0
    for x in (-0.2, 0.0, 0.1):
        for v in (0.01, 0.04, 0.09):
            for tau in (0.05, 0.5, 1.5):
                kappa, theta, xi, rho = p
                h = heston(x, v, tau, p)
                dt = 1e-4
                c_tau = (heston(x, v, tau + dt, p)["c"] - heston(x, v, tau - dt, p)["c"]) / (2 * dt)
                rhs = (0.5 * v * h["gam"] + rho * xi * v * h["cxv"]

                       + 0.5 * xi * xi * v * h["cvv"] + kappa * (theta - v) * h["cv"])
                worst = max(worst, abs(c_tau - rhs))
                scale = max(scale, abs(c_tau))
    print("    max |c_tau - rhs| = %.2e   (max |c_tau| = %.2e)" % (worst, scale))

def lhs_operator_on_bs(x, v, tau, p, wfun):
    """L[b] = b_tau - spatial operator, by finite differences on b(x, w(v, tau))."""
    kappa, theta, xi, rho = p

    def f(xx, vv, tt):
        return bs(xx, wfun(vv, tt, p))

    dx, dv, dt = 1e-4, 1e-5, 1e-6
    b0 = f(x, v, tau)
    b_tau = (f(x, v, tau + dt) - f(x, v, tau - dt)) / (2 * dt)
    b_x = (f(x + dx, v, tau) - f(x - dx, v, tau)) / (2 * dx)
    b_xx = (f(x + dx, v, tau) - 2 * b0 + f(x - dx, v, tau)) / (dx * dx)
    b_v = (f(x, v + dv, tau) - f(x, v - dv, tau)) / (2 * dv)
    b_vv = (f(x, v + dv, tau) - 2 * b0 + f(x, v - dv, tau)) / (dv * dv)
    b_xv = (f(x + dx, v + dv, tau) - f(x + dx, v - dv, tau)

            - f(x - dx, v + dv, tau) + f(x - dx, v - dv, tau)) / (4 * dx * dv)
    return (b_tau - 0.5 * v * (b_xx - b_x) - rho * xi * v * b_xv

            - 0.5 * xi * xi * v * b_vv - kappa * (theta - v) * b_v)

def check_source(p):
    print("T4  closed-form source terms vs operator applied to the control variate, params", p)
    for name, wfun, src in (("spot", w_spot, source_spot), ("expected", w_exp, source_exp)):
        worst, scale = 0.0, 0.0
        for x in (-0.1, 0.0, 0.05):
            for v in (0.02, 0.06):
                for tau in (0.1, 0.5, 1.0):
                    lhs = lhs_operator_on_bs(x, v, tau, p, wfun)
                    rhs = -src(x, v, tau, p)
                    worst = max(worst, abs(lhs - rhs))
                    scale = max(scale, abs(rhs))
        print("    %-8s max |L[b] + F| = %.2e   (max |F| = %.2e)" % (name, worst, scale))

def sup_over_band(src, v, tau, p, wfun):
    """Largest |F| over the band |x / sqrt(w)| <= 6."""
    s = math.sqrt(wfun(v, tau, p))
    return max(abs(src(0.05 * k * s, v, tau, p)) for k in range(-120, 121))

def table_sources(p):
    print("T5  largest |source| over the band |x/sqrt(w)| <= 6, params", p)
    print("    tau     v       sup|F_spot|  sup|F_expected|")
    for tau in (0.02, 0.1, 1.0):
        for v in (0.0001, 0.001, 0.01, 0.04):
            print("    %-6g  %-6g  %.4e   %.4e" % (
                tau, v, sup_over_band(source_spot, v, tau, p, w_spot),
                sup_over_band(source_exp, v, tau, p, w_exp)))
    print("    expected control variate at v = 0, tau = 0.1: F = %+.3e" % source_exp(0.0, 0.0, 0.1, p))

def table_two_cv(p):
    print("T8  correction size when v differs from theta: spot vs expected control variate, params", p)
    print("    v      tau    max|u_spot|  max|u_exp|   max|gam(u_spot)|  max|gam(u_exp)|  max gam(c)")
    for v in (0.01, 0.09):
        for tau in (0.1, 0.5):
            ws, we = w_spot(v, tau, p), w_exp(v, tau, p)
            s = math.sqrt(we)
            best = [0.0] * 5
            for k in range(-8, 9):
                x = 0.5 * k * s
                h = heston(x, v, tau, p)
                vals = [abs(h["c"] - bs(x, ws)), abs(h["c"] - bs(x, we)),
                        abs(h["gam"] - bs_gam(x, ws)), abs(h["gam"] - bs_gam(x, we)), h["gam"]]
                best = [max(a, b) for a, b in zip(best, vals)]
            print("    %-5g  %-5g  %.3e    %.3e    %8.4f          %8.4f         %8.4f" % (
                v, tau, best[0], best[1], best[2], best[3], best[4]))

def table_collapse(p):
    print("T9  correction rate u_exp/tau against z = x/sqrt(w) at short maturities, v = 0.04, params", p)
    v = 0.04
    print("    tau      z=-2      z=-1      z=0       z=+1      z=+2")
    for tau in (0.08, 0.02, 0.005):
        s = math.sqrt(w_exp(v, tau, p))
        row = []
        for z in (-2.0, -1.0, 0.0, 1.0, 2.0):
            x = z * s
            row.append((heston(x, v, tau, p)["c"] - bs(x, w_exp(v, tau, p))) / tau)
        print("    %-7g  %+.5f  %+.5f  %+.5f  %+.5f  %+.5f" % tuple([tau] + row))

def table_gamma(p):
    print("T6  normalized Gamma at the money: full price vs correction, v = 0.04, params", p)
    print("    tau     c          gamma(c)   u_exp       gamma(u_exp)  u_exp/tau   u_spot      gamma(u_spot)")
    v = 0.04
    for tau in (0.02, 0.05, 0.1, 0.5, 1.0, 2.0):
        h = heston(0.0, v, tau, p)
        we, ws = w_exp(v, tau, p), w_spot(v, tau, p)
        ue, us = h["c"] - bs(0.0, we), h["c"] - bs(0.0, ws)
        ge, gs = h["gam"] - bs_gam(0.0, we), h["gam"] - bs_gam(0.0, ws)
        print("    %-6g  %.6f  %8.4f   %+.3e  %+9.4f     %+.4f    %+.3e  %+9.4f" % (
            tau, h["c"], h["gam"], ue, ge, ue / tau, us, gs))

def table_profile(p):
    print("T7  correction across moneyness at tau = 0.02, v = 0.04 (expected control variate), params", p)
    print("    x       z=x/sqrt(w)  c          u_exp       u_exp/tau  gamma(c)   gamma(u_exp)")
    v, tau = 0.04, 0.02
    we = w_exp(v, tau, p)
    for x in (-0.15, -0.06, -0.03, 0.0, 0.03, 0.06, 0.15):
        h = heston(x, v, tau, p)
        ue = h["c"] - bs(x, we)
        print("    %+.2f   %+7.2f      %.6f  %+.3e  %+.4f    %8.4f   %+8.4f" % (
            x, x / math.sqrt(we), h["c"], ue, ue / tau, h["gam"], h["gam"] - bs_gam(x, we)))

def main():
    t0 = time.time()
    check_limit()
    for p in (P_FELLER_BAD, P_FELLER_OK):
        check_greeks(p)
        check_pde(p)
        check_source(p)
    table_sources(P_FELLER_BAD)
    table_gamma(P_FELLER_BAD)
    table_profile(P_FELLER_BAD)
    table_two_cv(P_FELLER_BAD)
    table_collapse(P_FELLER_BAD)
    print("done in %.1f s" % (time.time() - t0))

if __name__ == "__main__":
    main()
