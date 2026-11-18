# Control-Variate Physics-Informed Neural Networks for Heston Gamma and Vanna

**Working Benchmark Draft (Two-Week Synthetic Study — Phase 5 Deliverable)**

---

## Abstract

Physics-informed neural networks (PINNs) trained directly on option pricing PDEs without labeled solutions often achieve low pricing loss while exhibiting severe near-expiry inaccuracies in second-order Greeks—particularly normalized Gamma ($\Gamma_{\text{norm}} = c_{xx} - c_x$) and Vanna ($c_{xv}$)—because the call payoff corner at $x = 0$ generates an $O(\tau^{-1/2})$ curvature boundary layer as time to maturity $\tau \to 0$. We synthesize and benchmark **CV-RBA-PINN** (`M1`), a label-free Heston PDE solver that subtracts a normalized Black–Scholes carrier $b(x, w_{\text{exp}}(v, \tau))$ evaluated at expected integrated variance $w_{\text{exp}}(v, \tau) = \theta(\tau - \beta(\tau)) + v\beta(\tau)$, represents the remainder via an exact-terminal ansatz $u = \tau N_\phi(\cdot, \tanh(z/2))$ in the variance-scaled coordinate $z = x/\sqrt{w_{\text{exp}}}$, enforces the Ekström–Tysk degenerate transport limit $u_\tau - \kappa\theta u_v = 0$ on $v = 0$, and uses matched two-stage Cosine-Adam + frozen-batch Strong-Wolfe `L-BFGS` optimization in `float64`. Across an 875-point interior $(\tau, v, z)$ grid verified against a Lewis–Albrecher Fourier oracle ($\le 2.44 \times 10^{-12}$ discrepancy), `M1` beats a tuned baseline PINN (`B0`, `term_weight = 100.0`) on **5/5 random seeds in every bucket** for both Feller-satisfied (`P1`) and Feller-violated (`P2`) parameters: on short maturities ($\tau \le 0.1$), median `M1/B0` Gamma RMSE drops **22.8×** (`0.0438×`) on `P1` and **8.4×** (`0.1196×`) on `P2`, while Vanna RMSE drops **9.8×** (`0.1016×`) on `P1` and **3.8×** (`0.2602×`) on `P2`. Controlled ablations further show that expected-variance control variate `M1` outperforms the spot-variance carrier `A1` by **5.3×** on `P1` short-maturity Gamma and **2.2×** on `P2` overall Vanna, and that energy-locked residual-based attention (`A2b`) cuts fixed-point short-maturity Gamma and Vanna error by **31.2%** and **21.7%** on `P2` relative to unweighted fixed points (`A2a`).

---

## 1. Introduction

Physics-informed neural networks solve pricing partial differential equations (PDEs) by minimizing PDE residuals over collocation coordinates without labeled prices. For single fixed-parameter European options under standard constant-parameter Heston dynamics, semi-analytic Fourier quadrature or alternating-direction-implicit (ADI) finite differences already provide fast, well-characterized accuracy. PINNs become compelling in three regimes: amortized parametric surrogates, higher-dimensional multi-factor models, and continuous automatic-differentiation Greeks surfaces.

Yet second-order Greeks—especially normalized Gamma ($c_{xx} - c_x$) and mixed spot–variance Vanna ($c_{xv}$)—remain the weakest empirical link in financial PINNs. Close to expiration ($\tau \to 0$), the call payoff $\max(e^x - 1, 0)$ creates an $O(\tau^{-1/2})$ Gamma spike concentrated inside an $O(\sqrt{w})$ boundary layer around forward log-moneyness $x = 0$. Recent independent work confirms that even when terminal conditions are satisfied identically via $u = \max(e^x - 1, 0) + \tau\,\text{Softplus}(N)$ (DeepSVM; Chakhoyan, Kalici, & Malandain 2025) or when validation PDE residuals are small (BeTINN; Lara Rangel & Vazquez Mendoza 2026), accurate prices coexist with oscillating at-the-money (ATM) Delta and locally negative autodiff Gamma.

This draft tests a narrow, falsifiable PDE-solver synthesis rather than claiming novelty for Black–Scholes proxies in isolation:
1. **Label-free PDE operator cancellation down to $v = 0$**: Decomposing $c = b(x, w_{\text{exp}}(v, \tau)) + \tau N_\phi(\dots)$ inside the normalized Heston operator $L$ satisfies the kinked payoff identically at $\tau = 0$, cancels mean-reversion drift $\kappa(\theta - v)$ analytically (verified identically at $\xi = 0$, extending algebraically to smooth time-dependent parameters), keeps the inhomogeneous source $F_{\text{exp}}$ uniformly bounded and zero on the attainable boundary $v = 0$, and avoids the long-horizon Vanna inflation suffered by spot-variance $w_{\text{spot}} = v\tau$.
2. **Variance-scaled boundary-layer feature**: Supplying $\tanh(x/(2\sqrt{w_{\text{exp}}}))$ stabilizes the near-expiry correction rate $u/\tau$ in $z = x/\sqrt{w_{\text{exp}}}$ down to $v = 0$ and preserves exact $O(1/\tau)$ low-variance Vanna scaling without large network weights.
3. **Exact Fourier benchmarking under optimizer parity**: All five arms (`B0`, `M1`, `A1`, `A2a`, `A2b`) share double precision (`float64`), the $v = v_{\min}$ degenerate-boundary collocation strip, and identical two-stage Cosine-Adam + frozen-batch Strong-Wolfe `L-BFGS` optimization across 5 seeds on both Feller-satisfied (`P1`) and Feller-violated (`P2`) regimes, scored across 6 stratified buckets on an 875-point grid.

---

## 2. Related Work

Prior work spans four adjacent directions (audited across 22 sources, with full-text PDF verification completed for the four closest neural solvers under Stop Rule `S1` in [prior_art_check.md](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/runs/prior_art_check.md)):

- **Parametric and boundary-constrained Heston PINNs**: Hainaut and Casas (2024) solve the parametric Heston PDE over a 9D state–parameter vector using 4-phase Adam and soft terminal/lower-boundary penalties without a control variate, reporting price heatmaps without Greek error tables. Tan and Zhang (2026, *ICPINNs*) decompose the solution as $U_\theta = A(S, v, \tau) + S\tau\,\text{NN}_\theta$, where $A$ is a pre-trained 3×40 ELU network fitted to Dirichlet payoff/zero boundaries on a Feller-satisfied set ($2\kappa\theta = 0.16 > \sigma^2 = 0.1521$), combined with soft penalties at $S_{\max}$, $v_{\max}$, and $v=0$; it subtracts no Black–Scholes carrier and benchmarks option prices only (no Delta, Gamma, or Vanna). Lara Rangel and Vazquez Mendoza (2026, *BeTINN*) introduce multi-directional time–parameter weight modulation with soft terminal panels ($R_{\text{IC}}$ on 100,000 points) over $v \in [0.1, 1.0]$ and benchmark Price, $\Delta$, $K\Gamma$, and $u_v$ against a semi-analytic oracle, but evaluate neither $v = 0$ nor Vanna ($u_{xv}$) and subtract no Gaussian carrier.
- **Hard terminal enforcement without Gaussian heat-kernel subtraction**: Chakhoyan, Kalici, and Malandain (2025, *DeepSVM*) enforce the exact payoff via $u_{\text{pred}} = \max(e^x - 1, 0) + \tau\,\text{Softplus}(N)$ while enforcing $2\kappa\theta > \sigma^2$ during sampling ($\nu \in [0.01, 0.40]$). Their Figure 3 directly demonstrates ATM Delta oscillations and locally negative autodiff Gamma around $x = 0$ at $10^{-5}$ loss, and Section IV reports that an explicit soft positive-Gamma penalty failed to stabilize Greeks without degrading prices. Georgoulis, Papapantoleon, and Smaragdakis (2026) and Hsu et al. (2024) likewise subtract smoothed or raw intrinsic value rather than a Black–Scholes variance carrier, leaving the $O(\tau^{-1/2})$ Gaussian curvature inside the learned time value.
- **Supervised Black–Scholes variance carriers**: Sakuma (2026) embeds 0DTE Bates prices inside $C_{\text{BS}}$ via a nonlinear maturity-gated variance correction $V_{\text{eff}}(\mathbf{x}) = \max\{V + (1 - e^{-\tau/\tau_0})\Delta V_\phi(\mathbf{x}), \varepsilon\}$ (Eq. 9), trained by supervised differential machine learning on Fourier price and $(\Delta, \Gamma, \nu)$ labels plus a small PIDE penalty ($\omega_R = 0.01$) and supervised jump targets (no Vanna reported). Feng et al. (2026, *FI-DeepONet*) add a strike-line integrated-variance Black–Scholes carrier for 1D local volatility using supervised finite-difference price labels without a PDE residual loss or stochastic variance $v$. Unlike both, our formulation is label-free and cancels mean-reversion drift inside $L[b]$ analytically.
- **Asymptotic expansions, Deep BSDE control variates, and RBA**: Expected-variance Black–Scholes appears classically as the $\xi \to 0$ leading term in Hull and White (1987), Ball and Roma (1994), Lewis (2000), and Benhamou, Gobet, and Miri (2010), and inside pathwise Deep BSDE solvers (Takahashi, Tsuchida, and Yamada 2022). Anagnostopoulos et al. (2024) introduce cumulative residual-based attention (`RBA-PINNs`) on fixed collocation sets; our arm `A2b` adds explicit per-step energy locking to eliminate the $10^6\times$ global residual-weight scale drift `S(k)` over 10,000 steps.

---

## 3. Method

### 3.1 Normalized Heston PDE

Under the risk-neutral measure with spot $S_t$, variance $v_t$, rate $r$, dividend yield $q$, correlation $\rho$, mean reversion $\kappa$, long-run variance $\theta$, and vol-of-vol $\xi$, changing to forward log-moneyness $x = \ln(S e^{(r-q)\tau}/K)$, time to maturity $\tau = T - t$, and discounted-strike normalized price $c(x, v, \tau) = C(S, v, t)/(K e^{-r\tau})$ removes $K, r, q$ from the PDE:
$$L[c] := \frac{\partial c}{\partial \tau} - \frac{v}{2}\left(\frac{\partial^2 c}{\partial x^2} - \frac{\partial c}{\partial x}\right) - \rho\xi v \frac{\partial^2 c}{\partial x \partial v} - \frac{\xi^2 v}{2}\frac{\partial^2 c}{\partial v^2} - \kappa(\theta - v)\frac{\partial c}{\partial v} = 0, \qquad c(x, v, 0) = \max(e^x - 1, 0).$$

### 3.2 Control-Variate Split and Closed-Form Sources

Let $b(x, w) = e^x N(d_1) - N(d_2)$ with $d_{1,2} = x/\sqrt{w} \pm \frac{1}{2}\sqrt{w}$ be the normalized Black–Scholes call at total variance $w$, which satisfies $b_w = \frac{1}{2}(b_{xx} - b_x)$. Writing $c = b(x, w(v, \tau)) + u(x, v, \tau)$ yields $L[u] = F := -L[b]$ with $u(x, v, 0) = 0$.

- **Expected-variance carrier (`M1`, `A2a`, `A2b`)**: Setting $\beta(\tau) = (1 - e^{-\kappa\tau})/\kappa$ and $w_{\text{exp}}(v, \tau) = \theta(\tau - \beta(\tau)) + v\beta(\tau)$ satisfies $\partial_\tau w_{\text{exp}} = v + \kappa(\theta - v)\beta(\tau)$. Applying $L$ cancels both the Black–Scholes diffusion and the $\kappa(\theta - v)$ mean-reversion drift identically, leaving
  $$F_{\text{exp}}(x, v, \tau) = \rho\xi v\beta(\tau)\frac{\partial^2 b}{\partial x \partial w}(x, w_{\text{exp}}) + \frac{\xi^2 v\beta(\tau)^2}{2}\frac{\partial^2 b}{\partial w^2}(x, w_{\text{exp}}).$$
  Because $w_{\text{exp}}(0, \tau) = \theta(\tau - \beta(\tau)) > 0$ for $\tau > 0$ while every term carries factor $v$, $F_{\text{exp}}$ vanishes identically at $\xi = 0$ (Gate `G3`), vanishes on $v = 0$, and stays bounded by $0.043$ across $|z| \le 6$ on `P2` (Check `T5`).
- **Spot-variance carrier (`A1`)**: Using $w_{\text{spot}} = v\tau$ leaves an uncancelled drift term $\kappa(\theta - v)\tau b_w$ alongside $\rho\xi v\tau b_{xw} + \frac{1}{2}\xi^2 v\tau^2 b_{ww}$, which scales as $O(\sqrt{\tau/v})$ (reaching $0.940$ at $v = 10^{-4}, \tau = 1$ in Check `T5`) and requires a variance floor $v_{\min} = 10^{-3}$.

### 3.3 Network Ansatz, Boundary Conditions, and Optimization Parity

For control-variate arms (`M1`, `A1`, `A2a`, `A2b`), the correction is parameterized as
$$u_\phi(x, v, \tau) = \tau\, N_\phi\!\left(\frac{x}{x_{\max}},\, \frac{2v}{v_{\max}} - 1,\, \frac{2\tau}{\tau_{\max}} - 1,\, \tanh\!\left(\frac{z}{2}\right)\right), \qquad z = \frac{x}{\sqrt{w(v, \tau)}},$$
using a 4-hidden-layer, width-64 `Tanh` MLP in `torch.float64` (baseline `B0` omits the $z$ input and predicts $c$ directly).

On the computational box $[-2.5, 2.5] \times [v_{\min}, 0.5] \times [10^{-4}, 2.0]$:
1. **Degenerate $v = 0$ boundary (`D9`)**: Setting $v = 0$ in $L$ reduces the PDE to the Ekström–Tysk (2010) transport condition $u_\tau - \kappa\theta u_v = 0$ (satisfied to $\le 6.73 \times 10^{-9}$ by the Fourier reference in Check `T12`, whereas clamping $u = 0$ or $u_v = 0$ injects up to $0.409$ derivative error). Every arm appends $N_{\text{col}}/16 = 128$ collocation points on $v = v_{\min}$ directly into the PDE residual loss.
2. **Upper variance and lateral edges**: $v = v_{\max} = 0.5$ is left free (relying on negative mean-reversion drift; exact CIR exceedance from $v_0 = 0.16$ on `P2` peaks at $2.09 \times 10^{-3}$ in Check `T14`), while lateral edges $x = \pm 2.5$ (`14.2×` smaller truncation error than $x_{\max} = 1.5$ in Check `T13`) enforce $u = 0$ (`far_weight = 1.0`, $256$ points/step).
3. **Energy-locked RBA (`A2b`, `D13`)**: On a fixed collocation set (`A2a` vs. `A2b`), raw weights initialize at $\lambda^{(0)} = 0$ and update via $\lambda_i \leftarrow \gamma\lambda_i + \eta^* |r_i|/\max_j |r_j|$ ($\gamma = 0.999, \eta^* = 0.01$; Anagnostopoulos et al. 2024), then rescale inside `torch.no_grad()` to $w_{\text{eff}} = \lambda \sqrt{\text{mean}(r^2)/(\text{mean}((\lambda r)^2) + 10^{-30})}$ so effective PDE-to-edge energy stays constant across all 10,000 steps.
4. **Conditioning & two-stage optimizer (`D14`–`D16`)**: Small-$\tau$ cancellation in $\tau - \beta(\tau)$ uses `-expm1(-kappa*tau)` and a 4-term Taylor series for $\kappa\tau < 10^{-3}$ (relative error $\le 2.49 \times 10^{-16}$ in `T15`); `bs_call` clamps $w \ge 10^{-30}$ prior to `sqrt` and zeros $n(d_2)$ for $|d_2| > 38$. Every arm runs 10,000 `CosineAnnealingLR` Adam steps ($10^{-3} \to 10^{-4}$, $N_{\text{col}} = 2048$ interior) followed by up to 500 frozen-batch Strong-Wolfe `L-BFGS` iterations ($8192 + 512 + 1024$ points for resampled arms; fixed set + detached $w_{\text{eff}}$ for `A2a`/`A2b`) with finite-state safeguard.

---

## 4. Why It Should Work, and Where It Should Not

Pure-Python checks against `heston_ref.py` (`T6`–`T10`) establish both the mechanism and its limits before any network is trained:
- **ATM Gamma collapse**: At $v = 0.04$ on `P2`, exact Heston normalized Gamma rises nearly ninefold from $1.63$ ($\tau = 2.0$) to $14.15$ ($\tau = 0.02$), while expected-variance correction Gamma stays below $0.35$ at ATM across all maturities ($0.04$ at $\tau = 0.02$).
- **Scaled-$z$ boundary-layer stabilization**: Between $\tau = 0.08$ and $\tau = 0.005$ at $v = 0.04$, $u_{\text{exp}}/\tau$ at fixed $z = -1$ varies by under $2\%$ (`-0.0219` to `-0.0217`) even as raw moneyness width shrinks $4\times$.
- **Short-maturity Vanna reduction vs. long-horizon `A1` failure**: At $\tau = 0.02$, subtracting $b(x, w_{\text{exp}})$ cuts peak Vanna $7.1\times$ at $v = 0.04$ (`3.450` $\to$ `0.487`) and $11.0\times$ at $v = 0.09$ (`1.462` $\to$ `0.133`). Conversely, at $(v = 0.04, \tau = 2.0)$, $\partial_v w_{\text{spot}} = \tau$ ignores mean reversion ($1/\kappa = 0.667$), inflating `A1` peak remainder Vanna **56.8% above raw Heston Vanna** (`2.464` vs. `1.571`, while `M1` cuts it to `0.638`).
- **Low-variance Feller limit (`R3`)**: At $v \le 0.01$ on `P2`, CIR non-Gaussian kurtosis leaves $40\%$–$54\%$ of peak Gamma and Vanna in $u$ (`T8`, `T10`), and exact Fourier Vanna RMS is $8.6\times$ larger at $v \le 0.01$ (`10.5081`) than at $v \ge 0.04$ (`1.2164`). Consequently, low-variance and moderate-variance sub-buckets must be reported separately.

---

## 5. Experiments

We benchmark five arms (`B0`, `M1`, `A1`, `A2a`, `A2b`) across two parameter sets—`P1` ($\kappa=2.0, \theta=0.04, \xi=0.3, \rho=-0.5$, Feller-satisfied, $b_F(0) = +0.035$) and `P2` ($\kappa=1.5, \theta=0.04, \xi=0.5, \rho=-0.7$, Feller-violated, $b_F(0) = -0.065$)—and 5 seeds (`0–4`, **50 runs total**) on Apple M5 CPU (`float64`, `--workers 4`).

- **Pre-training gates (`G1`–`G3`)**: Vectorized 16-node Gauss–Legendre evaluator [heston_grid_vec.py](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/heston_grid_vec.py) matched scalar [heston_ref.py](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/heston_ref.py) across all 875 points (`7 × 5 × 25` grid: $\tau \in \{0.02, 0.05, 0.1, 0.25, 0.5, 1, 2\}$, $v \in \{0.005, 0.01, 0.04, 0.09, 0.16\}$, $z \in [-3, 3]$ step $0.25$) within $1.14 \times 10^{-12}$ (`P1`) and $2.44 \times 10^{-12}$ (`P2`, Gate `G1`). PyTorch autograd `max |L[b] + F|` passed at $\le 3.47 \times 10^{-15}$ (`expected`), $\le 4.86 \times 10^{-17}$ ($v=0$ strip), and $\le 9.84 \times 10^{-15}$ (`spot`, Gate `G2`), and $\xi = 0$ residuals vanished identically (`0.0`, Gate `G3`).
- **Pre-registered Day-4 tuning**: On `P2` seed 0, `M1` at $v_{\max} = 0.5$ vs. $0.75$ yielded nearly identical `overall` Gamma error (`0.0794` vs. `0.0793`) with $0.5$ slightly better on `short_mat` Vanna (`0.2235` vs. `0.2298`) and `low_var` sample share (`56.6%` vs. `46.2%`), locking $v_{\max} = 0.5$. Sweeping `B0` `term_weight` $\in \{1, 10, 100\}$ yielded `short_mat` Gamma relative RMSEs of `0.8491`, `0.8221`, and `0.7684`, locking `B0` at `term_weight = 100.0` in [pre_registration.json](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/runs/pre_registration.json).

---

## 6. Results

### 6.1 Stop Rule `S2` and Main Comparison (`B0` vs. `M1`)

Stop Rule `S2` passed on all 5 seeds on `P2` `short_mat` ($\tau \le 0.1$): `M1` outperformed `B0` in **5/5 seeds** on both Gamma and Vanna RMSE, with median Gamma RMSE ratio `0.1196` (`8.4×` reduction, far below the `0.50` bar) and median Vanna RMSE ratio `0.2602` (`3.8×` reduction). Across the full 50-run matrix ([tables.md](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/tables.md), [summary.json](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/runs/summary.json)), `M1` wins **5/5 seeds in every single bucket** on both `P1` and `P2`:

| Param | Bucket | `B0` Gamma Rel RMSE (Max Abs) | `M1` Gamma Rel RMSE (Max Abs) | Med. Ratio | `B0` Vanna Rel RMSE (Max Abs) | `M1` Vanna Rel RMSE (Max Abs) | Med. Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `P1` | `overall` | `0.7512 ± 0.0262` (`34.51`) | **`0.0341 ± 0.0012`** (`1.58`) | `0.046×` | `0.9935 ± 0.0062` (`24.57`) | **`0.1081 ± 0.0015`** (`2.80`) | `0.109×` |
| `P1` | `short_mat` | `0.7646 ± 0.0264` (`34.51`) | **`0.0335 ± 0.0012`** (`1.58`) | `0.044×` | `0.9993 ± 0.0041` (`24.57`) | **`0.1010 ± 0.0012`** (`2.80`) | `0.102×` |
| `P1` | `short_low_var` | `0.7975 ± 0.0248` (`34.51`) | **`0.0357 ± 0.0013`** (`1.58`) | `0.045×` | `0.9998 ± 0.0031` (`24.57`) | **`0.1021 ± 0.0012`** (`2.80`) | `0.103×` |
| `P1` | `short_mod_var` | `0.5214 ± 0.0442` (`9.68`) | **`0.0132 ± 0.0006`** (`0.20`) | `0.025×` | `0.9767 ± 0.0435` (`3.05`) | **`0.0397 ± 0.0006`** (`0.21`) | `0.040×` |
| `P2` | `overall` | `0.7829 ± 0.0198` (`40.98`) | **`0.0990 ± 0.0163`** (`5.85`) | `0.125×` | `0.9949 ± 0.0042` (`34.63`) | **`0.2730 ± 0.0149`** (`10.85`) | `0.282×` |
| `P2` | `short_mat` | `0.7934 ± 0.0201` (`40.98`) | **`0.0961 ± 0.0167`** (`5.85`) | `0.120×` | `0.9995 ± 0.0024` (`34.63`) | **`0.2540 ± 0.0179`** (`10.85`) | `0.260×` |
| `P2` | `short_low_var` | `0.8241 ± 0.0185` (`40.98`) | **`0.1021 ± 0.0177`** (`5.85`) | `0.122×` | `0.9998 ± 0.0018` (`34.63`) | **`0.2562 ± 0.0180`** (`10.85`) | `0.262×` |
| `P2` | `short_mod_var` | `0.5328 ± 0.0399` (`9.90`) | **`0.0327 ± 0.0060`** (`0.59`) | `0.059×` | `0.9809 ± 0.0334` (`3.97`) | **`0.0982 ± 0.0129`** (`0.77`) | `0.100×` |

In standard market variances (`short_mod_var`, $v \ge 0.04$), `M1` reaches **1.32%** Gamma and **3.97%** Vanna relative RMSE on `P1` (**40×** and **25×** median improvement over `B0`), and **3.27%** Gamma and **9.82%** Vanna on `P2` (**17×** and **10×** improvement). Figures 1 and 2 visualize this contrast on `P2` median seeds (`B0` seed 3 vs. `M1` seed 2): `B0` exhibits a severe ATM error ridge peaking above 40 in Gamma and 34 in Vanna, while `M1` drives error across $v \ge 0.04$ down to $10^{-2}$–$10^{-1}$ and isolates residual error to the $v \le 0.01$ Feller corner. Figure 3 confirms that `M1`'s learned correction rate $u_\phi/\tau$ collapses along the scaled coordinate $z$ across $\tau \in \{0.08, 0.02, 0.005\}$ with slight amplitude damping at the finest short-maturity tail.

### 6.2 Ablation Studies and Compute Cost

- **Expected vs. Spot Variance (`M1` vs. `A1`)**: On `P1`, leaving mean-reversion drift inside $F_{\text{spot}}$ degrades `A1` sharply: `short_mat` Gamma rises `5.3×` (`0.1791 ± 0.0257` vs. `0.0335 ± 0.0012`, `5/5` `M1` wins) and `overall` Vanna jumps `6.6×` (`0.7184 ± 0.1111`, max abs `31.71`, vs. `0.1081 ± 0.0015` for `M1`). On `P2`, `A1` and `M1` tie within $4\%$ on `short_mat` Vanna (`0.2440` vs. `0.2540`), yet across the full maturity range (`overall`), `A1` Vanna degrades to `0.6017 ± 0.0341` (max abs `33.07`) vs. `0.2730 ± 0.0149` (max abs `10.85`) for `M1` (`5/5` `M1` wins, median ratio `0.4544×`), directly confirming Check `T10`'s long-horizon Vanna divergence.
- **Energy-Locked RBA on Fixed Points (`A2b` vs. `A2a`)**: On `P1`, `A2b` yields modest gains (`4/5` wins, `short_mat` Gamma `0.0337` vs. `0.0360`). On the stiffer Feller-violating `P2` regime with fixed collocation points (`2,176` points), `A2b` beats `A2a` on **5/5 seeds across every bucket**: `short_mat` Gamma falls **31.2%** (`0.0902 ± 0.0122` vs. `0.1308 ± 0.0036`), Vanna falls **21.7%** (`0.2450 ± 0.0235` vs. `0.3047 ± 0.0049`), and `short_mod_var` reaches `0.0276` Gamma / `0.0875` Vanna—fully closing the gap between fixed-point and resampled `M1`.
- **Compute & Reliability**: Across all 50 runs, `L-BFGS` triggered **0/50 rollbacks**. Mean training time per run under 4-worker parallel saturation was `636–664 s` (`P1`) and `585–658 s` (`P2`) for resampled arms and `540–554 s` for fixed-point arms (`A2a`, `A2b`), with batched 875-point inference + all autodiff Greeks taking `5.8–14.4 ms` (specifically `7.97 ms` for `M1` on `P2` and `9.86 ms` on `P1`).

---

## 7. Limitations and Future Extensions

1. **Constant-parameter scope**: Constant-parameter Heston admits fast Fourier inversion (`0.37 s` on `P1`, `0.82 s` on `P2` for 875 points in `heston_grid_vec.py`). A full journal submission requires extending the solver where closed-form transforms fail: our verified smooth time-dependent generalization $w_{\text{td}}(v, \tau) = \alpha(\tau) + v\beta(\tau)$ (`Check T15`: max RK4 residual mismatch $5.00 \times 10^{-8}$) against an ADI finite-difference oracle, or an 8D/9D amortized parametric surrogate retaining $b(x, w_{\text{exp}}(v, \tau; \kappa, \theta))$.
2. **Low-variance corner ($v \le 0.01$)**: While `M1` beats `B0` by $8\times$ on `P2` `short_low_var`, `10.21%` Gamma and `25.62%` Vanna relative error remain there because uniform $x \in [-2.5, 2.5]$ places only a small fraction of samples inside the $O(\tau)$-wide $|z| \le 3$ transition zone as $v \to 0$. Stratified-$z$ sampling and localized low-variance `gPINN` residual-gradient terms remain natural Tier-2 upgrades.
3. **Real market data**: Optional SPX/SPXW parity calibration (`W8`, Tab 7.5) is deferred as a separate applicability study and cannot rank solver accuracy because Heston structural misfit is shared across arms.

---

## Appendix: Verification & Reproducibility Artifacts

- **Reference & Checks**: [heston_ref.py](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/heston_ref.py) (`T1`–`T9`), [verify_expansion.py](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/verify_expansion.py) (`T10`–`T15`), [heston_grid_vec.py](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/heston_grid_vec.py) (Gate `G1`), [cvpinn.py](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/cvpinn.py) (Gates `G2`–`G3`), and [test_cvpinn_gates.py](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/tests/test_cvpinn_gates.py).
- **Experiment Outputs**: [gates_report.json](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/runs/gates_report.json), [day4_tuning.json](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/runs/day4_tuning.json), [pre_registration.json](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/runs/pre_registration.json), [s2_report.json](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/runs/s2_report.json), [summary.json](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/runs/summary.json), and all 50 run JSONs & `.pt` checkpoints under [runs/](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/runs).
- **Tables & Figures**: [tables.md](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/tables.md), [fig1_gamma_error_surface_p2.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig1_gamma_error_surface_p2.png), [fig2_vanna_error_surface_p2.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig2_vanna_error_surface_p2.png), and [fig3_boundary_layer_collapse.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig3_boundary_layer_collapse.png).

---

## References (Selected Verified Bibliography)

- Anagnostopoulos, S. J., Toscano, J. D., Stergiopulos, N., & Karniadakis, G. E. (2024). Residual-based attention in physics-informed neural networks. *CMAME*, 421, 116805.
- Benhamou, E., Gobet, E., & Miri, M. (2010). Time dependent Heston model. *SIAM J. Financial Math.*, 1(1), 289–325.
- Chakhoyan, H., Kalici, S., & Malandain, K. A. (2025). DeepSVM: Learning stochastic volatility models with physics-informed deep operator networks. `arXiv:2512.07162`.
- Ekström, E., & Tysk, J. (2010). The Black–Scholes equation in stochastic volatility models. *J. Math. Anal. Appl.*, 368(2), 498–507.
- Feng, J., Hu, B., Jiang, Y., Mou, C., & Zhang, Y. (2026). Finance-informed operator learning for option pricing with quantum-compatible realizations. `arXiv:2609.33213`.
- Georgoulis, E. H., Papapantoleon, A., & Smaragdakis, C. (2026). A deep implicit-explicit minimizing movement method for option pricing in jump-diffusion models. *CNSNS*, 156, 109709.
- Hainaut, D., & Casas, A. (2024). Option pricing in the Heston model with physics inspired neural networks. *Annals of Finance*, 20(3), 353–376.
- Hsu, C.-W., Dai, T.-S., Wang, C.-J., & Chen, Y.-P. (2024). Accurate neural network option pricing methods with control variate techniques. *Proc. CIKM '24*, 860–869.
- Lara Rangel, J. M., & Vazquez Mendoza, M. (2026). Beyond time induction: Structured conditional neural solvers for parametric stochastic-volatility pricing (BeTINN). `arXiv:2610.19254`.
- Lewis, A. L. (2000). *Option Valuation under Stochastic Volatility*. Finance Press; Lewis, A. L. (2001), *SSRN 282110*; Albrecher et al. (2007), *Wilmott Magazine*, 83–92.
- Sakuma, T. (2026). Differential machine learning for 0DTE options with stochastic volatility and jumps. `arXiv:2603.07600`.
- Takahashi, A., Tsuchida, Y., & Yamada, T. (2022). Control variate method for Deep BSDE solver. *J. Comput. Phys.*, 454, 110956.
- Tan, J., & Zhang, X. (2026). Improved constrained physics-informed neural networks (ICPINNs) to solve PDE and its application to option pricing. *Math. Comput. Simul.*, 241A, 908–924.
