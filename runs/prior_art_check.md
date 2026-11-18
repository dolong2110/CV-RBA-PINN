# Workstream W1 — Day-1 Prior-Art Full-Text Verification (`S1`)

Status: **4/4 PDFs inspected in full — Stop Rule `S1` SIGNED OFF (CLEAR TO PROCEED).**

No surveyed work subtracts an expected-variance Black–Scholes carrier $b(x, w_{\text{exp}}(v, \tau))$ inside a label-free Heston PDE operator $L[u] = F_{\text{exp}}$ or benchmarks Heston Gamma and Vanna surfaces against a Fourier oracle across Feller-satisfied and Feller-violating regimes.

---

## 1. Sakuma, T. (Jul 2026, v5) — *Differential Machine Learning for 0DTE Options with Stochastic Volatility and Jumps* (`arXiv:2603.07600v5`)
File: [sakuma_2603.07600.pdf](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/docs/prior_art/sakuma_2603.07600.pdf) (20 pp., checked in full)

- [x] **Exact variance carrier formula**: Section 3, **Eq. (9)** (p. 3) sets
  $$V_{\text{eff}}(\mathbf{x}) = \max\{V + g(\tau)\,\Delta V_\phi(\mathbf{x}), \varepsilon\}, \qquad g(\tau) = 1 - \exp(-\tau/\tau_0),$$
  and passes $\sigma_{\text{eff}} = \sqrt{V_{\text{eff}}}$ into $u_\phi(\mathbf{x}) := C_{\text{BS}}(S = e^x, K = 1, r = 0.01, q = 0, \tau, \sigma_{\text{eff}})$. This is a **nonlinear implied-variance perturbation inside $C_{\text{BS}}$** (not an additive $b(x, w_{\text{exp}}) + \tau N$ decomposition canceling mean-reversion drift inside $L[u]=F_{\text{exp}}$).
- [x] **Supervision regime**: **Supervised differential machine learning (DML)** throughout (Sections 3.1–3.2, 4.1, **Eqs. (10), (13), (16)**, pp. 3–6). Requires Fourier reference prices and Greeks $(\Delta, \Gamma, \nu)$ (`w_G = 0.7`, `w_col = 0.1`, `w_R = 0.01`, `w_NA = 0.05`) plus supervised jump-operator targets $J^{\text{ref}}$ (**Eq. (12)**) in Stages 2–3. Not label-free.
- [x] **Greeks evaluated**: Price, Delta ($\Delta$), Gamma ($\Gamma$), and Vega ($\nu$) in Tables 1, 2, 4, and 8. **Vanna ($\partial^2 c / \partial x \partial v$) is not evaluated.**
- **Verdict for `S1`**: **Clear.**

---

## 2. Lara Rangel, J. M., & Vazquez Mendoza, M. (Oct 2026, v1) — *BeTINN* (`arXiv:2610.19254v1`)
File: [betinn_2610.19254.pdf](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/docs/prior_art/betinn_2610.19254.pdf) (30 pp., checked in full)

- [x] **Exact $\tau = 0$ parameterization**: Sections 4.1–4.2 (pp. 4–5) write $u_{\text{BeTINN}}(\tau, x, v, \mu) = f_{\Theta(\tau, \mu)}(x, v, \mu) + g_\psi(\tau, x, v, \mu)$ with affine subspace parameter modulation $\Theta_g(\tau, \mu) = B_g + \sum_{k=1}^q A_{g,k}\phi_{g,k}(\tau, \mu)$. Terminal payoff is enforced via a **soft MSE penalty** $R_{\text{IC}} = u_\omega(0, x, v; \mu) - (e^x - 1)^+$ on 100,000 sampled points (`Table 2`, p. 12). **No Black–Scholes carrier is subtracted.**
- [x] **Domain & boundary notes**: $v \in [0.1, 1.0]$ (`v_min = 0.1`, non-degenerate; Appendix A, p. 10 explicitly notes it avoids the $v = 0$ degenerate boundary). Trained in float64 with Levenberg–Marquardt (3,000 steps).
- [x] **Second-order sensitivities evaluated**: Section 3 & Appendix A/E (`Tables 1, 13, 15, 16`) report Price, $\Delta$, strike-scaled Gamma $K\Gamma = e^{-2x}(u_{xx} - u_x)$, variance sensitivity $u_v$, $2\sqrt{v}\,u_v$, $-u_\tau$, and $u_r$. **Vanna ($u_{xv}$) is not evaluated.**
- **Verdict for `S1`**: **Clear.**

---

## 3. Chakhoyan, H., Kalici, S., & Malandain, K. A. (Dec 2025, v1) — *DeepSVM* (`arXiv:2512.07162v1`)
File: [deepsvm_2512.07162.pdf](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/docs/prior_art/deepsvm_2512.07162.pdf) (7 pp., checked in full; note PDF author order is Chakhoyan, Kalici, Malandain)

- [x] **Feller condition during sampling**: **Table I** (p. 2) explicitly states: *"The Feller condition $2\kappa\theta > \sigma^2$ is enforced during sampling"* with $\nu \in [0.01, 0.40]$ (does not reach $v = 0$ or Feller-violating `P2`).
- [x] **Trial ansatz & ATM Gamma failure**: Section III.B, **Eq. (10)** (p. 3) uses $u_{\text{pred}}(x, \nu, \tau; \mu) = \phi(x) + \tau\,\text{Softplus}(\mathfrak{N}_\theta(\mu, x, \nu, \tau))$ with $\phi(x) = \max(e^x - 1, 0)$ (no Black–Scholes Gaussian carrier). **Section IV / Figure 3** (pp. 5–6) confirms ATM Delta oscillations and locally negative autodiff Gamma around $x = 0$ even at $10^{-5}$ loss, and reports that an explicit soft positive-Gamma penalty at ATM collocation points failed to stabilize Greeks and degraded pricing accuracy.
- **Verdict for `S1`**: **Clear** (directly validates our cut of the intrinsic-value baseline and necessity of subtracting the Gaussian carrier).

---

## 4. Tan, J., & Zhang, X. (2026) — *ICPINNs*, *Math. Comput. Simul.* 241A, 908–924 (`doi:10.1016/j.matcom.2025.10.005`)
File: [tan_zhang_2026_icpinn.pdf.pdf](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/docs/prior_art/tan_zhang_2026_icpinn.pdf.pdf) (17 pp., pp. 908–924, checked in full)

- [x] **Exact Heston trial ansatz (Section 3.2, Eqs. 3.8–3.16, pp. 915–917)**:
  $$U_\theta(S, v, \tau) = A(S, v, \tau) + B(S, v, \tau) \times \text{NN}(S, v, \tau; \theta), \qquad B(S, v, \tau) = S \times \tau,$$
  where $A(S, v, \tau)$ is a **small pre-trained 3×40 ELU network** (`Table 2`, `Table 4`, p. 912–913) fitted to the Dirichlet conditions $U(S, v, 0) = \max(S - K, 0)$ (**Eq. 3.9a**) and $U(0, v, \tau) = 0$ (**Eq. 3.9b**). **No Black–Scholes carrier (expected-variance or spot-variance) is subtracted anywhere.**
- [x] **Boundary conditions & parameter regime**: Works in raw spot coordinates $(S, v, \tau)$ on one fixed Feller-satisfied parameter set ($K=1, r=0.1, \kappa=1, \theta=0.08, \sigma=0.39, \rho=-0.93$; $2\kappa\theta = 0.16 > \sigma^2 = 0.1521$) with soft penalties at $S_{\max}=4K$ ($\partial_S U = 1$), $v_{\max}=1$ ($\partial_v U = 0$), and $v=0$ ($\mathcal{B}_{\text{deg}} U = 0$, **Eq. 3.10**, **Eq. 3.15**).
- [x] **Greeks benchmarked**: Evaluates option prices $U$ only (21×21×21 grid MSE `Fig. 3.6`, and `Table 6` at $\tau=1$). **No Delta, Gamma, or Vanna benchmarks.**
- **Verdict for `S1`**: **Clear.**
