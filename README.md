# CV-RBA-PINN: Control-Variate & Boundary-Layer Physics-Informed Neural Networks for Stochastic Volatility Pricing, Greeks, Hedging, and Fast SPX Calibration

This repository contains the complete PyTorch (`float64`) implementation, numerical PDE/Fourier reference oracles, trained checkpoints, empirical calibration pipelines, and Overleaf LaTeX manuscript for **CV-RBA-PINN** (`M1` $\to$ `M2_zv_lw` $\to$ Championship `M3_rar_80` / `M3_sech_80` and 10D amortized surrogates `M3_param_plus` / `M3_param_wide` / `M3_param_crisis`).

---

## Core Architectural Idea

Standard Physics-Informed Neural Networks (`B0`) fail to recover second-order option sensitivities (**Gamma**, **Vanna**, **Volga**) under realistic Feller-violating stochastic volatility ($2\kappa\theta < \xi^2$) because of two compounding singularities:
1. **The $\tau \to 0^+$ payoff corner** $(e^x - 1)^+$, whose Dirac delta second derivative $\delta(x)$ dominates collocation gradients.
2. **The $v \to 0^+$ degenerate drift boundary** $\kappa(\theta - v)c_v$, where parabolic diffusion vanishes.

**CV-RBA-PINN** decomposes the normalized option price as:

$$c(x, v, \tau) = b\big(x, w_{\mathrm{exp}}(v, \tau)\big) + \operatorname{sech}(\alpha z)\,\tilde{u}_\theta(x, v, \tau, z), \qquad z = \frac{x}{\sqrt{w_{\mathrm{exp}}(v, \tau)}}$$

where $b(x, w_{\mathrm{exp}})$ is the analytic Black–Scholes call evaluated at exact expected integrated variance $w_{\mathrm{exp}}(v, \tau) = \theta(\tau - \beta(\tau)) + v\beta(\tau)$. Because $b(x, w_{\mathrm{exp}})$ satisfies the $\tau = 0$ kink and mean-reversion drift identically, the PDE source term $S_{\mathrm{exp}} = \rho\xi v \beta b_{xw} + \frac{1}{2}\xi^2 v \beta^2 b_{ww}$ vanishes cleanly as $v \to 0^+$ and $\xi \to 0^+$ (`Gate G2 <= 1e-12`, `Gate G3 == 0.0`).

---

## Headline Results Summary

| Benchmark / Domain | Standard PINN (`B0`) | Championship **CV-RBA-PINN** | Key Gain / Takeaway |
| :--- | :--- | :--- | :--- |
| **Feller-Violated Heston (`P2`)** (`875`-pt Fourier oracle) | Gamma `79.6%`, Vanna `98.8%`, Vega `72.4%` | **`M3_sech_80`**: Price `0.03%`, **Gamma `1.66%`**, **Vega `1.39%`**, **Vanna `4.77%`**, `0/875` core neg. $\Gamma$ | **`48×` lower Gamma error**, **`52×` lower Vega error** |
| **Time-Dependent Heston (`T15`)** (`800×400` ADI oracle) | Gamma `80.1%`, Vanna `98.8%`, Vega `73.7%` | **`M3_rar_80`**: Price `0.03%`, **Gamma `2.65%`**, **Vega `1.70%`**, **Vanna `5.65%`** | **`30×` lower Gamma error** via spectral Chebyshev $w_{\mathrm{exp}}$ |
| **Non-Affine 2D `Heston-SLV`** (no closed-form CF) | Gamma `74.1%`, Vanna `98.4%`, Vega `59.3%` | **`M3_slv`**: Price `0.03%`, **Gamma `1.63%`**, **Vega `1.50%`**, **Vanna `3.45%`** | **`45.5×` lower Gamma error** where no Fourier solver exists |
| **Dynamic $\Delta$–$\Gamma$–Vega Hedging** (`15,000` exact CIR paths, $v_0=0.01$) | `\$0.2856` PnL RMSE (`\$0.2864` tracking err) | **`M3_rar_80` `\$0.0515`** / **0-shot `M3_param_plus` `\$0.0516`** (Exact Fourier: `\$0.0509`) | Matches exact Fourier replication within **`0.06` cents** |
| **Live Cboe SPX / SPXW** (`2026-11-19`, `2,533` quotes, $\xi^*=1.50$) | `108.38 bps` vs. Fourier (`82.0%` $\Gamma$) | **`M3_sech_80`**: **`1.58 bps` vs. Fourier** (`7.62 bps` vs. Cboe mid, `2.01%` $\Gamma_{\text{short,mod}}$) | **`68.8×` more accurate** on `4.9×` Feller-violated SPX |
| **168-Day Rolling SPX Archive** (`4.12M` quotes, `2024–2025`) | DE+Fourier: `~6.8 hours` (`111–208 s`/day) | **`M3_param_wide`**: **`486.5 s` total (`1.49 s`/day)**, **`9.86 bps`** med IS, **`10.10 bps`** med OOS | **`57×–119×` faster**; **`168/168` days violate Feller** |
| **Quarter-Century Wharton WRDS OptionMetrics** (`2000–2025`, `9` eras, `78` days, `1.28M` rows, `113,063` OOS quotes) | DE+Fourier: up to `343.4 s`/day; `B0` `71–316 bps` on crashes | **`M3_param_wide` + `M3_param_crisis_v2`**: **`10×–130×` faster** (`51.32` vs `53.46 bps` COVID OOS, `9.25` vs `9.03 bps` 2025 OOS); `M3_sech_80` **`3.31 bps`** (`2000`), **`12.44 bps`** (`2018` Volmageddon) | **`248/248` (`100%`) real SPX trading days (`2000–2026`, `5.44M+` quotes) violate Feller** |

---

## Repository Structure

- **Core PDE & Neural Solvers**:
  - `cvpinn.py`, `cvpinn_tier2.py`, `cvpinn_m3.py` — Baseline `B0`, paper base `M1`, boundary-layer `M2_zv_lw`, and multi-round `RAR-L-BFGS` + `sech` wing envelope `M3_rar_80` / `M3_sech_80`.
  - `cvpinn_td.py`, `heston_adi_2d.py` — Spectral Chebyshev time-dependent Heston solver & `800×400` Hundsdorfer–Verwer 2D ADI finite-difference oracle.
  - `cvpinn_slv.py` — Non-affine 2D Heston Stochastic-Local Volatility (`Heston-SLV`) CV-PINN & ADI oracle.
  - `cvpinn_param.py`, `train_wide_spx_surrogate.py` — 8D/10D amortized parametric surrogates (`M3_param_plus`, `M3_param_wide`, `M3_param_crisis`).
- **Hedging, Calibration & Empirical SPX Pipelines**:
  - `hedging_backtest.py` — Exact Broadie–Kaya CIR simulation & dynamic two-maturity $\Delta$–$\Gamma$–Vega replication backtest.
  - `diff_calibrate.py`, `spx_w8.py` — Millisecond batched PyTorch autograd + Strong-Wolfe `L-BFGS` Heston calibrator and Cboe / WRDS OptionMetrics / DataShop chain processor.
  - `real_spx_study.py`, `wrds_optionmetrics_study.py`, `run_wrds_option2_training.py` — Live Cboe `2026-11-19` study and quarter-century (`2000–2025`, `9` eras, `78` days) Wharton WRDS OptionMetrics IvyDB US crisis/bull/bear study.
- **Manuscript, Figures & Telemetry Logs**:
  - `paper/main.tex`, `paper/references.bib`, `paper/fig1..fig19*.png` — Self-contained Overleaf LaTeX manuscript with `19` publication figures and `24` verified BibTeX references.
  - `training_logger.py`, `paper/training_log.md`, `runs/training_history_log.jsonl` — Append-only master training telemetry registry (`33` logged checkpoints with loss trajectories, 6-Greek RMSEs, and market `bps` fit).
  - `paper/final_report_data.md`, `runs/final_report_log.json` — Master aggregated tables and JSON experiment logs.
  - `tests/` — `22` automated unit tests covering analytic PDE gates, Fourier/ADI parity, CIR martingale checks, and CLI calibration.

---

## Quickstart

```bash
# 1. Run the full verification & unit test suite (Gates G1-G3, ADI/Fourier parity, CLI):
.venv/bin/python -m unittest discover -s tests -v

# 2. Millisecond differentiable calibration on Cboe SPX/SPXW data:
.venv/bin/python spx_w8.py --t1 docs/market_data/spx_cboe_2026_11_19.csv \
                           --t2 docs/market_data/spx_cboe_all20months_2026_11_19.csv --diff-cal

# 3. Rebuild all master paper tables and summary logs:
.venv/bin/python paper/build_final_report_pack.py
```
