# CV-RBA-PINN Final Report Data Pack (Master Verified Reference)

## 1. Environment, Domain & Verification Gates

- **Compute Environment**: Python `3.10.21`, PyTorch `2.14.1` (`torch.float64`), NumPy `2.2.6`, SciPy `1.15.3`, Platform `macOS-26.6.2-arm64-arm-64bit`.
- **PDE Domain**: $x \in [-2.5, 2.5]$, $v \in [0, 0.5]$ ($v_{\min}=10^{-3}$ for spot-CV `A1`), $\tau \in [10^{-4}, 2.0]$.
- **Evaluation Grid**: `875` points (`7` maturities $\times$ `5` variances $\times$ `25` scaled moneyness $z \in [-3, 3]$) across `6` buckets.

| Gate / Validation Check | Metric / Value | Pass Criterion |
| --- | --- | --- |
| Gate `G1` (Vectorized Fourier vs Scalar `P1` / `P2`) | `1.144e-12` / `2.437e-12` | `<= 1e-11` |
| Gate `G2` (Const `P2` `expected` / $v=0$ strip / `spot`) | `3.372e-15` / `4.163e-17` / `9.836e-15` | `<= 1e-6` |
| Gate `G3` (Const `P1` / `P2` at $\xi=0$) | `0.000e+00` / `0.000e+00` | `== 0.0` |
| TD Chebyshev ODE vs 8k RK4 ($\alpha$ / $\beta$ max rel err) | `1.167e-13` / `1.538e-14` | `< 1e-12` |
| Gate `G2_td` (`expected` / $v=0$ strip / `spot`) | `1.434e-13` / `3.927e-14` / `9.832e-15` | `<= 1e-6` |
| Gate `G3_td` ($\xi(\tau) \equiv 0$) | `0.000e+00` | `== 0.0` |
| ADI Oracle vs Chained Riccati PWC (`c` / `gam` / `cxv`) | `7.149e-06` / `6.508e-04` / `1.762e-03` | Verified |
| ADI Grid Refinement Med vs Fine `800x400, 2700t` (`c` / `gam` / `cxv`) | `3.634e-06` / `5.263e-04` / `1.205e-03` | Converged |
| Gate `G2_param` (`expected` / `spot`) & `G3_param` (8D Box) | `2.549e-13` / `4.120e-13` & `0.000e+00` | `<= 1e-6`, `== 0.0` |
| Gate `G2_slv` & `G3_slv` (Non-Affine Heston-SLV) | `3.324e-15` & `0.000e+00` | `<= 1e-6`, `== 0.0` |
| SLV ADI Med vs Fine `800x400, 2700t` (`c` / `gam` / `cxv` / `cv`) | `3.299e-06` / `5.413e-04` / `1.240e-03` / `8.223e-05` | Converged |

## 2. Headline Model Progression (`B0` -> `M1` -> `M2_zv_lw` -> `M3_rar_80`)

| Regime | Arm (Seeds, Params) | Train (s) | Infer (ms) | Overall `c` | Overall `cx` | Overall **Gamma** (Max) | Overall **Vanna** (Max) | Overall **Vega** | Overall **Volga** | `short_low_var` **Gamma** | `short_mod_var` **Gamma** |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | `B0` (`n=5`, `12,801`) | `636` | `7.84` | `0.0466 ± 0.0123` | `0.1467 ± 0.0133` | `0.7512 ± 0.0262` (`34.51`) | `0.9935 ± 0.0062` (`24.57`) | `0.7064 ± 0.0615` | `1.0014 ± 0.0014` | `0.7975 ± 0.0248` | `0.5214 ± 0.0442` |
| P1 | `M1` (`n=5`, `12,865`) | `664` | `9.86` | `0.0010 ± 0.0000` | `0.0041 ± 0.0001` | `0.0341 ± 0.0012` (`1.58`) | `0.1081 ± 0.0015` (`2.80`) | `0.0565 ± 0.0017` | `0.2826 ± 0.0058` | `0.0357 ± 0.0013` | `0.0132 ± 0.0006` |
| P1 | `A1` (`n=5`, `12,865`) | `638` | `14.37` | `0.0095 ± 0.0033` | `0.0328 ± 0.0088` | `0.2265 ± 0.0318` (`12.07`) | `0.7184 ± 0.1111` (`31.71`) | `0.6168 ± 0.1469` | `4.5716 ± 0.3129` | `0.1926 ± 0.0279` | `0.0378 ± 0.0126` |
| P1 | `A2a` (`n=5`, `12,865`) | `554` | `6.72` | `0.0011 ± 0.0002` | `0.0043 ± 0.0002` | `0.0366 ± 0.0036` (`1.74`) | `0.1101 ± 0.0018` (`2.87`) | `0.0586 ± 0.0018` | `0.2841 ± 0.0039` | `0.0383 ± 0.0039` | `0.0147 ± 0.0016` |
| P1 | `A2b` (`n=5`, `12,865`) | `540` | `5.83` | `0.0011 ± 0.0002` | `0.0043 ± 0.0008` | `0.0345 ± 0.0057` (`1.56`) | `0.1070 ± 0.0041` (`2.72`) | `0.0568 ± 0.0028` | `0.2841 ± 0.0041` | `0.0360 ± 0.0059` | `0.0134 ± 0.0014` |
| P1 | `M2_zv_lw` (`n=5`, `20,081`) | `1234` | `13.77` | `0.0006 ± 0.0001` | `0.0031 ± 0.0003` | `0.0303 ± 0.0019` (`1.40`) | `0.0818 ± 0.0040` (`2.44`) | `0.0295 ± 0.0035` | `0.2041 ± 0.0102` | `0.0317 ± 0.0019` | `0.0105 ± 0.0007` |
| P1 | `M3_rar_80` (`n=3`, `20,081`) | `2581` | `8.46` | `0.0005 ± 0.0001` | `0.0023 ± 0.0003` | `0.0241 ± 0.0019` (`1.14`) | `0.0664 ± 0.0027` (`2.09`) | `0.0257 ± 0.0040` | `0.1718 ± 0.0074` | `0.0255 ± 0.0020` | `0.0092 ± 0.0005` |
| P2 | `B0` (`n=5`, `12,801`) | `585` | `7.07` | `0.0469 ± 0.0128` | `0.1571 ± 0.0113` | `0.7829 ± 0.0198` (`40.98`) | `0.9949 ± 0.0042` (`34.63`) | `0.6957 ± 0.0469` | `1.0009 ± 0.0019` | `0.8241 ± 0.0185` | `0.5328 ± 0.0399` |
| P2 | `M1` (`n=5`, `12,865`) | `658` | `7.97` | `0.0018 ± 0.0003` | `0.0099 ± 0.0013` | `0.0990 ± 0.0163` (`5.85`) | `0.2730 ± 0.0149` (`10.85`) | `0.1188 ± 0.0082` | `0.6395 ± 0.0256` | `0.1021 ± 0.0177` | `0.0327 ± 0.0060` |
| P2 | `A1` (`n=5`, `12,865`) | `654` | `11.84` | `0.0122 ± 0.0016` | `0.0339 ± 0.0022` | `0.1584 ± 0.0158` (`8.29`) | `0.6017 ± 0.0341` (`33.07`) | `0.7503 ± 0.1003` | `3.9917 ± 0.2898` | `0.1392 ± 0.0155` | `0.0435 ± 0.0064` |
| P2 | `A2a` (`n=5`, `12,865`) | `551` | `9.08` | `0.0032 ± 0.0005` | `0.0146 ± 0.0010` | `0.1346 ± 0.0039` (`7.93`) | `0.3221 ± 0.0056` (`13.32`) | `0.1541 ± 0.0090` | `0.7315 ± 0.0170` | `0.1389 ± 0.0039` | `0.0439 ± 0.0010` |
| P2 | `A2b` (`n=5`, `12,865`) | `546` | `7.97` | `0.0026 ± 0.0005` | `0.0102 ± 0.0019` | `0.0943 ± 0.0115` (`5.52`) | `0.2659 ± 0.0262` (`10.49`) | `0.1148 ± 0.0180` | `0.6287 ± 0.0522` | `0.0958 ± 0.0130` | `0.0276 ± 0.0083` |
| P2 | `M2_zv_lw` (`n=5`, `20,081`) | `907` | `19.41` | `0.0010 ± 0.0002` | `0.0058 ± 0.0012` | `0.0713 ± 0.0132` (`4.46`) | `0.1668 ± 0.0238` (`8.55`) | `0.0468 ± 0.0073` | `0.3812 ± 0.0338` | `0.0741 ± 0.0130` | `0.0245 ± 0.0032` |
| P2 | `M3_rar_80` (`n=3`, `20,081`) | `2570` | `10.12` | `0.0006 ± 0.0001` | `0.0023 ± 0.0001` | `0.0273 ± 0.0052` (`1.67`) | `0.0704 ± 0.0145` (`2.91`) | `0.0234 ± 0.0011` | `0.1821 ± 0.0171` | `0.0278 ± 0.0066` | `0.0116 ± 0.0023` |
| TD | `B0_td` (`n=5`, `12,801`) | `608` | `6.86` | `0.0365 ± 0.0088` | `0.1463 ± 0.0107` | `0.7846 ± 0.0195` (`40.96`) | `0.9957 ± 0.0034` (`34.31`) | — | — | `0.8263 ± 0.0180` | `0.5406 ± 0.0379` |
| TD | `M1_td` (`n=5`, `12,865`) | `695` | `12.00` | `0.0011 ± 0.0003` | `0.0079 ± 0.0013` | `0.0924 ± 0.0135` (`5.52`) | `0.2610 ± 0.0252` (`10.40`) | — | — | `0.0971 ± 0.0149` | `0.0293 ± 0.0057` |
| TD | `A1_td` (`n=5`, `12,865`) | `552` | `9.22` | `0.0065 ± 0.0013` | `0.0314 ± 0.0038` | `0.3404 ± 0.1791` (`21.86`) | `0.7648 ± 0.3225` (`44.83`) | — | — | `0.2972 ± 0.1894` | `0.0726 ± 0.0570` |
| TD | `A2b_td` (`n=5`, `12,865`) | `510` | `7.88` | `0.0017 ± 0.0003` | `0.0086 ± 0.0018` | `0.0980 ± 0.0281` (`5.75`) | `0.2703 ± 0.0473` (`10.47`) | — | — | `0.1029 ± 0.0304` | `0.0348 ± 0.0046` |
| TD | `M2_zv_lw_td` (`n=5`, `20,081`) | `994` | `15.96` | `0.0008 ± 0.0001` | `0.0043 ± 0.0007` | `0.0590 ± 0.0149` (`3.71`) | `0.1449 ± 0.0269` (`7.37`) | — | — | `0.0629 ± 0.0158` | `0.0201 ± 0.0033` |
| TD | `M3_rar_80` (`n=3`, `20,081`) | `2592` | `9.58` | `0.0005 ± 0.0000` | `0.0025 ± 0.0007` | `0.0355 ± 0.0132` (`2.18`) | `0.0999 ± 0.0298` (`5.14`) | — | — | `0.0376 ± 0.0147` | `0.0123 ± 0.0040` |

## 3. Financial Admissibility Audit & Tail Error Percentiles (`P2`)

| Arm | Core Neg. Gamma (`|z|<=2`) | Total Neg. Gamma (/875) | Worst Min Gamma | Floor Viol. (`c<payoff`) | Gamma MAE / P95 / P99 | Vanna MAE / P95 / P99 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `A0_intr` | `89.0` | `148.0` | `-9.664e-06` | `380.0` | `2.837` / `12.973` / `29.607` | `2.170` / `10.041` / `19.527` |
| `B0` | `13.0` | `51.6` | `-4.611e-01` | `155.6` | `1.975` / `8.724` / `25.195` | `2.097` / `9.848` / `19.335` |
| `M1_noz` | `6.0` | `59.0` | `-1.619e-01` | `141.0` | `0.657` / `3.301` / `6.945` | `0.814` / `4.071` / `7.487` |
| `A1` | `6.2` | `38.0` | `-2.144e+00` | `50.0` | `0.411` / `2.114` / `4.340` | `1.016` / `4.873` / `13.091` |
| `M1` | `6.6` | `86.4` | `-3.151e-01` | `148.2` | `0.236` / `1.096` / `2.808` | `0.497` / `2.584` / `6.283` |
| `A2b` | `12.2` | `96.0` | `-5.684e-01` | `98.6` | `0.226` / `1.050` / `2.778` | `0.481` / `2.405` / `6.165` |
| `M2_zv_lw` | `4.6` | `78.4` | `-1.692e-01` | `107.4` | `0.166` / `0.771` / `1.953` | `0.283` / `1.288` / `3.460` |
| `M3_rar_80` | `0.0` | `51.3` | `-1.011e-01` | `72.0` | `0.067` / `0.312` / `0.754` | `0.122` / `0.549` / `1.711` |
| `M3_sech_80` | `0.0` | `58.0` | `-1.437e-01` | `38.0` | `0.043` / `0.194` / `0.435` | `0.083` / `0.397` / `1.095` |
| `M3_rar_112` | `0.0` | `35.0` | `-7.707e-02` | `73.0` | `0.058` / `0.249` / `0.755` | `0.107` / `0.520` / `1.362` |
| `M3_rar_112_arb` | `0.0` | `39.0` | `-7.857e-03` | `32.0` | `0.095` / `0.468` / `1.012` | `0.182` / `0.855` / `2.527` |
| `M3_rar_112_silu` | `0.0` | `45.0` | `-1.177e-01` | `18.0` | `0.058` / `0.232` / `0.613` | `0.094` / `0.396` / `0.962` |

## 4. Per-Slice $(\tau, v)$ RMSE Matrices (`M3_rar_80` Median Seed)

### P2 — Normalized Gamma RMSE by $(\tau, v)$ (`seed=1`)

| $\tau \setminus v$ | `0.005` | `0.010` | `0.040` | `0.090` | `0.160` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.02` | `0.6769` | `0.3910` | `0.0753` | `0.0367` | `0.0236` |
| `0.05` | `0.3704` | `0.2785` | `0.0735` | `0.0364` | `0.0206` |
| `0.10` | `0.4121` | `0.1808` | `0.0675` | `0.0412` | `0.0206` |
| `0.25` | `0.2332` | `0.1276` | `0.0570` | `0.0263` | `0.0178` |
| `0.50` | `0.1628` | `0.1052` | `0.0384` | `0.0208` | `0.0151` |
| `1.00` | `0.0594` | `0.0582` | `0.0342` | `0.0244` | `0.0154` |
| `2.00` | `0.0346` | `0.0310` | `0.0268` | `0.0233` | `0.0206` |

### P2 — Normalized Vanna RMSE by $(\tau, v)$ (`seed=1`)

| $\tau \setminus v$ | `0.005` | `0.010` | `0.040` | `0.090` | `0.160` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.02` | `1.1508` | `0.4211` | `0.0348` | `0.0115` | `0.0061` |
| `0.05` | `1.0789` | `0.4633` | `0.0623` | `0.0220` | `0.0100` |
| `0.10` | `0.8195` | `0.3445` | `0.0925` | `0.0372` | `0.0154` |
| `0.25` | `0.4132` | `0.2670` | `0.0933` | `0.0365` | `0.0199` |
| `0.50` | `0.2401` | `0.2002` | `0.0727` | `0.0337` | `0.0225` |
| `1.00` | `0.0854` | `0.0765` | `0.0483` | `0.0300` | `0.0217` |
| `2.00` | `0.1152` | `0.1300` | `0.0550` | `0.0277` | `0.0194` |

### TD — Normalized Gamma RMSE by $(\tau, v)$ (`seed=1`)

| $\tau \setminus v$ | `0.005` | `0.010` | `0.040` | `0.090` | `0.160` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.02` | `0.6962` | `0.4425` | `0.0846` | `0.0383` | `0.0226` |
| `0.05` | `0.8427` | `0.4088` | `0.1134` | `0.0556` | `0.0288` |
| `0.10` | `0.6516` | `0.3511` | `0.0958` | `0.0556` | `0.0304` |
| `0.25` | `0.2055` | `0.1525` | `0.0578` | `0.0300` | `0.0196` |
| `0.50` | `0.0707` | `0.0523` | `0.0399` | `0.0214` | `0.0125` |
| `1.00` | `0.0342` | `0.0284` | `0.0128` | `0.0101` | `0.0069` |
| `2.00` | `0.0138` | `0.0123` | `0.0083` | `0.0071` | `0.0053` |

### TD — Normalized Vanna RMSE by $(\tau, v)$ (`seed=1`)

| $\tau \setminus v$ | `0.005` | `0.010` | `0.040` | `0.090` | `0.160` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.02` | `1.6646` | `0.6153` | `0.0473` | `0.0147` | `0.0062` |
| `0.05` | `1.8313` | `0.8529` | `0.0995` | `0.0326` | `0.0131` |
| `0.10` | `1.3070` | `0.7183` | `0.1313` | `0.0476` | `0.0199` |
| `0.25` | `0.4069` | `0.2814` | `0.1249` | `0.0472` | `0.0211` |
| `0.50` | `0.1280` | `0.1221` | `0.0842` | `0.0357` | `0.0179` |
| `1.00` | `0.0647` | `0.0603` | `0.0205` | `0.0136` | `0.0099` |
| `2.00` | `0.0522` | `0.0469` | `0.0108` | `0.0091` | `0.0072` |

## 5. Amortized 8D/10D Parametric Surrogate — Zero-Shot Evaluation Across 4 Regimes

| Test Regime | Model | Price Rel RMSE | Gamma Rel RMSE | Vanna Rel RMSE | Vega Rel RMSE | Volga Rel RMSE | `short_low_var` Gamma | Infer (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `P1` | `B0_param` | `0.0762` | `0.7423` | `0.9906` | `0.7760` | `1.0023` | `0.7892` | `13.03` |
| `P1` | `M_param` | `0.0015` | `0.0231` | `0.0517` | `0.0347` | `0.1550` | `0.0237` | `19.74` |
| `P1` | `M_param_plus` | `0.0014` | `0.0260` | `0.0728` | `0.0338` | `0.2106` | `0.0270` | `11.65` |
| `P1` | `M3_param` | `0.0008` | `0.0175` | `0.0453` | `0.0199` | `0.1422` | `0.0183` | `9.72` |
| `P1` | `M3_param_plus` | `0.0010` | `0.0155` | `0.0412` | `0.0291` | `0.1332` | `0.0160` | `10.06` |
| `P2` | `B0_param` | `0.0846` | `0.7808` | `0.9929` | `0.7947` | `1.0023` | `0.8226` | `11.66` |
| `P2` | `M_param` | `0.0020` | `0.0671` | `0.2021` | `0.0897` | `0.4959` | `0.0678` | `18.11` |
| `P2` | `M_param_plus` | `0.0027` | `0.0853` | `0.1906` | `0.0579` | `0.4378` | `0.0904` | `10.26` |
| `P2` | `M3_param` | `0.0015` | `0.0594` | `0.1763` | `0.0563` | `0.4511` | `0.0624` | `10.61` |
| `P2` | `M3_param_plus` | `0.0009` | `0.0401` | `0.1085` | `0.0372` | `0.2626` | `0.0434` | `9.78` |
| `P3_steep` | `B0_param` | `0.1369` | `0.8112` | `0.9949` | `0.8186` | `1.0026` | `0.8514` | `11.07` |
| `P3_steep` | `M_param` | `0.0034` | `0.1371` | `0.3606` | `0.1464` | `0.7753` | `0.1425` | `16.97` |
| `P3_steep` | `M_param_plus` | `0.0024` | `0.1450` | `0.3078` | `0.1088` | `0.6590` | `0.1536` | `9.96` |
| `P3_steep` | `M3_param` | `0.0021` | `0.1271` | `0.3379` | `0.1149` | `0.7410` | `0.1327` | `11.34` |
| `P3_steep` | `M3_param_plus` | `0.0012` | `0.0725` | `0.1718` | `0.0489` | `0.4056` | `0.0762` | `10.07` |
| `P4_calm` | `B0_param` | `0.0451` | `0.7432` | `0.9921` | `0.7715` | `1.0021` | `0.7892` | `10.39` |
| `P4_calm` | `M_param` | `0.0008` | `0.0184` | `0.0359` | `0.0173` | `0.1022` | `0.0191` | `17.12` |
| `P4_calm` | `M_param_plus` | `0.0011` | `0.0233` | `0.0515` | `0.0285` | `0.1433` | `0.0247` | `10.43` |
| `P4_calm` | `M3_param` | `0.0009` | `0.0126` | `0.0269` | `0.0197` | `0.0842` | `0.0129` | `11.21` |
| `P4_calm` | `M3_param_plus` | `0.0011` | `0.0136` | `0.0247` | `0.0236` | `0.1167` | `0.0139` | `9.79` |

## 6. Non-Affine 2D Heston Stochastic-Local Volatility (`Heston-SLV`) Benchmark

| Arm | Train (s) | Infer (ms) | Overall `c` | Overall `cx` | Overall **Gamma** | Overall **Vanna** | Overall **Vega** | Overall **Volga** | `short_low_var` **Gamma** | Core Neg. Gamma / Floor Viol. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B0_slv` | `560` | `8.94` | `0.0304` | `0.1321` | `0.7409` | `0.9835` | `0.5932` | `0.9989` | `0.7866` | `18` / `160` |
| `M3_slv` | `1997` | `9.87` | `0.0003` | `0.0014` | `0.0163` | `0.0345` | `0.0150` | `0.1286` | `0.0153` | `0` / `29` |

## 7. Dynamic Delta & Delta-Gamma-Vega Hedging Backtest (`P2`, 15,000 Exact CIR Paths)

| Scenario | Hedge Engine | Delta-Only RMSE ($) | DGV RMSE ($) | DGV 95% CVaR Loss ($) | DGV Tracking RMSE vs Exact ($) |
| --- | --- | ---: | ---: | ---: | ---: |
| Low Var ($v_0=0.01, T=0.05$) | `Exact Fourier` | `0.3343` | `0.0509` | `0.1248` | `0.0000` |
| Low Var ($v_0=0.01, T=0.05$) | `M3_rar_80` | `0.3339` | `0.0515` | `0.1235` | `0.0116` |
| Low Var ($v_0=0.01, T=0.05$) | `M3_param_plus` | `0.3332` | `0.0516` | `0.1212` | `0.0192` |
| Low Var ($v_0=0.01, T=0.05$) | `M2_zv_lw` | `0.3262` | `0.0520` | `0.1183` | `0.0263` |
| Low Var ($v_0=0.01, T=0.05$) | `M1` | `0.3246` | `0.0605` | `0.1365` | `0.0421` |
| Low Var ($v_0=0.01, T=0.05$) | `B0` | `0.5605` | `0.2856` | `0.2989` | `0.2864` |
| Mod Var ($v_0=0.04, T=0.05$) | `Exact Fourier` | `0.4218` | `0.0684` | `0.1505` | `0.0000` |
| Mod Var ($v_0=0.04, T=0.05$) | `M3_rar_80` | `0.4226` | `0.0691` | `0.1546` | `0.0089` |
| Mod Var ($v_0=0.04, T=0.05$) | `M3_param_plus` | `0.4226` | `0.0686` | `0.1443` | `0.0111` |
| Mod Var ($v_0=0.04, T=0.05$) | `M2_zv_lw` | `0.4220` | `0.0689` | `0.1463` | `0.0133` |
| Mod Var ($v_0=0.04, T=0.05$) | `M1` | `0.4151` | `0.0707` | `0.1584` | `0.0188` |
| Mod Var ($v_0=0.04, T=0.05$) | `B0` | `0.8091` | `0.4805` | `0.5444` | `0.4753` |

## 8. Millisecond Differentiable Neural Calibration (`M3_param_plus` vs SciPy DE + Fourier)

| Regime | Quotes ($t_1/t_2$) | DE+Fourier $t_1$ Time (ms) | Neural Autograd $t_1$ Time (ms) | Speedup | DE Day-2 Fit RMSE (bps) | Neural Day-2 Fit RMSE (bps) | Surrogate vs Fourier $t_2$ (bps) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P1_FellerOK` | `77/80` | `7088` | `256` | **`27.7x`** | `3.73` | `4.96` | `3.60` |
| `P2_FellerBad` | `79/82` | `9795` | `230` | **`42.5x`** | `1.88` | `3.07` | `2.93` |
| `P3_SteepSkew` | `79/76` | `15310` | `221` | **`69.2x`** | `2.08` | `4.37` | `3.00` |

## 9. Live Cboe SPX / SPXW Empirical Market Validation (`Nov 19, 2026`, `^SPX = 7,205.56`)

| Expiry | Series | $\tau$ (yrs) | Discount $D(\tau)$ | Forward $F(\tau)$ | Implied $r$ (%) | Parity $R^2$ | Split |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `2026-12-18` | `SPX` | `0.0788` | `0.99623` | `7225.12` | `4.79` | `0.9999999` | In-Sample Cal |
| `2027-01-15` | `SPX` | `0.1555` | `0.99246` | `7250.72` | `4.87` | `0.9999999` | Held-Out Test |
| `2027-02-19` | `SPX` | `0.2513` | `0.98782` | `7276.49` | `4.88` | `0.9999999` | In-Sample Cal |
| `2027-03-19` | `SPX` | `0.3280` | `0.98413` | `7295.32` | `4.88` | `0.9999999` | Held-Out Test |
| `2027-06-17` | `SPX` | `0.5744` | `0.97193` | `7381.64` | `4.96` | `1.0000000` | In-Sample Cal |
| `2027-06-30` | `SPXW` | `0.6107` | `0.97074` | `7390.71` | `4.86` | `0.9999995` | Held-Out Test |
| `2027-12-17` | `SPX` | `1.0754` | `0.94675` | `7538.66` | `5.09` | `1.0000000` | In-Sample Cal |
| `2028-01-21` | `SPX` | `1.1712` | `0.94173` | `7575.02` | `5.13` | `1.0000000` | In-Sample Cal |

| Live SPX Model | 2,533-Quote PINN vs Fourier (bps) | 2,533-Quote PINN vs Market (bps) | `short_mod_var` Gamma | `short_mod_var` Vega | `short_mod_var` Vanna | Overall Gamma | Overall Vega |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` | `108.38` | `108.82` | `0.5743` | `0.5797` | `0.9292` | `0.8199` | `0.5706` |
| `M3_sech_80` | `1.58` | `7.62` | `0.0201` | `0.0113` | `0.0397` | `0.0617` | `0.0207` |

### Multi-Year Historical SPX/SPXW Day-1 -> Day-2 Out-of-Sample Benchmark (`2023-2025`, `168` Days / `4.12M` Quotes)

| Regime | Quotes ($t_1/t_2$) | $v_0^*(t_1) \to v_0^*(t_2)$ | DE $t_1$ (s) | Autograd $t_1$ (s) | Speedup | DE $t_2$ OOS (bps) | Autograd $t_2$ OOS (bps) | Surr vs Fourier $t_2$ (bps) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2024_Jun_CalmBull` | `1177/1243` | `0.0106 -> 0.0093` | `207.7` | `1.75` | **`118.9x`** | `6.48` | `6.67` | `3.05` |
| `2024_Aug_YenCarrySpike` | `1316/1526` | `0.0784 -> 0.0634` | `110.6` | `1.93` | **`57.4x`** | `34.17` | `33.16` | `7.78` |
| `2024_Nov_USElection` | `1600/1185` | `0.0256 -> 0.0196` | `145.7` | `2.19` | **`66.7x`** | `7.14` | `7.19` | `4.72` |
| `2025_Mar_Q1Selloff` | `1438/1639` | `0.0791 -> 0.0736` | `124.0` | `1.97` | **`62.9x`** | `10.76` | `11.56` | `4.27` |

- **168-Day Full-Archive Rolling Summary (`M3_param_wide`, `4.12M` Quotes)**: `168` days calibrated in `486.5s` (`1493ms`/day), Feller violated on **`168/168` (`100.0%`)** days, median In-Sample RMSE **`9.86 bps`**, median Next-Day OOS RMSE **`10.1 bps`**, median Surrogate-vs-Fourier **`3.98 bps`**.

### Quarter-Century Wharton WRDS OptionMetrics IvyDB US Study (`2000-2025`, `1,281,892` Rows, `78` Days, `113,063` Held-Out OOS Quotes)

| Era (Anchor) | Surrogate | OOS Quotes | DE $t_1$ (s) | Autograd $t_1$ (s) | Speedup | DE Week OOS (bps) | Autograd Week OOS (bps) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2000_Apr_DotCom` (`2000-04-10`) | `M3_param_wide` | `790` | `13.5` | `1.40` | **`9.6x`** | `17.15` | `19.44` |
| `2008_Oct_GFC` (`2008-10-14`) | `M3_param_crisis_v2` | `598` | `9.0` | `0.43` | **`20.9x`** | `38.63` | `49.03` |
| `2011_Aug_USDowngrade` (`2011-08-08`) | `M3_param_crisis_v2` | `1165` | `26.8` | `0.87` | **`30.8x`** | `27.72` | `31.77` |
| `2015_Aug_FlashCrash` (`2015-08-24`) | `M3_param_wide` | `3682` | `37.3` | `1.00` | **`37.3x`** | `28.06` | `32.91` |
| `2018_Feb_Volmageddon` (`2018-02-05`) | `M3_param_crisis_v2` | `11254` | `42.7` | `0.99` | **`43.3x`** | `26.17` | `32.14` |
| `2020_Mar_COVID_FullMonth` (`2020-03-16`) | `M3_param_crisis_v2` | `9854` | `65.1` | `2.53` | **`25.8x`** | `53.46` | `51.32` |
| `2021_Jul_LowVol_Bull` (`2021-07-12`) | `M3_param_wide` | `24509` | `305.0` | `2.67` | **`114.4x`** | `10.31` | `10.03` |
| `2022_Oct_Inflation_Bear` (`2022-10-10`) | `M3_param_wide` | `27735` | `102.7` | `2.60` | **`39.4x`** | `14.31` | `17.96` |
| `2025_Aug_Latest_FullMonth` (`2025-08-25`) | `M3_param_wide` | `33476` | `343.4` | `2.64` | **`130.1x`** | `9.03` | `9.25` |

- **4-Crisis Single-Surface PINNs & Full-Month Rolling:** `2000-04-14` Dot-Com `3.31` vs `83.91 bps` (`25.4x`), `2011-08-08` Downgrade `36.49` vs `108.44 bps` (`3.0x`), `2018-02-05` Volmageddon `12.44` vs `71.27 bps` (`5.7x`), and `2020-03-16` COVID-19 (`VIX = 82.69`, $\sqrt{v_0^*} = 92.58\%$) **`57.36 bps`** vs `315.99 bps` (`5.5x`); Feller violated on **`78/78` (`100.0%`)** WRDS trading days (`248/248` total SPX days `2000-2026`).

## 10. Complete Publication Figure Index

- [fig1_gamma_error_surface_p2.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig1_gamma_error_surface_p2.png): P2 Gamma Error Surface (`B0` vs `M1`)
- [fig2_vanna_error_surface_p2.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig2_vanna_error_surface_p2.png): P2 Vanna Error Surface (`B0` vs `M1`)
- [fig3_boundary_layer_collapse.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig3_boundary_layer_collapse.png): Constant-Param Near-Expiry Boundary-Layer Collapse in Scaled Coordinate $z$
- [fig4_td_gamma_error_surface.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig4_td_gamma_error_surface.png): Time-Dependent Heston (`T15`) Gamma Error Surface vs. ADI Oracle (`B0_td` vs `M1_td` vs `A2b_td`)
- [fig5_td_vanna_error_surface.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig5_td_vanna_error_surface.png): Time-Dependent Heston (`T15`) Vanna Error Surface vs. ADI Oracle
- [fig6_td_boundary_layer_collapse.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig6_td_boundary_layer_collapse.png): Time-Dependent Near-Expiry Collapse in Spectral Coordinate $z$
- [fig7_tier2_accuracy_comparison.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig7_tier2_accuracy_comparison.png): Tier-2 (`M2_zv_lw` / `M2_zv_lw_td`) vs `M1` & `B0` Across All 6 Outputs
- [fig8_param8d_zeroshot.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig8_param8d_zeroshot.png): 8D/10D Amortized Parametric Surrogate Zero-Shot Accuracy Across 4 Regimes
- [fig9_m3_convergence_trajectories.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig9_m3_convergence_trajectories.png): Multi-Round RAR-L-BFGS Optimization & Greek Convergence Dynamics (`M3`)
- [fig10_m3_error_surfaces_p2_and_td.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig10_m3_error_surfaces_p2_and_td.png): Error Surface Collapse: `B0` vs `M1` vs Championship `M3_rar_80` on `P2` and `TD`
- [fig11_hardest_slice_profiles.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig11_hardest_slice_profiles.png): Hardest Corner Slice (`P2`, $\tau=0.05, v=0.005$): Exact vs `B0` vs `M1` vs `M3_rar_80` (Gamma/Vanna/Vega/Volga)
- [fig12_ablation_ladder_and_arbitrage.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig12_ablation_ladder_and_arbitrage.png): Full Mechanistic Ablation Ladder & Financial Admissibility Audit (`P2`)
- [fig13_dynamic_hedging_pnl.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig13_dynamic_hedging_pnl.png): Dynamic Delta & Delta-Gamma-Vega Hedging Replication PnL Backtest (`P2`, Exact CIR Paths)
- [fig14_slv_benchmark.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig14_slv_benchmark.png): Non-Affine 2D Heston Stochastic-Local Volatility (`Heston-SLV`): `B0_slv` vs `M3_slv`
- [fig15_diff_calibration.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig15_diff_calibration.png): Millisecond Differentiable Heston Calibration via 10D `M3_param_plus` vs SciPy DE + Fourier
- [fig16_real_spx_market.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig16_real_spx_market.png): Live Cboe SPX / SPXW Empirical Market Validation (Parity Curve, Held-Out Smiles, & Stressed Greeks)
- [fig17_spx_168day_rolling_calibration.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig17_spx_168day_rolling_calibration.png): Full-Archive 168-Trading-Day Rolling SPX Calibration & Next-Day OOS Backtest (`4.12M` Quotes, `M3_param_wide`)
- [fig18_wrds_multidecade_study.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig18_wrds_multidecade_study.png): Quarter-Century Wharton WRDS OptionMetrics IvyDB US 9-Era SPX Validation (`2000-2025`, `1.28M` Rows)
- [fig19_wrds_march2020_aug2025_trajectories.png](file:///Users/longchaudo/Desktop/Research/CV-RBA-PINN/paper/fig19_wrds_march2020_aug2025_trajectories.png): Continuous Full-Month WRDS OptionMetrics Rolling Trajectories: March 2020 COVID-19 (`22d`) vs August 2025 (`21d`)
