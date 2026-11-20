# SPX / SPXW Market Data Inventory (`2008–2026`, `295+` Trading Days, `4.6+M` Quotes)

All downloaded live and historical S&P 500 (`SPX` / `SPXW`) option chain datasets spanning **`2008–2026`** are stored locally in this folder and auto-detected by `spx_w8.py`:

## 1. Wharton WRDS OptionMetrics IvyDB US (`wrds_optionmetrics/`, `2008–2025`, `434,763` Rows)
Queried from Wharton Research Data Services (`optionm_all`, `SECID = 108105` S&P 500 Index, 19 columns including `best_bid`, `best_offer`, `am_settlement`, `impl_volatility`, `delta`, `gamma`, `vega`):
- `wrds_spx_2008_10_gfc.csv` (`2008-10-13` to `2008-10-17`, `8,209` rows — Global Financial Crisis, VIX `~70–80`)
- `wrds_spx_2020_03_covid.csv` (`2020-03-16` to `2020-03-20`, `102,490` rows — COVID-19 Black Monday, all-time closing VIX peak `82.69`)
- `wrds_spx_2021_07_bull.csv` (`2021-07-12` to `2021-07-16`, `101,014` rows — post-stimulus low-volatility bull market, $v_0 \approx 0.012$)
- `wrds_spx_2022_10_bear.csv` (`2022-10-10` to `2022-10-14`, `102,114` rows — Fed rate-hike bear market & Oct-13 CPI reversal)
- `wrds_spx_2025_08_latest.csv` (`2025-08-25` to `2025-08-29`, `120,936` rows — latest subscribed IvyDB US week, `45–50` parity slices/day)

*(Note: Per WRDS Terms of Use, raw licensed `wrds_optionmetrics/*.csv` files are excluded via `.gitignore` and never pushed to public GitHub; all derived calibration summaries, checkpoints, and figures in `runs/` and `paper/` are tracked.)*

## 2. Open & Cboe SPX / SPXW Datasets (`2023–2026`)
- **`spx_168days_2024_2025_csv/`** (`168` consecutive trading days, `2024-06-03.csv` through `2025-03-13.csv`, `4,119,715` quotes):
  - Full-term-structure daily AM-settled `SPX` + PM-settled `SPXW` option chains unpacked from `historical_archives/OptionData_SPX_2024_06_to_2025_03_parquet.zip` (`659 MB`; gitignored due to GitHub's `100 MB` file limit).
- **`spx_cboe_datashop_2023_08_25.csv`** (`19,366` `SPX`/`SPXW` rows, `44` clean parity slices, `6,926` OTM quotes — **tracked in Git**):
  - Official **Cboe DataShop** free sample snapshot (`2023-08-25` at `15:45 ET`), including synchronized `bid_1545`, `ask_1545`, `implied_volatility_1545`, and Cboe Greeks.
- **`spx_cboe_all20months_2026_11_19.csv`** (`12,522` quotes across **all 20/20** Cboe expiration months `2026 Nov` through `2031 Dec` LEAPS — **tracked in Git** alongside `spx_cboe_2026_11_19.csv` and `cboe_2026_11_19_by_month/`).
- **`spx_daily_2026_aug_nov/`** (`65` daily `0DTE–7DTE` CSVs, `2026-08-25` to `2026-11-19`, `89.3 MB`) and **`historical_archives/spxOptionsChain_2024_05_to_2024_07_intraday.zip`** (`59 MB`, `37` intraday days).

## 3. Example Commands
```bash
# 1. Millisecond differentiable PyTorch autograd Day-1 -> Day-2 calibration on WRDS OptionMetrics:
.venv/bin/python spx_w8.py --combined-csv docs/market_data/wrds_optionmetrics/wrds_spx_2025_08_latest.csv \
                           --d1 2025-08-25 --d2 2025-08-26 --diff-cal

# 2. Full 17-year (2008-2025) WRDS OptionMetrics training & out-of-sample benchmark:
.venv/bin/python wrds_optionmetrics_study.py

# 3. Full 168-day (4.12M-quote) rolling SPX calibration & OOS study:
.venv/bin/python train_wide_spx_surrogate.py
```
