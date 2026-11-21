# SPX / SPXW Market Data Inventory (`2000–2026`, `248+` Full-Surface Trading Days, `5.44M+` Quotes)

All downloaded live and historical S&P 500 (`SPX` / `SPXW`) option chain datasets spanning **27 calendar years (`2000–2026`)** are stored locally in this folder and auto-detected by `spx_w8.py`:

## 1. Wharton WRDS OptionMetrics IvyDB US (`wrds_optionmetrics/`, `2000–2025`, `1,281,892` Unique Rows Across `78` Trading Days)
Queried from Wharton Research Data Services (`optionm_all`, `SECID = 108105` S&P 500 Index, 19 columns including `best_bid`, `best_offer`, `am_settlement`, `impl_volatility`, `delta`, `gamma`, `vega`):
1. `wrds_spx_2000_04_dotcom.csv` (`2000-04-10` to `2000-04-14`, `2,651` rows — Dot-Com tech bust, `~6%` interest-rate era)
2. `wrds_spx_2008_10_gfc.csv` (`2008-10-13` to `2008-10-17`, `8,209` rows — Global Financial Crisis, VIX `~70–80`)
3. `wrds_spx_2011_08_downgrade.csv` (`2011-08-08` to `2011-08-12`, `10,430` rows — US AAA sovereign credit downgrade & Eurozone crisis)
4. `wrds_spx_2015_08_flashcrash.csv` (`2015-08-24` to `2015-08-28`, `43,458` rows — August 2015 China / ETF flash crash)
5. `wrds_spx_2018_02_volmageddon.csv` (`2018-02-05` to `2018-02-09`, `65,072` rows — February 2018 "Volmageddon" `+115%` single-day VIX spike)
6. `wrds_spx_2020_03_fullmonth.csv` (`2020-03-02` to `2020-03-31`, `432,450` rows — **complete 22-trading-day continuous month** of the COVID-19 crash & Fed QE turnaround, VIX `33 -> 82.69 -> 53`; also subset `wrds_spx_2020_03_covid.csv`)
7. `wrds_spx_2021_07_bull.csv` (`2021-07-12` to `2021-07-16`, `101,014` rows — post-stimulus low-volatility bull market, $v_0 \approx 0.012$)
8. `wrds_spx_2022_10_bear.csv` (`2022-10-10` to `2022-10-14`, `102,114` rows — Fed rate-hike bear market & Oct-13 CPI reversal)
9. `wrds_spx_2025_08_fullmonth.csv` (`2025-08-01` to `2025-08-29`, `516,494` rows — **complete 21-trading-day continuous month** of latest subscribed IvyDB US data, merged from `wrds_spx_2025_08_part1.csv` and `wrds_spx_2025_08_latest.csv`)

*(Note: Per WRDS Terms of Use, raw licensed `wrds_optionmetrics/*.csv` files are excluded via `.gitignore` and never pushed to public GitHub; all derived calibration summaries, checkpoints, training telemetry logs, and figures in `runs/` and `paper/` are tracked.)*

## 2. Open & Cboe SPX / SPXW Datasets (`2023–2026`)
- **`spx_168days_2024_2025_csv/`** (`168` consecutive trading days, `2024-06-03.csv` through `2025-03-13.csv`, `4,119,715` quotes; raw ZIP/CSVs gitignored due to GitHub's `100 MB` file limit).
- **`spx_cboe_datashop_2023_08_25.csv`** (`19,366` `SPX`/`SPXW` rows, `44` clean parity slices, `6,926` OTM quotes — **tracked in Git**).
- **`spx_cboe_all20months_2026_11_19.csv`** (`12,522` quotes across **all 20/20** Cboe expiration months `2026 Nov` through `2031 Dec` LEAPS — **tracked in Git** alongside `spx_cboe_2026_11_19.csv` and `cboe_2026_11_19_by_month/`).
- **`spx_daily_2026_aug_nov/`** (`65` daily `0DTE–7DTE` CSVs, `2026-08-25` to `2026-11-19`) and **`historical_archives/spxOptionsChain_2024_05_to_2024_07_intraday.zip`** (`37` intraday days).
