# SPX / SPXW Market Data Drop Folder (Workstream W8)

Drop two daily European-style SPX/SPXW option chain CSV snapshots (`t1` calibration day and `t2` out-of-sample $v_0$ day, ideally captured during regular liquid US hours `10:00–15:45 ET`) into this directory, e.g.:
- `docs/market_data/spx_t1.csv`
- `docs/market_data/spx_t2.csv`

Supported schemas (auto-detected by `spx_w8.py`):
- **WRDS OptionMetrics**: `date`, `exdate`, `cp_flag`, `strike_price` (auto-scaled if `>= 100000` or via `--strike-scale`), `best_bid`, `best_offer`, `am_settlement`, `symbol` / `root`
- **Cboe DataShop / Kaggle / Broker CSV**: `quote_date` / `date`, `expiration` / `exdate`, `option_type` / `cp_flag` (`C`/`P` or `call`/`put`), `strike`, `bid`, `ask`, optional `root` / `symbol` (`SPX` AM-SOQ vs `SPXW` PM-close)

Once both CSVs are placed here, run (adding `--train-pinn --arm M2_zv_lw` to benchmark the upgraded Tier-2 PINN, or `--param-ckpt runs/param/checkpoints/M_param_plus_seed0.pt` for zero-shot amortized evaluation):
```bash
.venv/bin/python spx_w8.py --t1 docs/market_data/spx_t1.csv --t2 docs/market_data/spx_t2.csv --train-pinn --arm M2_zv_lw
```
