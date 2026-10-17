# Data (not stored in this repository)

Both sources are licensed CC BY 4.0 by the University of Strathclyde.

| File | Where | Put it in |
|---|---|---|
| `REFITPowerData111215.7z` (386 MB, 20 houses, 8-second readings) or the cleaned release `CLEAN_REFIT_081116.7z` | [REFIT: Electrical Load Measurements](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/) | extract the `House*.csv` files to `data/raw/refit/` |
| `REFIT_anomalies.csv` (26 KB) | [Annotated load anomalies from the REFIT dataset](https://pureportal.strath.ac.uk/en/datasets/annotated-load-anomalies-from-the-refit-dataset/) | `data/raw/REFIT_anomalies.csv` |

Then build the 1-minute files (about 2 minutes):

```bash
python scripts/ingest_refit.py data/raw/refit data/interim
```

This project was run on `REFITPowerData111215.7z`. Its CSVs have no header: `unix_time_utc, aggregate, appliance1..9` in watts.
The appliance in each column is listed per house in the REFIT readme and mapped in `src/smart_energy/labels.py`.

Please cite: D. Murray et al., "An electrical load measurements dataset of United Kingdom households from a two-year
longitudinal study", Scientific Data 4, 160122 (2017); and H. Rashid, V. Stankovic, L. Stankovic, P. Singh, "Evaluation of
Non-Intrusive Load Monitoring Algorithms for Appliance-level Anomaly Detection", ICASSP 2019.
