# Where this project started: audit of my first smart-home notebook

`notebooks/original/00_original_smart_home_project.ipynb` is my original course project, kept unchanged. It used the
Kaggle "Smart Home Dataset with weather information" (one house, 2016, 1-minute appliance readings plus weather).
Before rebuilding I audited it. The checks below were re-run on the raw file; the numbers are reproducible.

## What it got right (kept)
- Noticed the `time` column counts seconds although rows are one minute apart, and rebuilt a minute index.
- Found the duplicate columns (`use` = `House overall`, `gen` = `Solar`), the text values `cloudCover` and the broken last row.
- Merged `Furnace 1+2` and `Kitchen 12/14/38`.

## What made its conclusions unreliable
| # | Problem | Evidence | Consequence |
|---|---|---|---|
| 1 | The minute index starts at `2016-01-01 05:00` (midnight local time written as UTC) and hours were then read as local time | solar output peaks at 16:00 under that labelling; physically it peaks near noon | "solar is highest 3-6 pm" and "consumption is highest 7 pm-4 am" are 5-hour shift artefacts; the hourly weather is also misaligned with the energy rows |
| 2 | `House overall` was predicted from the same hour's sub-meter readings | the sub-meters are parts of the house total; a plain linear fit explains 56% of its variance in-sample | R² = 0.83 measures reconstruction of a total from its parts, not a forecast; random `train_test_split` on hourly data adds leakage between neighbouring hours |
| 3 | Per-minute kW were summed per hour and treated as kWh | hourly mean 51.5 "kWh" is 0.86 kWh (kW-minutes / 60) | energy labels and pie-chart totals are off by a factor of 60 |
| 4 | Isolation Forest on daily/weekly/monthly totals, never evaluated | its strongest "anomalies" are 2016-12-16 (210 minutes of data) and December (a partial month) | the main detections are incomplete periods; nothing was checked against known anomalies |
| 5 | Bugs | Holt-Winters fitted on `np.array(train).reshape(-1,1)` of a shuffled 17-column frame; Lasso with `alpha=1` zeroed every coefficient; two skew "fixes" increased skew (Fridge 1.65 → 2.57, dew point −0.28 → 1.34) | those model results are meaningless |

## What changed in this rebuild
The biggest limitation was not a bug: **one house with no anomaly labels cannot show whether a detector finds real
anomalies.** This repository therefore moved to REFIT (20 UK homes, appliance-level IoT plugs) with anomalies annotated by
researchers, evaluates on homes the model never saw, measures time on one clock (UTC, verified against the labels), and
reports energy in kWh.
