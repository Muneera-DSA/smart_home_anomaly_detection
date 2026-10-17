# Model card: smart-home appliance anomaly detector (validated on fridges and freezers)

## What it does
Every hour, scores each fridge, freezer or fridge-freezer monitored by a smart plug for abnormal compressor behaviour
(stuck on, cycling far longer than usual). Scores above a threshold become alerts; alert hours less than 3 h apart merge
into one alert. It is supervised anomaly detection: trained on researcher-annotated anomalies, and compared
with rules and an unsupervised Isolation Forest, which caught far fewer (38%).

## Model
`HistGradientBoostingClassifier` (300 iterations, learning rate 0.05, 31 leaves, min 50 samples per leaf, L2 = 1.0),
class-balanced sample weights. Inputs: 20 features per appliance-hour (see `src/smart_energy/features.py`), all computed
from data up to the end of the hour. Hour of day and day of year are deliberately excluded.

## Training data
REFIT (University of Strathclyde, CC BY 4.0): 32 cold appliances in 19 UK homes, 2013-2015; 347,656 scored
appliance-hours, 243 annotated anomalies (Rashid et al., ICASSP 2019). Each appliance's first 30 days are a reference
period and are not scored.

## Evaluation
Leave-houses-out (5 folds); thresholds set on training homes only; event-level scoring (an alert within ± 2 h of an
annotated anomaly's span catches it).

| At ≤ 1 false alert per appliance-month (deployable) | HGB | On-timer rule |
|---|---|---|
| Anomalies caught | 210 / 243 (86.4%) | 190 / 243 (78.2%) |
| False alerts per appliance-month | 0.86 | 1.18 |
| Median delay after anomaly start | 2.5 h | 3.0 h |

At an identical false-alert rate: 87.2% (95% CI over homes 0.78-0.94) vs 77.4% (0.64-0.88). The per-home difference
is not statistically significant (Wilcoxon over 18 homes, Holm p = 0.075); earlier alerting is (p = 0.0001).

## Intended use
Household alerts or a monitoring dashboard for cold appliances on plug-level power sensors with readings at least
every minute. The threshold should be set from the false-alarm budget the product can tolerate (see the recall vs
false-alert curve in the README).

## Not intended for
Other appliances (too few labels to validate); aggregate (whole-house) meters; safety-critical cold chains (medical,
food retail) without local validation.

## Known limitations and risks
* Labels are incomplete: some "false" alerts are real unannotated anomalies, so false-alert rates are upper bounds.
* Short anomalies (≤ 4 h) are caught less often (68%).
* A frozen sensor looks like a stuck-on compressor; route alerts with unchanged readings (`flat_frac` ≥ 0.9) to a
  "check the sensor" message rather than a repair request.
* Needs about 3 days of history for its relative features and 30 days for its first-month reference; use the on-timer
  rule for new appliances.
* Trained on one UK region in 2013-2015; recalibrate the threshold when deploying elsewhere.

## Explainability
Exact TreeSHAP values (additive to the model's log-odds to within 1e-14) are produced per alert; the strongest drivers
are the running share over the last 6 and 24 hours and the running streak relative to the appliance's first month.
