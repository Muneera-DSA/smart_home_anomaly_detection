"""Paths and fixed choices. Every number that shapes a result lives here so it can be cited and changed in one place."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("REFIT_DATA", ROOT / "data"))
MINUTE_DIR = DATA / "interim"                     # house{N}.pkl from scripts/ingest_refit.py
LABELS_CSV = DATA / "raw" / "REFIT_anomalies.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"

SEED = 42

# appliance "on" threshold: half the median power of clearly-running minutes in the channel's FIRST 30 days
THRESHOLD_DAYS = 30
MIN_ON_W = 10.0

# past-only baselines
BASELINE_HOURS = 14 * 24          # two weeks of history
BASELINE_MIN_HOURS = 3 * 24       # need three days before any relative feature exists
WARMUP_HOURS = 30 * 24            # the first 30 days (the reference window) of every appliance are not scored

# labels
MIN_OVERLAP_MIN = 30              # an hour is positive if >= 30 minutes fall inside an annotated event
EVENT_TOLERANCE_H = 2             # an alert within 2 hours of an event's span counts as hitting it

# evaluation
N_FOLDS = 5                       # leave-houses-out folds (GroupKFold by house)
FALSE_ALERTS_PER_APPLIANCE_MONTH = 1.0   # alert budget used to set every detector's threshold on training houses
ALERT_GAP_H = 3                   # alert hours closer than this merge into one alert

# business assumptions (stated in the README, not measured)
PRICE_PER_KWH_GBP = 0.15          # approximate UK domestic unit rate in 2014-15
