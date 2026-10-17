"""Appliance-hour features for cold appliances (fridges, freezers, fridge-freezers), using only data up to the end of
each hour, so a detector could raise its alert at that moment.

Behaviour of a compressor in one hour:
  duty            share of minutes the compressor is running
  starts          number of off->on switches (a healthy unit cycles; a stuck one does not)
  run_max         longest running streak reaching into this hour, in minutes (a streak can start hours earlier)
  p_on, p_on_std  mean and spread of power while running
  flat_frac       share of running minutes whose reading did not change at all (a frozen sensor repeats one value)
  coverage        share of minutes with any reading; rate = sensor readings per minute
Recent history: duty_6h, duty_24h  (running share over the last 6 and 24 hours)
Relative to the SAME appliance's own past two weeks (rolling, shifted so the current hour is never in its baseline):
  duty_rel, run_rel, starts_rel, p_on_rel, duty_6h_rel, duty_24h_rel, rate_rel
Relative to the appliance's first month ("commissioning" reference, frozen so later faults cannot contaminate it):
  run_ref = run_max / 95th percentile of run_max in the first 30 days
Ratios to an appliance's own history are what let one model work across 19 different homes and 32 different fridges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TYPES = ["fridge", "freezer", "fridge_freezer"]
RAW = ["duty", "starts", "run_max", "p_on", "p_on_std", "flat_frac", "coverage", "rate"]
RECENT = ["duty_6h", "duty_24h"]
RELATIVE = ["duty_rel", "run_rel", "starts_rel", "p_on_rel", "duty_6h_rel", "duty_24h_rel", "rate_rel", "run_ref"]
# Calendar features (hour of day, day of year) are computed for EDA but NOT given to the models: with leave-houses-out
# validation they let a model learn WHEN annotators were busy across all homes (most labels fall in 2014) rather than how
# a faulty compressor behaves. Removing them raised out-of-fold recall (0.835 -> 0.864 at the same budget).
CONTEXT = ["is_freezer", "is_fridge_freezer"]
FEATURES = RAW + RECENT + RELATIVE + CONTEXT


def on_threshold(p: pd.Series) -> float:
    first = p[p.index < p.index[0] + pd.Timedelta(days=config.THRESHOLD_DAYS)]
    running = first[first > config.MIN_ON_W]
    if len(running) < 60:
        running = p[p > config.MIN_ON_W]
    return float(max(config.MIN_ON_W, 0.5 * running.median()))


def minute_signals(p: pd.Series, readings: pd.Series, thr: float) -> pd.DataFrame:
    have = p.notna()
    on = (p > thr) & have
    run = on.groupby((~on).cumsum()).cumsum()                    # minutes in the current running streak
    start = on & ~on.shift(1, fill_value=False)
    flat = on & (p.diff().abs() < 0.05)
    return pd.DataFrame({"have": have, "on": on, "run": run, "start": start, "flat": flat,
                         "p_on": p.where(on), "readings": readings})


def hourly(p: pd.Series, readings: pd.Series, appliance_type: str) -> pd.DataFrame:
    """p: 1-minute mean power (W) on a complete minute grid, naive UTC index."""
    thr = on_threshold(p)
    m = minute_signals(p, readings, thr)
    g = m.groupby(m.index.floor("h"))
    h = pd.DataFrame({
        "coverage": g["have"].mean(),
        "duty": g["on"].sum() / g["have"].sum().replace(0, np.nan),
        "starts": g["start"].sum().astype(float),
        "run_max": g["run"].max().astype(float),
        "p_on": g["p_on"].mean(),
        "p_on_std": g["p_on"].std(),
        "flat_frac": g["flat"].sum() / g["on"].sum().replace(0, np.nan),
        "rate": g["readings"].mean(),
    })
    h.loc[h["coverage"] == 0, ["duty", "starts", "run_max", "rate"]] = np.nan
    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    h = h.reindex(full)
    h["threshold_w"] = thr

    def base(col, fn="median"):
        r = h[col].rolling(config.BASELINE_HOURS, min_periods=config.BASELINE_MIN_HOURS)
        b = r.median() if fn == "median" else r.quantile(0.95)
        return b.shift(1)                                         # history only: the current hour is excluded

    eps = {"duty": 0.02, "run_max": 5.0, "starts": 0.5, "p_on": 5.0, "rate": 0.5}
    h["duty_rel"] = h["duty"] / (base("duty") + eps["duty"])
    h["run_rel"] = h["run_max"] / (base("run_max", "q95") + eps["run_max"])
    h["starts_rel"] = (h["starts"] + eps["starts"]) / (base("starts") + eps["starts"])
    h["p_on_rel"] = h["p_on"] / (base("p_on") + eps["p_on"])
    d6 = h["duty"].rolling(6, min_periods=3).mean()
    d24 = h["duty"].rolling(24, min_periods=12).mean()
    h["duty_6h"], h["duty_24h"] = d6, d24
    h["duty_6h_rel"] = d6 / (base("duty") + eps["duty"])
    h["duty_24h_rel"] = d24 / (base("duty") + eps["duty"])
    h["rate_rel"] = h["rate"] / (base("rate") + eps["rate"])
    first = h["run_max"][h.index < h.index[0] + pd.Timedelta(days=config.THRESHOLD_DAYS)]
    h["run_ref"] = h["run_max"] / (first.quantile(0.95) + eps["run_max"])

    h["hour"] = h.index.hour
    doy = h.index.dayofyear
    h["doy_sin"], h["doy_cos"] = np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366)
    h["is_freezer"] = float(appliance_type == "freezer")
    h["is_fridge_freezer"] = float(appliance_type == "fridge_freezer")
    h["hours_since_start"] = np.arange(len(h))
    h.index.name = "hour_utc"
    return h
