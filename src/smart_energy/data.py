"""Build the appliance-hour table: one row per (house, cold appliance, hour) with features and labels.

Labels (annotation times are UTC - see the alignment audit in experiments.alignment):
  y_fault  >= 30 min of the hour inside an annotated BEHAVIOURAL anomaly the annotators were sure of (status S):
           compressor stuck on, unusually long cycles, abnormal signature
  y_stuck  >= 30 min inside an annotated "sensor seems stuck" event (a data fault, not an appliance fault)
  ambiguous  inside a behavioural event marked "not sure" (NS) - excluded from fault training and scoring
Events whose end precedes their start (2 rows) are dropped as unusable; an exact duplicate row is removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, features, labels


def load_minutes(house: int) -> pd.DataFrame:
    d = pd.read_pickle(config.MINUTE_DIR / f"house{house}.pkl")
    d.index = d.index.tz_convert(None)
    return d


def events() -> pd.DataFrame:
    a = labels.load(config.LABELS_CSV)
    a = a[~a["end_before_start"]].drop_duplicates(["house", "Appliance", "start_local", "end_local"]).copy()
    a["start"], a["end"] = a["start_local"], a["end_local"]          # UTC (see alignment audit)
    a["family"] = np.where(a["kind"] == "sensor_stuck", "stuck", "fault")
    a["event_id"] = np.arange(len(a))
    return a


def cold_channels() -> list[tuple[int, str, str, str]]:
    return [(h, ch, t, name) for (h, name), (ch, t) in labels.CHANNEL.items() if t in labels.COLD]


def _overlap_minutes(idx: pd.DatetimeIndex, start, end) -> np.ndarray:
    s = np.maximum(idx.values, np.datetime64(start))
    e = np.minimum(idx.values + np.timedelta64(1, "h"), np.datetime64(end))
    return np.clip((e - s) / np.timedelta64(1, "m"), 0, 60)


def attach_labels(h: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    h["y_fault"], h["y_stuck"], h["ambiguous"] = 0, 0, 0
    h["fault_event"], h["stuck_event"] = -1, -1
    for _, e in ev.iterrows():
        ov = _overlap_minutes(h.index, e["start"], e["end"])
        hit = ov >= min(config.MIN_OVERLAP_MIN, max(1.0, (e["end"] - e["start"]) / pd.Timedelta(minutes=1)) / 2)
        if e["family"] == "stuck":
            h.loc[hit, "y_stuck"] = 1
            h.loc[hit, "stuck_event"] = e["event_id"]
        elif e["certain"]:
            h.loc[hit, "y_fault"] = 1
            h.loc[hit, "fault_event"] = e["event_id"]
        else:
            h.loc[hit, "ambiguous"] = 1
    return h


@dataclass
class Table:
    rows: pd.DataFrame          # every appliance-hour, features + labels
    events: pd.DataFrame        # annotated events on the cold appliances


def build(cache=None) -> Table:
    if cache is not None and cache.exists():
        return pd.read_pickle(cache)
    ev = events()
    parts = []
    minutes = {}
    for house, ch, typ, name in cold_channels():
        if house not in minutes:
            minutes = {house: load_minutes(house)}               # keep one house in memory at a time
        m = minutes[house]
        h = features.hourly(m[f"{ch}_w"], m["n_readings"], typ)
        h = attach_labels(h, ev[(ev["house"] == house) & (ev["channel"] == ch)])
        h["house"], h["channel"], h["appliance_type"], h["appliance"] = house, ch, typ, f"H{house}-{name}"
        parts.append(h.reset_index())
    rows = pd.concat(parts, ignore_index=True)
    rows["scored"] = (rows["hours_since_start"] >= config.WARMUP_HOURS) & (rows["coverage"] >= 0.5) & rows["duty_rel"].notna()
    cold = ev[ev["appliance_type"].isin(labels.COLD)].reset_index(drop=True)
    cold["appliance"] = "H" + cold["house"].astype(str) + "-" + cold["Appliance"]
    t = Table(rows, cold)
    if cache is not None:
        pd.to_pickle(t, cache)
    return t
