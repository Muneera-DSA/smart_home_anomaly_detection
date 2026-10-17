"""REFIT appliance-anomaly annotations (Rashid et al., ICASSP 2019) mapped onto the sensor channels.

The annotation file names appliances in its own words ("Freezer_2", "Fridge_garag", "Mix"); the data files only number
channels app1..app9. CHANNEL maps every (house, annotation name) to its channel, using the appliance list in the REFIT
readme. Where a house has two appliances of one type, the annotation suffix _1/_2 follows the readme order; this is
checked against the data in the alignment audit rather than assumed silently.
"""

from __future__ import annotations

import pandas as pd

# (house, annotation appliance name) -> (channel, appliance type)
CHANNEL = {
    (1, "Fridge"): ("app1", "fridge"), (1, "Freezer_1"): ("app2", "freezer"), (1, "Freezer_2"): ("app3", "freezer"),
    (1, "WashingMachine"): ("app5", "washing_machine"), (1, "ElectricHeater"): ("app9", "electric_heater"),
    (2, "Fridge-Freezer"): ("app1", "fridge_freezer"), (2, "Dishwasher"): ("app3", "dishwasher"),
    (3, "Fridge-Freezer"): ("app2", "fridge_freezer"), (3, "Freezer"): ("app3", "freezer"),
    (3, "TumbleDryer"): ("app4", "tumble_dryer"), (3, "Dishwasher"): ("app5", "dishwasher"),
    (4, "Freezer"): ("app2", "freezer"), (4, "Fridge-Freezer"): ("app3", "fridge_freezer"),
    (5, "Fridge-Freezer"): ("app1", "fridge_freezer"), (5, "TumbleDryer"): ("app2", "tumble_dryer"),
    (5, "WashingMachine"): ("app3", "washing_machine"), (5, "Microwave"): ("app7", "microwave"),
    (6, "Freezer"): ("app1", "freezer"),
    (7, "Fridge"): ("app1", "fridge"), (7, "Freezer_1"): ("app2", "freezer"),
    (8, "Fridge"): ("app1", "fridge"), (8, "Freezer"): ("app2", "freezer"),
    (9, "Fridge-Freezer"): ("app1", "fridge_freezer"),
    (10, "ChestFreezer"): ("app3", "freezer"), (10, "Fridge-Freezer"): ("app4", "fridge_freezer"),
    (10, "Dishwasher"): ("app6", "dishwasher"), (10, "Mix"): ("app9", "food_mixer"),
    (11, "Fridge"): ("app1", "fridge"), (11, "Fridge-Freezer"): ("app2", "fridge_freezer"),
    (12, "Fridge-Freezer"): ("app1", "fridge_freezer"), (12, "Microwave"): ("app3", "microwave"),
    (13, "Kettle"): ("app9", "kettle"),
    (15, "Fridge-Freezer"): ("app1", "fridge_freezer"),
    (16, "Fridge-Freezer_1"): ("app1", "fridge_freezer"), (16, "Fridge-Freezer_2"): ("app2", "fridge_freezer"),
    (17, "Freezer"): ("app1", "freezer"), (17, "Fridge-Freezer"): ("app2", "fridge_freezer"),
    (18, "Fridge_garag"): ("app1", "fridge"), (18, "Freezer_garage"): ("app2", "freezer"),
    (18, "Fridge-Freezer"): ("app3", "fridge_freezer"),
    (19, "FridgeFreezer"): ("app1", "fridge_freezer"),
    (20, "Fridge"): ("app1", "fridge"), (20, "Freezer"): ("app2", "freezer"),
    (21, "Fridge-Freezer"): ("app1", "fridge_freezer"), (21, "Dishwasher"): ("app4", "dishwasher"),
}

COLD = {"fridge", "freezer", "fridge_freezer"}


def _kind(expl: str, status: str) -> str:
    e = expl.lower()
    if "stuck" in e:
        return "sensor_stuck"
    if "continuous on" in e or "kept on" in e or "left on" in e:
        return "continuous_on"
    if "longer" in e or "cycles" in e:
        return "long_cycles"
    return "other_behaviour"


def load(path) -> pd.DataFrame:
    """One row per annotated event, with the channel, appliance type, local start/end and an anomaly kind."""
    a = pd.read_csv(path, index_col=0)
    a.columns = [c.strip() for c in a.columns]
    a["Appliance"] = a["Appliance"].str.strip()
    a["start_local"] = pd.to_datetime(a["start_anomaly"], format="%d/%m/%Y %H:%M")
    a["end_local"] = pd.to_datetime(a["end_anomaly"], format="%d/%m/%Y %H:%M")
    keys = list(zip(a["House_No"], a["Appliance"]))
    missing = sorted({k for k in keys if k not in CHANNEL})
    if missing:
        raise KeyError(f"unmapped appliances: {missing}")
    a["channel"] = [CHANNEL[k][0] for k in keys]
    a["appliance_type"] = [CHANNEL[k][1] for k in keys]
    a["certain"] = a["Status"].str.strip().eq("S")
    a["kind"] = [_kind(e, s) for e, s in zip(a["Explanation"].fillna(""), a["Status"])]
    a["end_before_start"] = a["end_local"] < a["start_local"]
    return a.rename(columns={"House_No": "house"})


# ---- UK local time without tzdata: BST runs 01:00 UTC on the last Sunday of March to 01:00 UTC on the last Sunday of October
def _last_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=31 if month in (3, 10) else 30)
    return d - pd.Timedelta(days=(d.weekday() + 1) % 7)


BST = {y: (_last_sunday(y, 3) + pd.Timedelta(hours=1), _last_sunday(y, 10) + pd.Timedelta(hours=1)) for y in (2013, 2014, 2015)}


def utc_to_uk(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    off = pd.Series(0, index=range(len(naive)))
    for y, (s, e) in BST.items():
        off[(naive >= s) & (naive < e)] = 1
    return naive + pd.to_timedelta(off.to_numpy(), unit="h")


def uk_to_utc(ts: pd.Series) -> pd.Series:
    """Local UK wall time -> naive UTC (ambiguous autumn hour resolved as BST)."""
    out = ts.copy()
    for y, (s, e) in BST.items():
        m = (ts >= s + pd.Timedelta(hours=1)) & (ts < e + pd.Timedelta(hours=1))
        out[m] = ts[m] - pd.Timedelta(hours=1)
    return out
