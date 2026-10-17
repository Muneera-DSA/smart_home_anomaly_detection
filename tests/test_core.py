"""Unit tests on synthetic signals: no REFIT data needed."""

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import HistGradientBoostingClassifier

from smart_energy import config, detect, features, labels, treeshap


def fridge(days=40, period=60, on=20, power=80.0, seed=0):
    """A healthy fridge: runs `on` minutes in every `period`, with small noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=days * 1440, freq="min")
    p = np.where((np.arange(len(idx)) % period) < on, power, 1.0) + rng.normal(0, 1.5, len(idx))
    return pd.Series(p, idx), pd.Series(9, index=idx)


def test_every_annotation_name_maps_to_a_channel(tmp_path):
    names = {("1", "Freezer_2"), ("10", "Mix"), ("18", "Fridge_garag"), ("19", "FridgeFreezer")}
    for h, n in names:
        assert (int(h), n) in labels.CHANNEL


def test_uk_summer_time_conversion():
    s = pd.Series(pd.to_datetime(["2014-07-01 12:00", "2014-01-10 12:00", "2014-03-30 03:00"]))
    out = labels.uk_to_utc(s)
    assert out[0] == pd.Timestamp("2014-07-01 11:00")      # BST: UTC+1
    assert out[1] == pd.Timestamp("2014-01-10 12:00")      # GMT
    assert out[2] == pd.Timestamp("2014-03-30 02:00")      # after the spring change (01:00 UTC)
    back = labels.utc_to_uk(pd.DatetimeIndex(out))
    assert (back == pd.DatetimeIndex(s)).all()


def test_healthy_fridge_features():
    p, r = fridge()
    h = features.hourly(p, r, "fridge")
    late = h.iloc[-24:]
    assert late["duty"].between(0.3, 0.37).all()
    assert (late["starts"] == 1).all()
    assert late["run_max"].max() <= 21
    assert late["duty_rel"].between(0.8, 1.2).all()


def test_stuck_on_compressor_raises_run_and_duty():
    p, r = fridge()
    p.iloc[-600:] = 80.0 + np.random.default_rng(1).normal(0, 0.5, 600)   # last 10 hours running non-stop
    h = features.hourly(p, r, "fridge")
    assert h["run_max"].iloc[-1] >= 590
    assert h["duty"].iloc[-1] == 1.0
    assert h["starts"].iloc[-1] == 0
    assert h["run_ref"].iloc[-1] > 20


def test_frozen_sensor_is_flat():
    p, r = fridge()
    p.iloc[-300:] = 80.0                                                      # identical reading for 5 hours
    h = features.hourly(p, r, "fridge")
    assert h["flat_frac"].iloc[-2] > 0.95
    assert h["p_on_std"].iloc[-2] == pytest.approx(0.0)


def test_baselines_use_only_the_past():
    """Changing the future must not change any feature of an earlier hour."""
    p, r = fridge()
    h1 = features.hourly(p, r, "fridge")
    p2 = p.copy()
    p2.iloc[-1440:] = 80.0                                                    # a fault in the last day
    h2 = features.hourly(p2, r, "fridge")
    cut = h1.index[-25]
    cols = [c for c in features.FEATURES if c != "run_ref"]
    pd.testing.assert_frame_equal(h1.loc[:cut, cols], h2.loc[:cut, cols])


def test_alert_events_merge_close_hours():
    hrs = pd.date_range("2014-01-01", periods=12, freq="h").to_numpy()
    alert = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1], bool)
    ev = detect.alert_events(hrs, alert, gap=3)
    assert len(ev) == 2
    assert ev[0] == (hrs[0], hrs[4])


def _toy():
    hrs = pd.date_range("2014-02-01", periods=24 * 30, freq="h")
    df = pd.DataFrame({"hour_utc": hrs, "appliance": "H1-Fridge", "y_fault": 0, "fault_event": -1})
    df.loc[100:105, ["y_fault", "fault_event"]] = [1, 7]
    ev = pd.DataFrame({"event_id": [7], "appliance": ["H1-Fridge"], "family": ["fault"],
                       "start": [hrs[100]], "end": [hrs[106]]})
    return df, ev


def test_event_scoring_counts_hits_false_alerts_and_delay():
    df, ev = _toy()
    s = np.zeros(len(df))
    s[102] = 1.0                      # alert two hours into the event
    s[400] = 1.0                      # an alert far from any event
    x = detect.score_events(df, s, 0.5, ev)
    assert x["hit"] == 1 and x["events"] == 1
    assert x["false_alerts"] == 1
    assert x["delays"][7] == pytest.approx(3.0)       # raised at the end of the hour that starts 2 h in


def test_threshold_respects_budget():
    df, ev = _toy()
    rng = np.random.default_rng(0)
    s = rng.random(len(df))
    s[100:106] += 2
    thr = detect.threshold_for_budget(df, s, ev, budget=0.0)
    x = detect.score_events(df, s, thr, ev)
    assert x["false_alerts"] == 0 and x["hit"] == 1


def test_config_budget_is_positive():
    assert config.FALSE_ALERTS_PER_APPLIANCE_MONTH > 0


def test_treeshap_additivity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    model = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X, y)
    phi, base = treeshap.shap_values(model, X[:10])
    np.testing.assert_allclose(base + phi.sum(1), model.decision_function(X[:10]), atol=1e-6)
