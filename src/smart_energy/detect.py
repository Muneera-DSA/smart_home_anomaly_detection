"""Detectors, alert thresholds and event-level scoring.

Every detector maps appliance-hours to a score (higher = more anomalous). A threshold turns scores into alert hours;
alert hours of one appliance closer than ALERT_GAP_H merge into one alert. An alert HITS an annotated fault event if it
overlaps the event's span widened by EVENT_TOLERANCE_H. Alerts that touch neither a fault nor a sensor-stuck event are
false alerts. The threshold of every detector is chosen on TRAINING houses only, as the lowest score whose false
alerts stay within the budget (FALSE_ALERTS_PER_APPLIANCE_MONTH).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, features

HOURS_PER_MONTH = 730.0


# ------------------------------------------------------------------ detectors

class RunTimer:
    """Domain rule a smart plug could run: alarm when the compressor has been running for too long."""
    name = "Rule: compressor-on timer (run_max)"
    needs_labels = False

    def fit(self, X, y=None, w=None):
        return self

    def score(self, X):
        return X["run_max"].to_numpy(dtype=float)


class RefRule:
    """The same timer, scaled to each appliance: running streak / its typical longest streak in its first month."""
    name = "Rule: on-timer vs own first month (run_ref)"
    needs_labels = False

    def fit(self, X, y=None, w=None):
        return self

    def score(self, X):
        return X["run_ref"].to_numpy(dtype=float)


class IForest:
    name = "Isolation Forest (unsupervised)"
    needs_labels = False
    cols = ["duty", "run_max", "starts", "duty_6h", "duty_24h", "duty_rel", "run_rel", "starts_rel", "p_on_rel", "duty_6h_rel",
            "duty_24h_rel", "run_ref", "flat_frac"]

    def fit(self, X, y=None, w=None):
        Z = self._z(X)
        rng = np.random.default_rng(config.SEED)
        idx = rng.choice(len(Z), min(len(Z), 60000), replace=False)
        self.m_ = IsolationForest(n_estimators=300, max_samples=1024, random_state=config.SEED).fit(Z[idx])
        return self

    def _z(self, X):
        Z = X[self.cols].to_numpy(dtype=float)
        Z = np.log1p(np.clip(Z, 0, None))
        return np.nan_to_num(Z, nan=0.0)

    def score(self, X):
        return -self.m_.score_samples(self._z(X))


class Supervised:
    needs_labels = True

    def __init__(self, name, make, cols=None):
        self.name, self.make, self.cols = name, make, cols or features.FEATURES

    def fit(self, X, y, w):
        self.m_ = self.make().fit(self._x(X), y, **self._w(w))
        return self

    def _w(self, w):
        m = self.make()
        return {f"{m.steps[-1][0]}__sample_weight": w} if hasattr(m, "steps") else {"sample_weight": w}

    def _x(self, X):
        return np.nan_to_num(X[self.cols].to_numpy(dtype=float), nan=-1.0)

    def score(self, X):
        return self.m_.predict_proba(self._x(X))[:, 1]


def supervised_models(cols=None) -> list:
    s = config.SEED
    return [
        Supervised("Logistic regression", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)), cols),
        Supervised("Random forest", lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_samples=0.3,
                                                                   n_jobs=2, random_state=s), cols),
        Supervised("Extra trees", lambda: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=20, max_samples=0.3, bootstrap=True,
                                                               n_jobs=2, random_state=s), cols),
        Supervised("Gradient boosting (HGB)", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                                                   min_samples_leaf=50, l2_regularization=1.0,
                                                                                   random_state=s), cols),
    ]


def all_detectors() -> list:
    return [RunTimer(), RefRule(), IForest(), *supervised_models()]


# ------------------------------------------------------------------ training weights

def training_weights(df: pd.DataFrame) -> np.ndarray:
    """Class-balanced weights: positive and negative hours carry equal total weight.
    (Weighting every annotated EVENT equally instead was tried and lowered event recall: 0.74 vs 0.80 at the budget.)"""
    y = df["y_fault"].to_numpy()
    return np.where(y == 1, (y == 0).sum() / max(y.sum(), 1), 1.0)


# ------------------------------------------------------------------ alerts and events

def alert_events(hours: np.ndarray, alert: np.ndarray, gap: int = config.ALERT_GAP_H) -> list[tuple]:
    """hours: sorted datetime64 of one appliance; alert: bool. Returns [(start, end)] merged alert spans."""
    t = hours[alert]
    if len(t) == 0:
        return []
    out, s, e = [], t[0], t[0]
    step = np.timedelta64(gap, "h")
    for x in t[1:]:
        if x - e <= step:
            e = x
        else:
            out.append((s, e))
            s = e = x
    out.append((s, e))
    return out


def score_events(df: pd.DataFrame, score: np.ndarray, thr: float, events: pd.DataFrame) -> dict:
    """Event-level outcome of one threshold on a set of appliances (df rows must be scored hours)."""
    tol = np.timedelta64(config.EVENT_TOLERANCE_H, "h")
    hit_events, delays, false_alerts, stuck_alerts, n_alerts = set(), {}, 0, 0, 0
    months = 0.0
    df = df.assign(_s=score)
    for app, g in df.groupby("appliance", sort=False):
        g = g.sort_values("hour_utc")
        months += len(g) / HOURS_PER_MONTH
        hrs = g["hour_utc"].to_numpy()
        al = alert_events(hrs, g["_s"].to_numpy() >= thr)
        n_alerts += len(al)
        ev = events[events["appliance"] == app]
        f = ev[ev["family"] == "fault"]
        st = ev[ev["family"] == "stuck"]
        fs, fe = f["start"].to_numpy(), f["end"].to_numpy()
        ss, se = st["start"].to_numpy(), st["end"].to_numpy()
        for a0, a1 in al:
            a1e = a1 + np.timedelta64(1, "h")
            hit = (a0 <= fe + tol) & (a1e >= fs - tol)
            if hit.any():
                for i in np.flatnonzero(hit):
                    eid = f["event_id"].iloc[i]
                    hit_events.add(eid)
                    # the alert is raised at the end of its first hour
                    lag = float((a0 + np.timedelta64(1, "h") - fs[i]) / np.timedelta64(1, "h"))
                    delays[eid] = min(delays.get(eid, np.inf), max(0.0, lag))
            elif ((a0 <= se + tol) & (a1e >= ss - tol)).any():
                stuck_alerts += 1
            else:
                false_alerts += 1
    scoreable = events[(events["family"] == "fault") & events["appliance"].isin(df["appliance"].unique())]
    scoreable = scoreable[scoreable["event_id"].isin(df.loc[df["y_fault"] == 1, "fault_event"].unique())]
    n = len(scoreable)
    return {"events": n, "hit": len(hit_events & set(scoreable["event_id"])), "recall": len(hit_events & set(scoreable["event_id"])) / max(n, 1),
            "false_alerts": false_alerts, "false_per_app_month": false_alerts / max(months, 1e-9), "alerts": n_alerts,
            "stuck_alerts": stuck_alerts, "median_delay_h": float(np.median(list(delays.values()))) if delays else np.nan,
            "delays": delays, "hit_ids": sorted(hit_events & set(scoreable["event_id"])), "months": months}


def threshold_for_budget(df: pd.DataFrame, score: np.ndarray, events: pd.DataFrame,
                         budget: float = config.FALSE_ALERTS_PER_APPLIANCE_MONTH) -> float:
    """Lowest threshold whose false alerts per appliance-month on these (training) appliances stay within budget."""
    s = score[np.isfinite(score)]
    cands = np.unique(np.quantile(s, np.linspace(0.90, 0.99995, 80)))
    lo, hi = 0, len(cands) - 1                      # false alerts fall as the threshold rises: binary search
    if score_events(df, score, cands[hi], events)["false_per_app_month"] > budget:
        return float(cands[hi])
    while lo < hi:
        mid = (lo + hi) // 2
        if score_events(df, score, cands[mid], events)["false_per_app_month"] <= budget:
            hi = mid
        else:
            lo = mid + 1
    return float(cands[lo])
