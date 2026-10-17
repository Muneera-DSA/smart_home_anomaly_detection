"""All experiments. Run: python -m smart_energy.experiments [stage ...]   (no stage = all, in order)

Stages: quality, eda, detect, ablation, stuck, hypotheses, explain, business
Outputs: results/*.json|csv and reports/figures/*.png. Every number quoted in the README comes from these files.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config, data, detect, features, labels, plot, stats

R, F = config.RESULTS, config.FIGURES
BUDGETS = [0.25, 0.5, 1.0, 2.0, 4.0]


def _dump(name, obj):
    R.mkdir(parents=True, exist_ok=True)
    (R / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _load(name):
    return json.loads((R / name).read_text())


class Ctx:
    def __init__(self):
        cache = config.DATA / "processed" / "appliance_hours.pkl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        self.table = data.build(cache)
        self.rows, self.events = self.table.rows, self.table.events
        r = self.rows
        self.d = r[r["scored"] & (r["y_stuck"] == 0) & (r["ambiguous"] == 0)].reset_index(drop=True)
        self.folds = list(GroupKFold(config.N_FOLDS).split(self.d, groups=self.d["house"]))


# ====================================================================== quality

def quality(c: Ctx):
    t0 = time.time()
    ing = pd.read_pickle(config.MINUTE_DIR / "ingest_summary.pkl")
    ing["days"] = ing["minutes"] / 1440
    ing["coverage"] = 1 - ing["minutes_without_data"] / ing["minutes"]
    ing["appliance_over_4000w"] = ing["impossible_total"]
    per_house = ing[["house", "rows", "days", "coverage", "median_interval_s", "duplicate_timestamps", "missing_values",
                     "negative_values", "appliance_over_4000w"]].round(4)
    per_house.to_csv(R / "data_quality_houses.csv", index=False)

    raw = labels.load(config.LABELS_CSV)
    dup = int(raw.duplicated(["house", "Appliance", "start_local", "end_local"]).sum())
    inv = raw[raw["end_before_start"]][["house", "Appliance", "start_anomaly", "end_anomaly"]].values.tolist()
    # system-wide outage: the same "sensor stuck" window labelled in many houses
    st = raw[raw["kind"] == "sensor_stuck"]
    day = st.groupby(st["start_local"].dt.floor("D"))["house"].nunique().sort_values(ascending=False)
    shared = {str(k.date()): int(v) for k, v in day.head(4).items()}
    rates = []
    for h in sorted(raw.loc[raw["start_local"].dt.date.astype(str) == "2014-08-01", "house"].unique()):
        m = data.load_minutes(int(h))
        w = m.loc["2014-08-01 18:00":"2014-08-02 13:00", "n_readings"].mean()
        b = m.loc["2014-07-25":"2014-07-31", "n_readings"].mean()
        rates.append({"house": int(h), "readings_per_min_outage": round(float(w), 2), "readings_per_min_week_before": round(float(b), 2)})

    ev = c.events
    r = c.rows
    first30 = []
    for app, g in r.groupby("appliance"):
        start = g["hour_utc"].min()
        e = ev[(ev["appliance"] == app) & (ev["start"] < start + pd.Timedelta(days=config.THRESHOLD_DAYS))]
        first30 += e["event_id"].tolist()

    # channel sanity: every mapped cold channel should look like a cycling compressor
    chan = r.groupby("appliance").agg(type=("appliance_type", "first"), threshold_w=("threshold_w", "first"),
                                      on_power_w=("p_on", "median"), duty=("duty", "median"), starts_per_day=("starts", "mean"),
                                      coverage=("coverage", "mean"))
    chan["starts_per_day"] *= 24
    chan.round(2).to_csv(R / "data_quality_channels.csv")

    out = {
        "houses": int(len(ing)), "raw_rows": int(ing["rows"].sum()), "days_mean": float(ing["days"].mean()),
        "minute_coverage_mean": float(ing["coverage"].mean()), "minute_coverage_min": float(ing["coverage"].min()),
        "median_interval_s": float(ing["median_interval_s"].median()), "duplicate_timestamps": int(ing["duplicate_timestamps"].sum()),
        "missing_values": int(ing["missing_values"].sum()), "negative_values": int(ing["negative_values"].sum()),
        "impossible_readings_removed": int(ing["impossible_total"].sum()),
        "impossible_share_of_appliance_readings": float(ing["impossible_total"].sum() / (ing["rows"].sum() * 9)),
        "annotation_rows": int(len(raw)), "annotation_duplicates": dup, "annotation_end_before_start": inv,
        "annotation_kinds": raw["kind"].value_counts().to_dict(), "annotation_status_sure": float(raw["certain"].mean()),
        "annotation_types": raw["appliance_type"].value_counts().to_dict(),
        "cold_share_of_annotations": float(raw["appliance_type"].isin(labels.COLD).mean()),
        "shared_stuck_days": shared, "outage_reading_rates": rates,
        "events_in_reference_month": len(set(first30)),
        "cold_channels": int(r["appliance"].nunique()), "cold_houses": int(r["house"].nunique()),
        "appliance_hours": int(len(r)), "scored_hours": int(r["scored"].sum()),
        "fault_events": int((ev["family"] == "fault").sum()), "stuck_events": int((ev["family"] == "stuck").sum()),
        "fault_events_scoreable": int(c.d.loc[c.d["y_fault"] == 1, "fault_event"].nunique()),
        "fault_hours": int(c.d["y_fault"].sum()), "fault_hour_rate": float(c.d["y_fault"].mean()),
    }
    out["alignment"] = alignment(c)
    _dump("data_quality.json", out)
    print(f"[quality] {time.time() - t0:.0f}s", flush=True)


def alignment(c: Ctx) -> dict:
    """Which clock are the annotations on? Shift every certain 'continuous ON' summer event by k hours and measure how
    sharply the compressor's running share changes at the shifted start/end. UK summer time is UTC+1, so if annotators
    wrote local time the peak would sit at -1 h."""
    ev = c.events
    ev = ev[(ev["kind"] == "continuous_on") & ev["certain"] & ((ev["end"] - ev["start"]) < pd.Timedelta(days=2))]
    summer = labels.uk_to_utc(ev["start"]) != ev["start"]
    ev = ev[summer]
    shifts = np.arange(-3, 3.01, 0.25)
    acc = {k: [] for k in shifts}
    for house, g in ev.groupby("house"):
        m = data.load_minutes(int(house))
        for _, e in g.iterrows():
            s = m[f"{e['channel']}_w"]
            thr = features.on_threshold(s)
            on = (s > thr).astype(float).where(s.notna())
            for k in shifts:
                st, en = e["start"] + pd.Timedelta(hours=k), e["end"] + pd.Timedelta(hours=k)
                hr = pd.Timedelta(hours=1)
                b = on[st:st + hr].mean() - on[st - hr:st].mean()
                a = on[en - hr:en].mean() - on[en:en + hr].mean()
                acc[k].append(np.nanmean([a, b]))
    curve = {float(k): float(np.nanmean(v)) for k, v in acc.items()}
    best = max(curve, key=curve.get)
    fig, ax = plot.plt.subplots(figsize=(6.4, 3.2))
    ax.plot(list(curve), list(curve.values()), color=plot.BLUE, marker="o", ms=3)
    ax.axvline(0, color=plot.GREY, lw=0.8, ls="--")
    ax.axvline(-1, color=plot.ORANGE, lw=0.8, ls=":")
    ax.text(-1, max(curve.values()) * 0.95, " UK summer time\n would peak here", color=plot.ORANGE, fontsize=8)
    plot.style(ax, f"Annotation clock: edge sharpness peaks at {best:+.2f} h (n={len(ev)} summer events)",
               "shift applied to annotated start/end (hours)", "running-share change at the edges")
    plot.save(fig, F / "label_alignment.png")
    return {"events": int(len(ev)), "best_shift_h": best, "curve": curve, "conclusion": "annotation times are UTC" if best == 0 else "check"}


# ====================================================================== EDA

def eda(c: Ctx):
    t0 = time.time()
    r, ev = c.rows, c.events
    s = r[r["scored"]]
    out = {}

    # 1. coverage by house and month
    cov = s.assign(month=s["hour_utc"].dt.to_period("M").astype(str)).pivot_table(index="house", columns="month", values="coverage", aggfunc="mean")
    allh = r.assign(month=r["hour_utc"].dt.to_period("M").astype(str)).pivot_table(index="house", columns="month", values="coverage", aggfunc="mean")
    fig, ax = plot.plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(allh.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(allh)), [f"H{h}" for h in allh.index])
    ax.set_xticks(range(0, len(allh.columns), 2), allh.columns[::2], rotation=60)
    fig.colorbar(im, ax=ax, label="share of minutes with readings")
    plot.style(ax, "Sensor coverage of the cold appliances by house and month (blank = not yet installed)")
    plot.save(fig, F / "eda_coverage.png")
    out["months_below_80pct_coverage"] = int((cov < 0.8).sum().sum())

    # 2. normal behaviour: duty by hour of day and by month, per appliance type
    normal = s[(s["y_fault"] == 0) & (s["y_stuck"] == 0)]
    fig, axes = plot.plt.subplots(1, 2, figsize=(11, 3.6))
    colors = {"fridge": plot.AQUA, "freezer": plot.BLUE, "fridge_freezer": plot.ORANGE}
    for t, g in normal.groupby("appliance_type"):
        axes[0].plot(g.groupby("hour")["duty"].mean(), label=t.replace("_", "-"), color=colors[t])
        axes[1].plot(g.groupby(g["hour_utc"].dt.month)["duty"].mean(), label=t.replace("_", "-"), color=colors[t], marker="o", ms=3)
    plot.style(axes[0], "Normal running share by hour of day (UTC)", "hour", "share of minutes running")
    plot.style(axes[1], "...and by month: compressors work harder in summer", "month", "share of minutes running")
    axes[0].legend(frameon=False, fontsize=8)
    plot.save(fig, F / "eda_normal_behaviour.png")
    m = normal.groupby([normal["appliance"], normal["hour_utc"].dt.month])["duty"].mean().unstack()
    out["duty_by_type"] = normal.groupby("appliance_type")["duty"].mean().round(3).to_dict()
    out["normal_run_max_p50_p99"] = [float(normal["run_max"].median()), float(normal["run_max"].quantile(0.99))]

    # 3. what anomalies look like: one example of each kind at minute resolution
    picks = {}
    for kind, title in [("continuous_on", "Compressor stuck ON (annotated, certain)"), ("long_cycles", "Unusually long cycles (annotated, certain)"),
                        ("sensor_stuck", "Sensor stuck (annotated data fault)")]:
        e = ev[(ev["kind"] == kind) & ((ev["end"] - ev["start"]).between(pd.Timedelta(hours=4), pd.Timedelta(hours=30)))]
        e = e[e["certain"]] if kind != "sensor_stuck" else e
        picks[kind] = (e.iloc[len(e) // 2], title)
    fig, axes = plot.plt.subplots(3, 1, figsize=(11, 7.5))
    for ax, (kind, (e, title)) in zip(axes, picks.items()):
        m = data.load_minutes(int(e["house"]))
        w = m.loc[e["start"] - pd.Timedelta(hours=18):e["end"] + pd.Timedelta(hours=10), f"{e['channel']}_w"]
        ax.plot(w.index, w.to_numpy(), color=plot.INK, lw=0.6)
        ax.axvspan(e["start"], e["end"], color=plot.ORANGE, alpha=0.18, lw=0)
        plot.style(ax, f"{title}: {e['appliance']}, {e['start']:%d %b %Y}", None, "W")
    plot.save(fig, F / "eda_anomaly_examples.png")

    # 4. features that separate normal / fault / stuck hours
    grp = np.select([s["y_stuck"] == 1, s["y_fault"] == 1], ["sensor stuck", "appliance anomaly"], "normal")
    sub = s.assign(group=grp)
    feats = [("run_max", "longest running streak (min)"), ("duty_6h", "running share, last 6 h"),
             ("starts", "compressor starts in the hour"), ("p_on_std", "power spread while running (W)")]
    fig, axes = plot.plt.subplots(1, 4, figsize=(13, 3.4))
    order = ["normal", "appliance anomaly", "sensor stuck"]
    for ax, (f, lab) in zip(axes, feats):
        vals = [(lambda v: v.sample(min(20000, len(v)), random_state=0))(sub.loc[sub["group"] == g, f].dropna()) for g in order]
        bp = ax.boxplot(vals, tick_labels=order, showfliers=False, patch_artist=True, widths=0.55)
        for patch, col in zip(bp["boxes"], [plot.LIGHT, plot.ORANGE, plot.BLUE]):
            patch.set_facecolor(col)
        ax.tick_params(axis="x", rotation=20)
        if f == "run_max":
            ax.set_yscale("symlog", linthresh=10)
        plot.style(ax, lab)
    plot.save(fig, F / "eda_feature_separation.png")
    out["feature_medians"] = sub.groupby("group")[[f for f, _ in feats] + ["flat_frac"]].median().round(3).to_dict()

    # 5. events per house and kind; durations
    e = ev.copy()
    e["dur_h"] = (e["end"] - e["start"]) / pd.Timedelta(hours=1)
    tab = e.pivot_table(index="house", columns="kind", values="event_id", aggfunc="count", fill_value=0)
    fig, axes = plot.plt.subplots(1, 2, figsize=(11.5, 3.6))
    bottom = np.zeros(len(tab))
    kc = {"continuous_on": plot.ORANGE, "long_cycles": plot.AQUA, "other_behaviour": plot.GREY, "sensor_stuck": plot.BLUE}
    for k in tab.columns:
        axes[0].bar([f"H{h}" for h in tab.index], tab[k], bottom=bottom, color=kc[k], label=k.replace("_", " "))
        bottom += tab[k].to_numpy()
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].tick_params(axis="x", rotation=60)
    plot.style(axes[0], "Annotated events on cold appliances, by house", None, "events")
    axes[1].hist(np.log10(e["dur_h"].clip(lower=0.25)), bins=30, color=plot.BLUE)
    axes[1].set_xticks([-0.5, 0, 0.5, 1, 1.5, 2, 3], ["0.3", "1", "3", "10", "30", "100", "1000"])
    plot.style(axes[1], f"Event durations (median {e['dur_h'].median():.1f} h)", "hours (log scale)", "events")
    plot.save(fig, F / "eda_events.png")
    out["events_per_house"] = tab.sum(axis=1).to_dict()
    out["event_duration_h"] = e["dur_h"].describe().round(2).to_dict()
    out["kind_by_type"] = e.pivot_table(index="appliance_type", columns="kind", values="event_id", aggfunc="count", fill_value=0).to_dict()

    # 6. how complete are the labels? long running streaks nobody annotated
    long = s["run_max"] >= 240
    lab = (s["y_fault"] == 1) | (s["y_stuck"] == 1)
    out["hours_running_4h_plus"] = int(long.sum())
    out["share_of_those_annotated"] = float((long & lab).sum() / long.sum())
    out["unannotated_4h_runs_by_appliance_top"] = s[long & ~lab].groupby("appliance").size().sort_values(ascending=False).head(5).to_dict()
    _dump("eda.json", out)
    print(f"[eda] {time.time() - t0:.0f}s", flush=True)


# ====================================================================== detection (leave-houses-out)

def _inner_scores(det, D):
    """Out-of-fold scores on the training houses (3 inner folds by house) - used only to choose the threshold."""
    if not det.needs_labels:
        return det.score(D)
    s = np.full(len(D), np.nan)
    for itr, ite in GroupKFold(3).split(D, groups=D["house"]):
        Di = D.iloc[itr]
        m = type(det)(det.name, det.make, det.cols).fit(Di, Di["y_fault"].to_numpy(), detect.training_weights(Di))
        s[ite] = m.score(D.iloc[ite])
    return s


def run_cv(c: Ctx, dets: list, tag: str) -> dict:
    d, ev = c.d, c.events
    res = {}
    for det in dets:
        t0 = time.time()
        oof = np.full(len(d), np.nan)
        per_fold, per_house, thresholds = [], [], {}
        for k, (tr, te) in enumerate(c.folds):
            D, T = d.iloc[tr], d.iloc[te]
            if det.needs_labels:
                det.fit(D, D["y_fault"].to_numpy(), detect.training_weights(D))
            else:
                det.fit(D)
            inner = _inner_scores(det, D)
            thr = {b: detect.threshold_for_budget(D, inner, ev, b) for b in BUDGETS}
            s = det.score(T)
            oof[te] = s
            for b in BUDGETS:
                x = detect.score_events(T, s, thr[b], ev)
                x.pop("delays_list", None)
                per_fold.append({"fold": k, "budget": b, **{kk: v for kk, v in x.items() if kk not in ("delays",)},
                                 "delays": {int(i): float(v) for i, v in x["delays"].items()}})
            for h, g in T.groupby("house"):
                x = detect.score_events(g, s[T["house"].to_numpy() == h], thr[1.0], ev)
                per_house.append({"house": int(h), "fold": k, "events": x["events"], "hit": x["hit"],
                                  "false_alerts": x["false_alerts"], "months": x["months"]})
            thresholds[k] = thr
        pf = pd.DataFrame(per_fold)
        summ = {}
        for b in BUDGETS:
            p = pf[pf["budget"] == b]
            delays = [v for dd in p["delays"] for v in dd.values()]
            summ[str(b)] = {"recall": p["hit"].sum() / p["events"].sum(), "hit": int(p["hit"].sum()), "events": int(p["events"].sum()),
                            "false_per_app_month": p["false_alerts"].sum() / p["months"].sum(), "false_alerts": int(p["false_alerts"].sum()),
                            "stuck_alerts": int(p["stuck_alerts"].sum()), "median_delay_h": float(np.median(delays)) if delays else None,
                            "hit_ids": sorted(int(i) for ids in p["hit_ids"] for i in ids),
                            "delays": {int(i): float(v) for dd in p["delays"] for i, v in dd.items()}}
        y = d["y_fault"].to_numpy()
        fin = np.isfinite(oof)
        long_ev = c.events.loc[(c.events["end"] - c.events["start"]) > pd.Timedelta(days=7), "event_id"]
        keep = fin & ~d["fault_event"].isin(long_ev).to_numpy()
        res[det.name] = {"by_budget": summ, "ap_hour": float(average_precision_score(y[fin], oof[fin])),
                         "ap_hour_excl_long": float(average_precision_score(y[keep], oof[keep])),
                         "roc_auc_hour": float(roc_auc_score(y[fin], oof[fin])),
                         "per_house": per_house, "thresholds": {str(k): v for k, v in thresholds.items()}, "seconds": time.time() - t0}
        pd.to_pickle(oof, _oof_path(tag, det.name))
        b1 = summ["1.0"]
        print(f"[{tag}] {det.name}: recall {b1['recall']:.3f} ({b1['hit']}/{b1['events']}), false/app-month {b1['false_per_app_month']:.2f}, "
              f"delay {b1['median_delay_h']}, AP {res[det.name]['ap_hour']:.3f}  {time.time() - t0:.0f}s", flush=True)
    return res


def _oof_path(tag, name):
    return R / f"_oof_{tag}_{name.split(' (')[0].replace(' ', '_').replace(':', '')}.pkl"


def detect_stage(c: Ctx):
    res = run_cv(c, detect.all_detectors(), "detect")
    _dump("detection.json", res)


# ====================================================================== matched false-alert rate

def matched(c: Ctx):
    """Deployable results (detect stage) set each threshold on training houses, so detectors end up at slightly different
    realised false-alert rates (0.5-1.3 per appliance-month) and their recalls are not directly comparable. Here every
    detector's OUT-OF-FOLD scores are thresholded so that all of them produce the SAME false-alert rate over the 19
    houses; per-house recall and delays at that common rate are what the hypothesis tests compare. (The threshold is
    chosen on the evaluation scores, equally for every detector, so these numbers rank detectors; the deployable numbers
    are the ones to expect in use.)"""
    d, ev = c.d, c.events
    det, abl = _load("detection.json"), _load("ablation.json")
    out = {}
    rng = np.random.default_rng(config.SEED)
    for tag, names in [("detect", list(det)), ("ablation", list(abl))]:
        for n in names:
            oof = pd.read_pickle(_oof_path(tag, n))
            curve = {}
            for b in BUDGETS:
                thr = detect.threshold_for_budget(d, oof, ev, b)
                x = detect.score_events(d, oof, thr, ev)
                curve[str(b)] = {"recall": x["recall"], "false_per_app_month": x["false_per_app_month"], "hit": x["hit"],
                                 "median_delay_h": x["median_delay_h"], "stuck_alerts": x["stuck_alerts"]}
                if b == 1.0:
                    thr1, x1 = thr, x
            ph = []
            for h, g in d.groupby("house"):
                xx = detect.score_events(g, oof[d["house"].to_numpy() == h], thr1, ev)
                ph.append({"house": int(h), "events": xx["events"], "hit": xx["hit"], "false_alerts": xx["false_alerts"], "months": xx["months"]})
            ph = pd.DataFrame(ph)
            # 95% CI of pooled recall, resampling houses
            boots = []
            for _ in range(2000):
                smp = ph.iloc[rng.integers(0, len(ph), len(ph))]
                boots.append(smp["hit"].sum() / max(smp["events"].sum(), 1))
            out[n] = {"tag": tag, "curve": curve, "recall_at_1": x1["recall"], "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                      "false_at_1": x1["false_per_app_month"], "delays_at_1": {int(k): float(v) for k, v in x1["delays"].items()},
                      "per_house": ph.to_dict("records")}
            print(f"[matched] {n}: recall@1 {x1['recall']:.3f} CI {out[n]['ci95'][0]:.2f}-{out[n]['ci95'][1]:.2f}", flush=True)
    _dump("matched.json", out)
    figures(c)


def figures(c: Ctx = None):
    det, m = _load("detection.json"), _load("matched.json")
    names = list(det)
    style = {"Rule: compressor-on timer (run_max)": (plot.INK, "-"), "Rule: on-timer vs own first month (run_ref)": (plot.GREY, "--"),
             "Isolation Forest (unsupervised)": (plot.AQUA, "-"), "Logistic regression": ("#7aa6e0", ":"), "Random forest": (plot.BLUE, "--"),
             "Extra trees": ("#123f7a", "-."), "Gradient boosting (HGB)": (plot.ORANGE, "-")}
    fig, axes = plot.plt.subplots(1, 2, figsize=(13, 4.4))
    for n in names:
        col, ls = style[n]
        cv = m[n]["curve"]
        axes[0].plot([cv[str(b)]["false_per_app_month"] for b in BUDGETS], [cv[str(b)]["recall"] for b in BUDGETS], marker="o", ms=3,
                     color=col, ls=ls, lw=2 if n in ("Gradient boosting (HGB)", "Rule: compressor-on timer (run_max)") else 1.1, label=n)
    axes[0].set_xscale("log")
    axes[0].set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    axes[0].minorticks_off()
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    plot.style(axes[0], "Anomalies caught vs false alerts, 19 homes, out-of-fold scores", "false alerts per appliance per month (log)",
               "share of annotated anomalies caught")
    rec = [det[n]["by_budget"]["1.0"]["recall"] for n in names]
    fa = [det[n]["by_budget"]["1.0"]["false_per_app_month"] for n in names]
    order = np.argsort(rec)
    y = np.arange(len(names))
    axes[1].barh(y, [rec[i] for i in order], color=[style[names[i]][0] for i in order], height=0.6)
    axes[1].set_yticks(y, [names[i] for i in order])
    for k, i in enumerate(order):
        axes[1].text(rec[i], k, f" {rec[i]:.2f}  ({fa[i]:.2f} false/mo)", va="center", fontsize=8, color=plot.INK)
    axes[1].set_xlim(0, 1.25)
    plot.style(axes[1], "Deployable: thresholds set on training homes (target 1 false/mo)")
    plot.save(fig, F / "detection_comparison.png")


# ====================================================================== ablation

def ablation(c: Ctx):
    sets = {"raw behaviour only": features.RAW + features.CONTEXT,
            "raw + recent duty": features.RAW + features.RECENT + features.CONTEXT,
            "relative-to-own-history only": features.RELATIVE + features.CONTEXT,
            "all features": features.FEATURES,
            "all features + calendar (rejected)": features.FEATURES + ["hour", "doy_sin", "doy_cos"]}
    hgb = detect.supervised_models()[3]
    dets = [detect.Supervised(f"HGB - {k}", hgb.make, v) for k, v in sets.items()]
    res = run_cv(c, dets, "ablation")
    _dump("ablation.json", {k: {"recall_b1": v["by_budget"]["1.0"]["recall"], "false_b1": v["by_budget"]["1.0"]["false_per_app_month"],
                                "ap_hour": v["ap_hour"], "ap_hour_excl_long": v["ap_hour_excl_long"], "per_house": v["per_house"]}
                            for k, v in res.items()})


# ====================================================================== sensor stuck vs real fault

def stuck(c: Ctx):
    r = c.rows
    s = r[r["scored"] & (r["coverage"] >= 0.5)]
    fault = s[(s["y_fault"] == 1) & (s["duty"] >= 0.9)]
    stk = s[(s["y_stuck"] == 1) & (s["duty"] >= 0.9)]
    pe_f = fault.groupby("fault_event")[["flat_frac", "p_on_std"]].median()
    pe_s = stk.groupby("stuck_event")[["flat_frac", "p_on_std"]].median()
    out = {"fault_events_running": int(len(pe_f)), "stuck_events_running": int(len(pe_s)),
           "median_flat_frac": {"fault": float(pe_f["flat_frac"].median()), "stuck": float(pe_s["flat_frac"].median())},
           "median_p_on_std": {"fault": float(pe_f["p_on_std"].median()), "stuck": float(pe_s["p_on_std"].median())}}
    # rule: a reading that has not changed for >= 90% of running minutes while running >= 90% of the hour
    rule = (s["flat_frac"] >= 0.9) & (s["duty"] >= 0.9)
    yy = np.r_[np.ones(len(pe_s)), np.zeros(len(pe_f))]
    out["event_auc_flat_frac"] = float(roc_auc_score(yy, np.r_[pe_s["flat_frac"].fillna(0), pe_f["flat_frac"].fillna(0)]))
    out["rule_hours"] = int(rule.sum())
    out["rule_precision_vs_stuck_labels"] = float(s.loc[rule, "y_stuck"].mean())
    out["rule_share_on_fault_labels"] = float(s.loc[rule, "y_fault"].mean())
    caught = s.loc[rule & (s["stuck_event"] >= 0), "stuck_event"].nunique()
    scoreable = s.loc[s["stuck_event"] >= 0, "stuck_event"].nunique()
    out["stuck_events_caught"] = int(caught)
    out["stuck_events_scoreable"] = int(scoreable)
    out["fault_events_wrongly_flagged_as_stuck"] = int(s.loc[rule & (s["fault_event"] >= 0), "fault_event"].nunique())
    out["unlabelled_rule_hours_by_appliance"] = s.loc[rule & (s["y_stuck"] == 0) & (s["y_fault"] == 0)].groupby("appliance").size().sort_values(ascending=False).head(6).to_dict()
    _dump("stuck_sensor.json", out)


# ====================================================================== hypotheses

def hypotheses(c: Ctx):
    det = _load("matched.json")
    abl = {k.replace("HGB - ", ""): v for k, v in det.items() if v["tag"] == "ablation"}
    det = {k: v for k, v in det.items() if v["tag"] == "detect"}
    TIMER = "Rule: compressor-on timer (run_max)"
    best_ml = max([k for k in det if k not in (TIMER, "Rule: on-timer vs own first month (run_ref)", "Isolation Forest (unsupervised)")],
                  key=lambda k: det[k]["recall_at_1"])

    def house_recall(entry):
        ph = pd.DataFrame(entry["per_house"])
        ph = ph[ph["events"] > 0]
        return ph.set_index("house")["hit"] / ph.set_index("house")["events"]

    tests = {}
    # H1: best supervised model catches more faults per house than the timer rule (same false-alert rate, out-of-fold)
    a, b = house_recall(det[best_ml]), house_recall(det[TIMER])
    a, b = a.align(b, join="inner")
    w = sps.wilcoxon(a, b, alternative="greater", zero_method="zsplit")
    tests["H1"] = {"claim": f"{best_ml} catches a larger share of anomalies per house than the timer rule at the same false-alert rate (1 per appliance-month)",
                   "n_houses": int(len(a)), "median_diff": float((a - b).median()), "mean_diff": float((a - b).mean()),
                   "houses_better": int((a > b).sum()), "houses_worse": int((a < b).sum()), "p": float(w.pvalue)}
    # H2: among faults both catch, the ML alert comes earlier
    da, db = det[best_ml]["delays_at_1"], det[TIMER]["delays_at_1"]
    both = sorted(set(da) & set(db))
    x, y = np.array([da[i] for i in both]), np.array([db[i] for i in both])
    w = sps.wilcoxon(x, y, alternative="less", zero_method="zsplit") if np.any(x != y) else None
    tests["H2"] = {"claim": f"{best_ml} raises its alert earlier than the timer rule on anomalies both catch", "n_events": len(both),
                   "median_delay_ml_h": float(np.median(x)), "median_delay_rule_h": float(np.median(y)),
                   "events_earlier": int((x < y).sum()), "events_later": int((x > y).sum()), "p": float(w.pvalue) if w else 1.0}
    # H3: features relative to each appliance's own history help a model generalise to unseen homes
    a, b = house_recall(abl["all features"]), house_recall(abl["raw behaviour only"])
    a, b = a.align(b, join="inner")
    w = sps.wilcoxon(a, b, alternative="greater", zero_method="zsplit")
    tests["H3"] = {"claim": "Adding features relative to the appliance's own history raises per-house recall over raw behaviour only",
                   "n_houses": int(len(a)), "median_diff": float((a - b).median()), "mean_diff": float((a - b).mean()), "p": float(w.pvalue)}
    # H4: a frozen sensor shows less power variation while 'running' than a genuinely stuck-on compressor
    r = c.rows[c.rows["scored"] & (c.rows["duty"] >= 0.9)]
    fs = r[r["y_fault"] == 1].groupby("fault_event")["p_on_std"].median().dropna()
    ss = r[r["y_stuck"] == 1].groupby("stuck_event")["p_on_std"].median().dropna()
    u = sps.mannwhitneyu(ss, fs, alternative="less")
    tests["H4"] = {"claim": "Sensor-stuck events show lower power variation while 'running' than real stuck-on compressors",
                   "n_stuck": int(len(ss)), "n_fault": int(len(fs)), "median_std_stuck": float(ss.median()), "median_std_fault": float(fs.median()),
                   "cliffs_delta": stats.cliffs_delta(ss, fs), "p": float(u.pvalue)}
    # H5: compressors run more in summer (Jun-Aug) than in winter (Dec-Feb), paired by appliance
    n = c.rows[c.rows["scored"] & (c.rows["y_fault"] == 0) & (c.rows["y_stuck"] == 0)]
    mon = n["hour_utc"].dt.month
    su = n[mon.isin([6, 7, 8])].groupby("appliance")["duty"].mean()
    wi = n[mon.isin([12, 1, 2])].groupby("appliance")["duty"].mean()
    su, wi = su.align(wi, join="inner")
    w = sps.wilcoxon(su, wi, alternative="greater")
    tests["H5"] = {"claim": "Cold appliances run a larger share of the time in summer than in winter", "n_appliances": int(len(su)),
                   "median_summer": float(su.median()), "median_winter": float(wi.median()), "appliances_higher_in_summer": int((su > wi).sum()),
                   "p": float(w.pvalue)}
    adj = stats.holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = adj[k]
        tests[k]["supported"] = bool(adj[k] < 0.05)
    tests["best_ml"] = best_ml
    _dump("hypotheses.json", tests)
    for k, v in tests.items():
        if isinstance(v, dict):
            print(k, v["claim"][:70], "p_holm=%.4f" % v["p_holm"], v["supported"])


# ====================================================================== explain (SHAP on the gradient-boosting model)

def explain(c: Ctx):
    from . import treeshap
    d = c.d
    cols = features.FEATURES
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
                                         random_state=config.SEED)
    # explain on held-out houses: fit on folds 1-4, explain fold 0
    tr, te = c.folds[0]
    D, T = d.iloc[tr], d.iloc[te]
    X = np.nan_to_num(D[cols].to_numpy(float), nan=-1.0)
    hgb.fit(X, D["y_fault"].to_numpy(), sample_weight=detect.training_weights(D))
    rng = np.random.default_rng(config.SEED)
    pos = np.flatnonzero(T["y_fault"].to_numpy() == 1)
    neg = np.flatnonzero(T["y_fault"].to_numpy() == 0)
    idx = np.r_[rng.choice(pos, min(150, len(pos)), replace=False), rng.choice(neg, 250, replace=False)]
    Xt = np.nan_to_num(T[cols].to_numpy(float), nan=-1.0)[idx]
    phi, base = treeshap.shap_values(hgb, Xt)
    raw = hgb.decision_function(Xt)
    check = float(np.max(np.abs(base + phi.sum(1) - raw)))
    imp = pd.Series(np.abs(phi).mean(0), index=cols).sort_values(ascending=False)
    pd.to_pickle({"phi": phi, "X": Xt, "cols": cols, "base": base, "rows": T.iloc[idx][["appliance", "hour_utc", "y_fault"]]}, R / "_shap_values.pkl")
    fig, axes = plot.plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    top = imp.head(12)[::-1]
    plot.barh(axes[0], list(top.index), list(top.values), [plot.BLUE] * len(top))
    plot.style(axes[0], f"Mean |SHAP| (log-odds), {len(idx)} hours, unseen homes")
    yv = T["y_fault"].to_numpy()[idx]
    for ax, f, lab in [(axes[1], "duty_6h", "running share over the last 6 h"), (axes[2], "run_ref", "running streak / own first-month p95 (log)")]:
        k = cols.index(f)
        xv = Xt[:, k] if f == "duty_6h" else np.log10(np.clip(Xt[:, k], 0.01, None))
        ax.scatter(xv[yv == 0], phi[yv == 0, k], s=7, color=plot.GREY, alpha=0.6, label="normal hour")
        ax.scatter(xv[yv == 1], phi[yv == 1, k], s=7, color=plot.ORANGE, alpha=0.7, label="annotated anomaly hour")
        ax.axhline(0, color=plot.GREY, lw=0.6)
        if f == "run_ref":
            ax.set_xticks([-2, -1, 0, 1, 2], ["0.01", "0.1", "1", "10", "100"])
        plot.style(ax, f"SHAP of {f}", lab, "contribution to anomaly score (log-odds)")
    axes[1].legend(frameon=False, fontsize=8)
    plot.save(fig, F / "shap_hgb.png")
    # one explained alert
    j = int(np.argmax(raw[: len(pos[:150])])) if len(pos) else 0
    contrib = pd.Series(phi[j], index=cols).sort_values(key=np.abs, ascending=False).head(6)
    row = T.iloc[idx[j]]
    _dump("shap.json", {"additivity_max_abs_error": check, "importance": imp.round(4).to_dict(), "houses_explained": sorted(T["house"].unique().tolist()),
                        "example": {"appliance": row["appliance"], "hour_utc": str(row["hour_utc"]), "fault_probability": float(1 / (1 + np.exp(-raw[j]))),
                                    "top_contributions": contrib.round(3).to_dict(),
                                    "values": {k: float(row[k]) for k in contrib.index}}})


# ====================================================================== business impact

def business(c: Ctx):
    """What an alert is worth. Two currencies: the electricity a fault wastes (small) and the time a fridge or freezer
    runs in a fault state before anyone notices (the real risk: food warming, a compressor wearing out)."""
    det = _load("detection.json")
    hyp = _load("hypotheses.json")
    TIMER = "Rule: compressor-on timer (run_max)"
    d = c.d
    normal = d[(d["y_fault"] == 0)]
    rows = []
    for eid, g in d[d["y_fault"] == 1].groupby("fault_event"):
        app, start = g["appliance"].iloc[0], g["hour_utc"].min()
        ref = normal[(normal["appliance"] == app) & (normal["hour_utc"] < start) & (normal["hour_utc"] >= start - pd.Timedelta(days=30))]
        if len(ref) < 72:
            ref = normal[normal["appliance"] == app]
        normal_duty = ref["duty"].median()                       # usual running share in the 30 days before the fault
        p_on = ref["p_on"].median()
        excess_h = (g["duty"] - normal_duty).clip(lower=0).sum()
        rows.append({"event_id": int(eid), "hours": len(g), "excess_kwh": float(excess_h * p_on / 1000), "appliance": app})
    e = pd.DataFrame(rows).set_index("event_id")
    out = {"events": int(len(e)), "fault_hours": int(e["hours"].sum()), "excess_kwh_median_event": float(e["excess_kwh"].median()),
           "excess_kwh_total": float(e["excess_kwh"].sum()), "excess_kwh_largest_event": float(e["excess_kwh"].max()),
           "price_gbp_per_kwh": config.PRICE_PER_KWH_GBP, "response_time_h_assumed": 2}
    out["cost_gbp_median_event"] = out["excess_kwh_median_event"] * config.PRICE_PER_KWH_GBP
    out["cost_gbp_total"] = out["excess_kwh_total"] * config.PRICE_PER_KWH_GBP
    for name in [TIMER, hyp["best_ml"]]:
        b = det[name]["by_budget"]["1.0"]
        hours_cut, kwh_cut = 0.0, 0.0
        for eid, delay in b["delays"].items():
            if int(eid) not in e.index:
                continue
            hrs = e.loc[int(eid), "hours"]
            cut = max(0.0, hrs - delay - out["response_time_h_assumed"])
            hours_cut += cut
            kwh_cut += e.loc[int(eid), "excess_kwh"] * cut / hrs
        out[name] = {"events_caught": b["hit"], "fault_hours_cut": hours_cut, "share_of_fault_hours_cut": hours_cut / out["fault_hours"],
                     "kwh_avoidable": kwh_cut, "gbp_avoidable": kwh_cut * config.PRICE_PER_KWH_GBP,
                     "false_alerts_per_appliance_year": b["false_per_app_month"] * 12, "alerts_on_frozen_sensors": b["stuck_alerts"]}
    _dump("business.json", out)


STAGES = {"quality": quality, "eda": eda, "detect": detect_stage, "ablation": ablation, "matched": matched, "stuck": stuck,
          "hypotheses": hypotheses,
          "explain": explain, "business": business, "figures": figures}


def main(argv):
    F.mkdir(parents=True, exist_ok=True)
    R.mkdir(parents=True, exist_ok=True)
    c = Ctx()
    for s in argv or list(STAGES):
        STAGES[s](c)


if __name__ == "__main__":
    main(sys.argv[1:])
