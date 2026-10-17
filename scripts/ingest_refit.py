"""Convert raw REFIT house CSVs (8-second readings, no header) into compact 1-minute files.

Input row:  unix_time_utc, aggregate, appliance1 ... appliance9   (watts)
Output:     data/interim/house{N}.pkl  - DataFrame indexed by UTC minute with
              <channel>_w   mean power over the minute (W)
              n_readings    number of raw readings in the minute (0 = no data)
Minutes with no readings are kept (NaN power) so gaps stay visible.
Cleaning: appliance readings above 4000 W are physically impossible for a plug monitor (the REFIT authors apply the same
rule in their cleaned release) and are corrupted values such as 65535 or 98301; they are set to missing and counted.

Usage: python scripts/ingest_refit.py RAW_DIR OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["unix", "aggregate"] + [f"app{i}" for i in range(1, 10)]
MAX_APPLIANCE_W = 4000


def convert(src: Path, dst: Path) -> dict:
    raw = pd.read_csv(src, header=None, names=COLS, dtype={c: "float32" for c in COLS[1:]} | {"unix": "int64"}, engine="c")
    na = int(raw[COLS[1:]].isna().sum().sum())
    apps = COLS[2:]
    max_raw = {c: float(raw[c].max()) for c in COLS[1:]}
    impossible = {c: int((raw[c] > MAX_APPLIANCE_W).sum()) for c in apps}
    raw[apps] = raw[apps].where(raw[apps] <= MAX_APPLIANCE_W)
    raw = raw.sort_values("unix")
    dup = int(raw.duplicated("unix").sum())
    raw = raw.drop_duplicates("unix", keep="last")
    minute = raw["unix"] // 60 * 60
    g = raw.drop(columns="unix").groupby(minute.to_numpy())
    out = g.mean().astype("float32")
    out.columns = [f"{c}_w" for c in out.columns]
    out["n_readings"] = g.size().astype("int16")
    full = np.arange(out.index.min(), out.index.max() + 60, 60)
    out = out.reindex(full)
    out["n_readings"] = out["n_readings"].fillna(0).astype("int16")
    out.index = pd.to_datetime(out.index, unit="s", utc=True)
    out.index.name = "minute_utc"
    out.to_pickle(dst)
    dt = np.diff(raw["unix"].to_numpy())
    return {"rows": len(raw), "duplicate_timestamps": dup, "start": out.index[0], "end": out.index[-1],
            "minutes": len(out), "minutes_without_data": int((out["n_readings"] == 0).sum()),
            "median_interval_s": float(np.median(dt)), "missing_values": na, "max_w_raw": max_raw, "impossible_readings": impossible,
            "impossible_total": int(sum(impossible.values())),
            "negative_values": int((raw[COLS[1:]] < 0).sum().sum())}


if __name__ == "__main__":
    raw_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(raw_dir.glob("House*.csv"), key=lambda p: int(p.stem[5:])):
        n = int(f.stem[5:])
        info = convert(f, out_dir / f"house{n}.pkl")
        info["house"] = n
        rows.append(info)
        print(n, info["rows"], info["start"], info["end"], info["minutes_without_data"], info["median_interval_s"], flush=True)
    pd.DataFrame(rows).to_pickle(out_dir / "ingest_summary.pkl")
