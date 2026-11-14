"""Leakage-free 2021 SMAP backtest using the research-work SMAI definition.

SMAI_t = (theta_t - mu_hist) / sigma_hist, where mu/sigma are the same
calendar dekad (10-day period) over 2015-2020. The 2021 target year is excluded
from the baseline. The platform currently uses the native SMAP 9 km product and
does not yet implement the SMAP -> 10 m downscaling from the research work.
Therefore the documented 18.06.2021 / SMAI -1.68 target is kept as a reference,
not overwritten by a hardcoded value.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import GRAIN_BELT_REGIONS, PROCESSED_DIR

OFFICIAL_DATE = pd.Timestamp("2021-07-15")
RESEARCH_TARGET_DATE = pd.Timestamp("2021-06-18")
RESEARCH_TARGET_SMAI = -1.68
THRESHOLD = -1.5
BASELINE_START = 2015
BASELINE_END = 2020
SEASON_START = pd.Timestamp("2021-05-01")
SEASON_END = pd.Timestamp("2021-09-30")


def _dekad(frame):
    month = frame["date"].dt.month
    dekad = ((frame["date"].dt.day - 1) // 10 + 1).astype(int)
    return month * 10 + dekad


def build():
    source = pd.read_parquet(PROCESSED_DIR / "hydroborder_regions.parquet")
    source["date"] = pd.to_datetime(source["date"])
    frames = []
    summaries = []
    for region_id in GRAIN_BELT_REGIONS:
        region = source[source["region_id"] == region_id].copy()
        if region.empty:
            print(f"{region_id}: no SMAP rows")
            continue
        region["year"] = region["date"].dt.year
        region["period"] = _dekad(region)
        period_mean = (
            region.groupby(["year", "period"], as_index=False)
            .agg(date=("date", "max"), soil_moisture=("soil_moisture", "mean"))
        )
        baseline = period_mean[
            (period_mean["year"] >= BASELINE_START) & (period_mean["year"] <= BASELINE_END)
        ]
        stats = baseline.groupby("period")["soil_moisture"].agg(
            sm_mean="mean", sm_std="std"
        ).reset_index()
        stats["sm_std"] = stats["sm_std"].clip(lower=0.01)
        target = period_mean[period_mean["year"] == 2021].merge(stats, on="period", how="left")
        target["smai"] = (target["soil_moisture"] - target["sm_mean"]) / target["sm_std"]
        target = target[
            (target["date"] >= SEASON_START) & (target["date"] <= SEASON_END)
        ].copy()
        triggers = target[target["smai"] < THRESHOLD]
        trigger = triggers.iloc[0]["date"] if not triggers.empty else None
        lead_days = int((OFFICIAL_DATE - trigger).days) if trigger is not None else None
        target["region_id"] = region_id
        frames.append(target[["region_id", "date", "soil_moisture", "smai"]])
        summaries.append({
            "region_id": region_id,
            "trigger_date": trigger,
            "official_date": OFFICIAL_DATE,
            "lead_days": lead_days,
            "threshold": THRESHOLD,
            "min_smai": float(target["smai"].min()),
            "baseline": f"{BASELINE_START}-{BASELINE_END} dekad",
            "research_target_date": RESEARCH_TARGET_DATE,
            "research_target_smai": RESEARCH_TARGET_SMAI,
            "source": "NASA/SMAP/SPL3SMP_E/006 native 9 km; no downscaling yet",
        })
        print(f"{region_id}: trigger={trigger.date() if trigger is not None else None}, "
              f"lead_days={lead_days}, min_smai={target['smai'].min():.2f}")

    if not frames:
        raise RuntimeError("No SMAP backtest rows produced")
    result = pd.concat(frames, ignore_index=True).sort_values(["region_id", "date"])
    summary = pd.DataFrame(summaries)
    result.to_parquet(PROCESSED_DIR / "backtest_2021_regions.parquet", index=False)
    summary.to_parquet(PROCESSED_DIR / "backtest_2021_summary.parquet", index=False)
    print("Wrote backtest_2021_regions.parquet and backtest_2021_summary.parquet")
    return result, summary


if __name__ == "__main__":
    build()
