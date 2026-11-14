"""Research-reference 2021 series from the legacy NASA-USDA 10 km SMAP product.

This product is the source named in the research work. It is discontinued in
GEE (superseded by SPL4SMGP), so it is used only for historical validation,
not for current 2026 monitoring.
"""

import sys
from pathlib import Path

import ee
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import GRAIN_BELT_REGIONS, PROCESSED_DIR
from pipeline.gee_client import gee


def _reduce(image, geometry):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10000,
        maxPixels=1_000_000_000,
        tileScale=8,
    )
    return ee.Feature(None, {
        "date": image.date().format("YYYY-MM-dd"),
        "ssm": stats.get("ssm"),
        "susm": stats.get("susm"),
        "smp": stats.get("smp"),
        "ssma": stats.get("ssma"),
        "susma": stats.get("susma"),
    })


def build():
    if not gee.initialize():
        raise RuntimeError("GEE initialization failed")
    frames = []
    summaries = []
    for region_id, info in GRAIN_BELT_REGIONS.items():
        geometry = ee.Geometry.Polygon(info["polygon"])
        collection = gee.get_smap_usda(geometry, "2021-04-01", "2021-10-31")
        features = ee.FeatureCollection(collection.map(lambda image: _reduce(image, geometry))).getInfo()
        rows = [f["properties"] for f in features.get("features", [])
                if f.get("properties", {}).get("date")]
        df = pd.DataFrame(rows)
        if df.empty:
            print(region_id, "no NASA-USDA rows")
            continue
        df["date"] = pd.to_datetime(df["date"])
        df["region_id"] = region_id
        frames.append(df)
        first = df[df["ssma"] < -1.5].sort_values("date")
        summaries.append({
            "region_id": region_id,
            "first_ssma_trigger": first.iloc[0]["date"] if not first.empty else pd.NaT,
            "min_ssma": float(df["ssma"].min()),
            "source": "NASA_USDA/HSL/SMAP10KM_soil_moisture (legacy)",
        })
        print(f"{region_id}: first SSMA < -1.5 = {first.iloc[0]['date'].date() if not first.empty else None}, "
              f"min={df['ssma'].min():.2f}")
    if not frames:
        raise RuntimeError("No NASA-USDA data returned")
    result = pd.concat(frames, ignore_index=True).sort_values(["region_id", "date"])
    summary = pd.DataFrame(summaries)
    result.to_parquet(PROCESSED_DIR / "backtest_2021_usda_regions.parquet", index=False)
    summary.to_parquet(PROCESSED_DIR / "backtest_2021_usda_summary.parquet", index=False)
    print("Wrote NASA-USDA 2021 reference files")


if __name__ == "__main__":
    build()
