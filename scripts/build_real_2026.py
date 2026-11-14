"""
AgriCascade - scripts/build_real_2026.py
=========================================
Real (non-synthetic) GEE ingestion for the four grain-belt regions of
Kazakhstan (sko, kostanay, akmola, pavlodar) and the 2026 vegetation season.

New outputs only (existing processed parquet files are never touched):
    data/processed/ndvi_current_2026.parquet      region_id,date,doy,year,ndvi
    data/processed/ndvi_baseline_regions.parquet  same schema, 2015..2025
    data/processed/phenological_norm_regions.parquet
    data/processed/phenoshift_regions.parquet     baseline years + 2026
    data/processed/hydroborder_regions.parquet    SMAP per region (basin mapped)
    data/processed/smap_current_2026.parquet      daily SMAP, 2026 only

Sources:
    COPERNICUS/S2_SR_HARMONIZED (Sentinel-2 L2A, harmonized), 500 m current / 1000 m baseline
    NASA/SMAP/SPL3SMP_E/006 (SMAP L3 9 km), band soil_moisture_am

Conventions:
    * Every reduceRegion call passes an explicit scale and never bestEffort.
    * All output writes are atomic (tmp file + os.replace).
    * Per region/year checkpoints under data/raw/gee_cache/real2026_v1 make
      reruns cheap and resumable; use --force to re-download from GEE.
    * No random / synthetic fill anywhere. Missing GEE data is omitted and
      reported.

Usage:
    python scripts/build_real_2026.py [--force]
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import ee
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    GRAIN_BELT_REGIONS,
    PROCESSED_DIR,
    SMAP_BAND_SURFACE,
    SMAP_ROLLING_WINDOW_DAYS,
    VEGETATION_DOY_START,
    VEGETATION_DOY_END,
)
from pipeline.gee_client import gee
from modules.m1_phenoshift import PhenoShiftModule

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
REGION_IDS = ("sko", "kostanay", "akmola", "pavlodar")

TARGET_YEAR = 2026
BASELINE_NDVI_YEARS = list(range(2015, 2026))       # 2015..2025
SMAI_BASELINE_YEARS = list(range(2015, 2026))       # 2015..2025

SMAP_START_YEAR = 2015
SMAP_END_YEAR = 2026

NDVI_SCALE_M = 500
NDVI_BASELINE_SCALE_M = 1000
SMAP_SCALE_M = 9000
MAX_CLOUD_PCT = 20
NDVI_MIN, NDVI_MAX = -0.1, 1.0

SEASON_START_MD = "04-01"
SEASON_END_MD = "11-01"      # exclusive -> last day is Oct 31

IMAGE_CHUNK = 50             # scenes per synchronous getInfo request
REQUEST_RETRIES = 3

S2_SOURCE = "COPERNICUS/S2_SR_HARMONIZED"
SMAP_SOURCE = "NASA/SMAP/SPL3SMP_E/006"

# Dashboard contract: region_id -> (basin id, basin name)
BASIN_BY_REGION = {
    "sko": ("ishim", "р. Ишим (Есиль)"),
    "kostanay": ("tobol", "р. Тобол"),
    "akmola": ("nura", "р. Нура"),
    "pavlodar": ("irtysh", "р. Иртыш (Ертис)"),
}

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "gee_cache" / "real2026_v1"

PATH_NDVI_CURRENT = PROCESSED_DIR / "ndvi_current_2026.parquet"
PATH_NDVI_BASELINE = PROCESSED_DIR / "ndvi_baseline_regions.parquet"
PATH_NORM = PROCESSED_DIR / "phenological_norm_regions.parquet"
PATH_PSI = PROCESSED_DIR / "phenoshift_regions.parquet"
PATH_HYDRO = PROCESSED_DIR / "hydroborder_regions.parquet"
PATH_SMAP_CURRENT = PROCESSED_DIR / "smap_current_2026.parquet"

NDVI_COLUMNS = ["region_id", "date", "doy", "year", "ndvi", "n_scenes", "source"]
NORM_COLUMNS = ["region_id", "doy", "ndvi_mean", "ndvi_std", "ndvi_count", "ndvi_upper", "ndvi_lower"]
PSI_COLUMNS = [
    "region_id", "year", "psi", "risk", "delay_days",
    "ndvi_mean_current", "ndvi_mean_norm",
]
HYDRO_COLUMNS = [
    "region_id", "basin", "basin_name", "date", "soil_moisture",
    "sm_rolling_90d", "month", "year", "sm_mean", "sm_std", "smai",
    "anomaly_level", "source",
]

MISSING = []  # [(dataset, region_id, year, reason)]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def init_ee() -> None:
    """Initialize Earth Engine using the project from config/env."""
    if not gee.initialize():
        raise RuntimeError("Earth Engine initialization failed. Run: earthengine authenticate")
    logger.info("GEE ready (project={})", os.getenv("GEE_PROJECT", "daryn-505019"))


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet atomically (tmp file then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def get_info_with_retry(ee_object, label: str) -> dict:
    """getInfo() with a small exponential-backoff retry loop."""
    last_error = None
    for attempt in range(REQUEST_RETRIES):
        try:
            return ee_object.getInfo()
        except ee.EEException as exc:
            last_error = exc
            wait = 2 ** attempt
            logger.warning("GEE request failed ({}, attempt {}/{}): {}. Retry in {}s",
                           label, attempt + 1, REQUEST_RETRIES, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"GEE request failed after {REQUEST_RETRIES} attempts: {label}") from last_error


def geometry_for(region_id: str) -> ee.Geometry:
    return ee.Geometry.Polygon(GRAIN_BELT_REGIONS[region_id]["polygon"])


def season_dates(year: int) -> tuple[str, str]:
    return f"{year}-{SEASON_START_MD}", f"{year}-{SEASON_END_MD}"


def empty_ndvi_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "region_id": pd.Series(dtype="object"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "doy": pd.Series(dtype="int32"),
        "year": pd.Series(dtype="int32"),
        "ndvi": pd.Series(dtype="float64"),
        "n_scenes": pd.Series(dtype="int32"),
        "source": pd.Series(dtype="object"),
    })


def validate_frame(df: pd.DataFrame, keys: list, required: list, name: str) -> None:
    """Fail loudly on duplicate keys or NaN in required columns."""
    duplicated = int(df.duplicated(keys).sum())
    if duplicated:
        raise ValueError(f"{name}: {duplicated} duplicate rows on {keys}")
    nan_counts = {col: int(df[col].isna().sum()) for col in required if df[col].isna().any()}
    if nan_counts:
        raise ValueError(f"{name}: NaN in required columns {nan_counts}")
    logger.info("{}: {} rows, keys {} unique, no NaN in {}", name, len(df), keys, required)


# ─────────────────────────────────────────────
# NDVI
# ─────────────────────────────────────────────

def _make_ndvi_reducer(geometry: ee.Geometry, scale: int):
    """Per-scene region stats: valid-pixel mean NDVI and valid-pixel count."""

    def reduce_image(image: ee.Image) -> ee.Feature:
        stats = image.select("NDVI").reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=geometry,
            scale=scale,
            maxPixels=int(1e9),
            tileScale=16,
        )
        return ee.Feature(None, {
            "acq_date": image.date().format("YYYY-MM-dd"),
            "ndvi_mean": stats.get("NDVI_mean"),
            "n_valid": stats.get("NDVI_count"),
        })

    return reduce_image


def extract_ndvi_region_year(region_id: str, year: int, force: bool = False) -> pd.DataFrame:
    """Daily NDVI series (valid-pixel weighted scene aggregation) for one region-year."""
    ckpt = CACHE_DIR / f"ndvi_{region_id}_{year}.parquet"
    if ckpt.exists() and not force:
        df = pd.read_parquet(ckpt)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("cache  NDVI {} {}: {} rows", region_id, year, len(df))
        return df
    if ckpt.exists() and force:
        ckpt.unlink()

    geometry = geometry_for(region_id)
    start, end = season_dates(year)
    collection = (
        gee.get_sentinel2(
            geometry=geometry,
            start_date=start,
            end_date=end,
            max_cloud=MAX_CLOUD_PCT,
            apply_cloud_mask=True,
        )
        .map(gee.add_ndvi_s2)
    )

    t0 = time.perf_counter()
    n_scenes = int(collection.size().getInfo())
    rows: list[tuple] = []
    if n_scenes:
        reducer = _make_ndvi_reducer(geometry, NDVI_SCALE_M if year == TARGET_YEAR else NDVI_BASELINE_SCALE_M)
        for offset in range(0, n_scenes, IMAGE_CHUNK):
            chunk = ee.ImageCollection(collection.toList(IMAGE_CHUNK, offset))
            payload = get_info_with_retry(
                ee.FeatureCollection(chunk.map(reducer)),
                f"NDVI {region_id}/{year} offset={offset}",
            )
            for feature in payload.get("features", []):
                props = feature.get("properties", {})
                acq_date = props.get("acq_date")
                mean = props.get("ndvi_mean")
                n_valid = props.get("n_valid")
                # GEE drops null properties -> .get() is None for fully masked scenes.
                if acq_date is None or mean is None or n_valid is None or n_valid <= 0:
                    continue
                rows.append((acq_date, float(mean), int(n_valid)))

    elapsed = time.perf_counter() - t0
    if not rows:
        logger.warning("NDVI {} {}: no valid scenes ({} scenes scanned, {:.1f}s)",
                       region_id, year, n_scenes, elapsed)
        MISSING.append(("ndvi", region_id, year, "no valid scenes"))
        df = empty_ndvi_frame()
        atomic_write_parquet(df, ckpt)
        return df

    scenes = pd.DataFrame(rows, columns=["date", "ndvi_mean", "n_valid"])
    scenes["wsum"] = scenes["ndvi_mean"] * scenes["n_valid"]
    grouped = scenes.groupby("date", as_index=False).agg(
        wsum=("wsum", "sum"),
        n_total=("n_valid", "sum"),
        n_scenes=("n_valid", "size"),
    )
    grouped["ndvi"] = grouped["wsum"] / grouped["n_total"]
    grouped["date"] = pd.to_datetime(grouped["date"])
    grouped = grouped[(grouped["ndvi"] >= NDVI_MIN) & (grouped["ndvi"] <= NDVI_MAX)]
    grouped["region_id"] = region_id
    grouped["doy"] = grouped["date"].dt.dayofyear.astype("int32")
    grouped["year"] = grouped["date"].dt.year.astype("int32")
    grouped["n_scenes"] = grouped["n_scenes"].astype("int32")
    grouped["source"] = S2_SOURCE
    df = (
        grouped[NDVI_COLUMNS]
        .sort_values("date")
        .reset_index(drop=True)
    )
    df = df[df["year"] == year].reset_index(drop=True)

    if df.empty:
        MISSING.append(("ndvi", region_id, year, "all values outside [-0.1, 1.0]"))
        logger.warning("NDVI {} {}: all values filtered out", region_id, year)

    atomic_write_parquet(df, ckpt)
    logger.info(
        "GEE    NDVI {} {}: {} scenes -> {} daily rows in {:.1f}s",
        region_id, year, n_scenes, len(df), elapsed,
    )
    return df


def extract_ndvi_baseline_monthly(region_id: str, year: int, force: bool = False) -> pd.DataFrame:
    """Monthly Sentinel-2 median composites for the historical baseline.

    Monthly composites keep the real seasonal signal while avoiding hundreds of
    per-scene reduceRegion calls per region-year. The current 2026 season remains
    scene-level daily.
    """
    ckpt = CACHE_DIR / f"ndvi_monthly_{region_id}_{year}.parquet"
    if ckpt.exists() and not force:
        df = pd.read_parquet(ckpt)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("cache  NDVI-monthly {} {}: {} rows", region_id, year, len(df))
        return df

    geometry = geometry_for(region_id)
    start, end = season_dates(year)
    collection = (
        gee.get_sentinel2(geometry=geometry, start_date=start, end_date=end,
                          max_cloud=MAX_CLOUD_PCT, apply_cloud_mask=True)
        .map(gee.add_ndvi_s2)
    )
    rows = []
    for month in range(4, 11):
        m_start = f"{year}-{month:02d}-01"
        m_end = f"{year + (month == 12)}-{month + 1 if month < 12 else 1:02d}-01"
        monthly = collection.filterDate(m_start, m_end)
        count = int(monthly.size().getInfo())
        if not count:
            continue
        stats = monthly.median().select("NDVI").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=NDVI_BASELINE_SCALE_M,
            maxPixels=int(1e9),
            tileScale=16,
        )
        value = stats.get("NDVI").getInfo()
        if value is None or not np.isfinite(float(value)) or not NDVI_MIN <= float(value) <= NDVI_MAX:
            continue
        date = pd.Timestamp(year=year, month=month, day=15)
        rows.append((date, float(value), count))

    if not rows:
        MISSING.append(("ndvi", region_id, year, "no monthly composites"))
        df = empty_ndvi_frame()
        atomic_write_parquet(df, ckpt)
        return df

    df = pd.DataFrame(rows, columns=["date", "ndvi", "n_scenes"])
    df["region_id"] = region_id
    df["doy"] = df["date"].dt.dayofyear.astype("int32")
    df["year"] = year
    df["source"] = S2_SOURCE
    df = df[NDVI_COLUMNS].sort_values("date").reset_index(drop=True)
    atomic_write_parquet(df, ckpt)
    logger.info("GEE    NDVI-monthly {} {}: {} composites", region_id, year, len(df))
    return df


def print_ndvi_summary(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        logger.warning("{}: empty", name)
        return
    print(f"\n== {name} ==")
    summary = df.groupby("region_id").agg(
        rows=("ndvi", "size"),
        date_min=("date", "min"),
        date_max=("date", "max"),
        ndvi_min=("ndvi", "min"),
        ndvi_mean=("ndvi", "mean"),
        ndvi_max=("ndvi", "max"),
        scenes=("n_scenes", "sum"),
    )
    print(summary.to_string())
    print()


def build_ndvi_files(force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_frames, baseline_frames = [], []

    # 2026 season
    for region_id in REGION_IDS:
        df = extract_ndvi_region_year(region_id, TARGET_YEAR, force)
        if not df.empty:
            current_frames.append(df)

    # Baseline seasons (region-specific, monthly median composites)
    for region_id in REGION_IDS:
        for year in BASELINE_NDVI_YEARS:
            df = extract_ndvi_baseline_monthly(region_id, year, force)
            if not df.empty:
                baseline_frames.append(df)

    current = pd.concat(current_frames, ignore_index=True) if current_frames else empty_ndvi_frame()
    baseline = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else empty_ndvi_frame()

    if current.empty:
        raise RuntimeError("No 2026 NDVI data for any region")
    if baseline.empty:
        raise RuntimeError("No baseline NDVI data for any region")

    validate_frame(current, ["region_id", "date"], ["region_id", "date", "doy", "year", "ndvi"], "ndvi_current_2026")
    validate_frame(baseline, ["region_id", "date"], ["region_id", "date", "doy", "year", "ndvi"], "ndvi_baseline_regions")

    atomic_write_parquet(current, PATH_NDVI_CURRENT)
    atomic_write_parquet(baseline, PATH_NDVI_BASELINE)
    logger.success("Wrote {} and {}", PATH_NDVI_CURRENT.name, PATH_NDVI_BASELINE.name)

    print_ndvi_summary(current, "ndvi_current_2026")
    print_ndvi_summary(baseline, "ndvi_baseline_regions (2015-2023)")
    return current, baseline


# ─────────────────────────────────────────────
# Phenology: norm + PSI
# ─────────────────────────────────────────────

def build_norm_regions(baseline: pd.DataFrame) -> pd.DataFrame:
    """Per-region phenological norm via PhenoShiftModule.build_phenological_norm."""
    module = PhenoShiftModule()
    frames = []
    for region_id in REGION_IDS:
        region_df = baseline[baseline["region_id"] == region_id]
        if region_df.empty:
            MISSING.append(("norm", region_id, None, "no baseline NDVI"))
            logger.warning("Norm {}: no baseline data", region_id)
            continue
        norm = module.build_phenological_norm(region_df[["doy", "ndvi"]].copy())
        norm["region_id"] = region_id
        norm["ndvi_count"] = norm["ndvi_count"].fillna(0)
        frames.append(norm)

    if not frames:
        raise RuntimeError("Could not build phenological norm for any region")

    out = pd.concat(frames, ignore_index=True)[NORM_COLUMNS]
    out["doy"] = out["doy"].astype("int32")
    out = out.sort_values(["region_id", "doy"]).reset_index(drop=True)
    validate_frame(out, ["region_id", "doy"],
                   ["region_id", "doy", "ndvi_mean", "ndvi_std", "ndvi_count", "ndvi_upper", "ndvi_lower"],
                   "phenological_norm_regions")
    d_min = int(out.loc[out["doy"] == out["doy"].min(), "doy"].iloc[0])
    d_max = int(out["doy"].max())
    if d_min < VEGETATION_DOY_START or d_max > VEGETATION_DOY_END:
        raise ValueError(f"Norm DOY outside [{VEGETATION_DOY_START}, {VEGETATION_DOY_END}]: {d_min}..{d_max}")
    atomic_write_parquet(out, PATH_NORM)
    logger.success("Wrote {} ({} rows, DOY {}..{})", PATH_NORM.name, len(out), d_min, d_max)

    print("\n== phenological_norm_regions ==")
    print(out.groupby("region_id").agg(
        rows=("doy", "size"),
        doy_min=("doy", "min"),
        doy_max=("doy", "max"),
        ndvi_mean=("ndvi_mean", "mean"),
        ndvi_std_min=("ndvi_std", "min"),
    ).to_string())
    print()
    return out


def build_psi_regions(baseline: pd.DataFrame, current: pd.DataFrame, norm: pd.DataFrame) -> pd.DataFrame:
    """Per-region PSI for baseline years + 2026 using PhenoShiftModule.compute_psi."""
    module = PhenoShiftModule()
    rows = []
    for region_id in REGION_IDS:
        norm_region = norm[norm["region_id"] == region_id][["doy", "ndvi_mean", "ndvi_std"]]
        if norm_region.empty:
            continue
        for year in BASELINE_NDVI_YEARS + [TARGET_YEAR]:
            source = current if year == TARGET_YEAR else baseline
            year_df = source[(source["region_id"] == region_id) & (source["year"] == year)]
            if year_df.empty:
                MISSING.append(("psi", region_id, year, "no NDVI for year"))
                continue
            result = module.compute_psi(norm_region, year_df[["doy", "ndvi"]].copy())
            if result.get("psi") is None:
                MISSING.append(("psi", region_id, year, result.get("risk", "NO_PSI")))
                logger.warning("PSI {} {}: {} ({} points)", region_id, year,
                               result.get("risk"), len(year_df))
                continue
            rows.append({
                "region_id": region_id,
                "year": int(year),
                "psi": float(result["psi"]),
                "risk": result["risk"],
                "delay_days": int(result["delay_days"]),
                "ndvi_mean_current": float(result["ndvi_mean_current"]),
                "ndvi_mean_norm": float(result["ndvi_mean_norm"]),
            })

    if not rows:
        raise RuntimeError("No PSI rows produced")
    out = pd.DataFrame(rows)[PSI_COLUMNS].sort_values(["region_id", "year"]).reset_index(drop=True)
    validate_frame(out, ["region_id", "year"], PSI_COLUMNS, "phenoshift_regions")
    atomic_write_parquet(out, PATH_PSI)
    logger.success("Wrote {} ({} rows, years {}..{})", PATH_PSI.name, len(out),
                   int(out["year"].min()), int(out["year"].max()))

    print("\n== phenoshift_regions ==")
    print(out.to_string(index=False))
    print()
    return out


# ─────────────────────────────────────────────
# SMAP / HydroBorder
# ─────────────────────────────────────────────

def _make_smap_reducer(geometry: ee.Geometry):
    def reduce_image(image: ee.Image) -> ee.Feature:
        stats = image.select(SMAP_BAND_SURFACE).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=SMAP_SCALE_M,
            maxPixels=int(1e9),
            tileScale=16,
        )
        return ee.Feature(None, {
            "acq_date": image.date().format("YYYY-MM-dd"),
            "soil_moisture": stats.get(SMAP_BAND_SURFACE),
        })

    return reduce_image


def extract_smap_region_year(region_id: str, year: int, force: bool = False) -> pd.DataFrame:
    """Daily region-mean SMAP soil moisture for one region-year."""
    ckpt = CACHE_DIR / f"smap_{region_id}_{year}.parquet"
    if ckpt.exists() and not force:
        df = pd.read_parquet(ckpt)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("cache  SMAP {} {}: {} rows", region_id, year, len(df))
        return df
    if ckpt.exists() and force:
        ckpt.unlink()

    geometry = geometry_for(region_id)
    collection = gee.get_smap(geometry, f"{year}-01-01", f"{year}-12-31")

    t0 = time.perf_counter()
    n_images = int(collection.size().getInfo())
    rows: list[tuple] = []
    if n_images:
        reducer = _make_smap_reducer(geometry)
        for offset in range(0, n_images, IMAGE_CHUNK):
            chunk = ee.ImageCollection(collection.toList(IMAGE_CHUNK, offset))
            payload = get_info_with_retry(
                ee.FeatureCollection(chunk.map(reducer)),
                f"SMAP {region_id}/{year} offset={offset}",
            )
            for feature in payload.get("features", []):
                props = feature.get("properties", {})
                acq_date = props.get("acq_date")
                value = props.get("soil_moisture")
                if acq_date is None or value is None:
                    continue
                value = float(value)
                if not np.isfinite(value) or value < 0.0 or value > 1.0:
                    continue
                rows.append((acq_date, value))

    elapsed = time.perf_counter() - t0
    if not rows:
        logger.warning("SMAP {} {}: no valid retrievals ({} images, {:.1f}s)",
                       region_id, year, n_images, elapsed)
        MISSING.append(("smap", region_id, year, "no valid retrievals"))
        df = pd.DataFrame({
            "region_id": pd.Series(dtype="object"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "soil_moisture": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="object"),
        })
        atomic_write_parquet(df, ckpt)
        return df

    df = pd.DataFrame(rows, columns=["date", "soil_moisture"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.groupby("date", as_index=False)["soil_moisture"].mean()
    df["region_id"] = region_id
    df["source"] = SMAP_SOURCE
    df = df.sort_values("date").reset_index(drop=True)

    atomic_write_parquet(df, ckpt)
    logger.info("GEE    SMAP {} {}: {} images -> {} daily rows in {:.1f}s",
                region_id, year, n_images, len(df), elapsed)
    return df


def build_hydro(force: bool) -> pd.DataFrame:
    """SMAI per region with the m2 formula (90d rolling, monthly baseline)."""
    frames = []
    smap_current_frames = []

    for region_id in REGION_IDS:
        basin_id, basin_name = BASIN_BY_REGION[region_id]
        year_frames = []
        for year in range(SMAP_START_YEAR, SMAP_END_YEAR + 1):
            df = extract_smap_region_year(region_id, year, force)
            if not df.empty:
                year_frames.append(df)

        if not year_frames:
            MISSING.append(("hydro", region_id, None, "no SMAP data for region"))
            logger.warning("Hydro {}: no SMAP data at all, region omitted", region_id)
            continue

        raw = pd.concat(year_frames, ignore_index=True)
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.groupby("date", as_index=False)["soil_moisture"].mean()
        raw = raw[(raw["soil_moisture"] >= 0.0) & (raw["soil_moisture"] <= 1.0)]
        if raw.empty:
            MISSING.append(("hydro", region_id, None, "all SMAP values outside [0, 1]"))
            continue
        raw = raw.sort_values("date").set_index("date")

        raw["sm_rolling_90d"] = (
            raw["soil_moisture"]
            .rolling(window=SMAP_ROLLING_WINDOW_DAYS, min_periods=10)
            .mean()
        )
        raw["month"] = raw.index.month.astype("int32")
        raw["year"] = raw.index.year.astype("int32")

        baseline_mask = raw["year"].isin(SMAI_BASELINE_YEARS)
        monthly_stats = (
            raw[baseline_mask]
            .groupby("month")["sm_rolling_90d"]
            .agg(sm_mean="mean", sm_std="std")
            .reset_index()
        )
        monthly_stats["sm_std"] = monthly_stats["sm_std"].fillna(0.02).clip(lower=0.01)

        out = raw.reset_index()
        out = out.merge(monthly_stats, on="month", how="left")
        out["smai"] = (out["sm_rolling_90d"] - out["sm_mean"]) / out["sm_std"]

        conditions = [
            out["smai"] < -2.0,
            out["smai"] < -1.0,
            out["smai"] < -0.5,
            out["smai"] > 2.0,
            out["smai"] > 1.0,
        ]
        choices = ["SEVERE_DRY", "DRY", "SLIGHTLY_DRY", "WET", "SLIGHTLY_WET"]
        out["anomaly_level"] = np.select(conditions, choices, default="NORMAL")

        out["region_id"] = region_id
        out["basin"] = basin_id
        out["basin_name"] = basin_name
        out["source"] = SMAP_SOURCE

        before = len(out)
        out = out.dropna(subset=["soil_moisture", "sm_rolling_90d", "sm_mean", "sm_std", "smai"])
        dropped = before - len(out)
        if dropped:
            logger.info("Hydro {}: dropped {} rows with NaN rolling/baseline stats", region_id, dropped)

        if out.empty:
            MISSING.append(("hydro", region_id, None, "empty after rolling/baseline joins"))
            continue

        frames.append(out[HYDRO_COLUMNS])
        smap_current_frames.append(
            out.loc[out["year"] == TARGET_YEAR, ["region_id", "date", "soil_moisture", "source"]]
        )

    if not frames:
        raise RuntimeError("No hydroborder rows produced for any region")

    hydro = pd.concat(frames, ignore_index=True)
    hydro["date"] = pd.to_datetime(hydro["date"])
    hydro = hydro.sort_values(["region_id", "date"]).reset_index(drop=True)
    validate_frame(hydro, ["region_id", "date"],
                   ["region_id", "basin", "date", "soil_moisture", "sm_rolling_90d",
                    "sm_mean", "sm_std", "smai", "anomaly_level"],
                   "hydroborder_regions")
    atomic_write_parquet(hydro, PATH_HYDRO)
    logger.success("Wrote {} ({} rows)", PATH_HYDRO.name, len(hydro))

    print("\n== hydroborder_regions ==")
    print(hydro.groupby(["region_id", "basin"]).agg(
        rows=("date", "size"),
        date_min=("date", "min"),
        date_max=("date", "max"),
        sm_mean=("soil_moisture", "mean"),
        smai_min=("smai", "min"),
        smai_mean=("smai", "mean"),
        dry_days=("anomaly_level", lambda s: int(s.isin(["SEVERE_DRY", "DRY", "SLIGHTLY_DRY"]).sum())),
    ).to_string())
    print()

    if smap_current_frames:
        smap_current = pd.concat(smap_current_frames, ignore_index=True)
        smap_current = smap_current.sort_values(["region_id", "date"]).reset_index(drop=True)
        smap_current = smap_current.dropna(subset=["soil_moisture"])
        validate_frame(smap_current, ["region_id", "date"],
                       ["region_id", "date", "soil_moisture"], "smap_current_2026")
        atomic_write_parquet(smap_current, PATH_SMAP_CURRENT)
        logger.success("Wrote {} ({} rows)", PATH_SMAP_CURRENT.name, len(smap_current))
        print("== smap_current_2026 ==")
        print(smap_current.groupby("region_id").agg(
            rows=("date", "size"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            sm_mean=("soil_moisture", "mean"),
        ).to_string())
        print()

    return hydro


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build real GEE datasets for 4 grain-belt regions")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached region/year checkpoints and re-download from GEE")
    parser.add_argument("--skip-smap", action="store_true",
                        help="Skip SMAP/HydroBorder stage (debug only)")
    args = parser.parse_args()

    t_start = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    init_ee()

    logger.info("Stage 1/3: NDVI (Sentinel-2 SR harmonized, 500 m current / 1000 m baseline)")
    current, baseline = build_ndvi_files(args.force)

    logger.info("Stage 2/3: phenological norm + PSI per region")
    norm = build_norm_regions(baseline)
    build_psi_regions(baseline, current, norm)

    if not args.skip_smap:
        logger.info("Stage 3/3: SMAP / HydroBorder per region")
        build_hydro(args.force)
    else:
        logger.warning("SMAP stage skipped by flag")

    elapsed = time.perf_counter() - t_start
    logger.success("Done in {:.1f} min", elapsed / 60.0)

    if MISSING:
        print("\n== Missing / omitted data ==")
        for dataset, region_id, year, reason in MISSING:
            print(f"  {dataset:6s} {region_id:9s} {str(year):>5s}  {reason}")
    else:
        print("\n== Missing / omitted data == none")


if __name__ == "__main__":
    main()
