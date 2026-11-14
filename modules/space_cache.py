"""Persistent offline cache for Phase 7 observations and local archive readers."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import PROCESSED_DIR
from modules.space_segment import validate_polygon


CACHE_SCHEMA = "agricascade.space-cache/1"
CACHE_DIR = PROCESSED_DIR / "space_cache"
NOAA_CACHE = CACHE_DIR / "noaa_kp.json"


class SpaceCacheError(ValueError):
    """Raised when a cache file exists but cannot be trusted."""


def aoi_sha256(geometry):
    polygon = validate_polygon(geometry)
    canonical = json.dumps({"type": "Polygon", "coordinates": [list(polygon.exterior.coords)] +
                            [list(ring.coords) for ring in polygon.interiors]},
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_query(*, region_id, district_id, year, orbit_pass, start_date, end_date, geometry):
    if orbit_pass not in ("ASCENDING", "DESCENDING"):
        raise ValueError("Invalid orbit pass")
    if not isinstance(region_id, str) or not region_id.strip():
        raise ValueError("Invalid region")
    if not isinstance(district_id, str) or not district_id.strip():
        raise ValueError("Invalid district")
    if isinstance(year, bool) or not isinstance(year, int) or not 2000 <= year <= 2100:
        raise ValueError("Invalid year")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise ValueError("Invalid cache date range")
    return {
        "region_id": region_id,
        "district_id": district_id,
        "year": year,
        "orbit_pass": orbit_pass,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "aoi_sha256": aoi_sha256(geometry),
    }


def _cache_path(query):
    stem = f"{query['region_id']}_{query['district_id']}_{query['year']}_{query['orbit_pass']}"
    digest = hashlib.sha256(json.dumps(query, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return CACHE_DIR / f"{stem}_{digest}.json"


def load_space_cache(query):
    path = _cache_path(query)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SpaceCacheError("Offline cache is unreadable") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA:
        raise SpaceCacheError("Offline cache schema mismatch")
    if payload.get("query") != query:
        raise SpaceCacheError("Offline cache query mismatch")
    observations = payload.get("observations")
    if not isinstance(observations, dict) or not isinstance(observations.get("sar"), list) or not isinstance(observations.get("optical"), list):
        raise SpaceCacheError("Offline cache observations are invalid")
    try:
        cached_at = pd.Timestamp(payload["cached_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise SpaceCacheError("Offline cache timestamp is invalid") from error
    if pd.isna(cached_at) or cached_at.tzinfo is None:
        raise SpaceCacheError("Offline cache timestamp must be timezone-aware")
    return payload


def save_space_cache(query, observations, *, provider="Google Earth Engine", sources=None):
    if not isinstance(observations, dict):
        raise ValueError("Observations must be a mapping")
    if not isinstance(observations.get("sar"), list) or not isinstance(observations.get("optical"), list):
        raise ValueError("Observations must include SAR and optical lists")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Cache provider is required")
    sources = sources or {}
    payload = {
        "schema_version": CACHE_SCHEMA,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "observations": observations,
        "provenance": {
            "provider": provider,
            "sar": sources.get("sar", "COPERNICUS/S1_GRD"),
            "optical": sources.get("optical", "COPERNICUS/S2_SR_HARMONIZED"),
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = _cache_path(query)
    fd, temporary = tempfile.mkstemp(prefix=".space-cache-", suffix=".json", dir=CACHE_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return destination


def _validate_noaa(observation):
    required = {"kp", "observed_at", "source", "stale", "product"}
    if not isinstance(observation, dict) or not required.issubset(observation):
        raise ValueError("Invalid NOAA observation")
    kp = float(observation["kp"])
    if not 0 <= kp <= 9 or not isinstance(observation["stale"], bool):
        raise ValueError("Invalid NOAA Kp")
    if not isinstance(observation["source"], str) or not observation["source"].strip():
        raise ValueError("Invalid NOAA source")
    if not isinstance(observation["product"], str) or not observation["product"].strip():
        raise ValueError("Invalid NOAA product")
    stamp = pd.Timestamp(observation["observed_at"])
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Invalid NOAA timestamp")
    return {**observation, "kp": kp}


def save_noaa_kp(observation):
    observation = _validate_noaa(observation)
    payload = {"schema_version": "agricascade.noaa-kp-cache/1", "observation": observation}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".noaa-kp-", suffix=".json", dir=CACHE_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, NOAA_CACHE)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def load_noaa_kp():
    if not NOAA_CACHE.exists():
        return None
    try:
        payload = json.loads(NOAA_CACHE.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "agricascade.noaa-kp-cache/1":
            raise ValueError("Schema mismatch")
        observation = _validate_noaa(payload["observation"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SpaceCacheError("NOAA cache is unreadable") from error
    return observation


def offline_archive_status():
    files = []
    for path in sorted(PROCESSED_DIR.glob("*.parquet")):
        stat = path.stat()
        files.append({
            "name": path.name,
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
    return files


def _read_parquet(name, required):
    path = PROCESSED_DIR / name
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if not set(required).issubset(frame.columns):
        return None
    return frame


def archive_ndvi_climatology(region_id):
    """Return the regional SKO climatology only; never represent it as a field measurement."""
    if region_id != "sko":
        return None
    frame = _read_parquet("phenological_norm.parquet", ("doy", "ndvi_mean"))
    if frame is None or frame.empty:
        return None
    frame = frame[["doy", "ndvi_mean"]].copy()
    frame["doy"] = pd.to_numeric(frame["doy"], errors="coerce")
    frame["ndvi_mean"] = pd.to_numeric(frame["ndvi_mean"], errors="coerce")
    frame = frame.dropna().sort_values("doy")
    if frame.empty:
        return None
    return {
        "frame": frame.reset_index(drop=True),
        "label_kk": "NDVI климатологиясы 2015–2023 · СҚО (өлшем емес)",
        "label_ru": "Климатология NDVI 2015–2023 · СКО (не измерение)",
        "source": "data/processed/phenological_norm.parquet",
    }


def archive_smap_series(basin_id):
    frame = _read_parquet("smap_timeseries.parquet", ("date", "soil_moisture", "basin"))
    if frame is None or basin_id not in set(frame["basin"]):
        return None
    frame = frame[frame["basin"] == basin_id].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["soil_moisture"] = pd.to_numeric(frame["soil_moisture"], errors="coerce")
    frame = frame.dropna(subset=["date", "soil_moisture"]).sort_values("date")
    if frame.empty:
        return None
    latest = frame.iloc[-1]
    return {
        "frame": frame[["date", "soil_moisture"]].reset_index(drop=True),
        "latest": {"date": latest["date"].isoformat(), "soil_moisture": float(latest["soil_moisture"])},
        "source": "data/processed/smap_timeseries.parquet",
        "basin": basin_id,
    }
