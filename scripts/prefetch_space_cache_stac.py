"""Prefill the Phase 7 offline cache from public Copernicus Sentinel-1 RTC STAC.

This is a fallback for the defense when the configured GEE project is not yet
registered. It writes only data/processed/space_cache; existing parquet files
are never modified. The cache is explicitly labelled with its STAC provider.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import GRAIN_BELT_REGIONS
from modules.gis_engine import DISTRICT_REGISTRY
from modules.space_cache import build_query, load_space_cache, save_space_cache


STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel1euwestrtc/sentinel1-grd-rtc"
PROVIDER = "Copernicus Sentinel-1 RTC · Microsoft Planetary Computer"


def _acquisition_key(scene_id):
    parts = scene_id.split("_")
    if len(parts) < 7:
        return scene_id
    return parts[4][:8], parts[-3], parts[-2]


def _sas_token():
    response = requests.get(SAS_TOKEN, timeout=30)
    response.raise_for_status()
    token = response.json()["token"]
    return token


def _search(items, orbit_state):
    response = requests.post(STAC_SEARCH, json=items, timeout=60)
    response.raise_for_status()
    features = response.json().get("features", [])
    selected = []
    for feature in features:
        props = feature.get("properties", {})
        if str(props.get("sat:orbit_state", "")).upper() != orbit_state:
            continue
        if not {"vv", "vh"}.issubset(feature.get("assets", {})):
            continue
        selected.append(feature)
    selected.sort(key=lambda feature: feature["properties"]["datetime"])
    return selected


def _read_power(url, token, lonlat_bounds, tags_out):
    separator = "&" if "?" in url else "?"
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff",
        GDAL_HTTP_TIMEOUT=60,
        GDAL_HTTP_MAX_RETRY=1,
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=50_000_000,
    ):
        with rasterio.open(url + separator + token) as source:
            transformer = Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
            xs, ys = transformer.transform(
                [lonlat_bounds[0], lonlat_bounds[2], lonlat_bounds[2], lonlat_bounds[0]],
                [lonlat_bounds[1], lonlat_bounds[1], lonlat_bounds[3], lonlat_bounds[3]],
            )
            window = from_bounds(min(xs), min(ys), max(xs), max(ys), transform=source.transform)
            out_shape = (max(1, min(256, int(window.height))), max(1, min(256, int(window.width))))
            data = source.read(1, window=window, out_shape=out_shape, resampling=Resampling.average).astype("float64")
            tags_out.update(source.tags())
    valid = np.isfinite(data) & (data > 0) & (data != -32768)
    if not valid.any():
        raise ValueError("No valid RTC pixels in AOI")
    return float(data[valid].mean())


def prefetch(region_id, district_id, year, orbit_state, relative_orbit, limit):
    district = next((item for item in DISTRICT_REGISTRY[region_id] if item["id"] == district_id), None)
    if district is None:
        raise ValueError(f"Unknown district: {district_id}")
    polygon = {"type": "Polygon", "coordinates": [district["polygon"]]}
    lons = [point[0] for point in district["polygon"]]
    lats = [point[1] for point in district["polygon"]]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    start, end = f"{year}-04-01T00:00:00Z", f"{year}-10-01T00:00:00Z"
    query = build_query(
        region_id=region_id, district_id=district_id, year=year, orbit_pass=orbit_state,
        start_date=start[:10], end_date=end[:10], geometry=polygon,
    )
    existing = load_space_cache(query)
    observations = existing["observations"] if existing else {"sar": [], "optical": [], "relative_orbit": None, "scale_m": 256}
    known = {row["scene_id"] for row in observations.get("sar", [])}
    known_acquisitions = {_acquisition_key(scene_id) for scene_id in known}

    items = {
        "collections": ["sentinel-1-rtc"],
        "bbox": bbox,
        "datetime": f"{start}/{end}",
        "limit": 100,
    }
    features = _search(items, orbit_state)
    if relative_orbit is not None:
        features = [feature for feature in features
                    if int(feature["properties"].get("sat:relative_orbit", -1)) == relative_orbit]
    if limit:
        features = features[:limit]
    if not features:
        raise RuntimeError("No Sentinel-1 RTC scenes match this query")
    print(f"Scenes: {len(features)} (existing cached rows: {len(known)})", flush=True)

    token = _sas_token()
    token_time = time.time()

    for index, feature in enumerate(features, 1):
        props = feature["properties"]
        scene_id = feature["id"]
        acquisition = _acquisition_key(scene_id)
        if scene_id in known or acquisition in known_acquisitions:
            print(f"[{index}/{len(features)}] {scene_id}: cached/duplicate acquisition", flush=True)
            continue
        if time.time() - token_time > 45 * 60:
            token = _sas_token()
            token_time = time.time()
        tags = {}
        try:
            vv_mean = _read_power(feature["assets"]["vv"]["href"], token, bbox, tags)
            vh_mean = _read_power(feature["assets"]["vh"]["href"], token, bbox, tags)
            near = tags.get("IncidenceNearAngle")
            far = tags.get("IncidenceFarAngle")
            if near is None or far is None:
                raise ValueError("Missing incidence angle tags")
            incidence = (float(near) + float(far)) / 2.0
            relative = int(props["sat:relative_orbit"])
        except (KeyError, TypeError, ValueError, rasterio.errors.RasterioIOError, requests.RequestException) as error:
            print(f"[{index}/{len(features)}] {scene_id}: skipped ({type(error).__name__}: {error})", flush=True)
            continue
        observations["sar"].append({
            "date": props["datetime"],
            "scene_id": scene_id,
            "vv_db": float(10 * np.log10(vv_mean)),
            "vh_db": float(10 * np.log10(vh_mean)),
            "orbit_pass": orbit_state,
            "relative_orbit": relative,
            "incidence_angle_deg": incidence,
            "source": "COPERNICUS/S1_GRD RTC (Microsoft Planetary Computer)",
        })
        observations["sar"].sort(key=lambda row: (row["date"], row["scene_id"]))
        observations["relative_orbit"] = relative
        save_space_cache(
            query, observations, provider=PROVIDER,
            sources={"sar": "COPERNICUS/S1_GRD RTC (Microsoft Planetary Computer)", "optical": "none"},
        )
        known.add(scene_id)
        known_acquisitions.add(acquisition)
        print(
            f"[{index}/{len(features)}] {scene_id}: VV={observations['sar'][-1]['vv_db']:.2f} dB, "
            f"VH={observations['sar'][-1]['vh_db']:.2f} dB, cached={len(observations['sar'])}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="sko", choices=list(GRAIN_BELT_REGIONS))
    parser.add_argument("--district", default="kyzylzhar")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--orbit-state", default="DESCENDING", choices=["ASCENDING", "DESCENDING"])
    parser.add_argument("--relative-orbit", type=int, default=122)
    parser.add_argument("--limit", type=int, default=0, help="0 means all matching scenes")
    args = parser.parse_args()
    prefetch(args.region, args.district, args.year, args.orbit_state, args.relative_orbit, args.limit)


if __name__ == "__main__":
    main()
