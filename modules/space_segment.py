import copy
import math
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from shapely.geometry import shape, mapping


NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SATELLITES = {
    "KazEOSat-1": {
        "orbit_altitude_km": 750, "max_off_nadir_deg": 35,
        "resolution_m": 1.0, "multispectral_resolution_m": 4.0, "swath_km": 20,
        "bands": ("PAN", "Blue", "Green", "Red", "NIR"),
        "source": "https://www.gharysh.kz/infrastruktura/AboutKA_DZZ22_jnot/",
    },
    "KazEOSat-2": {
        "orbit_altitude_km": 630, "max_off_nadir_deg": 35,
        "resolution_m": 6.5, "multispectral_resolution_m": 6.5, "swath_km": 77,
        "bands": ("Blue", "Green", "Red", "RedEdge", "NIR"),
        "source": "https://www.gharysh.kz/infrastruktura/AboutKA_DZZ22_jnot/",
    },
}
CHANNEL_LABELS = {
    "kk": {"ndvi": "🌿 Оптика NDVI", "vv_db": "📡 Радар: ылғалға сезімтал VV",
           "vh_db": "🌾 Радар: биомассаға сезімтал VH", "rvi": "⚡ Радарлық индекс RVI",
           "psi_radar": "📐 PSI_radar (VH/(VV+VH))"},
    "ru": {"ndvi": "🌿 Оптика NDVI", "vv_db": "📡 Радар: чувствительность к влаге VV",
           "vh_db": "🌾 Радар: чувствительность к биомассе VH", "rvi": "⚡ Радарный индекс RVI",
           "psi_radar": "📐 PSI_radar (VH/(VV+VH))"},
}
CHANNEL_RANGES = {"ndvi": (-1, 1), "vv_db": (-30, 0), "vh_db": (-35, -5), "rvi": (0, 4), "psi_radar": (0, 1)}
SAR_COLUMNS = ("date", "scene_id", "vv_db", "vh_db", "orbit_pass", "relative_orbit", "incidence_angle_deg")


def _bounded(value, low, high, name):
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name}: expected finite value in [{low}, {high}]")
    return number


def db_to_linear(db):
    values = np.asarray(db, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Backscatter must be finite")
    with np.errstate(over="ignore", under="ignore"):
        power = np.power(10.0, values / 10.0)
    if not np.isfinite(power).all() or np.any(power == 0):
        raise ValueError("Backscatter is outside floating-point power range")
    return power


def calculate_rvi(vv, vh, *, input_db=True):
    vv = db_to_linear(vv) if input_db else np.asarray(vv, dtype=float)
    vh = db_to_linear(vh) if input_db else np.asarray(vh, dtype=float)
    if vv.shape != vh.shape:
        raise ValueError("VV and VH must have identical shapes")
    if not np.isfinite(vv).all() or not np.isfinite(vh).all() or np.any(vv < 0) or np.any(vh < 0):
        raise ValueError("Linear backscatter must be finite and nonnegative")
    scale = np.maximum(vv, vh)
    if np.any(scale == 0):
        raise ValueError("VV + VH must be positive")
    # Scaling preserves the ratio without overflowing the power sum.
    vv, vh = vv / scale, vh / scale
    return 4.0 * vh / (vv + vh)


def calculate_radar_psi(vv, vh, *, input_db=True):
    """Plant Structure Index from the research work: VH/(VV+VH), 0..1."""
    vv = db_to_linear(vv) if input_db else np.asarray(vv, dtype=float)
    vh = db_to_linear(vh) if input_db else np.asarray(vh, dtype=float)
    if vv.shape != vh.shape:
        raise ValueError("VV and VH must have identical shapes")
    if not np.isfinite(vv).all() or not np.isfinite(vh).all() or np.any(vv < 0) or np.any(vh < 0):
        raise ValueError("Linear backscatter must be finite and nonnegative")
    denominator = vv + vh
    if np.any(denominator == 0):
        raise ValueError("VV + VH must be positive")
    return vh / denominator


def prepare_sar_observations(frame):
    missing = set(SAR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing SAR columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise", format="ISO8601")
    for column in ("vv_db", "vh_db", "relative_orbit", "incidence_angle_deg"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column]).all():
            raise ValueError(f"Nonfinite SAR column: {column}")
    if result["date"].isna().any() or result["scene_id"].isna().any() or result["scene_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing acquisition identity")
    if result["scene_id"].duplicated().any():
        raise ValueError("Duplicate SAR scene")
    if not result["orbit_pass"].isin(["ASCENDING", "DESCENDING"]).all():
        raise ValueError("Invalid orbit pass")
    orbit = result["relative_orbit"]
    if not (orbit.between(1, 175) & orbit.mod(1).eq(0)).all():
        raise ValueError("Invalid relative orbit")
    if not result["incidence_angle_deg"].between(0, 90, inclusive="neither").all():
        raise ValueError("Invalid incidence angle")
    result["rvi"] = calculate_rvi(result["vv_db"], result["vh_db"])
    result["psi_radar"] = calculate_radar_psi(result["vv_db"], result["vh_db"])
    result["vh_vv_ratio"] = db_to_linear(result["vh_db"] - result["vv_db"])
    return result.sort_values("date").reset_index(drop=True)


def prepare_optical_observations(frame):
    if not {"date", "scene_id", "ndvi"}.issubset(frame.columns):
        raise ValueError("Missing optical columns: date, scene_id, ndvi")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise", format="ISO8601")
    result["ndvi"] = pd.to_numeric(result["ndvi"], errors="raise")
    if result["date"].isna().any() or not result["ndvi"].between(-1, 1).all():
        raise ValueError("Invalid optical observation")
    if result["scene_id"].isna().any() or result["scene_id"].astype(str).str.strip().eq("").any() or result["scene_id"].duplicated().any():
        raise ValueError("Invalid optical scene identity")
    return result.sort_values("date").reset_index(drop=True)


def validate_polygon(geometry):
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
        raise ValueError("AOI must be a GeoJSON Polygon")
    rings = geometry.get("coordinates", [])
    if not rings:
        raise ValueError("Empty polygon")
    for ring in rings:
        if len(ring) < 4 or ring[0] != ring[-1]:
            raise ValueError("Polygon rings must be closed")
        for point in ring:
            if len(point) != 2:
                raise ValueError("Expected WGS84 longitude, latitude")
            _bounded(point[0], -180, 180, "longitude")
            _bounded(point[1], -90, 90, "latitude")
    polygon = shape(geometry)
    if not polygon.is_valid or polygon.is_empty or polygon.area == 0:
        raise ValueError("Invalid polygon topology")
    if polygon.bounds[2] - polygon.bounds[0] > 180:
        raise ValueError("Antimeridian AOIs require splitting")
    return polygon


def estimate_off_nadir_deg(ground_distance_km, altitude_km):
    distance = _bounded(ground_distance_km, 0, 20000, "ground distance")
    altitude = _bounded(altitude_km, 1, 40000, "altitude")
    radius = 6371.0
    angle = distance / radius
    if angle >= math.acos(radius / (radius + altitude)):
        raise ValueError("Target is beyond the geometric horizon")
    return math.degrees(math.atan2(radius * math.sin(angle), radius + altitude - radius * math.cos(angle)))


def generate_tasking_request(geometry, satellite, *, bands=None, smai=None,
                            anomaly_source=None, observed_at=None, context=None,
                            ground_distance_km=None, now=None):
    polygon = validate_polygon(geometry)
    if satellite not in SATELLITES:
        raise ValueError("Unknown satellite")
    spec = SATELLITES[satellite]
    bands = list(bands if bands is not None else (["PAN", "Red", "NIR"] if satellite == "KazEOSat-1" else ["Red", "RedEdge", "NIR"]))
    if not bands or len(set(bands)) != len(bands) or not set(bands).issubset(spec["bands"]):
        raise ValueError("Unsupported spectral bands")
    timestamp = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if timestamp.tzinfo is None or pd.isna(timestamp):
        raise ValueError("Creation time must be timezone-aware")
    trigger = {"type": "manual_planning", "smai": None}
    if smai is not None:
        smai = float(smai)
        if not math.isfinite(smai) or smai >= -1.5:
            raise ValueError("SMAI trigger requires a finite value < -1.5")
        if not isinstance(anomaly_source, str) or not anomaly_source.strip() or observed_at is None:
            raise ValueError("Anomaly source and observation time are required")
        observation_time = pd.Timestamp(observed_at)
        if pd.isna(observation_time) or observation_time.tzinfo is None or observation_time > timestamp:
            raise ValueError("Invalid anomaly observation time")
        trigger = {"type": "user_supplied_smai", "smai": smai, "source": anomaly_source.strip(),
                   "observed_at": observation_time.isoformat()}
    off_nadir = None if ground_distance_km is None else estimate_off_nadir_deg(ground_distance_km, spec["orbit_altitude_km"])
    if off_nadir is not None and off_nadir > spec["max_off_nadir_deg"]:
        raise ValueError("Estimated off-nadir exceeds the published 35 degree limit")
    return {
        "schema_version": "agricascade.tasking/1.0", "request_id": str(uuid4()),
        "created_at": timestamp.tz_convert("UTC").isoformat(),
        "status": "draft_not_submitted", "operator": "Қазақстан Ғарыш Сапары",
        "satellite": satellite, "aoi": copy.deepcopy(mapping(polygon)),
        "crs": "EPSG:4326", "centroid": [polygon.centroid.x, polygon.centroid.y],
        "sensor": {**spec, "bands": bands},
        "requested_resolution_m": 1.0 if "PAN" in bands else spec["multispectral_resolution_m"],
        "trigger": trigger, "analysis_context": copy.deepcopy(context or {}),
        "planning": {"estimated_off_nadir_deg": off_nadir, "ground_distance_km": ground_distance_km,
                     "roll_deg": None, "acquisition_window_utc": None, "coverage_percent": None,
                     "status": "operator_ephemeris_and_cloud_check_required"},
    }


def assess_space_weather(*, kp=None, tec_tecu=None, b_parallel_ut=None, frequency_hz=1.41e9):
    frequency = _bounded(frequency_hz, 1e6, 1e12, "frequency Hz")
    kp = None if kp is None else _bounded(kp, 0, 9, "Kp")
    tec = None if tec_tecu is None else _bounded(tec_tecu, 0, 1000, "slant TECU")
    field = None if b_parallel_ut is None else _bounded(b_parallel_ut, -100, 100, "B parallel microtesla")
    delay = None if tec is None else 40.3 * tec * 1e16 / frequency ** 2
    faraday = None if tec is None or field is None else 2.36e4 * tec * 1e16 * field * 1e-6 / frequency ** 2
    return {
        "kp": kp, "tec_tecu": tec, "frequency_hz": frequency,
        "geomagnetic_status": "unknown" if kp is None else "storm" if kp >= 5 else "unsettled" if kp >= 4 else "quiet",
        "group_delay_m": delay, "faraday_rotation_rad": faraday,
        "faraday_rotation_deg": None if faraday is None else math.degrees(faraday),
        "faraday_status": "tec_required" if tec is None else "geometry_required" if field is None else "estimated_not_corrected",
        "retrieval_validated": False,
    }


def parse_noaa_kp(payload, *, now=None):
    if not isinstance(payload, list) or not payload:
        raise ValueError("Empty NOAA Kp response")
    if isinstance(payload[0], list):
        header = payload[0]
        if not {"time_tag", "Kp"}.issubset(header):
            raise ValueError("Invalid NOAA Kp schema")
        rows = [dict(zip(header, row)) for row in payload[1:] if isinstance(row, list) and len(row) == len(header)]
    else:
        rows = payload
    records = []
    for row in rows:
        if not isinstance(row, dict) or not {"time_tag", "Kp"}.issubset(row):
            continue
        try:
            kp = _bounded(row["Kp"], 0, 9, "Kp")
            stamp = pd.to_datetime(row["time_tag"], utc=True, errors="raise")
            if pd.isna(stamp):
                continue
        except (TypeError, ValueError):
            continue
        records.append((stamp, kp))
    current = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if current.tzinfo is None:
        raise ValueError("Current time must be timezone-aware")
    records = [(stamp, kp) for stamp, kp in records if stamp <= current]
    if not records:
        raise ValueError("No valid NOAA Kp observations")
    stamp, kp = max(records, key=lambda record: record[0])
    return {"kp": kp, "observed_at": stamp.isoformat(), "source": NOAA_KP_URL,
            "stale": (current - stamp).total_seconds() > 6 * 3600,
            "product": "NOAA estimated planetary Kp, 3-hour interval"}


def fetch_noaa_kp():
    response = requests.get(NOAA_KP_URL, timeout=(5, 15))
    response.raise_for_status()
    return parse_noaa_kp(response.json())


def build_observation_map(geometry, channel, value=None, *, lang="kk"):
    polygon = validate_polygon(geometry)
    label = CHANNEL_LABELS[lang][channel]
    fig = go.Figure()
    if value is not None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Map value must be finite")
        low, high = CHANNEL_RANGES[channel]
        fig.add_trace(go.Choropleth(
            geojson={"type": "FeatureCollection", "features": [{"type": "Feature", "id": "aoi", "geometry": mapping(polygon), "properties": {}}]},
            locations=["aoi"], z=[value], zmin=low, zmax=high, colorscale="Viridis",
            marker_line_width=2, colorbar_title=channel,
            hovertemplate=f"{label}: %{{z:.3f}}<extra></extra>",
        ))
    else:
        x, y = polygon.exterior.xy
        fig.add_trace(go.Scattergeo(lon=list(x), lat=list(y), mode="lines", fill="toself",
                                   line={"color": "#8a9bb0", "width": 2}, name="AOI"))
    west, south, east, north = polygon.bounds
    fig.update_layout(geo={"projection_type": "mercator", "center": {"lat": polygon.centroid.y, "lon": polygon.centroid.x},
                           "lonaxis": {"range": [west - 0.2, east + 0.2], "showgrid": True},
                           "lataxis": {"range": [south - 0.2, north + 0.2], "showgrid": True},
                           "showland": True, "landcolor": "#162236", "showcountries": True},
                      template="plotly_dark", height=340, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig


def build_data_cube(*, lang="kk", root_depth_m=1.2, cloud_cover=70, channel="rvi", value=None):
    root = _bounded(root_depth_m, 0.1, 5, "root depth")
    cloud = _bounded(cloud_cover, 0, 100, "cloud cover")
    if channel not in CHANNEL_RANGES:
        raise ValueError("Unknown channel")
    kk = lang == "kk"
    labels = [
        f"{'Тамыр аймағы: модель' if kk else 'Корневая зона: модель'} · {root:g} м",
        "Топырақтың беткі қабаты · 0–5 см" if kk else "Поверхностная почва · 0–5 см",
        "Өсімдік жамылғысы · VH / RVI" if kk else "Растительный покров · VH / RVI",
        f"{'Бұлттылық сценарийі' if kk else 'Сценарий облачности'} · 2–5 км · {cloud:g}%",
        "Ионосфера · 300 км · TEC" if kk else "Ионосфера · 300 км · TEC",
        "Sentinel-1 · 693 км",
    ]
    colors = ["#9b704c", "#b79a64", "#2ca975", "#aab9cb", "#7068b5", "#39bde0"]
    fig = go.Figure()
    for z, (label, color) in enumerate(zip(labels, colors)):
        opacity = 0.12 + 0.7 * cloud / 100 if z == 3 else 0.45
        fig.add_trace(go.Mesh3d(x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[z] * 4,
                               i=[0, 0], j=[1, 2], k=[2, 3], color=color, opacity=opacity,
                               name=label, hovertemplate=label + "<extra></extra>", showlegend=True))
    optical = channel == "ndvi"
    end_z = 3 if optical and cloud > 0 else 2 if optical else 1
    beam_name = ("Оптикалық жол" if kk else "Оптический путь") if optical else "C-band SAR"
    fig.add_trace(go.Scatter3d(x=[0.5, 0.5], y=[0.5, 0.5], z=[5, end_z], mode="lines+markers",
                              line={"color": "#58d7ff", "width": 7}, name=beam_name))
    fig.add_trace(go.Scatter3d(x=[0.5, 0.5], y=[0.5, 0.5], z=[1, 0], mode="lines",
                              line={"color": "#ffbd59", "width": 5, "dash": "dash"},
                              name="Су балансы моделі; тікелей өлшем емес" if kk else "Модель водного баланса; не прямой замер"))
    title = CHANNEL_LABELS[lang][channel]
    if value is not None:
        if not math.isfinite(float(value)):
            raise ValueError("Cube value must be finite")
        title += f" · {float(value):.3f}"
    fig.update_layout(title=title, height=560, template="plotly_dark",
                      scene={"xaxis": {"visible": False}, "yaxis": {"visible": False},
                             "zaxis": {"tickvals": list(range(6)), "ticktext": ["−1.2 м" if root == 1.2 else f"−{root:g} м", "−0.05 м", "0 м", "3 км", "300 км", "693 км"],
                                       "title": "Шартты масштаб" if kk else "Условный масштаб"},
                             "aspectratio": {"x": 1, "y": 1, "z": 1.6}},
                      legend={"orientation": "h", "y": -0.05}, margin={"l": 0, "r": 0, "t": 45, "b": 0})
    return fig
