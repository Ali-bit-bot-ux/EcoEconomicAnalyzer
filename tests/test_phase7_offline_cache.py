"""Offline-cache contracts; numeric fixtures are synthetic unit-test inputs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import space_cache
from modules.gis_engine import DISTRICT_REGISTRY
from modules.space_cache import (
    SpaceCacheError, archive_ndvi_climatology, archive_smap_series, build_query,
    load_noaa_kp, load_space_cache, offline_archive_status, save_noaa_kp, save_space_cache,
)

AOI = {"type": "Polygon", "coordinates": [DISTRICT_REGISTRY["sko"][0]["polygon"]]}
SAR = [{
    "date": "2026-06-01T10:00:00.000Z", "scene_id": "unit-test-sar-1",
    "vv_db": -12.0, "vh_db": -18.0, "orbit_pass": "DESCENDING",
    "relative_orbit": 42, "incidence_angle_deg": 35.0,
}]
OPTICAL = [{
    "date": "2026-07-01T10:00:00.000Z", "scene_id": "unit-test-optical-1", "ndvi": 0.52,
}]
OBSERVATIONS = {"sar": SAR, "optical": OPTICAL, "relative_orbit": 42, "scale_m": 100}


class TestPhase7OfflineCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache_patch = patch.object(space_cache, "CACHE_DIR", self.root / "space_cache")
        self.noaa_patch = patch.object(space_cache, "NOAA_CACHE", self.root / "noaa_kp.json")
        self.processed_patch = patch.object(space_cache, "PROCESSED_DIR", self.root)
        self.cache_patch.start()
        self.noaa_patch.start()
        self.processed_patch.start()
        self.query = build_query(
            region_id="sko", district_id="kyzylzhar", year=2026, orbit_pass="DESCENDING",
            start_date="2026-04-01", end_date="2026-10-01", geometry=AOI,
        )

    def tearDown(self):
        self.processed_patch.stop()
        self.noaa_patch.stop()
        self.cache_patch.stop()
        self.temp.cleanup()

    def test_cache_round_trip_is_atomic_and_query_bound(self):
        path = save_space_cache(self.query, OBSERVATIONS)
        self.assertTrue(path.exists())
        self.assertEqual(path.parent, self.root / "space_cache")
        payload = load_space_cache(self.query)
        self.assertEqual(payload["query"], self.query)
        self.assertEqual(payload["observations"], OBSERVATIONS)
        self.assertEqual(payload["provenance"]["provider"], "Google Earth Engine")
        self.assertEqual(list(path.parent.glob(".space-cache-*")), [])
        save_space_cache(self.query, OBSERVATIONS, provider="Unit-test STAC provider",
                         sources={"sar": "unit-test-sar-source", "optical": "unit-test-optical-source"})
        provenance = load_space_cache(self.query)["provenance"]
        self.assertEqual(provenance["provider"], "Unit-test STAC provider")
        self.assertEqual(provenance["sar"], "unit-test-sar-source")

    def test_cache_mismatch_and_corruption_are_not_silently_reused(self):
        path = save_space_cache(self.query, OBSERVATIONS)
        other = build_query(
            region_id="sko", district_id="kyzylzhar", year=2026, orbit_pass="ASCENDING",
            start_date="2026-04-01", end_date="2026-10-01", geometry=AOI,
        )
        self.assertIsNone(load_space_cache(other))
        path.write_text("{not-json", encoding="utf-8")
        with self.assertRaises(SpaceCacheError):
            load_space_cache(self.query)
        path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
        with self.assertRaises(SpaceCacheError):
            load_space_cache(self.query)

    def test_cache_rejects_invalid_queries_and_observations(self):
        for kwargs in (
            {"orbit_pass": "SIDEWAYS"},
            {"year": 1999},
            {"start_date": "2026-10-01", "end_date": "2026-04-01"},
        ):
            params = {
                "region_id": "sko", "district_id": "kyzylzhar", "year": 2026,
                "orbit_pass": "DESCENDING", "start_date": "2026-04-01",
                "end_date": "2026-10-01", "geometry": AOI,
            }
            params.update(kwargs)
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_query(**params)
        with self.assertRaises(ValueError):
            save_space_cache(self.query, {"sar": []})
        with self.assertRaises(ValueError):
            save_space_cache(self.query, {"sar": [], "optical": [], "bad": float("nan")})

    def test_noaa_cache_round_trip_and_corruption_rejection(self):
        observation = {
            "kp": 1.8, "observed_at": "2026-11-12T00:00:00+00:00",
            "source": "synthetic unit-test NOAA fixture", "stale": False,
            "product": "unit-test 3-hour Kp",
        }
        save_noaa_kp(observation)
        self.assertEqual(load_noaa_kp(), observation)
        for bad in (
            {**observation, "kp": 10},
            {**observation, "observed_at": "2026-11-12T00:00:00"},
            {**observation, "stale": "False"},
            {**observation, "source": ""},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    save_noaa_kp(bad)
        (self.root / "noaa_kp.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(SpaceCacheError):
            load_noaa_kp()

    def test_archive_readers_label_exact_coverage_and_do_not_mutate_sources(self):
        pd.DataFrame({"doy": [120, 180], "ndvi_mean": [0.2, 0.6]}).to_parquet(
            self.root / "phenological_norm.parquet", index=False
        )
        pd.DataFrame({
            "date": pd.to_datetime(["2025-07-01", "2025-08-01"]),
            "soil_moisture": [0.18, 0.22], "basin": ["tobol", "tobol"],
        }).to_parquet(self.root / "smap_timeseries.parquet", index=False)
        ndvi = archive_ndvi_climatology("sko")
        self.assertEqual(ndvi["frame"]["ndvi_mean"].tolist(), [0.2, 0.6])
        self.assertIn("2015–2023", ndvi["label_ru"])
        self.assertIsNone(archive_ndvi_climatology("kostanay"))
        smap = archive_smap_series("tobol")
        self.assertAlmostEqual(smap["latest"]["soil_moisture"], 0.22)
        self.assertIsNone(archive_smap_series("ishim"))
        self.assertEqual(len(offline_archive_status()), 2)

    def test_streamlit_loads_cache_without_network_and_keeps_values(self):
        from streamlit.testing.v1 import AppTest
        save_space_cache(self.query, OBSERVATIONS)
        app = AppTest.from_string(
            "from dashboard.space_segment_tab import render_space_segment_tab\n"
            "render_space_segment_tab(lang='ru', region_id='sko', crop_id='spring_wheat', year=2026, shock_percent=0)"
        ).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertIn("0.520", [metric.value for metric in app.metric])
        self.assertTrue(any("Оффлайн-кэш" in info.value for info in app.info))

    def test_streamlit_noaa_failure_uses_timestamped_local_copy(self):
        from streamlit.testing.v1 import AppTest
        from dashboard import space_segment_tab
        import requests
        save_noaa_kp({
            "kp": 1.8, "observed_at": "2026-11-12T00:00:00+00:00",
            "source": "synthetic unit-test NOAA fixture", "stale": False,
            "product": "unit-test 3-hour Kp",
        })
        with patch.object(space_segment_tab, "fetch_noaa_kp", side_effect=requests.RequestException("offline unit test")):
            app = AppTest.from_string(
                "from dashboard.space_segment_tab import render_space_segment_tab\n"
                "render_space_segment_tab(lang='ru', region_id='sko', crop_id='spring_wheat', year=2026, shock_percent=0)"
            ).run(timeout=20)
            app.button(key="space_weather_fetch").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("1.80", [metric.value for metric in app.metric])
        self.assertTrue(any("локальный кэш" in caption.value for caption in app.caption))

    def test_streamlit_fetch_failure_falls_back_to_cached_rows(self):
        from streamlit.testing.v1 import AppTest
        from dashboard import space_segment_tab
        save_space_cache(self.query, OBSERVATIONS)
        with patch.object(space_segment_tab, "_fetch_observations", side_effect=RuntimeError("offline unit test")):
            app = AppTest.from_string(
                "from dashboard.space_segment_tab import render_space_segment_tab\n"
                "render_space_segment_tab(lang='ru', region_id='sko', crop_id='spring_wheat', year=2026, shock_percent=0)"
            ).run(timeout=20)
            app.button(key="space_fetch").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("GEE недоступен" in warning.value for warning in app.warning))
        self.assertIn("0.520", [metric.value for metric in app.metric])

    def test_streamlit_successful_fetch_persists_the_offline_copy(self):
        from streamlit.testing.v1 import AppTest
        from dashboard import space_segment_tab
        with patch.object(space_segment_tab, "_fetch_observations", return_value=OBSERVATIONS):
            app = AppTest.from_string(
                "from dashboard.space_segment_tab import render_space_segment_tab\n"
                "render_space_segment_tab(lang='ru', region_id='sko', crop_id='spring_wheat', year=2026, shock_percent=0)"
            ).run(timeout=20)
            app.button(key="space_fetch").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("сохранены" in success.value for success in app.success))
        self.assertIsNotNone(load_space_cache(self.query))


if __name__ == "__main__":
    unittest.main()
