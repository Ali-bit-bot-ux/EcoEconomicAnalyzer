"""Contracts for the region-aware 2026 GEE datasets used by the dashboard."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import GRAIN_BELT_REGIONS, PROCESSED_DIR

REGIONS = set(GRAIN_BELT_REGIONS)
FILES = {
    "ndvi_current": "ndvi_current_2026.parquet",
    "ndvi_baseline": "ndvi_baseline_regions.parquet",
    "norm": "phenological_norm_regions.parquet",
    "psi": "phenoshift_regions.parquet",
    "hydro": "hydroborder_regions.parquet",
    "vulnerability": "vuln_map_regions.parquet",
    "backtest": "backtest_2021_regions.parquet",
    "backtest_summary": "backtest_2021_summary.parquet",
}


class TestReal2026Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in FILES.values() if not (PROCESSED_DIR / name).exists()]
        if missing:
            raise unittest.SkipTest("2026 regional datasets not generated yet: " + ", ".join(missing))
        cls.current = pd.read_parquet(PROCESSED_DIR / FILES["ndvi_current"])
        cls.baseline = pd.read_parquet(PROCESSED_DIR / FILES["ndvi_baseline"])
        cls.norm = pd.read_parquet(PROCESSED_DIR / FILES["norm"])
        cls.psi = pd.read_parquet(PROCESSED_DIR / FILES["psi"])
        cls.hydro = pd.read_parquet(PROCESSED_DIR / FILES["hydro"])
        cls.vulnerability = pd.read_parquet(PROCESSED_DIR / FILES["vulnerability"])
        cls.backtest = pd.read_parquet(PROCESSED_DIR / FILES["backtest"])
        cls.backtest_summary = pd.read_parquet(PROCESSED_DIR / FILES["backtest_summary"])

    def test_schemas_and_four_regions(self):
        expected = {
            "ndvi_current": {"region_id", "date", "doy", "year", "ndvi"},
            "ndvi_baseline": {"region_id", "date", "doy", "year", "ndvi"},
            "norm": {"region_id", "doy", "ndvi_mean", "ndvi_std", "ndvi_upper", "ndvi_lower"},
            "psi": {"region_id", "year", "psi", "risk", "delay_days"},
            "hydro": {"region_id", "basin", "date", "soil_moisture", "smai", "anomaly_level"},
        }
        frames = {
            "ndvi_current": self.current, "ndvi_baseline": self.baseline,
            "norm": self.norm, "psi": self.psi, "hydro": self.hydro,
        }
        for name, columns in expected.items():
            with self.subTest(dataset=name):
                self.assertTrue(columns.issubset(frames[name].columns))
                self.assertEqual(set(frames[name]["region_id"]), REGIONS)
                self.assertFalse(frames[name].empty)

    def test_current_ndvi_is_2026_and_physical(self):
        self.assertEqual(set(self.current["year"]), {2026})
        self.assertTrue(self.current["ndvi"].between(-0.1, 1.0).all())
        self.assertTrue(self.current["date"].notna().all())
        self.assertEqual(self.current.groupby("region_id")["date"].size().min() > 0, True)

    def test_baseline_has_no_future_years_and_norm_is_complete(self):
        self.assertTrue((self.baseline["year"] <= 2025).all())
        self.assertTrue(self.baseline["ndvi"].between(-0.1, 1.0).all())
        self.assertTrue(self.norm["ndvi_std"].gt(0).all())
        self.assertTrue((self.norm["ndvi_upper"] > self.norm["ndvi_lower"]).all())
        self.assertEqual(self.norm.groupby("region_id").size().nunique(), 1)

    def test_psi_has_baseline_plus_current_and_no_synthetic_source(self):
        self.assertIn(2026, set(self.psi["year"]))
        self.assertTrue(self.psi["psi"].dropna().between(-10, 10).all())
        self.assertIn("source", self.current.columns)
        self.assertEqual(set(self.current["source"]), {"COPERNICUS/S2_SR_HARMONIZED"})
        self.assertIn("source", self.hydro.columns)
        self.assertEqual(set(self.hydro["source"]), {"NASA/SMAP/SPL3SMP_E/006"})

    def test_backtest_uses_real_2021_series_and_computed_triggers(self):
        self.assertEqual(set(self.backtest["region_id"]), REGIONS)
        self.assertTrue(self.backtest["date"].between("2021-05-01", "2021-09-30").all())
        self.assertTrue(np.isfinite(self.backtest["smai"]).all())
        self.assertTrue({"trigger_date", "official_date", "lead_days", "threshold",
                         "min_smai", "baseline"}.issubset(self.backtest_summary.columns))
        self.assertEqual(set(self.backtest_summary["region_id"]), REGIONS)
        self.assertTrue(self.backtest_summary["baseline"].str.startswith("2015-2020").all())
        self.assertTrue((self.backtest_summary["research_target_date"] == "2021-06-18").all())
        self.assertTrue((self.backtest_summary["research_target_smai"] == -1.68).all())

    def test_vulnerability_uses_real_district_registry_not_synthetic_grid(self):
        self.assertEqual(len(self.vulnerability), 20)
        self.assertEqual(set(self.vulnerability["region_id"]), REGIONS)
        self.assertEqual(set(self.vulnerability["source"]), {"district_registry"})
        self.assertTrue(self.vulnerability["vuln_score"].between(0, 100).all())
        self.assertTrue(self.vulnerability["nearest_elevator_km"].ge(0).all())
        self.assertNotIn("synthetic_grid", set(self.vulnerability["source"]))

    def test_hydro_smai_is_physical_and_basins_are_complete(self):
        self.assertTrue(self.hydro["soil_moisture"].between(0, 1).all())
        self.assertTrue(self.hydro["smai"].notna().all())
        self.assertEqual(set(self.hydro["basin"]), {"ishim", "tobol", "nura", "irtysh"})
        self.assertTrue(2026 in set(self.hydro["year"]))
        self.assertFalse(self.hydro.duplicated(["region_id", "date"]).any())


if __name__ == "__main__":
    unittest.main()
