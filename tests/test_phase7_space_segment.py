"""Offline contracts with synthetic test inputs, never production satellite observations."""

import copy
import json
import math
import sys
import unittest
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.space_segment import (
    CHANNEL_LABELS,
    SATELLITES,
    assess_space_weather,
    build_data_cube,
    build_observation_map,
    calculate_radar_psi,
    calculate_rvi,
    db_to_linear,
    estimate_off_nadir_deg,
    generate_tasking_request,
    parse_noaa_kp,
    prepare_optical_observations,
    prepare_sar_observations,
    validate_polygon,
)


class TestPhase7SpaceSegment(unittest.TestCase):
    """Independent mathematics, validation, provenance, and Plotly contracts."""

    def setUp(self):
        self.now = pd.Timestamp("2026-11-11T12:00:00Z")
        # Deliberately simple WGS84 geometry, not a surveyed field boundary.
        self.aoi = {
            "type": "Polygon",
            "coordinates": [[[60, 50], [64, 50], [64, 54], [60, 54], [60, 50]]],
        }

    def _sar_frame(self):
        # Paired VV/VH values belong to the same synthetic scene in each row.
        return pd.DataFrame({
            "date": ["2026-11-10T12:00:00Z", "2026-11-09T12:00:00Z"],
            "scene_id": ["unit-test-sar-later", "unit-test-sar-earlier"],
            "vv_db": [-10.0, -20.0],
            "vh_db": [-20.0, -10.0],
            "orbit_pass": ["ASCENDING", "DESCENDING"],
            "relative_orbit": [1, 175],
            "incidence_angle_deg": [30.0, 45.0],
            "input_kind": ["synthetic unit-test input"] * 2,
        })

    def _optical_frame(self):
        return pd.DataFrame({
            "date": ["2026-11-10T12:00:00Z", "2026-11-09T12:00:00Z"],
            "scene_id": ["unit-test-optical-later", "unit-test-optical-earlier"],
            "ndvi": [-1.0, 1.0],
        })

    def _request(self, satellite="KazEOSat-1", **kwargs):
        return generate_tasking_request(self.aoi, satellite, now=self.now, **kwargs)

    def _noaa_payload(self, *rows):
        # Wire-format parser fixture only; not a fetched NOAA observation.
        return [["time_tag", "Kp"], *[list(row) for row in rows]]

    def _assert_serializable(self, figure):
        payload = json.loads(figure.to_json())
        self.assertIsInstance(payload["data"], list)
        self.assertIsInstance(payload["layout"], dict)
        self.assertTrue(figure.to_html(full_html=False, include_plotlyjs=False))
        return payload

    def _cube_plane(self, figure, level):
        matches = [trace for trace in figure.data
                   if trace.type == "mesh3d" and set(trace.z) == {level}]
        self.assertEqual(len(matches), 1)
        return matches[0]

    # RVI = 4 * VH / (VV + VH), with POWER (10**(dB/10)), not amplitude.
    def test_db_conversion_has_independent_power_reference(self):
        result = db_to_linear([-20, -10, 0, 10])
        np.testing.assert_allclose(result, [0.01, 0.1, 1, 10], rtol=1e-14)
        self.assertAlmostEqual(float(db_to_linear(-10)), 0.1)

    def test_rvi_db_pair_matches_hand_computed_four_elevenths(self):
        # 4 * 0.01 / (0.1 + 0.01) = 4/11.
        self.assertAlmostEqual(float(calculate_rvi(-10, -20)), 4 / 11, places=14)
        self.assertAlmostEqual(
            float(calculate_rvi(0.1, 0.01, input_db=False)), 4 / 11, places=14
        )

    def test_rvi_true_range_is_zero_to_four_without_clipping_at_one(self):
        vv = np.array([1.0, 3.0, 1.0, 1.0, 0.0])
        vh = np.array([0.0, 1.0, 1.0, 3.0, 1.0])
        result = calculate_rvi(vv, vh, input_db=False)
        np.testing.assert_allclose(result, [0, 1, 2, 3, 4])
        self.assertTrue(np.all((result >= 0) & (result <= 4)))
        self.assertAlmostEqual(float(calculate_rvi(-20, -10)), 40 / 11)
        self.assertEqual(float(calculate_rvi(0, 0)), 2.0)  # 0 dB is unit power.

    def test_radar_psi_matches_research_formula_and_stays_in_unit_interval(self):
        self.assertAlmostEqual(float(calculate_radar_psi(-10, -20)), 0.01 / 0.11, places=14)
        result = calculate_radar_psi(np.array([1.0, 3.0, 0.0]), np.array([1.0, 1.0, 1.0]), input_db=False)
        np.testing.assert_allclose(result, [0.5, 0.25, 1.0])
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        with self.assertRaises(ValueError):
            calculate_radar_psi(0, 0, input_db=False)

    def test_rvi_preserves_multidimensional_shape_and_values(self):
        vv = np.array([[1, 3], [0, 1]], dtype=float)
        vh = np.array([[1, 1], [2, 0]], dtype=float)
        result = calculate_rvi(vv, vh, input_db=False)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, [[2, 1], [4, 0]])

    def test_rvi_rejects_shape_mismatch_instead_of_broadcasting(self):
        for vv, vh in [(1, [1]), ([1, 2], [1]), ([[1], [2]], [[1, 2]])]:
            for input_db in (False, True):
                with self.subTest(vv=vv, vh=vh, input_db=input_db):
                    with self.assertRaises(ValueError):
                        calculate_rvi(vv, vh, input_db=input_db)

    def test_rvi_zero_denominator_and_negative_linear_power_are_rejected(self):
        for vv, vh in [(0, 0), (-1, 2), (2, -1), ([1, 0], [1, 0])]:
            with self.subTest(vv=vv, vh=vh):
                with self.assertRaises(ValueError):
                    calculate_rvi(vv, vh, input_db=False)

    def test_backscatter_nonfinite_values_are_rejected_in_either_channel(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(conversion=bad):
                with self.assertRaises(ValueError):
                    db_to_linear([0, bad])
            for input_db in (False, True):
                for vv, vh in [([1, bad], [1, 1]), ([1, 1], [1, bad])]:
                    with self.subTest(bad=bad, input_db=input_db, vv=vv, vh=vh):
                        with self.assertRaises(ValueError):
                            calculate_rvi(vv, vh, input_db=input_db)

    def test_empty_backscatter_arrays_remain_empty(self):
        for shape in [(0,), (2, 0)]:
            empty = np.empty(shape)
            self.assertEqual(db_to_linear(empty).shape, shape)
            for input_db in (False, True):
                result = calculate_rvi(empty, empty, input_db=input_db)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.size, 0)

    def test_extreme_linear_power_avoids_overflow_and_preserves_ratio(self):
        largest = np.finfo(float).max
        smallest = np.nextafter(0.0, 1.0)
        vv = [largest, largest, smallest, largest, smallest]
        vh = [largest, largest / 2, smallest, smallest, largest]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = calculate_rvi(vv, vh, input_db=False)
        np.testing.assert_allclose(result, [2, 4 / 3, 2, 0, 4], atol=1e-14)
        self.assertTrue(np.isfinite(result).all())

    def test_extreme_db_power_is_either_representable_or_explicitly_rejected(self):
        result = calculate_rvi([3000, -3000], [3000, -3000])
        np.testing.assert_allclose(result, [2, 2])
        for bad in (4000, -4000, np.finfo(float).max, -np.finfo(float).max):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    db_to_linear([0, bad])
                with self.assertRaises(ValueError):
                    calculate_rvi([0, bad], [0, 0])

    def test_sar_preparation_sorts_without_unpairing_scenes_or_mutating_input(self):
        frame = self._sar_frame()
        original = frame.copy(deep=True)
        prepared = prepare_sar_observations(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(str(prepared["date"].dt.tz), "UTC")
        self.assertTrue(prepared["date"].is_monotonic_increasing)
        self.assertEqual(prepared.index.tolist(), [0, 1])
        self.assertEqual(prepared["scene_id"].tolist(),
                         ["unit-test-sar-earlier", "unit-test-sar-later"])
        np.testing.assert_allclose(prepared["rvi"], [40 / 11, 4 / 11])
        np.testing.assert_allclose(prepared["vh_vv_ratio"], [10, 0.1])
        self.assertEqual(prepared["orbit_pass"].tolist(), ["DESCENDING", "ASCENDING"])
        self.assertEqual(prepared["relative_orbit"].tolist(), [175, 1])
        self.assertEqual(prepared["incidence_angle_deg"].tolist(), [45, 30])
        self.assertEqual(prepared["input_kind"].tolist(),
                         ["synthetic unit-test input"] * 2)

    def test_sar_requires_all_paired_measurement_and_metadata_columns(self):
        for column in ("date", "scene_id", "vv_db", "vh_db", "orbit_pass",
                       "relative_orbit", "incidence_angle_deg"):
            with self.subTest(missing=column):
                with self.assertRaises(ValueError):
                    prepare_sar_observations(self._sar_frame().drop(columns=column))

    def test_sar_rejects_missing_or_invalid_acquisition_dates(self):
        for bad in (None, pd.NaT, "", "not-a-date", "2026-02-30T00:00:00Z"):
            frame = self._sar_frame()
            frame.loc[0, "date"] = bad
            with self.subTest(date=bad):
                with self.assertRaises(ValueError):
                    prepare_sar_observations(frame)

    def test_sar_rejects_duplicate_and_blank_scene_identities(self):
        for bad in (None, "", "  ", "unit-test-sar-earlier"):
            frame = self._sar_frame()
            frame.loc[0, "scene_id"] = bad
            with self.subTest(scene_id=bad):
                with self.assertRaises(ValueError):
                    prepare_sar_observations(frame)

    def test_distinct_sar_scenes_at_same_time_are_not_false_duplicates(self):
        frame = self._sar_frame()
        frame.loc[0, "date"] = frame.loc[1, "date"]
        prepared = prepare_sar_observations(frame)
        self.assertEqual(len(prepared), 2)
        indexed = prepared.set_index("scene_id")
        self.assertAlmostEqual(indexed.loc["unit-test-sar-later", "rvi"], 4 / 11)
        self.assertAlmostEqual(indexed.loc["unit-test-sar-earlier", "rvi"], 40 / 11)

    def test_sar_validates_orbit_pass_relative_orbit_and_incidence(self):
        invalid = {
            "orbit_pass": [None, "", "ascending", "SIDEWAYS"],
            "relative_orbit": [0, -1, 176, 1.5, np.nan, np.inf],
            "incidence_angle_deg": [0, 90, -1, 91, np.nan, np.inf],
        }
        for column, values in invalid.items():
            for bad in values:
                frame = self._sar_frame().astype({column: object})
                frame.loc[0, column] = bad
                with self.subTest(column=column, value=bad):
                    with self.assertRaises(ValueError):
                        prepare_sar_observations(frame)
        frame = self._sar_frame()
        frame["incidence_angle_deg"] = [0.001, 89.999]
        self.assertEqual(len(prepare_sar_observations(frame)), 2)

    def test_sar_rejects_nonfinite_or_nonnumeric_power_columns(self):
        for column in ("vv_db", "vh_db"):
            for bad in (None, np.nan, np.inf, -np.inf, "not-a-number"):
                frame = self._sar_frame().astype({column: object})
                frame.loc[0, column] = bad
                with self.subTest(column=column, value=bad):
                    with self.assertRaises(ValueError):
                        prepare_sar_observations(frame)

    def test_empty_sar_and_optical_frames_have_no_fabricated_observations(self):
        sar = prepare_sar_observations(self._sar_frame().iloc[:0])
        optical = prepare_optical_observations(self._optical_frame().iloc[:0])
        self.assertTrue(sar.empty)
        self.assertTrue(optical.empty)
        self.assertTrue({"rvi", "vh_vv_ratio"}.issubset(sar.columns))

    def test_optical_preparation_preserves_ndvi_endpoints_and_input(self):
        frame = self._optical_frame()
        original = frame.copy(deep=True)
        prepared = prepare_optical_observations(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(prepared["ndvi"].tolist(), [1, -1])
        self.assertEqual(str(prepared["date"].dt.tz), "UTC")
        self.assertTrue(prepared["date"].is_monotonic_increasing)
        self.assertEqual(prepared["scene_id"].tolist(),
                         ["unit-test-optical-earlier", "unit-test-optical-later"])

    def test_optical_rejects_ndvi_outside_minus_one_to_one_or_nonfinite(self):
        for bad in (-1.0001, 1.0001, np.nan, np.inf, -np.inf, None, "bad"):
            frame = self._optical_frame().astype({"ndvi": object})
            frame.loc[0, "ndvi"] = bad
            with self.subTest(ndvi=bad):
                with self.assertRaises(ValueError):
                    prepare_optical_observations(frame)

    def test_optical_requires_columns_dates_and_unique_scene_ids(self):
        for column in ("date", "scene_id", "ndvi"):
            with self.subTest(missing=column):
                with self.assertRaises(ValueError):
                    prepare_optical_observations(self._optical_frame().drop(columns=column))
        for column, values in {
            "date": [None, "", "bad", "2026-02-30T00:00:00Z"],
            "scene_id": [None, "", "  ", "unit-test-optical-earlier"],
        }.items():
            for bad in values:
                frame = self._optical_frame()
                frame.loc[0, column] = bad
                with self.subTest(column=column, value=bad):
                    with self.assertRaises(ValueError):
                        prepare_optical_observations(frame)

    def test_polygon_wgs84_bounds_centroid_and_hole_are_preserved(self):
        geometry = copy.deepcopy(self.aoi)
        geometry["coordinates"].append([[61, 51], [61, 53], [63, 53], [63, 51], [61, 51]])
        original = copy.deepcopy(geometry)
        polygon = validate_polygon(geometry)
        self.assertEqual(polygon.bounds, (60, 50, 64, 54))
        self.assertAlmostEqual(polygon.area, 12)  # 4*4 minus a 2*2 hole; square degrees.
        self.assertAlmostEqual(polygon.centroid.x, 62)
        self.assertAlmostEqual(polygon.centroid.y, 52)
        self.assertEqual(len(polygon.interiors), 1)
        request = generate_tasking_request(geometry, "KazEOSat-2", now=self.now)
        self.assertEqual(request["crs"], "EPSG:4326")
        self.assertEqual(request["centroid"], [62, 52])
        self.assertEqual(json.loads(json.dumps(request["aoi"])), original)
        self.assertEqual(geometry, original)

    def test_polygon_rejects_non_polygon_empty_open_and_degenerate_aoi(self):
        invalid = [
            None, [], {}, {"type": "Polygon"},
            {"type": "Polygon", "coordinates": []},
            {"type": "Point", "coordinates": [62, 52]},
            {"type": "Feature", "geometry": self.aoi},
            {"type": "MultiPolygon", "coordinates": [self.aoi["coordinates"]]},
            {"type": "Polygon", "coordinates": [[[60, 50], [64, 50], [64, 54], [60, 54]]]},
            {"type": "Polygon", "coordinates": [[[60, 50], [61, 50], [60, 50]]]},
            {"type": "Polygon", "coordinates": [[[60, 50], [61, 50], [62, 50], [60, 50]]]},
        ]
        for geometry in invalid:
            with self.subTest(geometry=geometry):
                with self.assertRaises(ValueError):
                    validate_polygon(geometry)

    def test_polygon_rejects_self_intersections_and_invalid_holes(self):
        bow_tie = {"type": "Polygon", "coordinates": [
            [[60, 50], [64, 54], [60, 54], [64, 50], [60, 50]]
        ]}
        outside_hole = copy.deepcopy(self.aoi)
        outside_hole["coordinates"].append(
            [[65, 51], [66, 51], [66, 52], [65, 52], [65, 51]]
        )
        open_hole = copy.deepcopy(self.aoi)
        open_hole["coordinates"].append([[61, 51], [62, 51], [62, 52], [61, 52]])
        crossing_hole = copy.deepcopy(self.aoi)
        crossing_hole["coordinates"].append(
            [[61, 51], [63, 53], [61, 53], [63, 51], [61, 51]]
        )
        for geometry in (bow_tie, outside_hole, open_hole, crossing_hole):
            with self.subTest(geometry=geometry):
                with self.assertRaises(ValueError):
                    validate_polygon(geometry)

    def test_polygon_rejects_invalid_coordinate_dimensions_and_wgs84_bounds(self):
        invalid_points = [
            [181, 51], [-181, 51], [61, 91], [61, -91],
            [np.nan, 51], [61, np.inf], [-np.inf, 51], [61], [61, 51, 100],
        ]
        for point in invalid_points:
            geometry = copy.deepcopy(self.aoi)
            geometry["coordinates"][0][1] = point
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    validate_polygon(geometry)
        for lon, lat in [(179, 89), (-180, -90)]:
            geometry = {"type": "Polygon", "coordinates": [[
                [lon, lat], [lon + 1, lat], [lon + 1, lat + 1],
                [lon, lat + 1], [lon, lat],
            ]]}
            self.assertTrue(validate_polygon(geometry).is_valid)

    def test_polygon_rejects_unsplit_antimeridian_aoi(self):
        geometry = {"type": "Polygon", "coordinates": [[
            [-179, 50], [179, 50], [179, 54], [-179, 54], [-179, 50]
        ]]}
        with self.assertRaises(ValueError):
            validate_polygon(geometry)

    def test_tasking_both_satellites_publish_correct_bands_and_resolutions(self):
        expected = {
            "KazEOSat-1": {
                "bands": {"PAN", "Blue", "Green", "Red", "NIR"},
                "swath": 20, "pan": 1.0, "ms": 4.0,
                "default": ["PAN", "Red", "NIR"],
            },
            "KazEOSat-2": {
                "bands": {"Blue", "Green", "Red", "RedEdge", "NIR"},
                "swath": 77, "pan": 6.5, "ms": 6.5,
                "default": ["Red", "RedEdge", "NIR"],
            },
        }
        for satellite, values in expected.items():
            with self.subTest(satellite=satellite):
                spec = SATELLITES[satellite]
                self.assertEqual(set(spec["bands"]), values["bands"])
                self.assertEqual(spec["resolution_m"], values["pan"])
                self.assertEqual(spec["multispectral_resolution_m"], values["ms"])
                request = self._request(satellite)
                self.assertEqual(request["satellite"], satellite)
                self.assertEqual(request["sensor"]["bands"], values["default"])
                self.assertEqual(request["sensor"]["orbit_altitude_km"], spec["orbit_altitude_km"])
                self.assertEqual(request["sensor"]["swath_km"], values["swath"])
                self.assertEqual(request["requested_resolution_m"], values["pan"])
                for band in sorted(values["bands"]):
                    single = self._request(satellite, bands=[band])
                    self.assertEqual(single["sensor"]["bands"], [band])
                    self.assertEqual(single["requested_resolution_m"],
                                     1.0 if band == "PAN" else values["ms"])

    def test_tasking_ke1_multispectral_is_four_meters_not_pan_resolution(self):
        request = self._request(bands=["Blue", "Green", "Red", "NIR"])
        self.assertEqual(request["requested_resolution_m"], 4.0)
        self.assertEqual(request["sensor"]["resolution_m"], 1.0)
        self.assertEqual(request["sensor"]["multispectral_resolution_m"], 4.0)

    def test_tasking_uses_operator_altitudes_and_rejects_unreachable_pointing(self):
        for satellite, altitude in (("KazEOSat-1", 750), ("KazEOSat-2", 630)):
            with self.subTest(satellite=satellite):
                self.assertEqual(SATELLITES[satellite]["orbit_altitude_km"], altitude)
                self.assertEqual(SATELLITES[satellite]["max_off_nadir_deg"], 35)
                with self.assertRaises(ValueError):
                    self._request(satellite, ground_distance_km=1000)

    def test_tasking_rejects_unknown_satellite_and_unsupported_duplicate_empty_bands(self):
        with self.assertRaises(ValueError):
            self._request("unit-test-unknown-satellite")
        for satellite, bands in [
            ("KazEOSat-1", ["RedEdge"]), ("KazEOSat-2", ["PAN"]),
            ("KazEOSat-1", ["SWIR"]), ("KazEOSat-2", ["NIR", "NIR"]),
            ("KazEOSat-1", []), ("KazEOSat-2", []),
        ]:
            with self.subTest(satellite=satellite, bands=bands):
                with self.assertRaises(ValueError):
                    self._request(satellite, bands=bands)

    def test_tasking_is_local_draft_without_invented_pass_roll_coverage_or_quality(self):
        for satellite in ("KazEOSat-1", "KazEOSat-2"):
            with self.subTest(satellite=satellite):
                request = self._request(satellite)
                self.assertEqual(request["schema_version"], "agricascade.tasking/1.0")
                self.assertEqual(request["status"], "draft_not_submitted")
                self.assertEqual(request["trigger"], {"type": "manual_planning", "smai": None})
                self.assertEqual(request["created_at"], "2026-11-11T12:00:00+00:00")
                self.assertEqual(UUID(request["request_id"]).version, 4)
                planning = request["planning"]
                self.assertEqual(planning["status"], "operator_ephemeris_and_cloud_check_required")
                for field in ("roll_deg", "acquisition_window_utc", "coverage_percent",
                              "estimated_off_nadir_deg", "ground_distance_km"):
                    self.assertIsNone(planning[field])
                encoded = json.dumps(request, allow_nan=False)
                for unsupported_claim in ("signal_quality", "quality_percent", "next_pass", "retrieval_validated"):
                    self.assertNotIn(unsupported_claim, encoded)
                self.assertNotEqual(request["request_id"], self._request(satellite)["request_id"])

    def test_tasking_requires_aware_creation_time_and_normalizes_to_utc(self):
        request = generate_tasking_request(
            self.aoi, "KazEOSat-1", now="2026-11-11T17:00:00+05:00"
        )
        self.assertEqual(request["created_at"], "2026-11-11T12:00:00+00:00")
        for bad in ("2026-11-11T12:00:00", "not-a-date", pd.NaT):
            with self.subTest(now=bad):
                with self.assertRaises(ValueError):
                    generate_tasking_request(self.aoi, "KazEOSat-1", now=bad)

    def test_smai_trigger_is_strictly_below_minus_one_point_five_with_provenance(self):
        request = self._request(
            smai=-1.500001, anomaly_source="  synthetic unit-test SMAI input  ",
            observed_at="2026-11-11T11:00:00Z",
        )
        self.assertEqual(request["trigger"], {
            "type": "user_supplied_smai", "smai": -1.500001,
            "source": "synthetic unit-test SMAI input", "observed_at": "2026-11-11T11:00:00+00:00",
        })
        self.assertEqual(request["status"], "draft_not_submitted")
        for bad in (-1.5, -1.499999, 0, 2, np.nan, np.inf, -np.inf):
            with self.subTest(smai=bad):
                with self.assertRaises(ValueError):
                    self._request(smai=bad, anomaly_source="unit-test input", observed_at=self.now)

    def test_smai_requires_nonblank_source_and_aware_nonfuture_observation_time(self):
        for source in (None, "", "   ", 42):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    self._request(smai=-2, anomaly_source=source, observed_at=self.now)
        for time in (None, pd.NaT, "", "bad", "2026-11-11T11:00:00", "2026-11-11T12:00:01Z"):
            with self.subTest(observed_at=time):
                with self.assertRaises(ValueError):
                    self._request(smai=-2, anomaly_source="unit-test input", observed_at=time)
        # Equality is valid even when the supplied time has a different UTC offset.
        request = self._request(smai=-2, anomaly_source="unit-test input",
                                observed_at="2026-11-11T17:00:00+05:00")
        self.assertEqual(pd.Timestamp(request["trigger"]["observed_at"]), self.now)

    def test_tasking_deep_copies_analysis_context_aoi_and_selected_bands(self):
        context = {"input_kind": "synthetic unit-test input", "selection": {
            "region": "unit-test-region", "crops": ["unit-test-crop"], "stress": {"factor": 0.2},
        }}
        bands = ["Red", "NIR"]
        original_context = copy.deepcopy(context)
        original_aoi = copy.deepcopy(self.aoi)
        request = self._request(context=context, bands=bands)
        self.assertEqual(request["analysis_context"], original_context)
        context["selection"]["crops"].append("unit-test-second-crop")
        bands.append("Blue")
        self.aoi["coordinates"][0][1][0] = 65
        self.assertEqual(request["analysis_context"], original_context)
        self.assertEqual(request["sensor"]["bands"], ["Red", "NIR"])
        self.assertEqual(json.loads(json.dumps(request["aoi"])), original_aoi)
        request["analysis_context"]["selection"]["stress"]["factor"] = 0.9
        self.assertEqual(context["selection"]["stress"]["factor"], 0.2)
        request["sensor"]["bands"].append("Green")
        self.assertEqual(tuple(SATELLITES["KazEOSat-1"]["bands"]),
                         ("PAN", "Blue", "Green", "Red", "NIR"))

    def test_off_nadir_uses_hand_derived_spherical_geometry_not_flat_earth(self):
        # Unit-test orbit r_sat = 2R; target central angle 30 degrees.
        # Horizontal leg = R/2; vertical leg = R*(2 - sqrt(3)/2).
        # tan(off_nadir) = 1/(4 - sqrt(3)); no production helper in oracle.
        radius = 6371.0
        distance = radius * math.pi / 6
        expected = math.degrees(math.atan(1 / (4 - math.sqrt(3))))
        actual = estimate_off_nadir_deg(distance, radius)
        self.assertAlmostEqual(actual, expected, places=12)
        flat_earth = math.degrees(math.atan(distance / radius))
        self.assertGreater(abs(actual - flat_earth), 1)
        self.assertEqual(estimate_off_nadir_deg(0, 675), 0)

    def test_off_nadir_horizon_and_physical_input_bounds(self):
        radius = 6371.0
        horizon = radius * math.acos(0.5)  # r_sat=2R: central horizon angle=60 degrees.
        near_horizon = estimate_off_nadir_deg(horizon - 0.001, radius)
        self.assertAlmostEqual(near_horizon, 30.0, places=8)
        for distance in (horizon, horizon + 0.001, radius * math.pi / 2):
            with self.subTest(distance=distance):
                with self.assertRaises(ValueError):
                    estimate_off_nadir_deg(distance, radius)
        for distance, altitude in [
            (-1, 675), (20001, 675), (np.nan, 675), (np.inf, 675),
            (0, 0), (0, -1), (0, 40001), (0, np.nan), (0, np.inf),
        ]:
            with self.subTest(distance=distance, altitude=altitude):
                with self.assertRaises(ValueError):
                    estimate_off_nadir_deg(distance, altitude)

    def test_tasking_off_nadir_estimate_never_becomes_actual_roll_or_pass(self):
        for satellite in ("KazEOSat-1", "KazEOSat-2"):
            with self.subTest(satellite=satellite):
                # Mission altitude is configuration, geometry is independently derived.
                altitude = SATELLITES[satellite]["orbit_altitude_km"]
                # Small-angle spherical construction via independent vector dot product.
                angle = 100 / 6371
                vertical = altitude + 6371 * (1 - math.cos(angle))
                horizontal = 6371 * math.sin(angle)
                expected = math.degrees(math.acos(vertical / math.hypot(vertical, horizontal)))
                request = self._request(satellite, ground_distance_km=100)
                self.assertAlmostEqual(request["planning"]["estimated_off_nadir_deg"], expected, places=10)
                self.assertEqual(request["planning"]["ground_distance_km"], 100)
                self.assertIsNone(request["planning"]["roll_deg"])
                self.assertIsNone(request["planning"]["acquisition_window_utc"])
                with self.assertRaises(ValueError):
                    self._request(satellite, ground_distance_km=10000)

    def test_space_weather_missing_inputs_do_not_manufacture_measurements(self):
        result = assess_space_weather()
        for field in ("kp", "tec_tecu", "group_delay_m", "faraday_rotation_rad", "faraday_rotation_deg"):
            self.assertIsNone(result[field])
        self.assertEqual(result["geomagnetic_status"], "unknown")
        self.assertEqual(result["faraday_status"], "tec_required")
        tec_only = assess_space_weather(tec_tecu=14)
        self.assertIsNotNone(tec_only["group_delay_m"])
        self.assertIsNone(tec_only["faraday_rotation_rad"])
        self.assertEqual(tec_only["faraday_status"], "geometry_required")
        field_only = assess_space_weather(b_parallel_ut=50)
        self.assertIsNone(field_only["faraday_rotation_rad"])
        self.assertEqual(field_only["faraday_status"], "tec_required")

    def test_zero_tec_gives_exactly_zero_delay_and_rotation_not_missing_values(self):
        result = assess_space_weather(kp=0, tec_tecu=0, b_parallel_ut=50)
        self.assertEqual(result["kp"], 0)
        self.assertEqual(result["tec_tecu"], 0)
        for field in ("group_delay_m", "faraday_rotation_rad", "faraday_rotation_deg"):
            self.assertEqual(result[field], 0.0)
        self.assertEqual(result["faraday_status"], "estimated_not_corrected")

    def test_fourteen_tecu_has_hand_computed_delay_and_faraday_rotation(self):
        # Unit-test inputs: 14 TECU = 14e16 electrons/m^2, B_parallel=50e-6 T.
        # f^2 = 1.9881e18 Hz^2: delay = 5.642/1.9881 = 56420/19881 m;
        # rotation = 0.1652/1.9881 = 1652/19881 rad (single-pass convention).
        result = assess_space_weather(tec_tecu=14, b_parallel_ut=50, frequency_hz=1.41e9)
        self.assertAlmostEqual(result["group_delay_m"], 56420 / 19881, places=12)
        self.assertAlmostEqual(result["faraday_rotation_rad"], 1652 / 19881, places=14)
        self.assertAlmostEqual(result["faraday_rotation_deg"], 1652 / 19881 * 180 / math.pi, places=12)
        self.assertEqual(result["faraday_status"], "estimated_not_corrected")

    def test_faraday_field_sign_zero_and_inverse_frequency_squared(self):
        positive = assess_space_weather(tec_tecu=14, b_parallel_ut=50, frequency_hz=1.41e9)
        negative = assess_space_weather(tec_tecu=14, b_parallel_ut=-50, frequency_hz=1.41e9)
        zero = assess_space_weather(tec_tecu=14, b_parallel_ut=0, frequency_hz=1.41e9)
        doubled = assess_space_weather(tec_tecu=14, b_parallel_ut=50, frequency_hz=2.82e9)
        self.assertAlmostEqual(negative["faraday_rotation_rad"], -positive["faraday_rotation_rad"])
        self.assertAlmostEqual(negative["group_delay_m"], positive["group_delay_m"])
        self.assertEqual(zero["faraday_rotation_rad"], 0)
        self.assertEqual(zero["faraday_status"], "estimated_not_corrected")
        for field in ("group_delay_m", "faraday_rotation_rad", "faraday_rotation_deg"):
            self.assertAlmostEqual(doubled[field], positive[field] / 4, places=13)

    def test_space_weather_rejects_invalid_tec_field_and_frequency(self):
        for argument, invalid in {
            "tec_tecu": [-1, 1001, np.nan, np.inf, -np.inf],
            "b_parallel_ut": [-101, 101, np.nan, np.inf, -np.inf],
            "frequency_hz": [0, -1, 999999, 1e12 + 1, np.nan, np.inf],
        }.items():
            for bad in invalid:
                with self.subTest(argument=argument, value=bad):
                    with self.assertRaises(ValueError):
                        assess_space_weather(**{argument: bad})
        for field in (-100, 100):
            for frequency in (1e6, 1e12):
                result = assess_space_weather(tec_tecu=1000, b_parallel_ut=field, frequency_hz=frequency)
                self.assertTrue(math.isfinite(result["group_delay_m"]))
                self.assertTrue(math.isfinite(result["faraday_rotation_rad"]))

    def test_kp_range_and_storm_threshold_at_five(self):
        for kp, expected in [(0, "quiet"), (3.999, "quiet"), (4, "unsettled"),
                             (4.999, "unsettled"), (5, "storm"), (9, "storm")]:
            with self.subTest(kp=kp):
                self.assertEqual(assess_space_weather(kp=kp)["geomagnetic_status"], expected)
        for kp in (-0.001, 9.001, np.nan, np.inf, -np.inf):
            with self.subTest(invalid_kp=kp):
                with self.assertRaises(ValueError):
                    assess_space_weather(kp=kp)

    def test_weather_estimates_never_validate_retrieval_or_invent_quality_percent(self):
        for kwargs in ({}, {"kp": 0}, {"kp": 9}, {"tec_tecu": 0, "b_parallel_ut": 0},
                       {"kp": 1.8, "tec_tecu": 14, "b_parallel_ut": 50}):
            with self.subTest(inputs=kwargs):
                result = assess_space_weather(**kwargs)
                self.assertIs(result["retrieval_validated"], False)
                serialized = json.dumps(result, allow_nan=False).lower()
                for unsupported in ("percent", "%", "quality", "99.2", "confidence"):
                    self.assertNotIn(unsupported, serialized)

    def test_noaa_parser_selects_latest_past_row_not_payload_order(self):
        payload = self._noaa_payload(
            ("2026-11-11 09:00:00", "4.67"),
            ("2026-11-11 15:00:00", "9"),  # Future fixture must not win.
            ("2026-11-11 03:00:00", "1"),
            ("2026-11-11 06:00:00", "2"),
        )
        original = copy.deepcopy(payload)
        result = parse_noaa_kp(payload, now=self.now)
        self.assertEqual(result["kp"], 4.67)
        self.assertEqual(result["observed_at"], "2026-11-11T09:00:00+00:00")
        self.assertEqual(result["source"], "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")
        self.assertIn("3-hour", result["product"])
        self.assertIs(result["stale"], False)
        self.assertEqual(payload, original)
        reversed_rows = [payload[0], *reversed(payload[1:])]
        self.assertEqual(parse_noaa_kp(reversed_rows, now=self.now), result)

    def test_noaa_parser_handles_reordered_header_extra_columns_and_zero_kp(self):
        payload = [["Kp", "unit_test_extra", "time_tag"],
                   ["0", "synthetic unit-test input", "2026-11-11T17:00:00+05:00"]]
        result = parse_noaa_kp(payload, now=self.now)
        self.assertEqual(result["kp"], 0)
        self.assertEqual(result["observed_at"], "2026-11-11T12:00:00+00:00")
        self.assertFalse(result["stale"])

    def test_noaa_dictionary_rows_match_table_and_ignore_malformed_records(self):
        # Alternate NOAA wire format; these dictionary rows are parser inputs only.
        rows = [
            {"time_tag": "2026-11-11T09:00:00Z", "Kp": "2.33"},
            {"time_tag": "2026-11-11T15:00:00Z", "Kp": "9"},
            {"time_tag": "2026-11-11T06:00:00Z", "Kp": "1"},
            None, "invalid-row", {},
            {"time_tag": "2026-11-11T12:00:00Z"},
            {"Kp": "3"}, {"time_tag": None, "Kp": "3"},
            {"time_tag": "invalid-time", "Kp": "3"},
            {"time_tag": "2026-11-11T12:00:00Z", "Kp": "nan"},
            {"time_tag": "2026-11-11T12:00:00Z", "Kp": "inf"},
            {"time_tag": "2026-11-11T12:00:00Z", "Kp": "10"},
        ]
        original = copy.deepcopy(rows)
        result = parse_noaa_kp(rows, now=self.now)
        expected = parse_noaa_kp(
            self._noaa_payload(("2026-11-11T09:00:00Z", "2.33")), now=self.now
        )
        self.assertEqual(result, expected)
        self.assertEqual(rows, original)
        self.assertEqual(parse_noaa_kp(list(reversed(rows)), now=self.now), expected)

    def test_noaa_parser_skips_malformed_rows_without_losing_valid_record(self):
        payload = self._noaa_payload(("2026-11-11T09:00:00Z", "2.33"))
        payload.extend([
            None, {}, "not-a-row", [], ["2026-11-11T12:00:00Z"],
            ["2026-11-11T12:00:00Z", "2", "unexpected"],
            ["bad-date", "4"], [None, "4"], ["NaT", "4"],
            ["2026-11-11T12:00:00Z", None],
            ["2026-11-11T12:00:00Z", "nan"],
            ["2026-11-11T12:00:00Z", np.nan],
            ["2026-11-11T12:00:00Z", "inf"],
            ["2026-11-11T12:00:00Z", "-inf"],
            ["2026-11-11T12:00:00Z", "bad-kp"],
            ["2026-11-11T12:00:00Z", -1],
            ["2026-11-11T12:00:00Z", 10],
        ])
        result = parse_noaa_kp(payload, now=self.now)
        self.assertEqual(result["kp"], 2.33)
        self.assertEqual(result["observed_at"], "2026-11-11T09:00:00+00:00")

    def test_noaa_parser_rejects_empty_bad_schema_and_all_invalid_or_future_rows(self):
        invalid = [
            None, {}, [], [["time_tag", "Kp"]],
            ["not-a-header", ["2026-11-11T09:00:00Z", "1"]],
            [["time_tag", "kp"], ["2026-11-11T09:00:00Z", "1"]],
            [["Kp"], ["1"]],
            self._noaa_payload(("bad-date", "2")),
            self._noaa_payload((None, "2")),
            self._noaa_payload(("2026-11-11T09:00:00Z", "nan")),
            self._noaa_payload(("2026-11-11T09:00:00Z", "inf")),
            self._noaa_payload(("2026-11-11T12:00:01Z", "2")),
            [["time_tag", "Kp"], None, [], {}],
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse_noaa_kp(payload, now=self.now)

    def test_noaa_staleness_uses_deterministic_now_and_strict_six_hour_boundary(self):
        for stamp, stale in [("2026-11-11T12:00:00Z", False),
                             ("2026-11-11T06:00:00Z", False),
                             ("2026-11-11T05:59:59Z", True),
                             ("2026-11-10T12:00:00Z", True)]:
            with self.subTest(stamp=stamp):
                result = parse_noaa_kp(self._noaa_payload((stamp, "3")), now=self.now)
                self.assertIs(result["stale"], stale)
                self.assertEqual(result["kp"], 3)  # Stale is flagged, not fabricated/replaced.
        payload = self._noaa_payload(("2026-11-11T06:00:00Z", "3"))
        self.assertEqual(parse_noaa_kp(payload, now="2026-11-11T17:00:00+05:00"),
                         parse_noaa_kp(payload, now=self.now))

    def test_noaa_requires_valid_timezone_aware_now(self):
        payload = self._noaa_payload(("2026-11-11T09:00:00Z", "1"))
        for now in ("2026-11-11T12:00:00", "bad", pd.NaT):
            with self.subTest(now=now):
                with self.assertRaises(ValueError):
                    parse_noaa_kp(payload, now=now)

    def test_maps_serialize_all_channels_in_both_languages_with_supplied_unit_test_values(self):
        # Synthetic unit-test channel values, never populated as production observations.
        channels = {"ndvi": (0.5, -1, 1), "vv_db": (-10, -30, 0),
                    "vh_db": (-20, -35, -5), "rvi": (3, 0, 4),
                    "psi_radar": (0.6, 0, 1)}
        for lang in ("kk", "ru"):
            self.assertEqual(set(CHANNEL_LABELS[lang]), set(channels))
            for channel, (value, low, high) in channels.items():
                with self.subTest(lang=lang, channel=channel):
                    fig = build_observation_map(self.aoi, channel, value, lang=lang)
                    payload = self._assert_serializable(fig)
                    trace = payload["data"][0]
                    self.assertEqual(trace["type"], "choropleth")
                    self.assertEqual(trace["z"], [value])
                    self.assertEqual((trace["zmin"], trace["zmax"]), (low, high))
                    self.assertIn(CHANNEL_LABELS[lang][channel], trace["hovertemplate"])
                    self.assertEqual(trace["geojson"]["features"][0]["geometry"], self.aoi)
                    self.assertEqual(payload["layout"]["geo"]["center"], {"lon": 62, "lat": 52})
                    self.assertEqual(payload["layout"]["geo"]["projection"]["type"], "mercator")

    def test_empty_maps_have_aoi_outline_but_no_fabricated_measurement(self):
        for lang in ("kk", "ru"):
            for channel in ("ndvi", "vv_db", "vh_db", "rvi", "psi_radar"):
                with self.subTest(lang=lang, channel=channel):
                    payload = self._assert_serializable(
                        build_observation_map(self.aoi, channel, None, lang=lang)
                    )
                    self.assertEqual(len(payload["data"]), 1)
                    self.assertEqual(payload["data"][0]["type"], "scattergeo")
                    self.assertNotIn("z", payload["data"][0])
                    self.assertNotIn("colorbar", payload["data"][0])
                    self.assertEqual(payload["data"][0]["lon"], [60, 64, 64, 60, 60])

    def test_maps_react_to_aoi_channel_and_values_including_zero_and_rvi_above_one(self):
        shifted = copy.deepcopy(self.aoi)
        for point in shifted["coordinates"][0]:
            point[0] += 8
        first = build_observation_map(self.aoi, "rvi", 0)
        second = build_observation_map(shifted, "rvi", 3)
        third = build_observation_map(self.aoi, "vv_db", -10)
        self.assertEqual(list(first.data[0].z), [0])
        self.assertEqual(list(second.data[0].z), [3])
        self.assertEqual(second.data[0].zmax, 4)
        self.assertEqual(first.layout.geo.center.lon, 62)
        self.assertEqual(second.layout.geo.center.lon, 70)
        self.assertNotEqual(first.layout.geo.lonaxis.range, second.layout.geo.lonaxis.range)
        self.assertEqual((third.data[0].zmin, third.data[0].zmax), (-30, 0))
        self.assertNotEqual(first.to_json(), second.to_json())
        self.assertNotEqual(first.to_json(), third.to_json())

    def test_maps_keep_polygon_holes_in_serialized_geojson(self):
        geometry = copy.deepcopy(self.aoi)
        geometry["coordinates"].append([[61, 51], [61, 53], [63, 53], [63, 51], [61, 51]])
        for lang in ("kk", "ru"):
            with self.subTest(lang=lang):
                payload = self._assert_serializable(build_observation_map(geometry, "rvi", 2, lang=lang))
                self.assertEqual(payload["data"][0]["geojson"]["features"][0]["geometry"], geometry)

    def test_maps_reject_nonfinite_measurements(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    build_observation_map(self.aoi, "rvi", bad)

    def test_cubes_serialize_both_languages_all_channels_and_empty_values(self):
        for lang in ("kk", "ru"):
            for channel, value in [("ndvi", 0.5), ("vv_db", -10), ("vh_db", -20), ("rvi", 3), ("psi_radar", 0.6)]:
                for selected in (None, 0, value):
                    with self.subTest(lang=lang, channel=channel, value=selected):
                        fig = build_data_cube(lang=lang, channel=channel, value=selected)
                        payload = self._assert_serializable(fig)
                        title = payload["layout"]["title"]["text"]
                        expected = CHANNEL_LABELS[lang][channel]
                        if selected is not None:
                            expected += f" · {selected:.3f}"
                        self.assertEqual(title, expected)
                        self.assertEqual(sum(t["type"] == "mesh3d" for t in payload["data"]), 6)
                        self.assertEqual(sum(t["type"] == "scatter3d" for t in payload["data"]), 2)
                        axis = payload["layout"]["scene"]["zaxis"]
                        self.assertEqual(axis["title"]["text"],
                                         "Шартты масштаб" if lang == "kk" else "Условный масштаб")
                        self.assertNotIn("nan", title.lower())

    def test_cube_root_depth_changes_model_label_and_depth_axis_not_sensor_penetration(self):
        for lang in ("kk", "ru"):
            with self.subTest(lang=lang):
                shallow = build_data_cube(lang=lang, root_depth_m=0.8)
                deep = build_data_cube(lang=lang, root_depth_m=2.0)
                self.assertIn("0.8 м", self._cube_plane(shallow, 0).name)
                self.assertIn("2 м", self._cube_plane(deep, 0).name)
                self.assertEqual(shallow.layout.scene.zaxis.ticktext[0], "−0.8 м")
                self.assertEqual(deep.layout.scene.zaxis.ticktext[0], "−2 м")
                self.assertNotEqual(shallow.to_json(), deep.to_json())
                for fig in (shallow, deep):
                    beam = next(trace for trace in fig.data if trace.name == "C-band SAR")
                    self.assertEqual(tuple(beam.z), (5, 1))

    def test_cube_cloud_cover_changes_opacity_and_blocks_optics_not_radar(self):
        for lang in ("kk", "ru"):
            for channel in ("ndvi", "vv_db", "vh_db", "rvi"):
                with self.subTest(lang=lang, channel=channel):
                    clear = build_data_cube(lang=lang, channel=channel, cloud_cover=0)
                    cloudy = build_data_cube(lang=lang, channel=channel, cloud_cover=100)
                    self.assertAlmostEqual(self._cube_plane(clear, 3).opacity, 0.12)
                    self.assertAlmostEqual(self._cube_plane(cloudy, 3).opacity, 0.82)
                    self.assertIn("0%", self._cube_plane(clear, 3).name)
                    self.assertIn("100%", self._cube_plane(cloudy, 3).name)
                    clear_beam = next(trace for trace in clear.data
                                      if trace.type == "scatter3d" and trace.mode == "lines+markers")
                    cloudy_beam = next(trace for trace in cloudy.data
                                       if trace.type == "scatter3d" and trace.mode == "lines+markers")
                    self.assertEqual(tuple(clear_beam.z), (5, 2 if channel == "ndvi" else 1))
                    self.assertEqual(tuple(cloudy_beam.z), (5, 3 if channel == "ndvi" else 1))
                    self.assertNotEqual(clear.to_json(), cloudy.to_json())

    def test_no_sar_beam_reaches_roots_and_root_link_is_explicitly_modelled(self):
        for lang in ("kk", "ru"):
            for channel in ("vv_db", "vh_db", "rvi", "psi_radar"):
                for root in (0.1, 1.2, 5.0):
                    with self.subTest(lang=lang, channel=channel, root=root):
                        fig = build_data_cube(lang=lang, channel=channel, root_depth_m=root)
                        beam = next(trace for trace in fig.data if trace.name == "C-band SAR")
                        self.assertEqual(tuple(beam.z), (5, 1))
                        self.assertNotIn(0, beam.z)
                        model = [trace for trace in fig.data
                                 if trace.type == "scatter3d" and 0 in trace.z]
                        self.assertEqual(len(model), 1)
                        self.assertEqual(tuple(model[0].z), (1, 0))
                        self.assertEqual(model[0].line.dash, "dash")
                        self.assertIn("тікелей өлшем емес" if lang == "kk" else "не прямой замер", model[0].name)
                        self.assertIn("модель", self._cube_plane(fig, 0).name.lower())

    def test_cube_rejects_invalid_root_cloud_and_channel(self):
        for argument, invalid in {
            "root_depth_m": [0, -1, 0.099, 5.001, np.nan, np.inf],
            "cloud_cover": [-0.001, 100.001, np.nan, np.inf],
            "channel": ["invalid-channel", "VV", ""],
        }.items():
            for bad in invalid:
                with self.subTest(argument=argument, value=bad):
                    with self.assertRaises(ValueError):
                        build_data_cube(**{argument: bad})

    def test_cube_rejects_nonfinite_measurements_instead_of_showing_nan_or_inf(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    build_data_cube(value=bad)

    def test_csv_filters_aoi_before_validating_per_scene_uniqueness(self):
        import io
        from dashboard.space_segment_tab import _uploaded_frame
        frame = self._sar_frame()
        frame["region_id"] = "sko"
        frame["district_id"] = "kyzylzhar"
        other = frame.copy()
        other["region_id"] = "kostanay"
        other["district_id"] = "karabalyk"
        combined = pd.concat([frame, other]).to_csv(index=False)
        selected = _uploaded_frame(io.StringIO(combined), "sar", "sko", "kyzylzhar", 2026)
        self.assertEqual(len(selected), 2)
        self.assertTrue(_uploaded_frame(io.StringIO(combined), "sar", "sko", "kyzylzhar", 2025).empty)
        self.assertTrue(_uploaded_frame(io.StringIO(combined), "sar", "pavlodar", "unknown", 2026).empty)

    def test_ui_tasking_invalidation_and_tec_reactivity_in_both_languages(self):
        from streamlit.testing.v1 import AppTest
        for lang in ("kk", "ru"):
            with self.subTest(lang=lang):
                app = AppTest.from_string(
                    "from dashboard.space_segment_tab import render_space_segment_tab\n"
                    f"render_space_segment_tab(lang='{lang}', region_id='sko', crop_id='spring_wheat', year=2026, shock_percent=50)"
                ).run(timeout=15)
                self.assertEqual(len(app.exception), 0)
                self.assertIsNone(app.number_input(key="space_tec").value)
                app.button(key="space_tasking").click().run()
                request = app.session_state["space_tasking_result"][1]
                self.assertEqual(request["status"], "draft_not_submitted")
                self.assertEqual(request["analysis_context"]["scenario_moisture_drop_percent"], 50)
                self.assertFalse(request["analysis_context"]["scenario_is_observation"])
                self.assertEqual(len(app.success), 1)
                app.selectbox(key="space_satellite").select("KazEOSat-2").run()
                self.assertEqual(len(app.success), 0)
                app.button(key="space_tasking").click().run()
                self.assertEqual(app.session_state["space_tasking_result"][1]["requested_resolution_m"], 6.5)
                app.number_input(key="space_smai").set_value(-1.5).run()
                app.button(key="space_tasking").click().run()
                self.assertEqual(len(app.error), 1)
                self.assertEqual(len(app.success), 0)
                app.number_input(key="space_tec").set_value(14).run()
                app.number_input(key="space_bparallel").set_value(50).run()
                self.assertIn("2.838", [metric.value for metric in app.metric])
                self.assertIn("4.761°", [metric.value for metric in app.metric])
                app.number_input(key="space_tec").set_value(0).run()
                self.assertIn("0.000°", [metric.value for metric in app.metric])
                self.assertEqual(len(app.exception), 0)

    def test_ui_all_regions_crops_and_languages_without_network(self):
        import tempfile
        from itertools import product
        from unittest.mock import patch
        from config.settings import GRAIN_BELT_REGIONS, TARGET_CROPS
        from modules import space_cache
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as temporary:
            empty_cache = Path(temporary) / "space_cache"
            with patch.object(space_cache, "CACHE_DIR", empty_cache), \
                 patch.object(space_cache, "NOAA_CACHE", empty_cache / "noaa.json"):
                for lang, region_id, crop_id in product(("kk", "ru"), GRAIN_BELT_REGIONS, TARGET_CROPS):
                    with self.subTest(lang=lang, region=region_id, crop=crop_id):
                        app = AppTest.from_string(
                            "from dashboard.space_segment_tab import render_space_segment_tab\n"
                            f"render_space_segment_tab(lang='{lang}', region_id='{region_id}', crop_id='{crop_id}', year=2026, shock_percent=0)"
                        ).run(timeout=15)
                        self.assertEqual(len(app.exception), 0)
                        self.assertEqual(len(app.radio(key="space_channel").options), 5)
                        self.assertEqual(len(app.get("plotly_chart")), 3)
                        self.assertTrue(any(GRAIN_BELT_REGIONS[region_id]["name_" + lang] in caption.value for caption in app.caption))


if __name__ == "__main__":
    unittest.main()
