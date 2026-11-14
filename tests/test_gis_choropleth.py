"""
AgriCascade — tests/test_gis_choropleth.py
==========================================
Автоматизированный верификационный тест для Phase 4 (ГИС-картограмма рисков):
- Проверка генерации GeoJSON FeatureCollection для всех 4 регионов зернового пояса
- Проверка структуры полигонов, площадей пашни и типов почв
- Проверка генерации HTML Rich-тултипов (KK и RU)
- Проверка реактивности What-If симулятора (динамическое перекрашивание в красный/янтарный)
"""

import sys
from pathlib import Path
import unittest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from modules.gis_engine import compute_district_choropleth_geojson, DISTRICT_REGISTRY


class TestGISChoropleth(unittest.TestCase):

    def test_registry_integrity(self):
        """Проверка целостности реестра районов 4 областей (20 районов)."""
        expected_regions = ["sko", "kostanay", "akmola", "pavlodar"]
        self.assertEqual(sorted(list(DISTRICT_REGISTRY.keys())), sorted(expected_regions))

        total_districts = sum(len(dists) for dists in DISTRICT_REGISTRY.values())
        self.assertEqual(total_districts, 20, "Ожидалось ровно 20 оцифрованных районов (по 5 в 4 областях)")

        for reg_id, dists in DISTRICT_REGISTRY.items():
            for d in dists:
                self.assertIn("polygon", d)
                self.assertGreaterEqual(len(d["polygon"]), 4)
                self.assertGreater(d["arable_ha"], 100000, f"Нереалистичная площадь пашни в {d['id']}")
                self.assertGreater(d["base_awc_mm"], 50, f"Нереалистичная влагоемкость AWC в {d['id']}")

    def test_geojson_generation_all_regions_and_languages(self):
        """Проверка генерации GeoJSON для всех 4 областей на KK и RU."""
        for lang in ["kk", "ru"]:
            for reg_id in ["sko", "kostanay", "akmola", "pavlodar"]:
                gj = compute_district_choropleth_geojson(
                    region_id=reg_id,
                    lang=lang,
                    current_psi=-0.8,
                    crop_kc=1.15,
                    crop_tolerance=3.5,
                    crop_name="Жаздық бидай" if lang == "kk" else "Пшеница яровая",
                    shock_factor=0.0,
                )
                self.assertEqual(gj["type"], "FeatureCollection")
                self.assertEqual(len(gj["features"]), 5, f"Ожидалось 5 районов в {reg_id}")

                for feat in gj["features"]:
                    props = feat["properties"]
                    self.assertIn(props["fill_color"], ["#34C759", "#FF9500", "#FF3B30"])
                    self.assertIn("tooltip_html", props)
                    self.assertIn("📍", props["tooltip_html"])
                    self.assertTrue(15 <= props["available_moisture_pct"] <= 95)

    def test_district_layer_is_deterministic_and_soil_driven(self):
        """Одинаковые входы дают одинаковый GeoJSON; вариация районов идёт от AWC, не от random."""
        kwargs = dict(region_id="sko", lang="ru", current_psi=-0.4, crop_kc=1.15,
                      crop_tolerance=3.5, crop_name="Пшеница яровая", shock_factor=0.0)
        first = compute_district_choropleth_geojson(**kwargs)
        second = compute_district_choropleth_geojson(**kwargs)
        self.assertEqual(first, second)
        psi_values = {feature["properties"]["id"]: feature["properties"]["psi"] for feature in first["features"]}
        self.assertGreater(len(set(psi_values.values())), 1)
        awc = {d["id"]: d["base_awc_mm"] for d in DISTRICT_REGISTRY["sko"]}
        worst = min(psi_values, key=psi_values.get)
        self.assertEqual(worst, min(awc, key=awc.get))

    def test_what_if_stress_reactivity(self):
        """Проверка реактивного перекрашивания районов при стресс-тесте."""
        # 1. Базовый сценарий (без шока)
        gj_base = compute_district_choropleth_geojson(
            region_id="kostanay",
            lang="ru",
            current_psi=-0.2,
            crop_kc=1.15,
            crop_tolerance=3.5,
            crop_name="Пшеница яровая",
            shock_factor=0.0,
        )
        base_high_risk = sum(1 for f in gj_base["features"] if f["properties"]["risk_level"] == "HIGH")

        # 2. Экстремальный шок 40% (-40% влажности)
        gj_shock = compute_district_choropleth_geojson(
            region_id="kostanay",
            lang="ru",
            current_psi=-0.2,
            crop_kc=1.15,
            crop_tolerance=3.5,
            crop_name="Пшеница яровая",
            shock_factor=0.40,
        )
        shock_high_risk = sum(1 for f in gj_shock["features"] if f["properties"]["risk_level"] == "HIGH")

        # При шоке число районов в критической красной зоне должно вырасти
        self.assertGreater(shock_high_risk, base_high_risk, "Шок 40% обязан увеличивать количество районов в красной зоне!")
        
        # Средний индекс риска должен вырасти минимум на 20 пунктов
        avg_base_r = sum(f["properties"]["risk_score"] for f in gj_base["features"]) / 5.0
        avg_shock_r = sum(f["properties"]["risk_score"] for f in gj_shock["features"]) / 5.0
        self.assertGreater(avg_shock_r - avg_base_r, 20.0, f"Прирост индекса риска недостаточен: {avg_shock_r - avg_base_r}")


if __name__ == "__main__":
    unittest.main()

