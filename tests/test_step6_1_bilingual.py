import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.elevator_logistics import (
    calculate_dynamic_silo_loads,
    calculate_price_transmission_chain,
    ALL_207_ELEVATORS,
)
from modules.gis_engine import compute_district_choropleth_geojson, DISTRICT_REGISTRY


class TestStep61Bilingual(unittest.TestCase):
    """Автоматизированная проверка двуязычия (KK / RU) согласно Step 6.1."""

    def test_district_choropleth_bilingual_output(self):
        """Проверка генерации GeoJSON на казахском и русском языках."""
        for reg in ["sko", "kostanay", "akmola", "pavlodar"]:
            # Казахский язык
            geo_kk = compute_district_choropleth_geojson(region_id=reg, lang="kk")
            self.assertGreater(len(geo_kk["features"]), 0)
            f_kk = geo_kk["features"][0]["properties"]
            self.assertIn("ауданы", f_kk["name"].lower())
            self.assertGreater(len(f_kk["soil"]), 5)
            self.assertIn("Мәртебе", f_kk["tooltip_html"])
            self.assertIn("Дақыл", f_kk["tooltip_html"])
            self.assertIn("Егістік", f_kk["tooltip_html"])
            self.assertIn("Ұсыныс", f_kk["tooltip_html"])

            # Русский язык
            geo_ru = compute_district_choropleth_geojson(region_id=reg, lang="ru")
            f_ru = geo_ru["features"][0]["properties"]
            self.assertTrue(
                "район" in f_ru["name"].lower() or "обл" in f_ru["name"].lower()
            )
            self.assertIn("Статус", f_ru["tooltip_html"])
            self.assertIn("Культура", f_ru["tooltip_html"])
            self.assertIn("Пашня", f_ru["tooltip_html"])
            self.assertIn("Действие", f_ru["tooltip_html"])

    def test_price_chain_bilingual_recommendations(self):
        """Проверка рекомендаций цепочки цен на казахском и русском языках."""
        # Высокий риск (> 65 баллов: шок -45%)
        high = calculate_price_transmission_chain(yield_shock_pct=-45.0)
        self.assertEqual(high["risk_tier"], "HIGH")
        self.assertIn("ИНФЛЯЦИЯНЫҢ КРИТИКАЛЫҚ ҚАУПІ", high["recommendation_kk"])
        self.assertIn("Азық-түлік корпорациясы", high["recommendation_kk"])
        self.assertIn("КРИТИЧЕСКИЙ РИСК ИНФЛЯЦИИ", high["recommendation_ru"])
        self.assertIn("Продкорпорации", high["recommendation_ru"])

        # Средний риск (30-65 баллов: шок -25%)
        med = calculate_price_transmission_chain(yield_shock_pct=-25.0)
        self.assertEqual(med["risk_tier"], "MEDIUM")
        self.assertIn("ОРТАША ҚАУІП", med["recommendation_kk"])
        self.assertIn("ҚТЖ", med["recommendation_kk"])
        self.assertIn("УМЕРЕННЫЙ РИСК", med["recommendation_ru"])
        self.assertIn("КТЖ", med["recommendation_ru"])

        # Низкий риск (< 30 баллов: шок -5%)
        low = calculate_price_transmission_chain(yield_shock_pct=-5.0)
        self.assertEqual(low["risk_tier"], "LOW")
        self.assertIn("БАҒА ТҰРАҚТЫЛЫҒЫ", low["recommendation_kk"])
        self.assertIn("ЦЕНОВАЯ СТАБИЛЬНОСТЬ", low["recommendation_ru"])

    def test_silo_loads_bilingual_fields(self):
        """Проверка названий областей на казахском и русском для всех 207 ХПП."""
        df = calculate_dynamic_silo_loads()
        self.assertEqual(len(df), 207)
        self.assertTrue((df["region_name_kk"].str.contains("облысы")).all())
        self.assertTrue((df["region_name_ru"].str.contains("область")).all())
        self.assertTrue((df["status_label"].str.contains("🟢|🟡|🔴")).all())

    def test_app_syntax_integrity(self):
        """Проверка синтаксической чистоты app.py после внесения правок двуязычия."""
        app_path = PROJECT_ROOT / "dashboard" / "app.py"
        import ast
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()
        parsed = ast.parse(code)
        self.assertIsNotNone(parsed)


if __name__ == "__main__":
    unittest.main()
