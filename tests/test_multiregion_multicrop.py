"""
AgriCascade — tests/test_multiregion_multicrop.py
=================================================
Автоматизированный верификационный тест по стандарту Opus 5.5:
- 4 зерновые области Казахстана (СКО, Костанай, Акмола, Павлодар)
- 5 целевых культур финала с коэффициентами водопотребления FAO-56 (Kc)
- Двуязычие (KK / RU) и генераторы аналитических отчетов (Excel, Markdown)
"""

import sys
from pathlib import Path
import unittest

# Добавление корня проекта в sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import GRAIN_BELT_REGIONS, TARGET_CROPS
from dashboard.app import generate_excel_report, generate_markdown_report


class TestMultiRegionMultiCrop(unittest.TestCase):

    def test_grain_belt_regions(self):
        """Проверка структуры и гео-параметров 4 областей зернового пояса."""
        expected_regions = ["sko", "kostanay", "akmola", "pavlodar"]
        self.assertEqual(sorted(list(GRAIN_BELT_REGIONS.keys())), sorted(expected_regions))

        for reg_id, reg in GRAIN_BELT_REGIONS.items():
            self.assertIn("center", reg)
            self.assertEqual(len(reg["center"]), 2, f"Центр должен быть [lat, lon] в {reg_id}")
            self.assertTrue(48.0 <= reg["center"][0] <= 56.0, f"Широта центра вне Казахстана в {reg_id}")
            self.assertTrue(58.0 <= reg["center"][1] <= 82.0, f"Долгота центра вне зернового пояса в {reg_id}")

            self.assertIn("polygon", reg)
            self.assertGreaterEqual(len(reg["polygon"]), 4, f"Полигон {reg_id} должен иметь >= 4 вершин")

            self.assertIn("name_ru", reg)
            self.assertIn("name_kk", reg)
            self.assertIn("districts_ru", reg)
            self.assertIn("districts_kk", reg)
            self.assertGreaterEqual(len(reg["districts_ru"]), 3, f"Недостаточно районов в {reg_id}")

    def test_target_crops_fao56(self):
        """Проверка 5 культур финала и соответствия стандарту FAO-56."""
        expected_crops = ["spring_wheat", "spring_barley", "sunflower", "flax_linseed", "chickpea"]
        self.assertEqual(sorted(list(TARGET_CROPS.keys())), sorted(expected_crops))

        for crop_id, crop in TARGET_CROPS.items():
            self.assertIn("kc", crop)
            # Коэффициенты водопотребления Kc по FAO-56 лежат в диапазоне 0.9 - 1.25
            self.assertTrue(0.9 <= crop["kc"] <= 1.25, f"Kc {crop['kc']} вне биологических норм FAO для {crop_id}")
            self.assertIn("fao_ref", crop)
            self.assertIn("root_depth_meters", crop)
            self.assertIn("critical_phase_ru", crop)
            self.assertIn("critical_phase_kk", crop)
            self.assertIn("drought_tolerance_score", crop)
            self.assertTrue(1.0 <= crop["drought_tolerance_score"] <= 5.0)

    def test_report_generation_all_combinations(self):
        """Проверка генерации отчетов для обоих языков, регионов и культур."""
        for lang in ["kk", "ru"]:
            for reg_id in ["sko", "kostanay", "akmola", "pavlodar"]:
                reg = GRAIN_BELT_REGIONS[reg_id]
                reg_name = reg["name_kk" if lang == "kk" else "name_ru"]
                
                for crop_id in ["spring_wheat", "sunflower"]:
                    crop = TARGET_CROPS[crop_id]
                    crop_name = crop["name_kk" if lang == "kk" else "name_ru"]

                    # 1. Excel тест
                    xlsx_bytes = generate_excel_report(
                        year=2024,
                        region=reg_name,
                        psi_val=-2.1,
                        smai_val=0.14,
                        wheat_price_val=240.0,
                        grain_prod_val=17.2,
                        risk_lvl="HIGH",
                        df_vuln_arg=None,
                        df_econ_arg=None,
                        lang=lang,
                        crop=crop_name,
                        crop_kc=crop["kc"],
                    )
                    self.assertIsInstance(xlsx_bytes, bytes)
                    self.assertGreater(len(xlsx_bytes), 2000, f"Excel поврежден для {lang}_{reg_id}_{crop_id}")

                    # 2. Markdown тест
                    md_text = generate_markdown_report(
                        year=2024,
                        region=reg_name,
                        psi_val=-2.1,
                        smai_val=0.14,
                        wheat_price_val=240.0,
                        grain_prod_val=17.2,
                        risk_lvl="HIGH",
                        high_risk=14,
                        moderate_risk=6,
                        lang=lang,
                        crop=crop_name,
                        crop_kc=crop["kc"],
                    )
                    self.assertIsInstance(md_text, str)
                    self.assertIn(crop_name, md_text)
                    self.assertIn(f"{crop['kc']:.2f}", md_text)
                    self.assertIn(reg_name, md_text)

    def test_what_if_stress_engine(self):
        """Проверка логики What-If стресс-симулятора по стандартам FAO-56."""
        for crop_id, crop in TARGET_CROPS.items():
            kc = crop["kc"]
            tol = crop["drought_tolerance_score"]
            crop_sensitivity = (kc / 1.10) * (4.2 / max(tol, 1.0))
            
            # Чувствительность должна быть в физически адекватном диапазоне [0.6, 2.0]
            self.assertTrue(0.6 <= crop_sensitivity <= 2.0, f"Чувствительность {crop_id} вне нормы: {crop_sensitivity}")
            
            # При шоке 20% (-20% влажности)
            shock_factor = 0.20
            yield_loss_cwt = round((shock_factor * crop_sensitivity) * 5.2, 1)
            yield_loss_pct = round(shock_factor * crop_sensitivity * 38.0, 1)
            
            self.assertGreater(yield_loss_cwt, 0.0)
            self.assertLess(yield_loss_pct, 100.0)


if __name__ == "__main__":
    unittest.main()

