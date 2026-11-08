import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.executive_audit import generate_executive_audit_markdown


class TestExecutiveAudit(unittest.TestCase):
    """Тестирование генератора институциональной аналитической записки (Step 6.2)."""

    def test_executive_audit_all_four_regions_kk(self):
        """Проверка формирования записки на казахском языке для 4 зерновых областей."""
        for reg_key in ["sko", "kostanay", "akmola", "pavlodar"]:
            report_kk = generate_executive_audit_markdown(
                year=2024,
                region_key=reg_key,
                crop_key="wheat_spring",
                current_psi=-0.62,
                smai_val=-0.45,
                shock_factor=0.20,
                lang="kk",
            )
            self.assertIsInstance(report_kk, str)
            self.assertGreater(len(report_kk), 1500, f"Отчет для {reg_key} слишком короткий")
            
            # Проверка заголовков и институциональных блоков
            self.assertIn("ҚАЗАҚСТАН РЕСПУБЛИКАСЫ АУЫЛ ШАРУАШЫЛЫҒЫ МИНИСТРЛІГІ", report_kk)
            self.assertIn("ЖЕДЕЛ ТАЛДАМАЛЫҚ АНЫҚТАМА", report_kk)
            self.assertIn("1. Спутниктік және климаттық мониторинг", report_kk)
            self.assertIn("2. Аудандардың микроаймақтық осалдығы", report_kk)
            self.assertIn("3. Элеваторлық инфрақұрылым және логистика балансы", report_kk)
            self.assertIn("4. Баға берілу тізбегі және азық-түлік қауіпсіздігі", report_kk)
            self.assertIn("5. Әкімдік пен Ауыл шаруашылығы министрлігі үшін ресми шешімдер", report_kk)

            # Проверка ключевых терминов
            self.assertIn("PhenoShift Index", report_kk)
            self.assertIn("NASA SMAP", report_kk)
            self.assertIn("ҚТЖ", report_kk)
            self.assertIn("вагон-астық тасығыш", report_kk)
            self.assertIn("USD", report_kk)

    def test_executive_audit_all_four_regions_ru(self):
        """Проверка формирования записки на русском языке для 4 зерновых областей."""
        for reg_key in ["sko", "kostanay", "akmola", "pavlodar"]:
            report_ru = generate_executive_audit_markdown(
                year=2024,
                region_key=reg_key,
                crop_key="wheat_spring",
                current_psi=-0.85,
                smai_val=-0.60,
                shock_factor=0.30,
                lang="ru",
            )
            self.assertIsInstance(report_ru, str)
            self.assertGreater(len(report_ru), 1500)

            # Проверка институциональных блоков
            self.assertIn("МИНИСТЕРСТВО СЕЛЬСКОГО ХОЗЯЙСТВА РЕСПУБЛИКИ КАЗАХСТАН", report_ru)
            self.assertIn("ОПЕРАТИВНАЯ АНАЛИТИЧЕСКАЯ ЗАПИСКА", report_ru)
            self.assertIn("1. Спутниковая космическая диагностика", report_ru)
            self.assertIn("2. Микрозональная уязвимость районов", report_ru)
            self.assertIn("3. Элеваторная инфраструктура и баланс логистики", report_ru)
            self.assertIn("4. Цепочка передачи цен и продовольственная безопасность", report_ru)
            self.assertIn("5. Комплекс управленческих и агрономических мер", report_ru)

            # Проверка метрик
            self.assertIn("вагонов-зерновозов КТЖ", report_ru)
            self.assertIn("Экономия ж/д тарифа", report_ru)
            self.assertIn("хлеб", report_ru.lower())

    def test_multi_crop_differentiation(self):
        """Проверка адаптации отчета под разные культуры (ячмень, подсолнечник, лен)."""
        rep_barley = generate_executive_audit_markdown(
            year=2024, region_key="sko", crop_key="barley_spring", lang="ru"
        )
        rep_sun = generate_executive_audit_markdown(
            year=2024, region_key="sko", crop_key="sunflower", lang="ru"
        )
        self.assertIn("Ячмень", rep_barley)
        self.assertIn("Подсолнечник", rep_sun)
        self.assertNotEqual(rep_barley, rep_sun)

    def test_what_if_stress_impact_in_report(self):
        """Проверка того, что при шоке засухи отчет динамически фиксирует потери и дефицит."""
        base_rep = generate_executive_audit_markdown(
            year=2024, region_key="kostanay", shock_factor=0.0, lang="ru"
        )
        stress_rep = generate_executive_audit_markdown(
            year=2024, region_key="kostanay", shock_factor=0.35, lang="ru"
        )
        # В стресс-отчете должно быть падение урожайности
        self.assertIn("-35.0%", stress_rep)
        self.assertNotEqual(base_rep, stress_rep)


if __name__ == "__main__":
    unittest.main()
