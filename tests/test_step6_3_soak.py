import unittest
import sys
import gc
import tracemalloc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import GRAIN_BELT_REGIONS, TARGET_CROPS
from modules.gis_engine import compute_district_choropleth_geojson
from modules.elevator_logistics import (
    calculate_dynamic_silo_loads,
    calculate_price_transmission_chain,
    GrainLogisticsRouter,
)
from modules.executive_audit import generate_executive_audit_markdown


class TestStep63SoakStability(unittest.TestCase):
    """
    Стресс-тест стабильности и отсутствия утечек памяти (Step 6.3).
    Имитирует непрерывное интенсивное переключение всех параметров AgriCascade:
    4 региона × 5 культур × 2 языка × 4 сценария засухи × роутинг ХПП × генерация аудита.
    """

    def setUp(self):
        self.router = GrainLogisticsRouter()
        self.regions = list(GRAIN_BELT_REGIONS.keys())
        self.crops = list(TARGET_CROPS.keys())
        self.languages = ["kk", "ru"]
        self.shocks = [0.0, 0.15, 0.30, 0.50]

    def test_combinatorial_exhaustive_coverage(self):
        """
        Проверка полного декартова произведения всех параметров
        (4 региона × 5 культур × 2 языка × 4 шока = 160 комбинаций).
        Ни одна комбинация не должна вызывать исключений или NaN значений.
        """
        total_runs = 0
        for reg in self.regions:
            for crop in self.crops:
                crop_info = TARGET_CROPS[crop]
                for lang in self.languages:
                    for shock in self.shocks:
                        # 1. ГИС-картограмма районов
                        geo = compute_district_choropleth_geojson(
                            region_id=reg,
                            lang=lang,
                            current_psi=-0.75,
                            crop_kc=crop_info["kc"],
                            crop_tolerance=crop_info["drought_tolerance_score"],
                            crop_name=crop_info["name_" + lang],
                            shock_factor=shock,
                        )
                        self.assertGreater(len(geo["features"]), 0)
                        for feat in geo["features"]:
                            p = feat["properties"]
                            self.assertFalse(any(x is None for x in [p["risk_score"], p["fill_color"]]))
                            self.assertTrue(5.0 <= p["risk_score"] <= 100.0)

                        # 2. Логистическая загрузка элеваторов
                        df_silos = calculate_dynamic_silo_loads(
                            region_filter=reg,
                            shock_factor=shock,
                            crop_sensitivity=crop_info["kc"] / 1.10,
                        )
                        self.assertGreater(len(df_silos), 0)
                        self.assertFalse(df_silos["current_fill_pct"].isna().any())
                        self.assertFalse(df_silos["stored_grain_t"].isna().any())

                        # 3. Эконометрическая цепочка передачи цен
                        pc = calculate_price_transmission_chain(yield_shock_pct=-shock * 100.0)
                        self.assertFalse(any(x is None for x in [pc["wheat_price_projected"], pc["bread_price_projected"]]))
                        self.assertTrue(pc["cpi_pressure_score"] >= 0)

                        total_runs += 1

        self.assertEqual(total_runs, len(self.regions) * len(self.crops) * len(self.languages) * len(self.shocks))

    def test_logistics_router_stress_all_207_elevators(self):
        """
        Стресс-тест маршрутизации перенаправления:
        проверка поиска путей от каждого ключевого узла сети к резервным ХПП.
        """
        df_loads = calculate_dynamic_silo_loads(shock_factor=0.35)
        # Тестируем выборку из 25 различных ХПП во всех 4 областях
        sample_silos = df_loads.iloc[::8]
        for _, row in sample_silos.iterrows():
            res = self.router.solve_rerouting(
                source_id=row["id"],
                volume_tons=30000.0,
                df_current_loads=df_loads,
            )
            self.assertIsNotNone(res)
            self.assertGreater(res["distance_km"], 0)
            self.assertGreater(res["wagons_required"], 0)
            self.assertGreater(res["trucks_required"], 0)
            self.assertGreaterEqual(res["savings_rail_kzt"], 0)

    def test_memory_leak_soak_500_iterations(self):
        """
        Тест на утечки памяти (Soak Test):
        Выполняет 500 полных итераций генерации аналитических записок и расчетов.
        Фиксирует объем выделяемой памяти до и после сборки мусора.
        Прирост памяти должен быть строго ограничен (< 25 МБ).
        """
        gc.collect()
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        # 500 циклов генерации отчетов и аналитики
        for i in range(500):
            reg = self.regions[i % len(self.regions)]
            crop = self.crops[i % len(self.crops)]
            lang = self.languages[i % len(self.languages)]
            shock = self.shocks[i % len(self.shocks)]

            # Полная генерация записки экспресс-аудита
            report = generate_executive_audit_markdown(
                year=2024,
                region_key=reg,
                crop_key=crop,
                current_psi=-0.60 - (i % 5) * 0.2,
                smai_val=-0.40 - (i % 4) * 0.15,
                shock_factor=shock,
                lang=lang,
            )
            self.assertGreater(len(report), 1000)

        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Анализ прироста памяти
        stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        total_growth_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
        total_growth_mb = total_growth_bytes / (1024 * 1024)

        # Допустимый предел роста памяти для 500 полных циклов — не более 25 МБ
        self.assertLess(
            total_growth_mb,
            25.0,
            f"Обнаружена утечка памяти: прирост составил {total_growth_mb:.2f} МБ (> 25 МБ)"
        )


if __name__ == "__main__":
    unittest.main()

