import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.elevator_logistics import (
    GrainLogisticsRouter,
    calculate_dynamic_silo_loads,
    KTZ_GRAIN_HOPPER_CAPACITY_T,
    TRUCK_GRAIN_CAPACITY_T,
    RAIL_TARIFF_KZT_PER_TKM,
    ROAD_TARIFF_KZT_PER_TKM,
)


class TestGrainLogisticsRerouting(unittest.TestCase):
    """Тестирование логистического роутинга и арбитража перенаправления зерна (Step 5.2)."""

    def setUp(self):
        self.router = GrainLogisticsRouter()
        self.df_loads = calculate_dynamic_silo_loads(shock_factor=0.35, crop_sensitivity=1.15)

    def test_logistics_graph_structure(self):
        """Проверка структуры и связности графа NetworkX."""
        g = self.router.graph
        self.assertEqual(len(g.nodes), 207, "Граф должен содержать ровно 207 узлов-элеваторов")
        self.assertGreater(len(g.edges), 100, "Граф должен иметь достаточную связанность ребер")

    def test_haversine_distance(self):
        """Проверка точности ортодромического расстояния между городами."""
        # Петропавловск (54.867, 69.131) <-> Кокшетау (53.288, 69.395) ~ 176 км
        dist = self.router.haversine_distance(54.867, 69.131, 53.288, 69.395)
        self.assertGreater(dist, 160.0)
        self.assertLess(dist, 195.0)

    def test_rerouting_solver(self):
        """Проверка поиска резервного элеватора и расчета параметров перевозки."""
        source_id = "sko_hpp_01"  # Петропавловский элеватор №1
        volume = 20000.0          # 20 тыс. тонн

        res = self.router.solve_rerouting(
            source_id=source_id,
            volume_tons=volume,
            df_current_loads=self.df_loads
        )

        self.assertIsNotNone(res)
        self.assertNotEqual(res["source_id"], res["target_id"], "Целевой элеватор должен отличаться от исходного")
        self.assertGreater(res["distance_km"], 0.0)
        self.assertGreater(len(res["path_coords"]), 1, "Должна быть полилиния пути минимум из 2 точек")
        self.assertGreater(res["target_free_cap_t"], 0.0, "Целевой элеватор должен иметь свободную емкость")

        # Проверка потребности в подвижном составе
        expected_wagons = int(-(-volume // KTZ_GRAIN_HOPPER_CAPACITY_T))
        expected_trucks = int(-(-volume // TRUCK_GRAIN_CAPACITY_T))
        self.assertEqual(res["wagons_required"], expected_wagons)
        self.assertEqual(res["trucks_required"], expected_trucks)

        # Проверка тарифов и экономии
        self.assertGreater(res["cost_road_kzt"], res["cost_rail_kzt"], "Автодоставка дороже ж/д доставки КТЖ")
        self.assertEqual(res["savings_rail_kzt"], res["cost_road_kzt"] - res["cost_rail_kzt"])
        self.assertGreater(res["savings_usd"], 0)

        # Проверка эко-эффекта
        self.assertGreater(res["co2_saved_t"], 0, "Ж/д перевозка должна сокращать выбросы CO2")


if __name__ == "__main__":
    unittest.main()

