import unittest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.elevator_logistics import (
    ALL_207_ELEVATORS,
    get_elevators_df,
    calculate_dynamic_silo_loads,
)
import pandas as pd


class TestElevatorLogisticsClustering(unittest.TestCase):
    """Тестирование реестра 207 ХПП и кластеризации (Step 5.1)."""

    def test_elevator_count_and_regions(self):
        """Проверка общего количества 207 ХПП и распределения по 4 областям."""
        self.assertEqual(len(ALL_207_ELEVATORS), 207, "Реестр должен содержать ровно 207 ХПП")
        
        df = get_elevators_df()
        self.assertEqual(len(df), 207)

        counts = df["region"].value_counts().to_dict()
        self.assertEqual(counts.get("sko"), 52, "СКО должна иметь 52 ХПП")
        self.assertEqual(counts.get("kostanay"), 65, "Костанайская область должна иметь 65 ХПП")
        self.assertEqual(counts.get("akmola"), 68, "Акмолинская область должна иметь 68 ХПП")
        self.assertEqual(counts.get("pavlodar"), 22, "Павлодарская область должна иметь 22 ХПП")

    def test_elevator_data_integrity(self):
        """Проверка целостности данных каждого ХПП."""
        df = get_elevators_df()

        # Проверка отсутствия пустых значений
        self.assertFalse(df["name"].isna().any(), "Имена ХПП не должны быть пустыми")
        self.assertFalse(df["lat"].isna().any(), "Широта не должна быть пустой")
        self.assertFalse(df["lon"].isna().any(), "Долгота не должна быть пустой")
        self.assertFalse(df["capacity_t"].isna().any(), "Вместимость не должна быть пустой")

        # Географические рамки зернового пояса Казахстана (lat: 50.0..55.6, lon: 60.0..80.0)
        self.assertTrue((df["lat"] >= 50.0).all() and (df["lat"] <= 55.6).all(), "Широта вне диапазона Казахстана")
        self.assertTrue((df["lon"] >= 60.0).all() and (df["lon"] <= 80.0).all(), "Долгота вне диапазона Казахстана")

        # Вместимость элеваторов (от 25 000 до 250 000 тонн)
        self.assertTrue((df["capacity_t"] >= 25000).all(), "Вместимость меньше 25 тыс тонн")
        self.assertTrue((df["capacity_t"] <= 250000).all(), "Вместимость превышает 250 тыс тонн")

        # Наличие ж/д тупиков (хотя бы 40% элеваторов с ж/д)
        rail_pct = df["railway_access"].mean()
        self.assertGreater(rail_pct, 0.40, "Доля элеваторов с ж/д доступом должна быть > 40%")

    def test_dynamic_what_if_silo_stress(self):
        """Проверка динамического стресс-пересчета заполненности (What-If)."""
        # Базовый расчет без стресса
        df_base = calculate_dynamic_silo_loads(shock_factor=0.0)
        base_mean_fill = df_base["current_fill_pct"].mean()

        # Стресс 40% при высокой чувствительности культуры (пшеница)
        df_stressed = calculate_dynamic_silo_loads(shock_factor=0.40, crop_sensitivity=1.20)
        stressed_mean_fill = df_stressed["current_fill_pct"].mean()

        self.assertLess(
            stressed_mean_fill,
            base_mean_fill,
            "При стрессе средняя заполненность элеваторов должна снижаться"
        )

        # Проверка статусов и цветов
        self.assertTrue(set(df_stressed["status_code"].unique()).issubset({"FULL", "OPTIMAL", "DEFICIT"}))
        self.assertTrue(set(df_stressed["color_hex"].unique()).issubset({"#34C759", "#FF9500", "#FF3B30"}))

    def test_parquet_sync(self):
        """Проверка экспорта и чтения датасета 207 ХПП."""
        df = calculate_dynamic_silo_loads()
        parquet_path = PROJECT_ROOT / "data" / "processed" / "silos_207.parquet"
        
        # Сохраняем датасет в parquet
        df.to_parquet(parquet_path)
        self.assertTrue(parquet_path.exists())

        # Читаем обратно
        df_read = pd.read_parquet(parquet_path)
        self.assertEqual(len(df_read), 207)
        self.assertIn("capacity_t", df_read.columns)
        self.assertIn("railway_access", df_read.columns)


if __name__ == "__main__":
    unittest.main()
