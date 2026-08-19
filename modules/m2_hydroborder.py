"""
Daryn — modules/m2_hydroborder.py
====================================
Модуль 2: HydroBorder — трансграничный водный риск

ЛОГИКА:
  1. Забирает ежедневные данные о влажности почвы (SMAP) в верховьях рек:
     - р. Или (Синьцзян → оз. Балхаш, трансграничная с Китаем)
     - р. Сырдарья (Кыргызстан → Казахстан)
     - р. Тобол (Россия → Казахстан / СКО)
  2. Строит 10-летнюю норму влажности (μ ± σ) по каждому бассейну
  3. Вычисляет Soil Moisture Anomaly Index (SMAI):
       SMAI = (SM_current_90d_avg - μ_baseline) / σ_baseline
  4. Проверяет predictive correlation:
       SMAI(t) →? wheat_price(t + 90 days)

ГИПОТЕЗА:
  Дефицит влажности в верховьях рек →
  снижение ирригационного потенциала →
  стресс посевов →
  падение урожайности через 90 дней →
  рост цен на пшеницу через 120–180 дней
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ee
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from loguru import logger
from typing import Optional

from pipeline.gee_client import gee
from config.settings import (
    ILI_RIVER_UPSTREAM,
    SYRDARYA_UPSTREAM,
    SMAP_COLLECTION,
    SMAP_BAND_SURFACE,
    SMAP_ROLLING_WINDOW_DAYS,
    SMAP_ANOMALY_THRESHOLD,
    BASELINE_YEARS,
    CURRENT_YEAR,
    ANALYSIS_START_YEAR,
    ANALYSIS_END_YEAR,
    PROCESSED_DIR,
)

# Добавляем Тобол (верховье, Россия → СКО)
TOBOL_UPSTREAM = [
    [62.0, 54.0], [65.0, 54.0], [65.0, 56.0], [62.0, 56.0], [62.0, 54.0]
]

WATER_BASINS = {
    "ili": {
        "name": "Река Или (верховье, Синьцзян → Балхаш)",
        "coords": ILI_RIVER_UPSTREAM,
        "transboundary_from": "Китай",
        "feeds_region": "Алматинская, юг Казахстана",
        "risk_weight": 0.45,
    },
    "syrdarya": {
        "name": "Сырдарья (верховье, Кыргызстан)",
        "coords": SYRDARYA_UPSTREAM,
        "transboundary_from": "Кыргызстан/Таджикистан",
        "feeds_region": "Южный Казахстан (орошение)",
        "risk_weight": 0.35,
    },
    "tobol": {
        "name": "Тобол (верховье, Россия → СКО)",
        "coords": TOBOL_UPSTREAM,
        "transboundary_from": "Россия",
        "feeds_region": "Северо-Казахстанская область",
        "risk_weight": 0.20,
    },
}


class HydroBorderModule:
    """
    Модуль 2: Мониторинг трансграничного водного стресса через SMAP.
    """

    OUTPUT_SMAP_PATH = PROCESSED_DIR / "smap_timeseries.parquet"
    OUTPUT_HYDRO_PATH = PROCESSED_DIR / "hydroborder.parquet"

    # ─────────────────────────────────────────────
    # Сбор данных SMAP
    # ─────────────────────────────────────────────

    def collect_smap_series(
        self, basin_id: str, year_start: int, year_end: int
    ) -> pd.DataFrame:
        """
        Собирает ежедневную влажность почв (SMAP) для конкретного бассейна.

        Args:
            basin_id: ключ из WATER_BASINS ('ili', 'syrdarya', 'tobol')
            year_start, year_end: диапазон лет

        Returns:
            DataFrame: [date, soil_moisture, basin]
        """
        basin = WATER_BASINS[basin_id]
        geometry = ee.Geometry.Polygon(basin["coords"])

        logger.info(f"SMAP: {basin['name']} ({year_start}–{year_end})")

        all_series = []
        for year in range(year_start, year_end + 1):
            collection = gee.get_smap(
                geometry=geometry,
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
            )

            series = gee.extract_time_series(
                collection=collection,
                geometry=geometry,
                band=SMAP_BAND_SURFACE,
                scale=9000,  # SMAP нативное разрешение 9 км
            )
            if series:
                all_series.extend(series)

        if not all_series:
            logger.warning(f"Нет данных SMAP для {basin_id}")
            return pd.DataFrame()

        df = pd.DataFrame(all_series)
        df.rename(columns={"value": "soil_moisture"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df["basin"] = basin_id
        df["basin_name"] = basin["name"]

        # Удаляем физически невозможные значения
        df = df[(df["soil_moisture"] >= 0) & (df["soil_moisture"] <= 1.0)]

        return df.sort_values("date").reset_index(drop=True)

    def collect_all_basins(self, force_refresh: bool = False) -> pd.DataFrame:
        """Собирает SMAP для всех трёх бассейнов, кэширует."""
        if self.OUTPUT_SMAP_PATH.exists() and not force_refresh:
            logger.info("SMAP: загрузка из кэша")
            return pd.read_parquet(self.OUTPUT_SMAP_PATH)

        all_dfs = []
        for basin_id in WATER_BASINS:
            df = self.collect_smap_series(
                basin_id,
                ANALYSIS_START_YEAR,
                ANALYSIS_END_YEAR,
            )
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            raise RuntimeError("Не удалось собрать данные SMAP ни по одному бассейну")

        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_parquet(self.OUTPUT_SMAP_PATH)
        logger.success(f"SMAP данные сохранены → {self.OUTPUT_SMAP_PATH}")
        return df_all

    # ─────────────────────────────────────────────
    # Вычисление SMAI (Soil Moisture Anomaly Index)
    # ─────────────────────────────────────────────

    def compute_smai(self, df_smap: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисляет Soil Moisture Anomaly Index для каждого бассейна.

        Алгоритм:
        1. Вычисляем 90-дневную скользящую среднюю (сглаживание)
        2. Для каждого месяца строим baseline норму (μ, σ) за 2015–2023
        3. SMAI = (SM_90d_avg - μ_month) / σ_month

        Returns:
            DataFrame: [date, basin, soil_moisture, sm_rolling_90d, smai, anomaly_level]
        """
        result_dfs = []

        for basin_id in df_smap["basin"].unique():
            df_basin = df_smap[df_smap["basin"] == basin_id].copy()
            df_basin = df_basin.sort_values("date").set_index("date")

            # 90-дневная скользящая средняя
            df_basin["sm_rolling_90d"] = (
                df_basin["soil_moisture"]
                .rolling(window=SMAP_ROLLING_WINDOW_DAYS, min_periods=10)
                .mean()
            )

            # Месяц для группировки
            df_basin["month"] = df_basin.index.month
            df_basin["year"] = df_basin.index.year

            # Baseline норма по месяцам (только BASELINE_YEARS)
            baseline_mask = df_basin["year"].isin(BASELINE_YEARS)
            monthly_stats = (
                df_basin[baseline_mask]
                .groupby("month")["sm_rolling_90d"]
                .agg(sm_mean="mean", sm_std="std")
                .reset_index()
            )
            monthly_stats["sm_std"] = monthly_stats["sm_std"].fillna(0.02).clip(lower=0.01)

            # Объединяем и считаем SMAI
            df_basin = df_basin.reset_index()
            df_basin = df_basin.merge(monthly_stats, on="month", how="left")
            df_basin["smai"] = (
                (df_basin["sm_rolling_90d"] - df_basin["sm_mean"]) / df_basin["sm_std"]
            )

            # Уровень аномалии
            conditions = [
                df_basin["smai"] < -2.0,
                df_basin["smai"] < -1.0,
                df_basin["smai"] < -0.5,
                df_basin["smai"] > 2.0,
                df_basin["smai"] > 1.0,
            ]
            choices = ["SEVERE_DRY", "DRY", "SLIGHTLY_DRY", "WET", "SLIGHTLY_WET"]
            df_basin["anomaly_level"] = np.select(conditions, choices, default="NORMAL")

            result_dfs.append(df_basin)

        df_result = pd.concat(result_dfs, ignore_index=True)
        return df_result

    # ─────────────────────────────────────────────
    # Корреляционный анализ SMAI → Цены пшеницы
    # ─────────────────────────────────────────────

    def analyze_hydro_price_correlation(
        self,
        df_smai: pd.DataFrame,
        df_economics: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Проверяет гипотезу:
        SMAI(t) при дефиците воды предсказывает рост цен на пшеницу через ~90 дней.

        Использует годовую агрегацию (т.к. экономические данные — годовые).

        Returns:
            dict с коэффициентами корреляции Пирсона для каждого бассейна
        """
        results = {}

        # Загружаем экономику если не передана
        if df_economics is None:
            econ_path = PROCESSED_DIR / "economics.parquet"
            if econ_path.exists():
                df_economics = pd.read_parquet(econ_path)
            else:
                logger.warning("Экономические данные не найдены. Пропускаем корреляцию.")
                return results

        # Годовой SMAI (среднее за год)
        df_smai["year"] = pd.to_datetime(df_smai["date"]).dt.year
        annual_smai = df_smai.groupby(["year", "basin"])["smai"].mean().reset_index()

        for basin_id in WATER_BASINS:
            basin_data = annual_smai[annual_smai["basin"] == basin_id].copy()
            merged = basin_data.merge(
                df_economics[["year", "wheat_price_usd_t", "wheat_price_yoy_pct"]],
                on="year",
                how="inner",
            )

            if len(merged) < 5:
                results[basin_id] = {"pearson_r": None, "p_value": None}
                continue

            r, p = pearsonr(merged["smai"], merged["wheat_price_yoy_pct"].fillna(0))
            results[basin_id] = {
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 4),
                "n": len(merged),
                "interpretation": (
                    "Значимая обратная корреляция (↓ влага → ↑ цена)"
                    if r < -0.4 and p < 0.1
                    else "Умеренная связь" if abs(r) > 0.3
                    else "Слабая связь"
                ),
            }
            logger.info(
                f"  {basin_id}: r={r:.3f}, p={p:.3f} — {results[basin_id]['interpretation']}"
            )

        return results

    # ─────────────────────────────────────────────
    # Основной запуск
    # ─────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Полный пайплайн HydroBorder."""
        logger.info("=" * 52)
        logger.info("МОДУЛЬ 2: HydroBorder (Трансграничный водный риск)")
        logger.info("=" * 52)

        # 1. Сбор SMAP
        df_smap = self.collect_all_basins()

        # 2. Вычисление SMAI
        df_smai = self.compute_smai(df_smap)

        # 3. Корреляция с ценами
        correlations = self.analyze_hydro_price_correlation(df_smai)

        # 4. Сводный риск-индекс (взвешенный SMAI по бассейнам)
        df_current = df_smai[df_smai["year"] == CURRENT_YEAR].copy()
        hydro_risk_score = 0.0
        for basin_id, basin_info in WATER_BASINS.items():
            basin_current = df_current[df_current["basin"] == basin_id]
            if not basin_current.empty:
                avg_smai = basin_current["smai"].mean()
                if not np.isnan(avg_smai):
                    hydro_risk_score += basin_info["risk_weight"] * max(0, -avg_smai)

        logger.info(f"\n{'─' * 52}")
        logger.info(f"HydroBorder Risk Score ({CURRENT_YEAR}): {hydro_risk_score:.3f}")
        logger.info(f"{'─' * 52}")
        for basin_id, corr in correlations.items():
            logger.info(f"  {basin_id}: r={corr.get('pearson_r')}, p={corr.get('p_value')}")
        logger.info(f"{'─' * 52}\n")

        # 5. Сохраняем
        df_smai.to_parquet(self.OUTPUT_HYDRO_PATH)
        logger.success(f"HydroBorder результаты сохранены → {self.OUTPUT_HYDRO_PATH}")

        return df_smai


if __name__ == "__main__":
    module = HydroBorderModule()
    result = module.run()
    print(result.tail(20))
