"""
Daryn — modules/m1_phenoshift.py
==================================
Модуль 1: PhenoShift Index (PSI)
Алгоритм выявления скрытого фенологического стресса растений.

МЕТРИКА:
  PSI(year, polygon) = Σ [ (NDVI_current(DOY) - μ_norm(DOY)) / σ_norm(DOY) ]
                        за DOY ∈ [вегетационный период]

ИНТЕРПРЕТАЦИЯ:
  PSI > +1.5  →  Опережение нормы (ранний урожай, риск заморозков)
  -1.5 < PSI < +1.5  →  Норма
  PSI < -1.5  →  Отставание (засуха, холодный старт, риск недобора)

ВАЛИДАЦИЯ:
  2021 год → засуха в Казахстане → должен показать PSI << -1.5
  2017 год → рекордный урожай → должен показать PSI > 0
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ee
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from loguru import logger
from typing import Optional

from pipeline.gee_client import gee
from config.settings import (
    SKO_POLYGON,
    BASELINE_YEARS,
    CURRENT_YEAR,
    ANALYSIS_START_YEAR,
    VEGETATION_DOY_START,
    VEGETATION_DOY_END,
    PSI_THRESHOLD_HIGH,
    PSI_THRESHOLD_LOW,
    MAX_CLOUD_COVER,
    PROCESSED_DIR,
    NDVI_SCALE,
)


class PhenoShiftModule:
    """
    Модуль 1: Вычисляет PhenoShift Index для заданного полигона.

    Пайплайн:
    1. Собирает NDVI time series за 10 лет (baseline)
    2. Для каждого DOY строит норму (μ, σ)
    3. Сравнивает текущий год с нормой → PSI
    4. Генерирует отчёт с риск-оценкой
    """

    OUTPUT_NDVI_PATH = PROCESSED_DIR / "ndvi_timeseries.parquet"
    OUTPUT_PSI_PATH = PROCESSED_DIR / "phenoshift.parquet"

    def __init__(self, polygon_coords: Optional[list] = None):
        """
        Args:
            polygon_coords: Список [lon, lat] пар. По умолчанию — СКО.
        """
        self.polygon_coords = polygon_coords or SKO_POLYGON
        self.geometry = ee.Geometry.Polygon(self.polygon_coords)

    # ─────────────────────────────────────────────
    # Шаг 1: Сбор NDVI Time Series
    # ─────────────────────────────────────────────

    def collect_ndvi_series(self, year: int) -> pd.DataFrame:
        """
        Собирает временной ряд NDVI для полигона за указанный год.
        Использует Sentinel-2 (2015+).

        Returns:
            DataFrame: [date, ndvi, doy, year]
        """
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        logger.info(f"Сбор NDVI за {year}...")

        # Получаем коллекцию Sentinel-2
        collection = gee.get_sentinel2(
            geometry=self.geometry,
            start_date=start,
            end_date=end,
            max_cloud=MAX_CLOUD_COVER,
            apply_cloud_mask=True,
        ).map(gee.add_ndvi_s2)

        # Извлекаем временной ряд
        series = gee.extract_time_series(
            collection=collection,
            geometry=self.geometry,
            band="NDVI",
            scale=NDVI_SCALE,
        )

        if not series:
            logger.warning(f"Нет данных NDVI за {year}")
            return pd.DataFrame(columns=["date", "ndvi", "doy", "year"])

        df = pd.DataFrame(series)
        df.rename(columns={"value": "ndvi"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df["doy"] = df["date"].dt.dayofyear
        df["year"] = year

        # Удаляем выбросы (NDVI вне [-0.1, 1.0] — артефакты)
        df = df[(df["ndvi"] >= -0.1) & (df["ndvi"] <= 1.0)]

        return df.sort_values("doy").reset_index(drop=True)

    def collect_baseline_ndvi(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Собирает NDVI за все базовые годы (2015–2023).
        Результат кэшируется в parquet для повторного использования.

        Returns:
            DataFrame: [date, ndvi, doy, year] за все baseline-годы
        """
        if self.OUTPUT_NDVI_PATH.exists() and not force_refresh:
            logger.info("NDVI baseline: загрузка из кэша")
            return pd.read_parquet(self.OUTPUT_NDVI_PATH)

        all_series = []
        for year in BASELINE_YEARS:
            df_year = self.collect_ndvi_series(year)
            if not df_year.empty:
                all_series.append(df_year)
                logger.success(f"  ✓ {year}: {len(df_year)} точек")

        if not all_series:
            raise RuntimeError("Не удалось собрать NDVI ни за один год")

        df_all = pd.concat(all_series, ignore_index=True)
        df_all.to_parquet(self.OUTPUT_NDVI_PATH)
        logger.success(f"NDVI baseline сохранён → {self.OUTPUT_NDVI_PATH}")
        return df_all

    # ─────────────────────────────────────────────
    # Шаг 2: Построение климатической нормы
    # ─────────────────────────────────────────────

    def build_phenological_norm(self, df_baseline: pd.DataFrame) -> pd.DataFrame:
        """
        Для каждого DOY вычисляет μ (среднее) и σ (стд. откл.) NDVI за baseline-годы.
        Применяет сглаживание Savitzky-Golay для получения плавной фенологической кривой.

        Returns:
            DataFrame: [doy, ndvi_mean, ndvi_std, ndvi_upper, ndvi_lower]
                       за DOY 1–365
        """
        logger.info("Построение фенологической нормы (μ ± σ)...")

        # Фильтр: только вегетационный период
        df_veg = df_baseline[
            (df_baseline["doy"] >= VEGETATION_DOY_START) &
            (df_baseline["doy"] <= VEGETATION_DOY_END)
        ].copy()

        # Группировка по DOY
        norm = df_veg.groupby("doy")["ndvi"].agg(
            ndvi_mean="mean",
            ndvi_std="std",
            ndvi_count="count",
        ).reset_index()

        # Заполняем пропущенные DOY интерполяцией
        full_doy = pd.DataFrame({"doy": range(VEGETATION_DOY_START, VEGETATION_DOY_END + 1)})
        norm = full_doy.merge(norm, on="doy", how="left")
        norm["ndvi_mean"] = norm["ndvi_mean"].interpolate(method="linear").bfill().ffill()
        norm["ndvi_std"] = norm["ndvi_std"].interpolate(method="linear").bfill().ffill().fillna(0.05)

        # Сглаживание Savitzky-Golay (убираем шум, сохраняем пики)
        window = min(21, len(norm) // 4 * 2 + 1)  # нечётное окно
        if len(norm) >= window:
            norm["ndvi_mean"] = savgol_filter(norm["ndvi_mean"], window, polyorder=3)
            norm["ndvi_std"] = savgol_filter(norm["ndvi_std"], window, polyorder=3)

        norm["ndvi_std"] = norm["ndvi_std"].abs().clip(lower=0.02)  # σ не может быть 0
        norm["ndvi_upper"] = norm["ndvi_mean"] + norm["ndvi_std"]
        norm["ndvi_lower"] = norm["ndvi_mean"] - norm["ndvi_std"]

        logger.success(f"Норма построена: DOY {VEGETATION_DOY_START}–{VEGETATION_DOY_END}")
        return norm

    # ─────────────────────────────────────────────
    # Шаг 3: Вычисление PhenoShift Index
    # ─────────────────────────────────────────────

    def compute_psi(
        self,
        df_norm: pd.DataFrame,
        df_current: pd.DataFrame,
    ) -> dict:
        """
        Вычисляет PhenoShift Index для текущего года относительно нормы.

        PSI = Σ [ (NDVI_current(DOY) - μ_norm(DOY)) / σ_norm(DOY) ]
              нормированное на длину вегетационного периода

        Args:
            df_norm: норма из build_phenological_norm()
            df_current: NDVI текущего года из collect_ndvi_series()

        Returns:
            dict с PSI, риском, задержкой в днях и деталями
        """
        # Оставляем только вегетационный период
        df_cur = df_current[
            (df_current["doy"] >= VEGETATION_DOY_START) &
            (df_current["doy"] <= VEGETATION_DOY_END)
        ].copy()

        if df_cur.empty:
            return {"psi": None, "risk": "NO_DATA", "delay_days": 0}

        # Объединяем с нормой
        merged = df_cur.merge(df_norm[["doy", "ndvi_mean", "ndvi_std"]], on="doy", how="inner")

        if len(merged) < 5:
            logger.warning("Недостаточно точек для PSI (< 5 совпадений DOY)")
            return {"psi": None, "risk": "INSUFFICIENT_DATA", "delay_days": 0}

        # Вычисляем z-score для каждого наблюдения
        merged["z_score"] = (merged["ndvi"] - merged["ndvi_mean"]) / merged["ndvi_std"]

        # PSI = нормированная сумма z-scores
        psi = float(merged["z_score"].mean())

        # ─── Оценка задержки фенологии ───
        # Находим дату пика NDVI в текущем году и в норме
        peak_current_doy = int(df_cur.loc[df_cur["ndvi"].idxmax(), "doy"])
        peak_norm_doy = int(df_norm.loc[df_norm["ndvi_mean"].idxmax(), "doy"])
        delay_days = peak_current_doy - peak_norm_doy

        # ─── Риск-оценка ───
        if psi > PSI_THRESHOLD_HIGH:
            risk = "HIGH_EARLY"
            risk_ru = "Опережение нормы (риск заморозков при раннем колошении)"
        elif psi < PSI_THRESHOLD_LOW:
            risk = "HIGH_LATE"
            risk_ru = "Отставание от нормы (риск засухи / недобора урожая)"
        elif psi < -1.0:
            risk = "MEDIUM_LATE"
            risk_ru = "Умеренное отставание вегетации"
        elif psi > 1.0:
            risk = "MEDIUM_EARLY"
            risk_ru = "Умеренное опережение вегетации"
        else:
            risk = "LOW"
            risk_ru = "Норма — существенных отклонений нет"

        result = {
            "psi": round(psi, 4),
            "risk": risk,
            "risk_description": risk_ru,
            "delay_days": delay_days,
            "peak_doy_current": peak_current_doy,
            "peak_doy_norm": peak_norm_doy,
            "n_observations": len(merged),
            "ndvi_mean_current": round(float(merged["ndvi"].mean()), 4),
            "ndvi_mean_norm": round(float(merged["ndvi_mean"].mean()), 4),
            "z_scores": merged[["doy", "z_score", "ndvi", "ndvi_mean"]].to_dict("records"),
        }

        return result

    # ─────────────────────────────────────────────
    # Детектор сдвига (текстовый вывод)
    # ─────────────────────────────────────────────

    @staticmethod
    def format_phenoshift_report(psi_result: dict, year: int, region_name: str = "СКО") -> str:
        """Форматирует человекочитаемый отчёт по PhenoShift."""
        psi = psi_result.get("psi")
        if psi is None:
            return f"⚠️  Данные за {year} недоступны для анализа"

        delay = psi_result["delay_days"]
        risk = psi_result["risk_description"]

        direction = "опережает" if delay < 0 else "отстаёт от"
        abs_delay = abs(delay)

        report = f"""
╔══════════════════════════════════════════════════════╗
║         PHENOSHIFT INDEX — Daryn Агро-Разведчик      ║
╠══════════════════════════════════════════════════════╣
║  Регион:    {region_name:<42} ║
║  Год:       {year:<42} ║
╠══════════════════════════════════════════════════════╣
║  PSI:       {psi:+.4f}                                    ║
║  Пик NDVI   {direction} норму на {abs_delay} дней              ║
╠══════════════════════════════════════════════════════╣
║  РИСК: {risk:<46} ║
╚══════════════════════════════════════════════════════╝
        """.strip()

        return report

    # ─────────────────────────────────────────────
    # Основной запуск
    # ─────────────────────────────────────────────

    def run(self, target_year: Optional[int] = None) -> pd.DataFrame:
        """
        Полный пайплайн:
        1. Собирает NDVI baseline (10 лет)
        2. Строит норму
        3. Собирает данные текущего года
        4. Вычисляет PSI + риск
        5. Сохраняет результат

        Args:
            target_year: год для анализа. По умолчанию CURRENT_YEAR из settings.

        Returns:
            DataFrame с PSI для каждого базового года + текущего (для графиков)
        """
        year = target_year or CURRENT_YEAR

        logger.info("=" * 52)
        logger.info("МОДУЛЬ 1: PhenoShift Index")
        logger.info("=" * 52)

        # Шаг 1: Baseline NDVI
        df_baseline_raw = self.collect_baseline_ndvi()
        
        # Защита от утечки данных (Data Leakage)
        df_baseline = df_baseline_raw[df_baseline_raw["year"] < year]
        if df_baseline.empty:
            logger.warning(f"Недостаточно исторических данных до {year} года. Используем все доступные базовые годы.")
            df_baseline = df_baseline_raw

        # Шаг 2: Норма
        df_norm = self.build_phenological_norm(df_baseline)

        # Шаг 3: Текущий год
        df_current = self.collect_ndvi_series(year)

        # Шаг 4: PSI
        psi_result = self.compute_psi(df_norm, df_current)

        # Печатаем отчёт
        report = self.format_phenoshift_report(psi_result, year)
        print(report)
        logger.info(f"PSI({year}) = {psi_result.get('psi')}")

        # Шаг 5: Рассчитываем PSI для всех годов (для графика тренда)
        all_psi_rows = []
        for yr in BASELINE_YEARS + [year]:
            df_yr = (
                df_current if yr == year
                else df_baseline[df_baseline["year"] == yr]
            )
            psi_r = self.compute_psi(df_norm, df_yr)
            all_psi_rows.append({
                "year": yr,
                "psi": psi_r.get("psi"),
                "risk": psi_r.get("risk", "N/A"),
                "delay_days": psi_r.get("delay_days", 0),
                "ndvi_mean_current": psi_r.get("ndvi_mean_current"),
                "ndvi_mean_norm": psi_r.get("ndvi_mean_norm"),
            })

        df_psi = pd.DataFrame(all_psi_rows)
        df_psi.to_parquet(self.OUTPUT_PSI_PATH)
        logger.success(f"PhenoShift результаты сохранены → {self.OUTPUT_PSI_PATH}")

        # Сохраняем норму для дашборда
        norm_path = PROCESSED_DIR / "phenological_norm.parquet"
        df_norm.to_parquet(norm_path)

        # Сохраняем текущий NDVI для дашборда
        current_path = PROCESSED_DIR / f"ndvi_current_{year}.parquet"
        df_current.to_parquet(current_path)

        return df_psi


if __name__ == "__main__":
    module = PhenoShiftModule()
    result = module.run(target_year=CURRENT_YEAR)
    print(result)
