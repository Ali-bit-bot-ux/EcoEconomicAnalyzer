"""
Daryn — modules/m3_silosentry.py
===================================
Модуль 3: SiloSentry — Мониторинг элеваторной инфраструктуры

ДВОЙНАЯ СТРАТЕГИЯ:

Метод A — OSM-источник:
  osmnx → теги: building=silo, man_made=grain_elevator
  Получаем: координаты, название, площадь

Метод B — Спутниковый анализ:
  Sentinel-2 → индекс застройки NDBI:
    NDBI = (SWIR - NIR) / (SWIR + NIR)
  Высокий NDBI → искусственные покрытия (крыши силосов)
  Изменение площади теней (Dark Feature Ratio) → proxy заполненности

ВЫХОД:
  GeoDataFrame с координатами элеваторов, статусом и
  оценочной заполненностью (%) на основе спутниковых данных.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ee
import json
import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False
    logger.warning("geopandas/shapely не установлены. GeoDataFrame недоступен.")

try:
    import osmnx as ox
    OSM_AVAILABLE = True
except ImportError:
    OSM_AVAILABLE = False
    logger.warning("osmnx не установлен. Метод OSM недоступен.")

from pipeline.gee_client import gee
from config.settings import (
    KNOWN_ELEVATORS,
    SKO_POLYGON,
    CURRENT_YEAR,
    PROCESSED_DIR,
)

# Регионы поиска элеваторов по OSM (bounding boxes Казахстана)
OSM_SEARCH_REGIONS = [
    {"name": "СКО", "bbox": (53.0, 65.5, 55.5, 70.5)},           # N, W, S, E (lat/lon)
    {"name": "Костанайская", "bbox": (51.5, 61.0, 54.5, 66.0)},
    {"name": "Акмолинская", "bbox": (50.0, 67.0, 52.5, 73.0)},
    {"name": "Павлодарская", "bbox": (51.0, 74.0, 53.5, 78.0)},
]


class SiloSentryModule:
    """
    Модуль 3: Обнаружение и мониторинг элеваторной инфраструктуры.
    """

    OUTPUT_PATH = PROCESSED_DIR / "silos.parquet"

    # ─────────────────────────────────────────────
    # Метод A: OpenStreetMap
    # ─────────────────────────────────────────────

    def fetch_elevators_osm(self) -> list[dict]:
        """
        Ищет элеваторы и зернохранилища в OpenStreetMap.
        Теги: building=silo, man_made=grain_elevator, building=warehouse

        Returns:
            Список dicts с полями: name, lat, lon, area_m2, source
        """
        if not OSM_AVAILABLE:
            logger.warning("osmnx недоступен, пропускаем OSM-поиск")
            return []

        results = []

        # Теги для поиска
        tags = {
            "man_made": ["grain_elevator", "silo"],
            "building": "silo",
        }

        for region in OSM_SEARCH_REGIONS:
            try:
                logger.info(f"OSM: поиск в регионе {region['name']}...")
                north, west, south, east = region["bbox"]

                # osmnx: получаем геометрии
                gdf = ox.features_from_bbox(
                    bbox=(north, south, east, west),
                    tags={"man_made": ["grain_elevator", "silo"]},
                )

                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    if hasattr(geom, "centroid"):
                        centroid = geom.centroid
                        lon, lat = centroid.x, centroid.y
                    elif hasattr(geom, "x"):
                        lon, lat = geom.x, geom.y
                    else:
                        continue

                    area = geom.area * 1e10 if hasattr(geom, "area") else None

                    results.append({
                        "name": row.get("name", f"Элеватор {region['name']} OSM"),
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "area_m2": area,
                        "region": region["name"],
                        "source": "OSM",
                        "capacity_t": None,
                        "osm_id": str(idx),
                    })

                logger.success(f"  {region['name']}: найдено {len(gdf)} объектов")

            except Exception as e:
                logger.warning(f"OSM ошибка для {region['name']}: {e}")

        return results

    # ─────────────────────────────────────────────
    # Метод B: Спутниковый анализ (NDBI + тени)
    # ─────────────────────────────────────────────

    def analyze_silo_from_satellite(
        self, lon: float, lat: float, name: str = "unknown"
    ) -> dict:
        """
        Анализирует спутниковые характеристики вокруг известного элеватора.

        Метрики:
        - NDBI (индекс застройки) = (SWIR - NIR) / (SWIR + NIR)
          Высокий NDBI → металлические/бетонные поверхности (крыши силосов)
        - Dark Feature Ratio (DFR) = доля тёмных пикселей вокруг силоса
          Тёмные области = тени от цилиндров → proxy заполненности

        Args:
            lon, lat: центр элеватора
            name: название для логирования

        Returns:
            dict с NDBI, DFR, estimated_fill_pct
        """
        # Полигон 500м вокруг точки
        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(500)  # 500 метров

        # Sentinel-2 за период уборки урожая текущего года (июль-октябрь)
        harvest_start = f"{CURRENT_YEAR}-07-01"
        harvest_end = f"{CURRENT_YEAR}-10-31"

        try:
            collection = gee.get_sentinel2(
                geometry=buffer,
                start_date=harvest_start,
                end_date=harvest_end,
                max_cloud=15,
            )

            # NDBI: (B11_SWIR - B8_NIR) / (B11_SWIR + B8_NIR)
            def add_ndbi(image):
                ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
                # Dark Feature Ratio: доля пикселей с яркостью < 0.1
                b4 = image.select("B4")
                dark = b4.lt(0.05)
                return image.addBands(ndbi).addBands(dark.rename("dark_pixels"))

            ndbi_collection = collection.map(add_ndbi)

            # Среднее NDBI за период
            ndbi_mean_img = ndbi_collection.select("NDBI").mean()
            dark_mean_img = ndbi_collection.select("dark_pixels").mean()

            ndbi_stats = ndbi_mean_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=10,
                maxPixels=1e6,
            ).getInfo()

            dark_stats = dark_mean_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=10,
                maxPixels=1e6,
            ).getInfo()

            ndbi_val = ndbi_stats.get("NDBI", None)
            dfr_val = dark_stats.get("dark_pixels", None)

            # Оценка заполненности: эмпирическая формула
            # Больше теней (DFR) = больше силосных цилиндров заполнено
            estimated_fill = None
            if dfr_val is not None:
                estimated_fill = min(100, max(0, dfr_val * 300))

            return {
                "ndbi": round(float(ndbi_val), 4) if ndbi_val else None,
                "dark_feature_ratio": round(float(dfr_val), 4) if dfr_val else None,
                "estimated_fill_pct": round(estimated_fill, 1) if estimated_fill else None,
                "satellite_analyzed": True,
            }

        except Exception as e:
            logger.warning(f"Спутниковый анализ для {name}: {e}")
            return {
                "ndbi": None,
                "dark_feature_ratio": None,
                "estimated_fill_pct": None,
                "satellite_analyzed": False,
            }

    # ─────────────────────────────────────────────
    # Сборка: объединение источников
    # ─────────────────────────────────────────────

    def build_elevator_registry(self) -> pd.DataFrame:
        """
        Объединяет:
        - Известные элеваторы (KNOWN_ELEVATORS из settings)
        - Найденные через OSM
        Дедуплицирует по расстоянию.

        Returns:
            DataFrame со всеми найденными элеваторами
        """
        # База: известные из конфига
        rows = []
        for elev in KNOWN_ELEVATORS:
            rows.append({
                "name": elev["name"],
                "lat": elev["lat"],
                "lon": elev["lon"],
                "capacity_t": elev.get("capacity_t"),
                "region": elev.get("region", "—"),
                "source": "config",
                "osm_id": None,
            })

        # OSM-дополнение
        osm_elevators = self.fetch_elevators_osm()
        for elev in osm_elevators:
            # Простая дедупликация: пропускаем если уже есть в ~1 км
            is_duplicate = False
            for existing in rows:
                dist_approx = (
                    (elev["lat"] - existing["lat"]) ** 2 +
                    (elev["lon"] - existing["lon"]) ** 2
                ) ** 0.5
                if dist_approx < 0.01:  # ~1 км в градусах
                    is_duplicate = True
                    break
            if not is_duplicate:
                rows.append(elev)

        df = pd.DataFrame(rows)
        logger.info(f"Реестр элеваторов: {len(df)} объектов")
        return df

    # ─────────────────────────────────────────────
    # Основной запуск
    # ─────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Полный пайплайн SiloSentry."""
        logger.info("=" * 52)
        logger.info("МОДУЛЬ 3: SiloSentry (Элеваторная инфраструктура)")
        logger.info("=" * 52)

        # 1. Реестр элеваторов
        df_elevators = self.build_elevator_registry()

        # 2. Спутниковый анализ для каждого известного элеватора
        logger.info("Запуск спутникового анализа элеваторов...")
        satellite_results = []

        for _, row in df_elevators.iterrows():
            logger.info(f"  Анализ: {row['name']} ({row['lon']:.2f}, {row['lat']:.2f})")
            sat_data = self.analyze_silo_from_satellite(
                lon=row["lon"],
                lat=row["lat"],
                name=row["name"],
            )
            satellite_results.append(sat_data)

        df_sat = pd.DataFrame(satellite_results)
        df_result = pd.concat([df_elevators.reset_index(drop=True), df_sat], axis=1)

        # 3. Итоговая оценка
        df_result["status"] = df_result["estimated_fill_pct"].apply(
            lambda x: "ACTIVE_FULL" if x and x > 70
            else "ACTIVE_PARTIAL" if x and x > 30
            else "ACTIVE_LOW" if x and x > 0
            else "UNKNOWN"
        )

        # 4. Сохраняем
        df_result.to_parquet(self.OUTPUT_PATH)
        logger.success(f"SiloSentry результаты сохранены → {self.OUTPUT_PATH}")
        logger.info(f"\n{df_result[['name', 'region', 'estimated_fill_pct', 'status']].to_string(index=False)}")

        return df_result


if __name__ == "__main__":
    module = SiloSentryModule()
    result = module.run()
    print(result)
