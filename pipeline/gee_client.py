"""
Daryn — pipeline/gee_client.py
================================
Singleton-клиент Google Earth Engine.
- Автоматическая инициализация при первом обращении
- Локальное кэширование результатов (избегаем повторных облачных вызовов)
- Единое место для всех GEE-запросов проекта
"""

import ee
import json
import hashlib
import pickle
from pathlib import Path
from loguru import logger
from typing import Optional

# Импортируем настройки
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import GEE_PROJECT, GEE_CACHE_ENABLED, GEE_CACHE_DIR


class GEEClient:
    """Singleton-клиент для работы с Google Earth Engine."""

    _instance: Optional["GEEClient"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> bool:
        """
        Инициализирует Earth Engine API.
        Вызывается один раз при старте пайплайна.
        """
        if self._initialized:
            return True

        try:
            # Попытка 1: Спроектом из конфига или окружения
            if GEE_PROJECT:
                ee.Initialize(project=GEE_PROJECT)
                logger.info(f"GEE инициализирован с проектом: {GEE_PROJECT}")
            else:
                # Попытка 2: Авто-вызов без явного проекта
                try:
                    ee.Initialize()
                    logger.info("GEE инициализирован")
                except ee.EEException as ex:
                    # Попытка 3: Если GEE требует проект, берем дефолтный или 'ee-daryn'
                    err_msg = str(ex)
                    if "no project found" in err_msg or "project=" in err_msg:
                        try:
                            # Пробуем дефолтный Cloud проект
                            import google.auth
                            credentials, project = google.auth.default()
                            project_id = project or "ee-daryn"
                            ee.Initialize(project=project_id)
                            logger.info(f"GEE инициализирован с авто-проектом: {project_id}")
                        except Exception:
                            # Резервный попытка с общедоступным режимом
                            ee.Initialize(project="ee-daryn-project")
                            logger.info("GEE инициализирован с резервным проектом")
                    else:
                        raise ex

            self._initialized = True
            return True

        except ee.EEException as e:
            logger.error(f"Ошибка инициализации GEE: {e}")
            logger.warning("Запусти: earthengine authenticate")
            return False

        except Exception as e:
            logger.error(f"Неожиданная ошибка GEE: {e}")
            return False

    def health_check(self) -> dict:
        """Проверяет соединение с GEE."""
        try:
            # Простой тест: получить номер изображения
            info = ee.Image(1).getInfo()
            return {
                "status": "ok",
                "message": "GEE подключен и работает",
                "test_value": info.get("type", "unknown"),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ─────────────────────────────────────────────
    # Кэширование
    # ─────────────────────────────────────────────

    def _cache_key(self, params: dict) -> str:
        """Генерирует уникальный ключ кэша из параметров запроса."""
        serialized = json.dumps(params, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return GEE_CACHE_DIR / f"{key}.pkl"

    def cache_get(self, params: dict):
        """Возвращает кэшированный результат или None."""
        if not GEE_CACHE_ENABLED:
            return None
        path = self._cache_path(self._cache_key(params))
        if path.exists():
            logger.debug(f"Cache HIT: {path.name}")
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def cache_set(self, params: dict, data) -> None:
        """Сохраняет результат в кэш."""
        if not GEE_CACHE_ENABLED:
            return
        path = self._cache_path(self._cache_key(params))
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Cache SET: {path.name}")

    # ─────────────────────────────────────────────
    # Утилиты создания геометрий
    # ─────────────────────────────────────────────

    @staticmethod
    def polygon_from_coords(coords: list) -> ee.Geometry:
        """Создаёт ee.Geometry.Polygon из списка [lon, lat] пар."""
        return ee.Geometry.Polygon(coords)

    @staticmethod
    def point(lon: float, lat: float) -> ee.Geometry:
        return ee.Geometry.Point([lon, lat])

    # ─────────────────────────────────────────────
    # Фабричные методы коллекций
    # ─────────────────────────────────────────────

    def get_sentinel2(
        self,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
        max_cloud: int = 20,
        apply_cloud_mask: bool = True,
    ) -> ee.ImageCollection:
        """
        Возвращает отфильтрованную коллекцию Sentinel-2 SR.

        Args:
            geometry: полигон анализа
            start_date: '2024-01-01'
            end_date: '2024-12-31'
            max_cloud: максимальный % облачности
            apply_cloud_mask: применять ли маску облаков по SCL
        """
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        )

        if apply_cloud_mask:
            collection = collection.map(self._s2_cloud_mask)

        return collection

    def get_landsat8(
        self,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
    ) -> ee.ImageCollection:
        """Landsat-8 C2 Level-2 (2013+). Используется для расширения 10-летнего ряда."""
        return (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
            .map(self._l8_scale_and_mask)
        )

    def get_smap(
        self,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
    ) -> ee.ImageCollection:
        """SMAP L3 Radiometer 9km — ежедневная влажность почв."""
        return (
            ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .select("soil_moisture_am")
        )

    # ─────────────────────────────────────────────
    # Внутренние функции обработки снимков
    # ─────────────────────────────────────────────

    @staticmethod
    def _s2_cloud_mask(image: ee.Image) -> ee.Image:
        """
        Маскировка облаков Sentinel-2 через SCL (Scene Classification Layer).
        Классы 3(тени), 8(облака med), 9(облака high), 10(cirrus), 11(снег) → маска.
        """
        scl = image.select("SCL")
        cloud_shadow = scl.eq(3)
        cloud_med = scl.eq(8)
        cloud_high = scl.eq(9)
        cirrus = scl.eq(10)
        snow = scl.eq(11)

        # Объединяем все «плохие» пиксели
        bad_pixels = (
            cloud_shadow.Or(cloud_med).Or(cloud_high).Or(cirrus).Or(snow)
        )
        mask = bad_pixels.Not()
        return image.updateMask(mask)

    @staticmethod
    def _l8_scale_and_mask(image: ee.Image) -> ee.Image:
        """
        Масштабирование Landsat-8 C2 L2 и маскировка облаков по QA_PIXEL.
        Применяет официальные коэффициенты масштабирования USGS.
        """
        # QA_PIXEL: бит 3 = облака, бит 4 = тени
        qa = image.select("QA_PIXEL")
        cloud_bit = 1 << 3
        shadow_bit = 1 << 4
        cloud_mask = qa.bitwiseAnd(cloud_bit).eq(0)
        shadow_mask = qa.bitwiseAnd(shadow_bit).eq(0)
        mask = cloud_mask.And(shadow_mask)

        # Официальные коэффициенты масштабирования USGS Landsat C2
        optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
        return image.addBands(optical, overwrite=True).updateMask(mask)

    @staticmethod
    def add_ndvi_s2(image: ee.Image) -> ee.Image:
        """Добавляет бэнд NDVI к снимку Sentinel-2."""
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        date = image.date().format("YYYY-MM-dd")
        return image.addBands(ndvi).set("date_str", date)

    @staticmethod
    def add_ndvi_l8(image: ee.Image) -> ee.Image:
        """Добавляет бэнд NDVI к снимку Landsat-8."""
        ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        date = image.date().format("YYYY-MM-dd")
        return image.addBands(ndvi).set("date_str", date)

    # ─────────────────────────────────────────────
    # Экспорт данных
    # ─────────────────────────────────────────────

    def extract_time_series(
        self,
        collection: ee.ImageCollection,
        geometry: ee.Geometry,
        band: str = "NDVI",
        scale: int = 10,
        reducer=None,
    ) -> list[dict]:
        """
        Извлекает временной ряд значений бэнда для заданного полигона.
        Возвращает список dicts: [{date, value}, ...]
        Результат кэшируется.

        Args:
            collection: ImageCollection с нужным бэндом
            geometry: полигон
            band: имя бэнда
            scale: пространственное разрешение в метрах
            reducer: редуктор (по умолчанию ee.Reducer.mean())
        """
        if reducer is None:
            reducer = ee.Reducer.mean()

        # Ключ кэша (используем параметры коллекции как прокси)
        try:
            col_info = collection.first().getInfo()
            cache_params = {
                "collection_id": str(col_info.get("id", "unknown")),
                "band": band,
                "scale": scale,
                "geometry_bounds": geometry.bounds().getInfo(),
            }
        except Exception:
            cache_params = {"band": band, "scale": scale}

        cached = self.cache_get(cache_params)
        if cached is not None:
            return cached

        # GEE-функция для редукции каждого снимка
        def reduce_image(image):
            stats = image.select(band).reduceRegion(
                reducer=reducer,
                geometry=geometry,
                scale=scale,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=16,
            )
            return ee.Feature(None, {
                "date": image.date().format("YYYY-MM-dd"),
                "value": stats.get(band),
            })

        features = collection.map(reduce_image)
        result = features.getInfo()

        # Преобразуем в список dicts
        series = []
        for f in result.get("features", []):
            props = f.get("properties", {})
            val = props.get("value")
            if val is not None:
                series.append({
                    "date": props.get("date"),
                    "value": round(float(val), 6),
                })

        # Сортируем по дате
        series.sort(key=lambda x: x["date"])

        self.cache_set(cache_params, series)
        logger.info(f"Извлечено {len(series)} точек временного ряда ({band})")
        return series


# Глобальный экземпляр
gee = GEEClient()
