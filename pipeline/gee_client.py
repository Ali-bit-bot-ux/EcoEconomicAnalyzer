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
import math
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
        """Initialize explicitly with existing credentials; return connection status."""
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
                    # Только отсутствие проекта допускает поиск существующего ADC.
                    err_msg = str(ex).lower()
                    if "no project found" not in err_msg and "project=" not in err_msg:
                        raise
                    import google.auth
                    credentials, project = google.auth.default()
                    if not project:
                        raise ex
                    ee.Initialize(credentials=credentials, project=project)
                    logger.info(f"GEE инициализирован с авто-проектом: {project}")

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

    def get_smap_usda(
        self,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
    ) -> ee.ImageCollection:
        """Legacy NASA-USDA Enhanced SMAP 10 km product used by the 2021 research work.

        The product was discontinued in August 2022 and is superseded by
        NASA/SMAP/SPL4SMGP/008; it is retained only for historical validation.
        """
        return (
            ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .select(["ssm", "susm", "smp", "ssma", "susma"])
        )

    def get_sentinel1(
        self,
        geometry: ee.Geometry,
        start_date: str,
        end_date: str,
    ) -> ee.ImageCollection:
        """
        Sentinel-1 SAR GRD. Всепогодный радар.
        Используется поляризация VV и VH.
        """
        return (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .select(['VV', 'VH'])
        )

    def get_space_observations(
        self,
        polygon_coords: list | dict,
        start_date: str,
        end_date: str,
        orbit_pass: str = "DESCENDING",
        relative_orbit: Optional[int] = None,
        scale: int = 100,
    ) -> dict:
        """Pair SAR powers on a common mask and orbit; retain independent optical acquisition times."""
        if orbit_pass not in ("ASCENDING", "DESCENDING"):
            raise ValueError("orbit_pass must be ASCENDING or DESCENDING")
        if relative_orbit is not None and (
            isinstance(relative_orbit, bool)
            or not isinstance(relative_orbit, int)
            or not 1 <= relative_orbit <= 175
        ):
            raise ValueError("relative_orbit must be an integer from 1 to 175 or None")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("scale must be a finite positive number of metres")
        if isinstance(polygon_coords, dict):
            if polygon_coords.get("type") != "Polygon":
                raise ValueError("ROI must be a GeoJSON Polygon geometry")
            polygon_coords = polygon_coords["coordinates"]

        if not self.initialize():
            raise RuntimeError("Earth Engine initialization is unavailable.")
        geometry = self.polygon_from_coords(polygon_coords)
        sar_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
            .select(["VV", "VH", "angle"])
        )
        if relative_orbit is None:
            histogram = sar_collection.aggregate_histogram(
                "relativeOrbitNumber_start"
            ).getInfo()
            if histogram:
                chosen = max(
                    histogram,
                    key=lambda orbit: (histogram[orbit], -int(float(orbit))),
                )
                relative_orbit = int(float(chosen))

        def reduce_sar(image):
            common_mask = image.select(["VV", "VH", "angle"]).mask().reduce(
                ee.Reducer.min()
            )
            paired_db = image.select(["VV", "VH"]).updateMask(common_mask)
            # exp(dB * ln(10) / 10) = 10**(dB/10), preserving the VV/VH grid.
            powers = paired_db.multiply(math.log(10) / 10).exp()
            values = powers.addBands(image.select("angle").updateMask(common_mask))
            stats = values.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                crs=image.select("VV").projection(),
                maxPixels=1e9,
                tileScale=16,
            )
            return ee.Feature(None, {
                "date": image.date().format("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", "UTC"),
                "scene_id": image.get("system:index"),
                "orbit_pass": image.get("orbitProperties_pass"),
                "relative_orbit": image.get("relativeOrbitNumber_start"),
                "incidence_angle_deg": stats.get("angle"),
                "vv_power": stats.get("VV"),
                "vh_power": stats.get("VH"),
            })

        sar = []
        if relative_orbit is not None:
            sar_collection = sar_collection.filter(
                ee.Filter.eq("relativeOrbitNumber_start", relative_orbit)
            ).sort("system:time_start")
            features = ee.FeatureCollection(sar_collection.map(reduce_sar)).getInfo()
            for feature in features["features"]:
                props = feature["properties"]
                required = (
                    "date", "scene_id", "orbit_pass", "relative_orbit",
                    "incidence_angle_deg", "vv_power", "vh_power",
                )
                if any(props.get(key) is None for key in required):
                    continue
                vv_power = float(props["vv_power"])
                vh_power = float(props["vh_power"])
                angle = float(props["incidence_angle_deg"])
                if not all(math.isfinite(v) for v in (vv_power, vh_power, angle)):
                    continue
                if vv_power <= 0 or vh_power <= 0 or not 0 <= angle <= 90:
                    continue
                sar.append({
                    "date": props["date"],
                    "scene_id": props["scene_id"],
                    "orbit_pass": props["orbit_pass"],
                    "relative_orbit": int(props["relative_orbit"]),
                    "incidence_angle_deg": angle,
                    "vv_db": 10 * math.log10(vv_power),
                    "vh_db": 10 * math.log10(vh_power),
                    "source": "COPERNICUS/S1_GRD",
                })

        optical_collection = self.get_sentinel2(
            geometry, start_date, end_date, apply_cloud_mask=True
        ).map(self.add_ndvi_s2).sort("system:time_start")

        def reduce_optical(image):
            stats = image.select("NDVI").reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                crs=image.select("B8").projection(),
                maxPixels=1e9,
                tileScale=16,
            )
            return ee.Feature(None, {
                "date": image.date().format("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", "UTC"),
                "scene_id": image.get("system:index"),
                "ndvi": stats.get("NDVI"),
            })

        optical = []
        features = ee.FeatureCollection(optical_collection.map(reduce_optical)).getInfo()
        for feature in features["features"]:
            props = feature["properties"]
            if any(props.get(key) is None for key in ("date", "scene_id", "ndvi")):
                continue
            ndvi = float(props["ndvi"])
            if math.isfinite(ndvi) and -1 <= ndvi <= 1:
                optical.append({
                    "date": props["date"],
                    "scene_id": props["scene_id"],
                    "ndvi": ndvi,
                    "source": "COPERNICUS/S2_SR_HARMONIZED",
                })

        sar.sort(key=lambda row: (row["date"], row["scene_id"]))
        optical.sort(key=lambda row: (row["date"], row["scene_id"]))
        return {
            "sar": sar,
            "optical": optical,
            "relative_orbit": relative_orbit,
            "scale_m": scale,
        }

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
        
        import time
        result = None
        for attempt in range(3):
            try:
                result = features.getInfo()
                break
            except ee.EEException as e:
                if attempt == 2:
                    logger.error(f"Критическая ошибка GEE getInfo(): {e}")
                    raise
                logger.warning(f"Таймаут GEE (попытка {attempt + 1}/3). Ждем {2 ** attempt}с...")
                time.sleep(2 ** attempt)

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
