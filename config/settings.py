"""
Daryn — Агро-Разведчик
Централизованная конфигурация проекта
======================================
Все параметры системы в одном месте.
Ключи API загружаются из .env (не попадают в git).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка .env файла (создай его рядом с этим скриптом)
load_dotenv()

# ─────────────────────────────────────────────
# Настройки Telegram бота (Модуль оповещений)
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ─────────────────────────────────────────────
# Пути проекта
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"

# Автоматически создать директории, если не существуют
for d in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Google Earth Engine
# ─────────────────────────────────────────────
GEE_PROJECT = os.getenv("GEE_PROJECT", "")          # Опционально: имя cloud-проекта GEE
GEE_CACHE_ENABLED = True                              # Кэшировать результаты локально
GEE_CACHE_DIR = RAW_DIR / "gee_cache"
GEE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Географические параметры
# ─────────────────────────────────────────────

# Тестовый полигон: Северо-Казахстанская область (Есильский р-н)
# Центр крупнейшего пшеничного пояса Казахстана (~2.5 млн га)
SKO_POLYGON = [
    [66.5, 53.3],
    [69.5, 53.3],
    [69.5, 54.5],
    [66.5, 54.5],
    [66.5, 53.3],
]
SKO_CENTER = [68.0, 53.9]  # [lon, lat]

# Полигон верховья реки Или (приток Балхаша, трансграничная)
ILI_RIVER_UPSTREAM = [
    [80.0, 43.0],
    [84.0, 43.0],
    [84.0, 44.5],
    [80.0, 44.5],
    [80.0, 43.0],
]

# Бассейн Сырдарьи (верховье, Кыргызстан/Казахстан)
SYRDARYA_UPSTREAM = [
    [69.0, 40.0],
    [73.0, 40.0],
    [73.0, 42.5],
    [69.0, 42.5],
    [69.0, 40.0],
]

import datetime

# Временные параметры анализа
# ─────────────────────────────────────────────
ANALYSIS_START_YEAR = 2015       # Начало: доступен Sentinel-2
CURRENT_YEAR = datetime.datetime.now().year  # Текущий календарный год в реальном времени
ANALYSIS_END_YEAR = CURRENT_YEAR
BASELINE_YEARS = list(range(2015, CURRENT_YEAR))  # Климатическая норма до текущего года

# Вегетационный период (DOY — день года)
VEGETATION_DOY_START = 120       # ~30 апреля
VEGETATION_DOY_END = 280         # ~7 октября

# Максимальный процент облачности при фильтрации снимков
MAX_CLOUD_COVER = 20             # %

# ─────────────────────────────────────────────
# Параметры PhenoShift Index (Модуль 1)
# ─────────────────────────────────────────────
PSI_THRESHOLD_HIGH = 1.5        # PSI > +1.5 → опережение (риск заморозков)
PSI_THRESHOLD_LOW = -1.5        # PSI < -1.5 → отставание (засуха / недобор)
PHENOSHIFT_TEMPORAL_RESOLUTION = 16  # дней между снимками (Sentinel-2 revisit)

# ─────────────────────────────────────────────
# Параметры SMAP / HydroBorder (Модуль 2)
# ─────────────────────────────────────────────
SMAP_COLLECTION = "NASA/SMAP/SPL3SMP_E/006"
SMAP_BAND_SURFACE = "soil_moisture_am"       # Утренний проход (более стабильный)
SMAP_ROLLING_WINDOW_DAYS = 90                # Скользящее среднее для сглаживания
SMAP_ANOMALY_THRESHOLD = -1.0               # Z-score аномалии (σ ниже нормы)

# ─────────────────────────────────────────────
# Параметры Granger Causality (Модуль 4)
# ─────────────────────────────────────────────
GRANGER_MAX_LAG = 12            # Максимум лагов (12 мес = 1 год при месячных данных)
GRANGER_SIGNIFICANCE = 0.05     # Порог p-value
VAR_MAX_LAGS = 6                # Лаги для VAR-модели

# ─────────────────────────────────────────────
# Параметры MicroVulnerability (Модуль 5)
# ─────────────────────────────────────────────
VULN_WEIGHTS = {
    "psi_anomaly": 0.40,
    "soil_moisture_anomaly": 0.30,
    "silo_distance_km": 0.20,
    "field_size_inv": 0.10,
}
VULN_HIGH_THRESHOLD = 70        # > 70 → красная зона (высокий риск)
VULN_MED_THRESHOLD = 40         # 40-70 → оранжевая зона

# ─────────────────────────────────────────────
# Датасеты спутников
# ─────────────────────────────────────────────
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
SENTINEL1_COLLECTION = "COPERNICUS/S1_GRD"


# NDVI: (NIR - Red) / (NIR + Red)
S2_NIR_BAND = "B8"
S2_RED_BAND = "B4"
L8_NIR_BAND = "SR_B5"
L8_RED_BAND = "SR_B4"

# SCL-классы облаков/теней (Sentinel-2 Scene Classification Layer)
S2_CLOUD_SCL_CLASSES = [3, 8, 9, 10, 11]  # тени, облака (med/high/cirrus), снег
NDVI_SCALE = 250  # Разрешение для извлечения временных рядов (м)

# ─────────────────────────────────────────────
# Элеваторы (координаты известных объектов)
# Дополнительно загружаются из regions.json
# ─────────────────────────────────────────────
KNOWN_ELEVATORS = [
    {"name": "Петропавловский элеватор", "lon": 69.1308, "lat": 54.8671, "capacity_t": 120000},
    {"name": "Кокшетауский элеватор", "lon": 69.3951, "lat": 53.2883, "capacity_t": 95000},
    {"name": "Костанайский ХПП", "lon": 63.6349, "lat": 53.2141, "capacity_t": 200000},
    {"name": "Акмолинский элеватор", "lon": 71.4419, "lat": 51.1801, "capacity_t": 150000},
    {"name": "Павлодарский элеватор", "lon": 76.9674, "lat": 52.2873, "capacity_t": 80000},
]

# ─────────────────────────────────────────────
# Экономические API
# ─────────────────────────────────────────────
FAOSTAT_AREA_KZ = "57"          # Код Казахстана в FAOSTAT
FAOSTAT_ITEM_WHEAT = "15"       # Код пшеницы
FAOSTAT_DOMAIN_PRICES = "PP"    # Producer Prices
WORLDBANK_INDICATORS = {
    "food_cpi": "FP.CPI.TOTL",
    "cereal_production": "AG.PRD.CREL.MT",
    "agricultural_land": "AG.LND.AGRI.ZS",
}
WORLDBANK_COUNTRY = "KAZ"

import logging
logging.getLogger(__name__).debug("Daryn configuration loaded successfully")
