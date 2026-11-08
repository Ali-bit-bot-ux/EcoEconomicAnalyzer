"""
AgriCascade — Мониторинг продовольственных рисков
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
# ─────────────────────────────────────────────
# Географические параметры: Зерновой пояс Казахстана (4 области)
# ─────────────────────────────────────────────

GRAIN_BELT_REGIONS = {
    "sko": {
        "id": "sko",
        "name_ru": "Северо-Казахстанская область (СКО)",
        "name_kk": "Солтүстік Қазақстан облысы (СҚО)",
        "short_name_ru": "СКО",
        "short_name_kk": "СҚО",
        "center": [54.87, 69.15],  # [lat, lon]
        "zoom": 7,
        "lat_bounds": [53.3, 55.4],
        "lon_bounds": [66.5, 71.5],
        "polygon": [
            [66.5, 53.3], [71.5, 53.3], [71.5, 55.4], [66.5, 55.4], [66.5, 53.3]
        ],
        "arable_area_ha": 2800000,
        "basin_ru": "р. Ишим / р. Тобол",
        "basin_kk": "Есіл / Тобыл өзендері",
        "dominant_soil_ru": "Черноземы южные и обыкновенные",
        "dominant_soil_kk": "Оңтүстік және кәдімгі қара топырақ",
        "districts_ru": ["Кызылжарский", "Есильский", "Мамлютский", "Акжарский", "Тайыншинский"],
        "districts_kk": ["Қызылжар", "Есіл", "Мамлют", "Ақжар", "Тайынша"],
    },
    "kostanay": {
        "id": "kostanay",
        "name_ru": "Костанайская область",
        "name_kk": "Қостанай облысы",
        "short_name_ru": "Костанайская обл.",
        "short_name_kk": "Қостанай обл.",
        "center": [53.21, 63.63],
        "zoom": 7,
        "lat_bounds": [51.5, 54.5],
        "lon_bounds": [61.0, 66.0],
        "polygon": [
            [61.0, 51.5], [66.0, 51.5], [66.0, 54.5], [61.0, 54.5], [61.0, 51.5]
        ],
        "arable_area_ha": 4200000,
        "basin_ru": "р. Тобол (Трансграничный РФ→РК)",
        "basin_kk": "Тобыл өзені (Трансшекаралық)",
        "dominant_soil_ru": "Темно-каштановые, солонцеватые",
        "dominant_soil_kk": "Күңгірт қара-қоңыр, сортаңдау",
        "districts_ru": ["Карабалыкский", "Федоровский", "Мендыкаринский", "Костанайский", "Аулиекольский"],
        "districts_kk": ["Қарабалық", "Федоров", "Меңдіқара", "Қостанай", "Әулиекөл"],
    },
    "akmola": {
        "id": "akmola",
        "name_ru": "Акмолинская область",
        "name_kk": "Ақмола облысы",
        "short_name_ru": "Акмолинская обл.",
        "short_name_kk": "Ақмола обл.",
        "center": [51.50, 70.50],
        "zoom": 7,
        "lat_bounds": [50.5, 53.5],
        "lon_bounds": [67.0, 73.5],
        "polygon": [
            [67.0, 50.5], [73.5, 50.5], [73.5, 53.5], [67.0, 53.5], [67.0, 50.5]
        ],
        "arable_area_ha": 3900000,
        "basin_ru": "р. Ишим и р. Нура",
        "basin_kk": "Есіл және Нұра өзендері",
        "dominant_soil_ru": "Темно-каштановые, супесчаные",
        "dominant_soil_kk": "Күңгірт қара-қоңыр, құмдауыт",
        "districts_ru": ["Зерендинский", "Сандыктауский", "Бурабайский", "Жаксынский", "Атбасарский"],
        "districts_kk": ["Зеренді", "Сандықтау", "Бурабай", "Жақсы", "Атбасар"],
    },
    "pavlodar": {
        "id": "pavlodar",
        "name_ru": "Павлодарская область",
        "name_kk": "Павлодар облысы",
        "short_name_ru": "Павлодарская обл.",
        "short_name_kk": "Павлодар обл.",
        "center": [52.30, 76.95],
        "zoom": 7,
        "lat_bounds": [50.5, 54.2],
        "lon_bounds": [74.5, 78.8],
        "polygon": [
            [74.5, 50.5], [78.8, 50.5], [78.8, 54.2], [74.5, 54.2], [74.5, 50.5]
        ],
        "arable_area_ha": 1400000,
        "basin_ru": "р. Иртыш (Ертіс)",
        "basin_kk": "Ертіс өзені",
        "dominant_soil_ru": "Светло-каштановые, супесчаные",
        "dominant_soil_kk": "Ашық қара-қоңыр, құмдақ",
        "districts_ru": ["Иртышский", "Железинский", "Актогайский", "Теренколь", "Успенский"],
        "districts_kk": ["Ертіс", "Железин", "Ақтоғай", "Тереңкөл", "Успен"],
    },
}

# ─────────────────────────────────────────────
# Целевые культуры финала (Multi-Crop Engine)
# Стандарт FAO-56 + Датасет Quareo
# ─────────────────────────────────────────────

TARGET_CROPS = {
    "spring_wheat": {
        "id": "spring_wheat",
        "name_ru": "Пшеница яровая",
        "name_kk": "Жаздық бидай",
        "kc": 1.15,
        "root_depth_meters": "1.2 - 1.6",
        "critical_phase_ru": "Выход в трубку — колошение",
        "critical_phase_kk": "Сабаққа шығу – бас шығару",
        "drought_tolerance_score": 3.0,
        "wilting_swc_percent": 35,
        "description_ru": "Базовая культура зернового пояса. Высокое водопотребление на наливе.",
        "description_kk": "Астық белдеуінің негізгі дақылы. Дән толысу кезеңінде суды көп қажет етеді.",
        "fao_ref": "FAO-56 Irrigation Paper, Table 12 (Kc mid = 1.15)",
    },
    "spring_barley": {
        "id": "spring_barley",
        "name_ru": "Ячмень яровой",
        "name_kk": "Жаздық арпа",
        "kc": 1.10,
        "root_depth_meters": "1.0 - 1.2",
        "critical_phase_ru": "Кущение — колошение (созревает на 10-14 дней раньше)",
        "critical_phase_kk": "Түптену – бас шығару (10-14 күн ерте піседі)",
        "drought_tolerance_score": 4.0,
        "wilting_swc_percent": 30,
        "description_ru": "Экономный расход влаги (-15% к пшенице). Успевает до пика июльской жары.",
        "description_kk": "Ылғалды үнемді жұмсайды. Шілдедегі аптап ыстыққа дейін пісіп үлгереді.",
        "fao_ref": "FAO-56 Irrigation Paper, Table 12 (Kc mid = 1.10)",
    },
    "sunflower": {
        "id": "sunflower",
        "name_ru": "Подсолнечник",
        "name_kk": "Күнбағыс",
        "kc": 1.15,
        "root_depth_meters": "1.8 - 2.4",
        "critical_phase_ru": "Цветение — налив семян (глубокий био-насос)",
        "critical_phase_kk": "Гүлдеу – дән толысу (терең тамырлы био-сорғы)",
        "drought_tolerance_score": 3.8,
        "wilting_swc_percent": 32,
        "description_ru": "Мощная корневая система. Выкачивает влагу до 2 м, иссушая подпочву.",
        "description_kk": "Қуатты тамыр жүйесі. Топырақ асты ылғалын 2 метрге дейін сорып алады.",
        "fao_ref": "FAO-56 Irrigation Paper, Table 12 (Kc mid = 1.15)",
    },
    "flax_linseed": {
        "id": "flax_linseed",
        "name_ru": "Лён масличный",
        "name_kk": "Майлы зығыр",
        "kc": 1.00,
        "root_depth_meters": "0.8 - 1.0",
        "critical_phase_ru": "Всходы — «елочка» (уязвим к почвенной корке)",
        "critical_phase_kk": "Көктеу – «шырша» кезеңі (топырақ қабығына сезімтал)",
        "drought_tolerance_score": 3.8,
        "wilting_swc_percent": 30,
        "description_ru": "Поверхностные корни. Критически уязвим к майско-июньской засухе верхнего слоя.",
        "description_kk": "Беткі тамырлар. Жоғарғы қабаттың мамыр-маусым құрғақшылығына сезімтал.",
        "fao_ref": "FAO-56 Irrigation Paper, Table 12 (Kc mid = 1.00)",
    },
    "chickpea": {
        "id": "chickpea",
        "name_ru": "Нут (турецкий горох)",
        "name_kk": "Ноқат (бұршақ)",
        "kc": 1.00,
        "root_depth_meters": "1.0 - 1.2",
        "critical_phase_ru": "Ветвление — цветение (азотфиксация)",
        "critical_phase_kk": "Бұтақтану – гүлдеу (азот фиксациясы)",
        "drought_tolerance_score": 4.5,
        "wilting_swc_percent": 28,
        "description_ru": "Высокая засухоустойчивость. Обогащает почву азотом, идеален для севооборота.",
        "description_kk": "Жоғары құрғақшылыққа төзімді. Топырақты азотпен байытады.",
        "fao_ref": "FAO-56 Irrigation Paper, Table 12 (Kc mid = 1.00)",
    },
}

# Обратная совместимость для существующих модулей (СКО)
SKO_POLYGON = GRAIN_BELT_REGIONS["sko"]["polygon"]
SKO_CENTER = [68.0, 53.9]  # [lon, lat]

# Бассейн реки Тобол (Костанайская обл., зерновой пояс)
TOBOL_BASIN = [
    [62.0, 52.0], [65.5, 52.0], [65.5, 55.5], [62.0, 55.5], [62.0, 52.0]
]

# Бассейн реки Ишим / Есиль (СКО, Петропавловск, Акмолинская)
ISHIM_BASIN = [
    [67.0, 51.0], [71.5, 51.0], [71.5, 55.0], [67.0, 55.0], [67.0, 51.0]
]

# Бассейн реки Нура (Акмолинская обл., Центрально-Северный)
NURA_BASIN = [
    [69.0, 49.8], [73.5, 49.8], [73.5, 51.8], [69.0, 51.8], [69.0, 49.8]
]

# Полигон верховья реки Или (приток Балхаша, юг)
ILI_RIVER_UPSTREAM = [
    [80.0, 43.0],
    [84.0, 43.0],
    [84.0, 44.5],
    [80.0, 44.5],
    [80.0, 43.0],
]

# Бассейн Сырдарьи (верховье, юг)
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
logging.getLogger(__name__).debug("AgriCascade configuration loaded successfully")
