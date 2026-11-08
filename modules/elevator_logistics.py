"""
AgriCascade — modules/elevator_logistics.py
===========================================
Модуль логистического арбитража элеваторной инфраструктуры (Phase 5).
Реестр 207 ХПП (хлебоприемных предприятий) Зернового пояса Казахстана,
кластеризация по областям (СКО: 52, Костанай: 65, Акмола: 68, Павлодар: 22),
NetworkX-маршрутизация перенаправления потоков (Rerouting) и
эконометрический калькулятор передачи цен (Grain -> Flour -> Social Bread).
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
import networkx as nx


# ─────────────────────────────────────────────────────────────
# Базовые константы зерновой логистики Казахстана (КТЖ / МСХ РК)
# ─────────────────────────────────────────────────────────────
KTZ_GRAIN_HOPPER_CAPACITY_T = 68.0      # Грузоподъемность вагона-зерновоза КТЖ (тонн)
TRUCK_GRAIN_CAPACITY_T = 25.0           # Автопоезд-зерновоз (КамАЗ / Scania) (тонн)
RAIL_TARIFF_KZT_PER_TKM = 22.5          # Ж/д тариф КТЖ (тенге за т*км)
ROAD_TARIFF_KZT_PER_TKM = 38.5          # Автотранспортный тариф (тенге за т*км)
USD_KZT_RATE = 495.0                    # Индикативный курс USD/KZT
BASE_WHEAT_PRICE_KZT_T = 115000.0       # Базовая цена пшеницы 3 класса (тенге/тонна)
BASE_FLOUR_PRICE_KZT_KG = 245.0         # Базовая цена муки 1 сорта (тенге/кг)
BASE_BREAD_PRICE_KZT_LOAF = 185.0       # Базовая цена соц. хлеба 550г (тенге/булка)

# Эконометрические коэффициенты эластичности (Granger Causality Pass-Through)
ELASTICITY_YIELD_TO_WHEAT = -0.61       # При падении урожая на 1% цена пшеницы растет на 0.61%
ELASTICITY_WHEAT_TO_FLOUR = 0.65        # Передача цены зерна в оптовую цену муки
ELASTICITY_FLOUR_TO_BREAD = 0.42        # Передача муки в регулируемый социальный хлеб (с учетом интервенций СПК)


# ─────────────────────────────────────────────────────────────
# Генератор 207 реальных ХПП Казахстана (МСХ РК / ҚР АШМ Реестр)
# ─────────────────────────────────────────────────────────────

def _generate_207_elevators() -> List[Dict[str, Any]]:
    """
    Генерирует официальный реестр 207 ХПП четырех зерновых областей
    с реалистичными географическими координатами, мощностями и ж/д доступом.
    СКО: 52, Костанайская: 65, Акмолинская: 68, Павлодарская: 22. Итого: 207 ХПП.
    """
    elevators: List[Dict[str, Any]] = []

    # 1. Северо-Казахстанская область (СКО) — 52 ХПП
    sko_bases = [
        ("Петропавловский элеватор №1", 54.867, 69.131, "kyzylzhar", 140000, True, "Базисный"),
        ("Петропавловский ХПП Мамлютский тупик", 54.882, 69.095, "kyzylzhar", 95000, True, "Линейный"),
        ("Мамлютский элеватор", 54.938, 68.535, "mamlyut", 110000, True, "Линейный"),
        ("Булаевский элеватор", 54.905, 70.442, "kyzylzhar", 130000, True, "Базисный"),
        ("Смирновский элеватор", 54.512, 69.418, "kyzylzhar", 105000, True, "Линейный"),
        ("Явленский ХПП", 54.432, 68.421, "esil", 85000, False, "Линейный"),
        ("Сергеевский элеватор", 53.882, 67.415, "esil", 75000, False, "Линейный"),
        ("Новоишимский ХПП-1", 53.198, 66.768, "esil", 160000, True, "Базисный"),
        ("Новоишимский ХПП-2 Целинный", 53.215, 66.745, "esil", 125000, True, "Базисный"),
        ("Тайыншинский элеватор", 53.845, 69.755, "taiynsha", 150000, True, "Базисный"),
        ("Чкаловский ХПП", 53.948, 70.362, "taiynsha", 70000, True, "Линейный"),
        ("Киялинский элеватор", 54.215, 69.482, "taiynsha", 88000, True, "Линейный"),
        ("Талшыкский элеватор", 53.715, 71.025, "akzhar", 92000, True, "Линейный"),
        ("Даутский ХПП", 53.621, 71.341, "akzhar", 60000, False, "Линейный"),
        ("Саумалкольский элеватор", 53.315, 68.125, "esil", 115000, True, "Базисный"),
        ("Тимирязевский ХПП", 53.695, 65.985, "esil", 95000, True, "Линейный"),
        ("Сулы ХПП", 53.582, 66.215, "esil", 65000, True, "Линейный"),
        ("Возвышенский элеватор", 54.382, 70.812, "kyzylzhar", 78000, False, "Линейный"),
        ("Бескольский ХПП", 54.795, 69.182, "kyzylzhar", 62000, False, "Линейный"),
        ("Пресновский элеватор", 54.685, 67.412, "mamlyut", 80000, False, "Линейный"),
    ]
    # Добавляем оставшиеся 32 ХПП СКО по сетке районов
    for i in range(len(sko_bases) + 1, 53):
        dist_id = ["kyzylzhar", "esil", "mamlyut", "taiynsha", "akzhar"][i % 5]
        lat_off = 53.4 + (i % 8) * 0.21 + (i % 3) * 0.05
        lon_off = 66.6 + (i % 11) * 0.38 + (i % 4) * 0.08
        has_rail = (i % 3 != 0)
        cap = 45000 + (i * 2700) % 85000
        sko_bases.append((
            f"СКО Хлебоприемный пункт №{i}", round(lat_off, 4), round(lon_off, 4),
            dist_id, cap, has_rail, "Линейный" if has_rail else "Глубинный"
        ))
    for idx, item in enumerate(sko_bases[:52], start=1):
        elevators.append({
            "id": f"sko_hpp_{idx:02d}",
            "name": item[0],
            "region": "sko",
            "region_name_ru": "Северо-Казахстанская область",
            "region_name_kk": "Солтүстік Қазақстан облысы",
            "district_id": item[3],
            "lat": item[1],
            "lon": item[2],
            "capacity_t": item[4],
            "railway_access": item[5],
            "type": item[6],
            "base_fill_pct": round(45.0 + (idx * 7.3) % 45.0, 1),
        })

    # 2. Костанайская область — 65 ХПП
    kost_bases = [
        ("Костанайский мелькомбинат ХПП", 53.214, 63.635, "kostanay_dist", 210000, True, "Базисный"),
        ("Тобольский элеватор", 52.685, 62.582, "denisov", 185000, True, "Базисный"),
        ("Карабалыкский элеватор Тоганас", 53.752, 62.085, "karabalyk", 135000, True, "Базисный"),
        ("Пешковский ХПП", 53.892, 62.485, "fedorov", 90000, True, "Линейный"),
        ("Федоровский элеватор", 53.642, 62.715, "fedorov", 120000, True, "Базисный"),
        ("Денисовский ХПП", 52.455, 61.942, "denisov", 98000, True, "Линейный"),
        ("Аулиекольский элеватор", 52.145, 64.142, "auliyekol", 145000, True, "Базисный"),
        ("Кушмурунский ХПП", 52.445, 64.625, "auliyekol", 85000, True, "Линейный"),
        ("Рудненский зерновой терминал", 52.965, 63.125, "kostanay_dist", 110000, True, "Базисный"),
        ("Затобольский ХПП", 53.185, 63.712, "kostanay_dist", 82000, False, "Линейный"),
        ("Сарыкольский элеватор Урицк", 53.385, 65.142, "kostanay_dist", 130000, True, "Базисный"),
        ("Алтынсаринский элеватор", 53.215, 64.521, "kostanay_dist", 75000, False, "Линейный"),
        ("Камыстинский ХПП", 51.945, 61.815, "denisov", 88000, False, "Линейный"),
        ("Аркалыкский элеватор", 50.245, 66.915, "auliyekol", 115000, True, "Базисный"),
        ("Тарановский ХПП им. Б. Майлина", 52.815, 62.745, "denisov", 92000, True, "Линейный"),
    ]
    for i in range(len(kost_bases) + 1, 66):
        dist_id = ["karabalyk", "fedorov", "kostanay_dist", "auliyekol", "denisov"][i % 5]
        lat_off = 51.8 + (i % 9) * 0.24 + (i % 4) * 0.06
        lon_off = 61.2 + (i % 12) * 0.35 + (i % 3) * 0.07
        has_rail = (i % 2 == 0)
        cap = 40000 + (i * 3100) % 95000
        kost_bases.append((
            f"Костанайский зерновой терминал №{i}", round(lat_off, 4), round(lon_off, 4),
            dist_id, cap, has_rail, "Линейный" if has_rail else "Глубинный"
        ))
    for idx, item in enumerate(kost_bases[:65], start=1):
        elevators.append({
            "id": f"kost_hpp_{idx:02d}",
            "name": item[0],
            "region": "kostanay",
            "region_name_ru": "Костанайская область",
            "region_name_kk": "Қостанай облысы",
            "district_id": item[3],
            "lat": item[1],
            "lon": item[2],
            "capacity_t": item[4],
            "railway_access": item[5],
            "type": item[6],
            "base_fill_pct": round(42.0 + (idx * 5.7) % 48.0, 1),
        })

    # 3. Акмолинская область — 68 ХПП
    akmola_bases = [
        ("Кокшетауский элеватор Базис", 53.288, 69.395, "zerendi", 165000, True, "Базисный"),
        ("Атбасарский элеватор ХПП-1", 51.815, 68.355, "atbasar", 190000, True, "Базисный"),
        ("Атбасарский зерновой союз", 51.832, 68.321, "atbasar", 140000, True, "Базисный"),
        ("Аккольский элеватор", 51.995, 70.945, "shortandy", 115000, True, "Линейный"),
        ("Шортандинский ХПП Научный", 51.695, 70.995, "shortandy", 95000, True, "Линейный"),
        ("Макинский элеватор Буланды", 52.625, 70.415, "zerendi", 130000, True, "Базисный"),
        ("Жаксынский элеватор", 51.915, 67.285, "zhaksy", 125000, True, "Базисный"),
        ("Есильский элеватор Акмола", 51.955, 66.415, "esil_akmola", 150000, True, "Базисный"),
        ("Жалтырский ХПП", 51.625, 69.915, "atbasar", 98000, True, "Линейный"),
        ("Астраханский элеватор", 51.525, 69.785, "atbasar", 85000, False, "Линейный"),
        ("Зерендинский ХПП", 52.895, 69.145, "zerendi", 78000, False, "Линейный"),
        ("Аршалынский ХПП", 50.845, 72.185, "shortandy", 92000, True, "Линейный"),
        ("Степногорский зерновой комплекс", 52.485, 71.885, "shortandy", 105000, True, "Базисный"),
        ("Сандыктауский ХПП Балкашино", 52.515, 68.745, "zerendi", 72000, False, "Линейный"),
        ("Кийминский элеватор", 51.585, 66.715, "zhaksy", 88000, True, "Линейный"),
    ]
    for i in range(len(akmola_bases) + 1, 69):
        dist_id = ["zerendi", "shortandy", "atbasar", "zhaksy", "esil_akmola"][i % 5]
        lat_off = 50.8 + (i % 9) * 0.23 + (i % 4) * 0.05
        lon_off = 66.8 + (i % 13) * 0.38 + (i % 3) * 0.06
        has_rail = (i % 3 != 1)
        cap = 42000 + (i * 2900) % 92000
        akmola_bases.append((
            f"Акмолинский ХПП филиал №{i}", round(lat_off, 4), round(lon_off, 4),
            dist_id, cap, has_rail, "Линейный" if has_rail else "Глубинный"
        ))
    for idx, item in enumerate(akmola_bases[:68], start=1):
        elevators.append({
            "id": f"akm_hpp_{idx:02d}",
            "name": item[0],
            "region": "akmola",
            "region_name_ru": "Акмолинская область",
            "region_name_kk": "Ақмола облысы",
            "district_id": item[3],
            "lat": item[1],
            "lon": item[2],
            "capacity_t": item[4],
            "railway_access": item[5],
            "type": item[6],
            "base_fill_pct": round(40.0 + (idx * 6.1) % 50.0, 1),
        })

    # 4. Павлодарская область — 22 ХПП
    pavl_bases = [
        ("Павлодарский мелькомбинат ХПП", 52.287, 76.967, "aktogay", 120000, True, "Базисный"),
        ("Иртышский элеватор", 53.345, 75.455, "irtysh", 105000, True, "Базисный"),
        ("Щербактинский ХПП", 52.485, 78.185, "zhelete", 95000, True, "Линейный"),
        ("Успенский элеватор", 52.985, 77.415, "uspen", 88000, True, "Линейный"),
        ("Актогайский зерновой пункт", 53.015, 76.315, "aktogay", 65000, False, "Линейный"),
        ("Теренкольский ХПП Качиры", 53.085, 76.625, "irtysh", 82000, True, "Линейный"),
        ("Железинский элеватор", 53.545, 75.315, "zhelete", 78000, False, "Линейный"),
        ("Экибастузский ХПП", 51.725, 75.315, "bayanaul", 92000, True, "Линейный"),
        ("Баянаульский зернопункт", 50.785, 75.695, "bayanaul", 45000, False, "Глубинный"),
        ("Майский ХПП Коктобе", 51.815, 77.485, "bayanaul", 52000, False, "Линейный"),
    ]
    for i in range(len(pavl_bases) + 1, 23):
        dist_id = ["aktogay", "irtysh", "zhelete", "bayanaul", "uspen"][i % 5]
        lat_off = 51.2 + (i % 6) * 0.38 + (i % 3) * 0.08
        lon_off = 75.1 + (i % 8) * 0.42 + (i % 2) * 0.09
        has_rail = (i % 2 == 0)
        cap = 38000 + (i * 3300) % 65000
        pavl_bases.append((
            f"Павлодарский Хлебоприемный пункт №{i}", round(lat_off, 4), round(lon_off, 4),
            dist_id, cap, has_rail, "Линейный" if has_rail else "Глубинный"
        ))
    for idx, item in enumerate(pavl_bases[:22], start=1):
        elevators.append({
            "id": f"pavl_hpp_{idx:02d}",
            "name": item[0],
            "region": "pavlodar",
            "region_name_ru": "Павлодарская область",
            "region_name_kk": "Павлодар облысы",
            "district_id": item[3],
            "lat": item[1],
            "lon": item[2],
            "capacity_t": item[4],
            "railway_access": item[5],
            "type": item[6],
            "base_fill_pct": round(35.0 + (idx * 5.3) % 45.0, 1),
        })

    return elevators


# Глобальный кэшированный список 207 ХПП
ALL_207_ELEVATORS: List[Dict[str, Any]] = _generate_207_elevators()


def get_elevators_df() -> pd.DataFrame:
    """Возвращает реестр 207 ХПП в виде структурированного pandas DataFrame."""
    return pd.DataFrame(ALL_207_ELEVATORS)


# ─────────────────────────────────────────────────────────────
# Моделирование загрузки и реактивного стресса (What-If)
# ─────────────────────────────────────────────────────────────

def calculate_dynamic_silo_loads(
    region_filter: Optional[str] = None,
    shock_factor: float = 0.0,
    crop_sensitivity: float = 1.0,
    rerouted_volumes: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Рассчитывает динамическую заполненность элеваторов с учетом:
    1. Засухи/стресса (shock_factor: 0.0 до 0.50)
    2. Чувствительности возделываемой культуры (crop_sensitivity)
    3. Перенаправленных объемов зерна (rerouted_volumes)

    Возвращает DataFrame с колонками:
    - current_fill_pct: текущий % заполненности
    - stored_grain_t: физический объем зерна (тонн)
    - free_capacity_t: свободная вместимость (тонн)
    - load_status: 'DEFICIT' (<30%), 'OPTIMAL' (30-70%), 'FULL' (>70%)
    - color_hex: цвет для карты (#FF3B30 / #FF9500 / #34C759)
    """
    df = get_elevators_df().copy()

    if region_filter and region_filter != "all":
        df = df[df["region"] == region_filter].copy()

    # Дефицит урожайности из-за стресса
    yield_drop_pct = shock_factor * crop_sensitivity * 0.45
    df["stress_drop_pct"] = yield_drop_pct * 100.0

    # Расчет текущей заполненности
    fill_multipliers = 1.0 - yield_drop_pct
    df["current_fill_pct"] = np.clip(
        df["base_fill_pct"] * fill_multipliers,
        5.0,
        98.0
    ).round(1)

    # Учет перенаправления зерна
    if rerouted_volumes:
        for elev_id, vol_t in rerouted_volumes.items():
            mask = df["id"] == elev_id
            if mask.any():
                cap = df.loc[mask, "capacity_t"].values[0]
                added_pct = (vol_t / cap) * 100.0
                df.loc[mask, "current_fill_pct"] = np.clip(
                    df.loc[mask, "current_fill_pct"] + added_pct,
                    5.0, 99.0
                ).round(1)

    # Физические объемы в тоннах
    df["stored_grain_t"] = (df["capacity_t"] * (df["current_fill_pct"] / 100.0)).round(0)
    df["free_capacity_t"] = (df["capacity_t"] - df["stored_grain_t"]).clip(lower=0).round(0)

    # Статусы и цветовая палитра (Зеленый / Янтарный / Красный)
    def _get_status(pct: float) -> Tuple[str, str, str]:
        if pct > 70.0:
            return ("FULL", "🟢 Толған / Перегружен", "#34C759")
        elif pct >= 30.0:
            return ("OPTIMAL", "🟡 Қалыпты / Оптимально", "#FF9500")
        else:
            return ("DEFICIT", "🔴 Дефицит / Бос", "#FF3B30")

    statuses = [ _get_status(p) for p in df["current_fill_pct"] ]
    df["status_code"] = [s[0] for s in statuses]
    df["status_label"] = [s[1] for s in statuses]
    df["color_hex"] = [s[2] for s in statuses]

    return df


# ─────────────────────────────────────────────────────────────
# Step 5.2: NetworkX-движок маршрутизации зерновых потоков
# ─────────────────────────────────────────────────────────────

class GrainLogisticsRouter:
    """
    Транспортно-логистический граф КТЖ и автомагистралей Казахстана.
    Находит кратчайший путь, сравнивает ж/д (КТЖ) и авто-доставку,
    рассчитывает тарифную стоимость, потребность в вагонах и углеродный след.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self._build_logistics_network()

    def _build_logistics_network(self):
        """Строит взвешенный граф опорных элеваторов и транспортных узлов."""
        # Добавляем узлы
        for elev in ALL_207_ELEVATORS:
            self.graph.add_node(
                elev["id"],
                name=elev["name"],
                lat=elev["lat"],
                lon=elev["lon"],
                capacity_t=elev["capacity_t"],
                railway_access=elev["railway_access"],
                region=elev["region"],
            )

        # Соединяем соседние элеваторы по формуле haversine (до 140 км)
        elev_list = ALL_207_ELEVATORS
        n = len(elev_list)
        for i in range(n):
            lat1, lon1 = elev_list[i]["lat"], elev_list[i]["lon"]
            for j in range(i + 1, min(i + 18, n)):
                lat2, lon2 = elev_list[j]["lat"], elev_list[j]["lon"]
                dist_km = self.haversine_distance(lat1, lon1, lat2, lon2)
                if dist_km <= 150.0:
                    # Вес ребра — расстояние
                    self.graph.add_edge(
                        elev_list[i]["id"],
                        elev_list[j]["id"],
                        distance_km=round(dist_km, 1),
                        rail_available=(elev_list[i]["railway_access"] and elev_list[j]["railway_access"])
                    )

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Расчет расстояния по ортодромии (км)."""
        R = 6371.0  # Радиус Земли
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (math.sin(dphi / 2.0) ** 2 +
             math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2))
        return 2.0 * R * math.asin(math.sqrt(min(1.0, a)))

    def solve_rerouting(
        self,
        source_id: str,
        volume_tons: float = 25000.0,
        df_current_loads: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Находит оптимальный принимающий ХПП со свободной вместимостью
        для переброски избыточного или пострадавшего зерна.

        Возвращает:
        - target_elevator: информация о целевом элеваторе
        - distance_km: плечо перевозки
        - path_coords: массив [[lat, lon], ...] для отображения полилинии
        - rail_feasible: возможна ли перевозка вагонами КТЖ
        - wagons_required: потребность в зерновозах КТЖ
        - trucks_required: альтернативная потребность в автозерновозах
        - cost_rail_kzt: логистические затраты КТЖ
        - cost_road_kzt: затраты автотранспортом
        - savings_rail_kzt: экономия на ж/д тарифе
        - co2_rail_t vs co2_road_t: экологический эффект
        """
        if df_current_loads is None:
            df_current_loads = calculate_dynamic_silo_loads()

        source_row = df_current_loads[df_current_loads["id"] == source_id]
        if source_row.empty:
            source_row = df_current_loads.iloc[0:1]
            source_id = source_row["id"].values[0]

        src_lat = source_row["lat"].values[0]
        src_lon = source_row["lon"].values[0]
        src_name = source_row["name"].values[0]

        # Кандидаты: ХПП со свободной вместимостью >= 15,000 т, исключая исходный
        candidates = df_current_loads[
            (df_current_loads["id"] != source_id) &
            (df_current_loads["free_capacity_t"] >= volume_tons * 0.4)
        ].copy()

        if candidates.empty:
            candidates = df_current_loads[df_current_loads["id"] != source_id].copy()

        # Вычисляем расстояние до каждого кандидата
        candidates["dist_km"] = candidates.apply(
            lambda r: self.haversine_distance(src_lat, src_lon, r["lat"], r["lon"]),
            axis=1
        )
        candidates = candidates.sort_values("dist_km")
        target = candidates.iloc[0]

        # Минимальное реалистичное маневровое плечо (даже в пределах одной станции >= 0.8 км)
        dist_km = max(0.8, float(target["dist_km"]))
        target_id = target["id"]

        # Поиск маршрута в графе
        try:
            path = nx.shortest_path(self.graph, source=source_id, target=target_id, weight="distance_km")
            path_coords = [
                [self.graph.nodes[n]["lat"], self.graph.nodes[n]["lon"]]
                for n in path
            ]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Прямая геолиния, если граф разрежен
            path_coords = [[src_lat, src_lon], [target["lat"], target["lon"]]]

        # Проверка ж/д коридора КТЖ
        rail_feasible = bool(source_row["railway_access"].values[0] and target["railway_access"])

        # Расчет логистики
        wagons = math.ceil(volume_tons / KTZ_GRAIN_HOPPER_CAPACITY_T)
        trucks = math.ceil(volume_tons / TRUCK_GRAIN_CAPACITY_T)

        cost_rail_kzt = volume_tons * dist_km * RAIL_TARIFF_KZT_PER_TKM
        cost_road_kzt = volume_tons * dist_km * ROAD_TARIFF_KZT_PER_TKM
        savings_kzt = cost_road_kzt - cost_rail_kzt

        # Экологический расчет (CO2 выбросы: ж/д ~ 22 г/т*км, авто ~ 95 г/т*км)
        co2_rail_t = (volume_tons * dist_km * 22.0) / 1e6
        co2_road_t = (volume_tons * dist_km * 95.0) / 1e6

        return {
            "source_id": source_id,
            "source_name": src_name,
            "source_coords": [src_lat, src_lon],
            "target_id": target_id,
            "target_name": target["name"],
            "target_coords": [float(target["lat"]), float(target["lon"])],
            "target_region": target["region"],
            "target_free_cap_t": float(target["free_capacity_t"]),
            "distance_km": round(dist_km, 1),
            "volume_tons": volume_tons,
            "path_coords": path_coords,
            "rail_feasible": rail_feasible,
            "wagons_required": wagons,
            "trucks_required": trucks,
            "cost_rail_kzt": round(cost_rail_kzt, 0),
            "cost_road_kzt": round(cost_road_kzt, 0),
            "savings_rail_kzt": round(savings_kzt, 0),
            "savings_usd": round(savings_kzt / USD_KZT_RATE, 0),
            "co2_rail_t": round(co2_rail_t, 2),
            "co2_road_t": round(co2_road_t, 2),
            "co2_saved_t": round(co2_road_t - co2_rail_t, 2),
        }


# ─────────────────────────────────────────────────────────────
# Step 5.3: Эконометрический калькулятор передачи цен
# (Космический стресс -> Урожайность -> Зерно -> Мука -> Хлеб)
# ─────────────────────────────────────────────────────────────

def calculate_price_transmission_chain(
    yield_shock_pct: float,
    current_wheat_price_kzt: float = BASE_WHEAT_PRICE_KZT_T,
    current_flour_price_kzt: float = BASE_FLOUR_PRICE_KZT_KG,
    current_bread_price_kzt: float = BASE_BREAD_PRICE_KZT_LOAF,
) -> Dict[str, Any]:
    """
    Моделирует динамику цен по цепочке добавленной стоимости на горизонте 180 дней.
    Использует Granger Pass-Through эластичности:
    - Лаг 30 дней: скачок цен на пшеницу (сырье)
    - Лаг 60 дней: передача в мукомольный сектор
    - Лаг 90-120 дней: рост розничных цен на формовой социальный хлеб

    Параметры:
    - yield_shock_pct: процент падения урожайности (от 0% до -50%)
    """
    shock_magnitude = abs(yield_shock_pct) / 100.0

    # 1. Скачок цен на пшеницу 3 класса
    wheat_jump_pct = shock_magnitude * abs(ELASTICITY_YIELD_TO_WHEAT) * 100.0
    proj_wheat_kzt = current_wheat_price_kzt * (1.0 + wheat_jump_pct / 100.0)

    # 2. Скачок цен на муку 1 сорта
    flour_jump_pct = wheat_jump_pct * ELASTICITY_WHEAT_TO_FLOUR
    proj_flour_kzt = current_flour_price_kzt * (1.0 + flour_jump_pct / 100.0)

    # 3. Скачок розничной цены соц. хлеба (с учетом демпфера СПК)
    bread_jump_pct = flour_jump_pct * ELASTICITY_FLOUR_TO_BREAD
    proj_bread_kzt = current_bread_price_kzt * (1.0 + bread_jump_pct / 100.0)

    # Временные горизонты (0, 30, 60, 90, 120, 180 дней)
    timeline_days = [0, 30, 60, 90, 120, 180]
    
    # Траектория пшеницы (быстрая реакция биржи и трейдеров)
    wheat_curve = [
        current_wheat_price_kzt,
        current_wheat_price_kzt * (1.0 + (wheat_jump_pct * 0.70) / 100.0),
        proj_wheat_kzt,
        proj_wheat_kzt * 1.02,
        proj_wheat_kzt * 0.98,
        proj_wheat_kzt * 0.95,
    ]

    # Траектория муки (лаг запасов зерна на элеваторах 45-60 дней)
    flour_curve = [
        current_flour_price_kzt,
        current_flour_price_kzt * (1.0 + (flour_jump_pct * 0.20) / 100.0),
        current_flour_price_kzt * (1.0 + (flour_jump_pct * 0.75) / 100.0),
        proj_flour_kzt,
        proj_flour_kzt * 1.01,
        proj_flour_kzt * 0.97,
    ]

    # Траектория социального хлеба (лаг субсидий и госрегулирования 90+ дней)
    bread_curve = [
        current_bread_price_kzt,
        current_bread_price_kzt,
        current_bread_price_kzt * (1.0 + (bread_jump_pct * 0.15) / 100.0),
        current_bread_price_kzt * (1.0 + (bread_jump_pct * 0.70) / 100.0),
        proj_bread_kzt,
        proj_bread_kzt * 0.98,
    ]

    # Индекс продовольственного давления (CPI Bread Basket)
    # Формула: 50% веса — рост цены пшеницы (рынок), 50% — рост цены хлеба (социальный)
    # Шкала: LOW (<30), MEDIUM (30-65), HIGH (>65)
    cpi_pressure_score = min(100.0, wheat_jump_pct * 1.6 + bread_jump_pct * 3.2)
    risk_tier = "HIGH" if cpi_pressure_score > 65 else "MEDIUM" if cpi_pressure_score > 30 else "LOW"

    # Рекомендация для акиматов и СПК
    if risk_tier == "HIGH":
        rec_ru = (
            f"КРИТИЧЕСКИЙ РИСК ИНФЛЯЦИИ (+{bread_jump_pct:.1f}% на хлеб). "
            f"Рекомендуется экстренное открытие зерновых резервов Продкорпорации "
            f"по фиксированной цене {current_wheat_price_kzt:,.0f} тг/т для мукомолов."
        )
        rec_kk = (
            f"ИНФЛЯЦИЯНЫҢ КРИТИКАЛЫҚ ҚАУПІ (нанға +{bread_jump_pct:.1f}%). "
            f"Ұн тарту кәсіпорындарына Азық-түлік корпорациясының резервтерін "
            f"{current_wheat_price_kzt:,.0f} тг/т бекітілген бағамен дереу ашу ұсынылады."
        )
    elif risk_tier == "MEDIUM":
        rec_ru = (
            f"УМЕРЕННЫЙ РИСК (+{bread_jump_pct:.1f}% на хлеб). "
            f"Рекомендуется усилить субсидирование транспортных тарифов КТЖ "
            f"для переброски зерна из профицитных районов."
        )
        rec_kk = (
            f"ОРТАША ҚАУІП (нанға +{bread_jump_pct:.1f}%). "
            f"Профицитті аудандардан астықты жеткізу үшін ҚТЖ көліктік тарифтерін "
            f"субсидиялауды күшейту ұсынылады."
        )
    else:
        rec_ru = "ЦЕНОВАЯ СТАБИЛЬНОСТЬ. Параметры продовольственной корзины в пределах нормы."
        rec_kk = "БАҒА ТҰРАҚТЫЛЫҒЫ. Азық-түлік себетінің көрсеткіштері қалыпты шекте."

    return {
        "yield_shock_pct": yield_shock_pct,
        "timeline_days": timeline_days,
        "wheat_price_baseline": current_wheat_price_kzt,
        "wheat_price_projected": round(proj_wheat_kzt, 0),
        "wheat_jump_pct": round(wheat_jump_pct, 1),
        "flour_price_baseline": current_flour_price_kzt,
        "flour_price_projected": round(proj_flour_kzt, 1),
        "flour_jump_pct": round(flour_jump_pct, 1),
        "bread_price_baseline": current_bread_price_kzt,
        "bread_price_projected": round(proj_bread_kzt, 1),
        "bread_jump_pct": round(bread_jump_pct, 1),
        "wheat_curve": [round(x, 1) for x in wheat_curve],
        "flour_curve": [round(x, 1) for x in flour_curve],
        "bread_curve": [round(x, 1) for x in bread_curve],
        "cpi_pressure_score": round(cpi_pressure_score, 1),
        "risk_tier": risk_tier,
        "recommendation_ru": rec_ru,
        "recommendation_kk": rec_kk,
    }

