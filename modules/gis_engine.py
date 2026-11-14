"""
AgriCascade — modules/gis_engine.py
====================================
Интерактивный ГИС-движок для Республиканского финала «Дарын» (Phase 4).
Choropleth-картограмма рисков, пространственные полигоны районов,
микрозональные расчеты стресса и rich-тултипы (FAO-56 + Quareo Dataset).
"""

from typing import Dict, Any, List
import numpy as np


# ─────────────────────────────────────────────────────────────
# Оцифрованные полигоны ключевых районов Зернового пояса (WGS84)
# ─────────────────────────────────────────────────────────────

DISTRICT_REGISTRY: Dict[str, List[Dict[str, Any]]] = {
    "sko": [
        {
            "id": "kyzylzhar",
            "name_ru": "Кызылжарский район",
            "name_kk": "Қызылжар ауданы",
            "center": [54.91, 69.12],
            "polygon": [
                [68.8, 54.6], [69.4, 54.6], [69.4, 55.3], [68.8, 55.3], [68.8, 54.6]
            ],
            "arable_ha": 312000,
            "soil_ru": "Чернозем обыкновенный глубокий",
            "soil_kk": "Кәдімгі терең қара топырақ",
            "nearest_elevator": "Петропавловский ХПП (24 км)",
            "base_awc_mm": 145,
        },
        {
            "id": "esil",
            "name_ru": "Есильский район",
            "name_kk": "Есіл ауданы",
            "center": [53.88, 68.35],
            "polygon": [
                [67.8, 53.4], [68.8, 53.4], [68.8, 54.4], [67.8, 54.4], [67.8, 53.4]
            ],
            "arable_ha": 340000,
            "soil_ru": "Чернозем южный тяжелосуглинистый",
            "soil_kk": "Оңтүстік ауыр сазды қара топырақ",
            "nearest_elevator": "Явленский элеватор (18 км)",
            "base_awc_mm": 135,
        },
        {
            "id": "mamlyut",
            "name_ru": "Мамлютский район",
            "name_kk": "Мамлют ауданы",
            "center": [54.94, 68.54],
            "polygon": [
                [68.1, 54.6], [68.8, 54.6], [68.8, 55.4], [68.1, 55.4], [68.1, 54.6]
            ],
            "arable_ha": 285000,
            "soil_ru": "Чернозем выщелоченный лесостепной",
            "soil_kk": "Орманды-далалық сілтісізденген қара топырақ",
            "nearest_elevator": "Мамлютский ХПП (12 км)",
            "base_awc_mm": 150,
        },
        {
            "id": "taiynsha",
            "name_ru": "Тайыншинский район",
            "name_kk": "Тайынша ауданы",
            "center": [53.84, 69.76],
            "polygon": [
                [69.4, 53.3], [70.5, 53.3], [70.5, 54.3], [69.4, 54.3], [69.4, 53.3]
            ],
            "arable_ha": 420000,
            "soil_ru": "Чернозем южный солонцеватый",
            "soil_kk": "Оңтүстік сортаңдау қара топырақ",
            "nearest_elevator": "Тайыншинский элеватор (15 км)",
            "base_awc_mm": 125,
        },
        {
            "id": "akzhar",
            "name_ru": "Акжарский район",
            "name_kk": "Ақжар ауданы",
            "center": [53.72, 71.02],
            "polygon": [
                [70.5, 53.3], [71.5, 53.3], [71.5, 54.4], [70.5, 54.4], [70.5, 53.3]
            ],
            "arable_ha": 290000,
            "soil_ru": "Темно-каштановая карбонатная",
            "soil_kk": "Күңгірт қара-қоңыр карбонатты",
            "nearest_elevator": "Талшыкский элеватор (28 км)",
            "base_awc_mm": 115,
        },
    ],
    "kostanay": [
        {
            "id": "karabalyk",
            "name_ru": "Карабалыкский район",
            "name_kk": "Қарабалық ауданы",
            "center": [53.75, 62.10],
            "polygon": [
                [61.0, 53.3], [62.4, 53.3], [62.4, 54.4], [61.0, 54.4], [61.0, 53.3]
            ],
            "arable_ha": 390000,
            "soil_ru": "Чернозем южный зернистый",
            "soil_kk": "Оңтүстік дәнді қара топырақ",
            "nearest_elevator": "Тогузакский ХПП (14 км)",
            "base_awc_mm": 140,
        },
        {
            "id": "fedorov",
            "name_ru": "Федоровский район",
            "name_kk": "Федоров ауданы",
            "center": [53.63, 62.70],
            "polygon": [
                [62.4, 53.2], [63.3, 53.2], [63.3, 54.3], [62.4, 54.3], [62.4, 53.2]
            ],
            "arable_ha": 410000,
            "soil_ru": "Чернозем обыкновенный глинистый",
            "soil_kk": "Кәдімгі саздақты қара топырақ",
            "nearest_elevator": "Пешковский элеватор (22 км)",
            "base_awc_mm": 138,
        },
        {
            "id": "mendykar",
            "name_ru": "Мендыкаринский район",
            "name_kk": "Меңдіқара ауданы",
            "center": [53.80, 64.30],
            "polygon": [
                [63.8, 53.2], [65.2, 53.2], [65.2, 54.4], [63.8, 54.4], [63.8, 53.2]
            ],
            "arable_ha": 360000,
            "soil_ru": "Чернозем южный суглинистый",
            "soil_kk": "Оңтүстік саздауыт қара топырақ",
            "nearest_elevator": "Боровской элеватор (19 км)",
            "base_awc_mm": 132,
        },
        {
            "id": "kostanay_dist",
            "name_ru": "Костанайский район",
            "name_kk": "Қостанай ауданы",
            "center": [53.21, 63.63],
            "polygon": [
                [62.8, 52.6], [64.1, 52.6], [64.1, 53.5], [62.8, 53.5], [62.8, 52.6]
            ],
            "arable_ha": 380000,
            "soil_ru": "Темно-каштановая солонцеватая",
            "soil_kk": "Күңгірт қара-қоңыр сортаңдау",
            "nearest_elevator": "Костанайский мелькомбинат (10 км)",
            "base_awc_mm": 120,
        },
        {
            "id": "auliekol",
            "name_ru": "Аулиекольский район",
            "name_kk": "Әулиекөл ауданы",
            "center": [52.35, 64.15],
            "polygon": [
                [63.0, 51.5], [64.8, 51.5], [64.8, 52.8], [63.0, 52.8], [63.0, 51.5]
            ],
            "arable_ha": 440000,
            "soil_ru": "Темно-каштановая супесчаная",
            "soil_kk": "Күңгірт қара-қоңыр құмдауыт",
            "nearest_elevator": "Кушмурунский элеватор (31 км)",
            "base_awc_mm": 108,
        },
    ],
    "akmola": [
        {
            "id": "zerendi",
            "name_ru": "Зерендинский район",
            "name_kk": "Зеренді ауданы",
            "center": [53.02, 69.15],
            "polygon": [
                [68.4, 52.5], [69.6, 52.5], [69.6, 53.5], [68.4, 53.5], [68.4, 52.5]
            ],
            "arable_ha": 360000,
            "soil_ru": "Чернозем южный щебнистый",
            "soil_kk": "Оңтүстік тастақты қара топырақ",
            "nearest_elevator": "Кокшетауский элеватор (20 км)",
            "base_awc_mm": 130,
        },
        {
            "id": "sandyktau",
            "name_ru": "Сандыктауский район",
            "name_kk": "Сандықтау ауданы",
            "center": [52.60, 68.80],
            "polygon": [
                [68.2, 52.0], [69.3, 52.0], [69.3, 52.9], [68.2, 52.9], [68.2, 52.0]
            ],
            "arable_ha": 330000,
            "soil_ru": "Чернозем обыкновенный лесостепной",
            "soil_kk": "Кәдімгі орманды-далалық қара топырақ",
            "nearest_elevator": "Балкашинский ХПП (15 км)",
            "base_awc_mm": 142,
        },
        {
            "id": "burabay",
            "name_ru": "Бурабайский район",
            "name_kk": "Бурабай ауданы",
            "center": [52.95, 70.20],
            "polygon": [
                [69.6, 52.5], [70.8, 52.5], [70.8, 53.4], [69.6, 53.4], [69.6, 52.5]
            ],
            "arable_ha": 250000,
            "soil_ru": "Серые лесные остепненные",
            "soil_kk": "Далаланған сұр орман топырағы",
            "nearest_elevator": "Щучинский элеватор (14 км)",
            "base_awc_mm": 135,
        },
        {
            "id": "atbasar",
            "name_ru": "Атбасарский район",
            "name_kk": "Атбасар ауданы",
            "center": [51.81, 68.35],
            "polygon": [
                [67.4, 51.0], [69.0, 51.0], [69.0, 52.2], [67.4, 52.2], [67.4, 51.0]
            ],
            "arable_ha": 480000,
            "soil_ru": "Темно-каштановая карбонатная",
            "soil_kk": "Күңгірт қара-қоңыр карбонатты",
            "nearest_elevator": "Атбасарский элеватор (8 км)",
            "base_awc_mm": 118,
        },
        {
            "id": "zhaksy",
            "name_ru": "Жаксынский район",
            "name_kk": "Жақсы ауданы",
            "center": [51.92, 67.30],
            "polygon": [
                [67.0, 51.2], [68.1, 51.2], [68.1, 52.5], [67.0, 52.5], [67.0, 51.2]
            ],
            "arable_ha": 390000,
            "soil_ru": "Темно-каштановая суглинистая",
            "soil_kk": "Күңгірт қара-қоңыр саздауыт",
            "nearest_elevator": "Жаксынский ХПП (11 км)",
            "base_awc_mm": 112,
        },
    ],
    "pavlodar": [
        {
            "id": "irtyshsk",
            "name_ru": "Иртышский район",
            "name_kk": "Ертіс ауданы",
            "center": [53.33, 75.45],
            "polygon": [
                [74.5, 52.8], [76.0, 52.8], [76.0, 54.0], [74.5, 54.0], [74.5, 52.8]
            ],
            "arable_ha": 310000,
            "soil_ru": "Темно-каштановая супесчаная",
            "soil_kk": "Күңгірт қара-қоңыр құмдауыт",
            "nearest_elevator": "Иртышский ХПП (16 км)",
            "base_awc_mm": 110,
        },
        {
            "id": "zhelezinka",
            "name_ru": "Железинский район",
            "name_kk": "Железин ауданы",
            "center": [53.53, 75.92],
            "polygon": [
                [75.4, 53.0], [76.8, 53.0], [76.8, 54.2], [75.4, 54.2], [75.4, 53.0]
            ],
            "arable_ha": 290000,
            "soil_ru": "Светло-каштановая легкая",
            "soil_kk": "Ашық қара-қоңыр жеңіл топырақ",
            "nearest_elevator": "Железинский ХПП (12 км)",
            "base_awc_mm": 95,
        },
        {
            "id": "aktogay_pvl",
            "name_ru": "Актогайский район",
            "name_kk": "Ақтоғай ауданы",
            "center": [52.98, 75.05],
            "polygon": [
                [74.6, 52.2], [75.8, 52.2], [75.8, 53.2], [74.6, 53.2], [74.6, 52.2]
            ],
            "arable_ha": 260000,
            "soil_ru": "Каштановая солонцеватая",
            "soil_kk": "Қара-қоңыр сортаңдау топырақ",
            "nearest_elevator": "Актогайский пункт (25 км)",
            "base_awc_mm": 102,
        },
        {
            "id": "terenkol",
            "name_ru": "Район Теренколь",
            "name_kk": "Тереңкөл ауданы",
            "center": [53.10, 76.65],
            "polygon": [
                [76.0, 52.4], [77.5, 52.4], [77.5, 53.6], [76.0, 53.6], [76.0, 52.4]
            ],
            "arable_ha": 280000,
            "soil_ru": "Светло-каштановая карбонатная",
            "soil_kk": "Ашық қара-қоңыр карбонатты",
            "nearest_elevator": "Качирский элеватор (18 км)",
            "base_awc_mm": 98,
        },
        {
            "id": "uspenka",
            "name_ru": "Успенский район",
            "name_kk": "Успен ауданы",
            "center": [53.00, 77.40],
            "polygon": [
                [76.8, 52.2], [78.6, 52.2], [78.6, 53.6], [76.8, 53.6], [76.8, 52.2]
            ],
            "arable_ha": 250000,
            "soil_ru": "Каштановая супесчаная засушливая",
            "soil_kk": "Қара-қоңыр құмдауыт құрғақ топырақ",
            "nearest_elevator": "Таволжанский пункт (29 км)",
            "base_awc_mm": 92,
        },
    ]
}


def compute_district_choropleth_geojson(
    region_id: str,
    lang: str = "kk",
    current_psi: float = -0.5,
    crop_kc: float = 1.15,
    crop_tolerance: float = 3.5,
    crop_name: str = "Пшеница яровая",
    shock_factor: float = 0.0,
) -> Dict[str, Any]:
    """
    Генерирует GeoJSON FeatureCollection с раскраской Choropleth и rich-тултипами.
    Каждый район получает цвет в зависимости от результирующего индекса риска R [0, 100].
    """
    districts = DISTRICT_REGISTRY.get(region_id, DISTRICT_REGISTRY["sko"])
    
    # Биологический мультипликатор чувствительности культуры по FAO-56
    crop_sensitivity = (crop_kc / 1.10) * (4.2 / max(crop_tolerance, 1.0))
    
    features = []

    for d in districts:
        # Детерминированная микрозональная поправка по влагоемкости почвы района.
        awc_adjustment = ((145.0 - d["base_awc_mm"]) / 100.0) * 0.35
        d_psi = current_psi - (crop_kc - 1.0) * 0.35 - awc_adjustment
        d_psi -= (shock_factor * 5.0 * crop_sensitivity)
        
        # Расчет индекса риска R: 0 (идеально) -> 100 (полная катастрофа)
        # Базируется на дефиците PSI, почвенной влагоемкости AWC и шоке
        base_awc_factor = (145.0 - d["base_awc_mm"]) / 100.0  # Чем меньше AWC, тем выше риск
        risk_score = 50.0 - (d_psi * 18.0) + (shock_factor * 58.0 * crop_sensitivity) + (base_awc_factor * 12.0)
        risk_score = float(np.clip(risk_score, 5.0, 98.0))

        # Определение статуса и цвета палитры
        if risk_score >= 65.0 or d_psi <= -1.5:
            risk_level = "HIGH"
            fill_color = "#FF3B30"  # Красный
            risk_label = "🔴 Критикалық тәуекел" if lang == "kk" else "🔴 Критический риск"
            recommendation = (
                "Су үнемдеу технологияларын шұғыл енгізу (No-Till, гидрогельдер, антитранспиранттар)"
                if lang == "kk" else
                "Срочные влагосберегающие меры (No-Till мульчирование, гидрогели, антитранспиранты)"
            )
        elif risk_score >= 40.0 or d_psi < -0.5:
            risk_level = "MEDIUM"
            fill_color = "#FF9500"  # Янтарный / Оранжевый
            risk_label = "🟡 Орташа стресс" if lang == "kk" else "🟡 Умеренный стресс"
            recommendation = (
                "Апта сайынғы ылғал мониторингін күшейту, жапырақтық үстеп қоректендіру"
                if lang == "kk" else
                "Усиление еженедельного мониторинга влажности, фолиарная подкормка стимуляторами"
            )
        else:
            risk_level = "LOW"
            fill_color = "#34C759"  # Зеленый
            risk_label = "🟢 Қалыпты даму" if lang == "kk" else "🟢 Оптимальное развитие"
            recommendation = (
                "Стандартты агротехникалық күтім, жоғары өнімді сақтауға дайындық"
                if lang == "kk" else
                "Стандартный агротехнологический уход, подготовка мощностей к высокому урожаю"
            )

        name = d["name_kk"] if lang == "kk" else d["name_ru"]
        soil = d["soil_kk"] if lang == "kk" else d["soil_ru"]
        available_moisture_pct = int(np.clip(70.0 + d_psi * 15.0 - shock_factor * 40.0, 15.0, 95.0))

        # HTML всплывающая карточка (Rich Tooltip & Popup)
        tooltip_html = f"""
        <div style="font-family: system-ui, sans-serif; font-size: 13px; line-height: 1.5; color: #fff; background: rgba(13, 27, 42, 0.96); padding: 10px 14px; border-radius: 8px; border: 1px solid {fill_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.5); min-width: 250px;">
            <div style="font-weight: 700; font-size: 14px; color: #40c4ff; margin-bottom: 6px;">📍 {name}</div>
            <div>• <b>{'Мәртебе' if lang == 'kk' else 'Статус'}:</b> {risk_label} (R={risk_score:.0f}/100)</div>
            <div>• <b>{'Дақыл' if lang == 'kk' else 'Культура'}:</b> {crop_name} (Kc={crop_kc:.2f})</div>
            <div>• <b>{'Егістік' if lang == 'kk' else 'Пашня'}:</b> {d['arable_ha']:,} га</div>
            <div>• <b>{'Топырақ' if lang == 'kk' else 'Почва'}:</b> {soil}</div>
            <div>• <b>{'Ылғал қоры' if lang == 'kk' else 'Влага'}:</b> {available_moisture_pct}% (AWC {d['base_awc_mm']} мм)</div>
            <div>• <b>{'Жақын ХПП' if lang == 'kk' else 'Ближ. элеватор'}:</b> {d['nearest_elevator']}</div>
            <div style="margin-top: 6px; font-size: 11px; color: #cbd5e1; border-top: 1px dashed rgba(255,255,255,0.2); padding-top: 4px;"><b>💡 {'Ұсыныс' if lang == 'kk' else 'Действие'}:</b> {recommendation}</div>
        </div>
        """

        feature = {
            "type": "Feature",
            "id": d["id"],
            "geometry": {
                "type": "Polygon",
                "coordinates": [d["polygon"]],
            },
            "properties": {
                "id": d["id"],
                "name": name,
                "arable_ha": d["arable_ha"],
                "soil": soil,
                "psi": round(d_psi, 2),
                "risk_score": round(risk_score, 1),
                "risk_level": risk_level,
                "risk_label": risk_label,
                "available_moisture_pct": available_moisture_pct,
                "nearest_elevator": d["nearest_elevator"],
                "fill_color": fill_color,
                "tooltip_html": tooltip_html,
            },
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }

