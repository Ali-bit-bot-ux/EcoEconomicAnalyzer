"""
AgriCascade — modules/executive_audit.py
========================================
Генератор институциональной аналитической записки экспресс-аудита региона (Step 6.2).
Формирует комплексный отчет для акиматов зерновых областей и МСХ РК:
1. Спутниковая космическая диагностика (Sentinel-1/2, NASA SMAP)
2. Микрозональный анализ уязвимости районов (Quareo Dataset, AWC)
3. Логистический баланс элеваторов (207 ХПП, КТЖ зерновозы, арбитраж)
4. Эконометрическая цепочка передачи цен (Granger Pass-Through)
5. Комплекс агрономических директив (Quareo AgroAdvisor)
"""

from typing import Dict, Any, Optional
import datetime
import pandas as pd
from config.settings import GRAIN_BELT_REGIONS, TARGET_CROPS
from modules.elevator_logistics import (
    calculate_dynamic_silo_loads,
    calculate_price_transmission_chain,
    GrainLogisticsRouter,
)
from modules.gis_engine import compute_district_choropleth_geojson


_AUDIT_ROUTER_SINGLETON: Optional[GrainLogisticsRouter] = None

def _get_audit_router() -> GrainLogisticsRouter:
    global _AUDIT_ROUTER_SINGLETON
    if _AUDIT_ROUTER_SINGLETON is None:
        _AUDIT_ROUTER_SINGLETON = GrainLogisticsRouter()
    return _AUDIT_ROUTER_SINGLETON


def generate_executive_audit_markdown(
    year: int,
    region_key: str,
    crop_key: str = "spring_wheat",
    current_psi: float = -0.5,
    smai_val: float = -0.4,
    shock_factor: float = 0.0,
    lang: str = "kk",
) -> str:
    """
    Генерирует полную аналитическую записку экспресс-аудита региона
    для акимата и Министерства сельского хозяйства РК в формате Markdown.
    """
    today_str = datetime.datetime.now().strftime("%d.%m.%Y")
    reg_info = GRAIN_BELT_REGIONS.get(region_key, GRAIN_BELT_REGIONS["sko"])

    # Нормализация ключа культуры
    crop_aliases = {
        "wheat_spring": "spring_wheat",
        "wheat": "spring_wheat",
        "barley_spring": "spring_barley",
        "barley": "spring_barley",
        "flax": "flax_linseed",
        "linseed": "flax_linseed",
    }
    canonical_crop_key = crop_aliases.get(crop_key, crop_key)
    crop_info = TARGET_CROPS.get(canonical_crop_key, TARGET_CROPS["spring_wheat"])

    reg_name = reg_info["name_kk" if lang == "kk" else "name_ru"]
    crop_name = crop_info["name_kk" if lang == "kk" else "name_ru"]
    crop_kc = crop_info["kc"]
    arable_ha = reg_info["arable_area_ha"]

    # 1. Данные по элеваторам региона
    df_silos = calculate_dynamic_silo_loads(
        region_filter=region_key if region_key != "all" else None,
        shock_factor=shock_factor,
        crop_sensitivity=crop_info.get("kc", 1.15) / 1.10,
    )
    total_cap = int(df_silos["capacity_t"].sum())
    total_stored = int(df_silos["stored_grain_t"].sum())
    total_free = int(df_silos["free_capacity_t"].sum())
    n_silos = len(df_silos)
    n_deficit = int((df_silos["status_code"] == "DEFICIT").sum())
    n_optimal = int((df_silos["status_code"] == "OPTIMAL").sum())
    n_full = int((df_silos["status_code"] == "FULL").sum())
    n_rail = int(df_silos["railway_access"].sum())
    avg_fill = float(df_silos["current_fill_pct"].mean())

    # 2. Логистический рероутинг при дефиците
    router = _get_audit_router()
    source_id = df_silos.iloc[0]["id"]
    reroute_res = router.solve_rerouting(
        source_id=source_id,
        volume_tons=25000.0,
        df_current_loads=df_silos,
    )

    # 3. Эконометрический расчет ценовой цепи
    yield_shock_pct = -shock_factor * 100.0 if shock_factor > 0 else 0.0
    price_chain = calculate_price_transmission_chain(yield_shock_pct=yield_shock_pct)

    # 4. Данные по районам (ГИС-картограмма)
    geo_data = compute_district_choropleth_geojson(
        region_id=region_key if region_key != "all" else "sko",
        lang=lang,
        current_psi=current_psi,
        crop_kc=crop_kc,
        crop_tolerance=crop_info.get("drought_tolerance_score", 3.5),
        crop_name=crop_name,
        shock_factor=shock_factor,
    )
    districts = geo_data["features"]

    if lang == "kk":
        risk_tier_lbl = "🔴 ЖОҒАРЫ ТӘУЕКЕЛ" if current_psi < -1.5 or shock_factor > 0.25 else ("🟡 ОРТАША СТРЕСС" if current_psi < -0.5 else "🟢 ҚАЛЫПТЫ ЖАҒДАЙ")
        
        # Генерация таблицы районов
        dist_rows = []
        for feat in districts:
            p = feat["properties"]
            risk_badge = "🔴 Жоғары" if p["risk_level"] == "HIGH" else ("🟡 Орташа" if p["risk_level"] == "MEDIUM" else "🟢 Қалыпты")
            dist_rows.append(
                f"| {p['name']} | {p['arable_ha']:,} га | {p['soil']} | {p['available_moisture_pct']}% | {p['risk_score']:.0f}/100 | {risk_badge} |"
            )
        dist_table_str = "\n".join(dist_rows)

        return f"""# ҚАЗАҚСТАН РЕСПУБЛИКАСЫ АУЫЛ ШАРУАШЫЛЫҒЫ МИНИСТРЛІГІ
## {reg_name.upper()} ӘКІМДІГІ
### ЖЕДЕЛ ТАЛДАМАЛЫҚ АНЫҚТАМА
**Тақырыбы:** «{reg_name} аумағындағы агро-климаттық тәуекелдер, элеваторлық инфрақұрылым және азық-түлік қауіпсіздігіне экспресс-аудит»

---

**Құрылған күні:** {today_str}  
**Талданатын маусым:** {year} жыл  
**Талдау өңірі:** {reg_name} (Астық белдеуі)  
**Егістік алқабы:** {arable_ha / 1e6:.2f} млн га  
**Негізгі бақылау дақылы:** {crop_name} (FAO-56 Kc={crop_kc:.2f})  
**Жүйе:** AgriCascade Early Warning Economic Platform  

---

## 1. Спутниктік және климаттық мониторинг (Космомониторинг)

- **PhenoShift Index (PSI):** `{current_psi:+.3f}` (10 жылдық климаттық қалыптан ауытқу).
- **NASA SMAP топырақ ылғалдылығы (SMAI):** `{smai_val:+.2f}` (тамыр аймағының тереңдігі 0–100 см).
- **Ерте ескерту гандикапы:** **21–27 күн** (NDVI жапырақтары сарғайғанға дейін топырақтағы ылғал тапшылығын радармен бекіту).
- **Жалпы агро-экологиялық мәртебе:** **{risk_tier_lbl}**.

---

## 2. Аудандардың микроаймақтық осалдығы ({len(districts)} аудан)

| Аудан | Егістік алқабы | Топырақ түрі | Ылғал қоры | Тәуекел индексі (R) | Мәртебесі |
|---|---|---|---|---|---|
{dist_table_str}

---

## 3. Элеваторлық инфрақұрылым және логистика балансы ({n_silos} ХПП)

- **Жалпы сыйымдылық:** **{total_cap / 1e6:.2f} млн тонна** ({n_silos} лицензияланған ХПП).
- **Ағымдағы сақталған астық:** **{total_stored / 1e3:,.0f} мың тонна** (орташа толымдылық: **{avg_fill:.1f}%**).
- **Резервтік бос сыйымдылық:** **{total_free / 1e3:,.0f} мың тонна**.
- **ХПП толымдылық жағдайы:** 🟢 Толық: {n_full} | 🟡 Қалыпты: {n_optimal} | 🔴 Тапшылық/Бос: {n_deficit} ХПП.
- **Теміржол қолжетімділігі:** {n_rail} ХПП ҚТЖ магистралімен байланысқан (қамту: **{n_rail / n_silos * 100:.0f}%**).

### 🔀 Астық ағынын қайта бағыттау жоспары (ҚТЖ логистикасы):
- **Маршрут:** `{reroute_res['source_name']}` $\\to$ `{reroute_res['target_name']}`.
- **Қашықтық:** **{reroute_res['distance_km']} км**.
- **Қажетті жылжымалы құрам:** **{reroute_res['wagons_required']} вагон-астық тасығыш (ҚТЖ)** (автобалама: {reroute_res['trucks_required']} КамАЗ).
- **ҚТЖ тарифінің үнемдеуі:** **{reroute_res['savings_usd']:,.0f} USD** ({reroute_res['savings_rail_kzt'] / 1e6:.1f} млн теңге автокөлікпен салыстырғанда).
- **Экологиялық әсер:** **{reroute_res['co2_saved_t']} тонна CO₂** эмиссиясы азайтылды.

---

## 4. Баға берілу тізбегі және азық-түлік қауіпсіздігі (Granger Pass-Through)

- **Болжамды өнімділік құлдырауы:** **{yield_shock_pct:.1f}%**.
- **Бидай бағасының өсуі (+30 күн лаг):** **{price_chain['wheat_jump_pct']:+.1f}%** ({price_chain['wheat_price_baseline']:,.0f} $\\to$ **{price_chain['wheat_price_projected']:,.0f} ₸/т**).
- **1-сұрып ұн бағасының өсуі (+60 күн лаг):** **{price_chain['flour_jump_pct']:+.1f}%** ({price_chain['flour_price_baseline']:.0f} $\\to$ **{price_chain['flour_price_projected']:.0f} ₸/кг**).
- **Формовой нанның өсуі (+90–120 күн лаг):** **{price_chain['bread_jump_pct']:+.1f}%** ({price_chain['bread_price_baseline']:.0f} $\\to$ **{price_chain['bread_price_projected']:.0f} ₸/дана**).
- **Тұтыну бағаларының қысым индексі (CPI Food Basket):** **{price_chain['cpi_pressure_score']:.0f}/100** (Мәртебе: **{price_chain['risk_tier']}**).

---

## 5. Әкімдік пен Ауыл шаруашылығы министрлігі үшін ресми шешімдер (Директивалар)

1. **Әлеуметтік нанды тұрақтандыру:**
   {price_chain['recommendation_kk']}
2. **Логистика және тасымалдау:**
   ҚТЖ арқылы профицитті элеваторларға теміржол тарифін субсидиялау тетігін шұғыл іске қосу.
3. **Агрономиялық ылғал үнемдеу:**
   Тапшылық аудандарында No-Till сабан мульчасын сақтау, топыраққа гидрогель түйіршіктерін енгізу және транспирация шығынын азайту үшін антитранспиранттар шашу.

---

*Аналитикалық анықтама AgriCascade спутниктік зияткерлік платформасымен бекітілген.*  
*Космомониторинг: Sentinel-1 SAR (ЕКА) · Sentinel-2 · NASA SMAP · ҚТЖ логистика графы.*  
*Верификация: Республикалық «Дарын» ғылыми жарыстары.*
"""

    else:
        # Русский язык
        risk_tier_lbl = "🔴 ВЫСОКИЙ РИСК" if current_psi < -1.5 or shock_factor > 0.25 else ("🟡 УМЕРЕННЫЙ СТРЕСС" if current_psi < -0.5 else "🟢 НОРМАЛЬНОЕ СОСТОЯНИЕ")
        dist_rows = []
        for feat in districts:
            p = feat["properties"]
            risk_badge = "🔴 Высокий" if p["risk_level"] == "HIGH" else ("🟡 Умеренный" if p["risk_level"] == "MEDIUM" else "🟢 Норма")
            dist_rows.append(
                f"| {p['name']} | {p['arable_ha']:,} га | {p['soil']} | {p['available_moisture_pct']}% | {p['risk_score']:.0f}/100 | {risk_badge} |"
            )
        dist_table_str = "\n".join(dist_rows)

        return f"""# МИНИСТЕРСТВО СЕЛЬСКОГО ХОЗЯЙСТВА РЕСПУБЛИКИ КАЗАХСТАН
## АКИМАТ {reg_name.upper()}
### ОПЕРАТИВНАЯ АНАЛИТИЧЕСКАЯ ЗАПИСКА
**Тема:** «Экспресс-аудит агроклиматических рисков, элеваторной инфраструктуры и продовольственной безопасности {reg_name}»

---

**Дата формирования:** {today_str}  
**Сезон анализа:** {year} год  
**Регион анализа:** {reg_name} (Зерновой пояс)  
**Площадь пашни:** {arable_ha / 1e6:.2f} млн га  
**Контрольная культура:** {crop_name} (FAO-56 Kc={crop_kc:.2f})  
**Платформа:** AgriCascade Early Warning Economic Platform  

---

## 1. Спутниковая космическая диагностика (Космомониторинг)

- **PhenoShift Index (PSI):** `{current_psi:+.3f}` (отклонение вегетации от 10-летней климатической нормы).
- **Почвенная влажность NASA SMAP (SMAI):** `{smai_val:+.2f}` (корневой горизонт 0–100 см).
- **Гандикап упреждения:** **21–27 дней** (фиксация влагодефицита радаром до оптического пожелтения листьев NDVI).
- **Общий агроэкологический статус:** **{risk_tier_lbl}**.

---

## 2. Микрозональная уязвимость районов ({len(districts)} районов)

| Район | Площадь пашни | Тип почвы | Запас влаги | Индекс риска (R) | Статус |
|---|---|---|---|---|---|
{dist_table_str}

---

## 3. Элеваторная инфраструктура и баланс логистики ({n_silos} ХПП)

- **Общая емкость хранения:** **{total_cap / 1e6:.2f} млн тонн** ({n_silos} лицензированных ХПП).
- **Текущий объем хранимого зерна:** **{total_stored / 1e3:,.0f} тыс. тонн** (средняя загрузка: **{avg_fill:.1f}%**).
- **Резервная свободная емкость:** **{total_free / 1e3:,.0f} тыс. тонн**.
- **Распределение ХПП по загрузке:** 🟢 Заполнены: {n_full} | 🟡 Оптимально: {n_optimal} | 🔴 Дефицит/Пустые: {n_deficit} ХПП.
- **Ж/д доступность:** {n_rail} ХПП подключены к магистральной сети КТЖ (покрытие: **{n_rail / n_silos * 100:.0f}%**).

### 🔀 План перенаправления зернопотоков (Логистический арбитраж КТЖ):
- **Маршрут переброски:** `{reroute_res['source_name']}` $\\to$ `{reroute_res['target_name']}`.
- **Расстояние переброски:** **{reroute_res['distance_km']} км**.
- **Потребность в подвижном составе:** **{reroute_res['wagons_required']} вагонов-зерновозов КТЖ** (автоальтернатива: {reroute_res['trucks_required']} КамАЗов).
- **Экономия ж/д тарифа КТЖ:** **{reroute_res['savings_usd']:,.0f} USD** ({reroute_res['savings_rail_kzt'] / 1e6:.1f} млн тенге по сравнению с автотранспортом).
- **Экологический эффект:** снижение углеродного следа на **{reroute_res['co2_saved_t']} тонн CO₂**.

---

## 4. Цепочка передачи цен и продовольственная безопасность (Granger Pass-Through)

- **Прогнозируемый шок урожайности:** **{yield_shock_pct:.1f}%**.
- **Рост цен на пшеницу (+30 дней лаг):** **{price_chain['wheat_jump_pct']:+.1f}%** ({price_chain['wheat_price_baseline']:,.0f} $\\to$ **{price_chain['wheat_price_projected']:,.0f} ₸/т**).
- **Рост оптовой цены муки 1 сорта (+60 дней лаг):** **{price_chain['flour_jump_pct']:+.1f}%** ({price_chain['flour_price_baseline']:.0f} $\\to$ **{price_chain['flour_price_projected']:.0f} ₸/кг**).
- **Рост цены формового социального хлеба (+90–120 дней лаг):** **{price_chain['bread_jump_pct']:+.1f}%** ({price_chain['bread_price_baseline']:.0f} $\\to$ **{price_chain['bread_price_projected']:.0f} ₸/булка**).
- **Индекс давления на корзину (CPI Food Basket):** **{price_chain['cpi_pressure_score']:.0f}/100** (Уровень: **{price_chain['risk_tier']}**).

---

## 5. Комплекс управленческих и агрономических мер (Директивы для акимата и МСХ)

1. **Стабилизация цен на социальный хлеб:**
   {price_chain['recommendation_ru']}
2. **Логистические директивы:**
   Задействовать субсидирование железнодорожного тарифа КТЖ для переброски зерна в дефицитные районы из профицитных элеваторов.
3. **Влагосберегающая агрономия (Quareo AgroAdvisor):**
   В районах с высоким риском применять стерневое мульчирование No-Till, вносить гидрогели в прикорневую зону и проводить обработку антитранспирантами для снижения потерь на непродуктивное испарение.

---

*Сгенерировано аналитической платформой AgriCascade.*  
*Источники данных: Sentinel-1 SAR (ЕКА) · Sentinel-2 · NASA SMAP · Сеть элеваторов и КТЖ.*  
*Верифицировано для Республиканского финала научных соревнований «Дарын».*
"""
