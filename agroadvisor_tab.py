# -*- coding: utf-8 -*-
"""
AgroAdvisor component for AgriCascade Streamlit Dashboard.
Connects the scientific Quareo Drought Dataset and AI Recommendation Engine
(16 crops, 7 soils, 17 practices, scientific citations from Gale/FAO/ICARDA/КазНИИЗиР).
Fully bilingual: every UI label and dataset value follows the selected `lang`.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add dashboard and cwd to sys.path
for p in [Path(__file__).resolve().parent, Path.cwd(), Path.cwd() / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from src.dataset_loader import DroughtDataset
    from src.drought_advisor import DroughtAdvisor
except ImportError:
    from dataset_loader import DroughtDataset
    from drought_advisor import DroughtAdvisor


@st.cache_resource
def get_advisor_engine():
    """Initializes and caches the DroughtDataset and DroughtAdvisor."""
    possible_paths = [
        PROJECT_ROOT / "dataset",
        Path(__file__).resolve().parent.parent / "dataset",
        Path(__file__).resolve().parent / "dataset",
        Path.cwd() / "dataset",
        Path.cwd() / "dashboard" / "dataset",
        Path("D:/legion/Projects/daryn/dataset"),
    ]
    dataset_dir = next((p for p in possible_paths if p.exists()), None)
    if dataset_dir is None:
        raise FileNotFoundError("Папка dataset не найдена ни в одном из путей.")
    ds = DroughtDataset(dataset_dir=str(dataset_dir))
    advisor = DroughtAdvisor(ds)
    return ds, advisor


def _T(obj: dict, key: str, lang: str, default=""):
    """Return the Kazakh *_kk field when lang == 'kk' (with Russian fallback)."""
    if lang == "kk":
        v = obj.get(key + "_kk")
        if v:
            return v
    return obj.get(key + "_ru", obj.get(key, default))


def render_agroadvisor_tab(lang: str = "kk"):
    """Renders the comprehensive AgroAdvisor tab in Streamlit."""
    kk = (lang == "kk")
    try:
        ds, advisor = get_advisor_engine()
    except Exception as e:
        st.error(f"Қате: Датасетті жүктеу мүмкін болмады: {e}" if kk else f"Ошибка загрузки датасета: {e}")
        return

    # Header
    if kk:
        st.markdown("## 🌾 Quareo AgroAdvisor — Агрономиялық шешімдер мен білім базасы")
        st.markdown(
            "Құрғақшылық жағдайында фермерлерге қолдау көрсететін ғылыми негізделген ұсынымдық жүйе. "
            "**Gale OneFile: Agriculture**, **FAO Land and Water**, **ICARDA** және "
            "**ҚазЕжӨҒЗИ (КазНИИЗиР)** зерттеулері негізінде жасақталған."
        )
    else:
        st.markdown("## 🌾 Quareo AgroAdvisor — Научно обоснованный датасет и система поддержки")
        st.markdown(
            "Рекомендательный ИИ-движок для поддержки фермерских хозяйств в условиях засухи и дефицита влаги. "
            "Создан на базе исследований **Gale OneFile: Agriculture**, **FAO Land and Water**, "
            "**ICARDA** и научных рекомендаций **КазНИИ земледелия и растениеводства**."
        )

    # Top KPI Metrics of the Dataset
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="🌾 Дақылдар" if kk else "🌾 Сельхозкультуры",
            value=f"{len(ds.crops)} " + ("түрі" if kk else "шт"),
            delta="Дәнді, майлы, бұршақ" if kk else "Зерновые, масличные, бобовые",
            delta_color="off"
        )
    with kpi2:
        st.metric(
            label="🌍 Топырақ түрлері" if kk else "🌍 Почвенные зоны",
            value=f"{len(ds.soils)} " + ("түрі" if kk else "типов"),
            delta="Қазақстанның астық белдеуі" if kk else "Север, сухие степи, юг",
            delta_color="off"
        )
    with kpi3:
        st.metric(
            label="🛡️ Агротехнологиялар" if kk else "🛡️ Агротехнологии",
            value=f"{len(ds.practices)} " + ("әдіс" if kk else "методов"),
            delta="Су үнемдеу 30-70%" if kk else "Водосбережение 30-70%",
            delta_color="normal"
        )
    with kpi4:
        st.metric(
            label="📚 Ғылыми мақалалар" if kk else "📚 Научные статьи",
            value=f"{len(ds.references)} " + ("дереккөз" if kk else "источников"),
            delta="Gale OneFile / FAO / ICARDA",
            delta_color="off"
        )

    st.divider()

    # Sub-tabs within Tab 7
    subtab1_title = "🎯 Құрғақшылыққа қарсы кеңесші" if kk else "🎯 Интерактивный советник фермера"
    subtab2_title = "📂 Датасет каталогы" if kk else "📂 Каталог датасета (Таблицы)"
    subtab3_title = "📖 Ғылыми дереккөздер (Gale & FAO)" if kk else "📖 Научные источники (Gale & FAO)"

    subtab1, subtab2, subtab3 = st.tabs([subtab1_title, subtab2_title, subtab3_title])

    # ──────────────────────────────────────────────────────────
    # Sub-tab 1: Interactive Recommendation Engine
    # ──────────────────────────────────────────────────────────
    with subtab1:
        st.markdown(
            "### 🚜 Дақыл мен топырақ бойынша шешім қабылдау" if kk
            else "### 🚜 Формирование дорожной карты защиты урожая"
        )

        col_sel1, col_sel2, col_sel3 = st.columns(3)

        # 1. Crops Selector
        crop_options = list(ds.crops.values())
        crop_labels = [
            f"{_T(c, 'name', lang)} ({c.get('latin_name', '')}) — "
            + ("құрғақшылыққа төзімділігі" if kk else "засухоустойчивость")
            + f": {c.get('drought_tolerance_score', '-')}/5"
            for c in crop_options
        ]
        with col_sel1:
            crop_idx = st.selectbox(
                "Дақылды таңдаңыз" if kk else "Сельскохозяйственная культура",
                options=range(len(crop_options)),
                format_func=lambda i: crop_labels[i],
                key="agroadvisor_crop_select"
            )
            selected_crop = crop_options[crop_idx]

        # 2. Soils Selector
        soil_options = list(ds.soils.values())
        soil_labels = [
            f"{_T(s, 'name', lang)} ({s.get('wrb_classification', '')}) — "
            f"{str(s.get('region_kk') if kk else s.get('region_kz', '')).split(';')[0][:30]}"
            for s in soil_options
        ]
        with col_sel2:
            soil_idx = st.selectbox(
                "Топырақ типін таңдаңыз" if kk else "Тип почвы региона",
                options=range(len(soil_options)),
                format_func=lambda i: soil_labels[i],
                key="agroadvisor_soil_select"
            )
            selected_soil = soil_options[soil_idx]

        # 3. Growth Stage (labels per language; the advisor always receives the
        #    canonical Russian phase string for internal matching)
        stage_options_kk = [
            "Барлық кезеңдер (жалпы жоспар)",
            "Өну және көктеу",
            "Түптену / Сабақтану",
            "Гүлдену (ең қауіпті кезең)",
            "Дән толысу (налив зерна)",
            "Пісу кезеңі"
        ]
        stage_options_ru = [
            "Все фазы (общий регламент)",
            "Всходы и кущение",
            "Выход в трубку",
            "Цветение (критическая фаза)",
            "Налив зерна",
            "Созревание"
        ]
        with col_sel3:
            stage_idx = st.selectbox(
                "Вегетация кезеңі" if kk else "Фаза вегетации",
                options=range(len(stage_options_ru)),
                format_func=lambda i: stage_options_kk[i] if kk else stage_options_ru[i],
                key="agroadvisor_stage_select"
            )

        # Generate roadmap in the selected language
        roadmap = advisor.generate_roadmap(
            crop_query=selected_crop.get("name_ru", ""),
            soil_query=selected_soil.get("name_ru", ""),
            growth_stage=stage_options_ru[stage_idx] if stage_idx > 0 else None,
            lang=lang
        )

        st.markdown("---")

        if roadmap.get("status") == "error":
            st.error(roadmap.get("message", "Қате" if kk else "Ошибка"))
            return

        action_map = roadmap.get("action_roadmap", {})
        emergency_acts = action_map.get("emergency_0_48h", [])
        tactical_acts = action_map.get("seasonal_tactical", [])
        strategic_acts = action_map.get("long_term_strategic", [])
        detailed_practices = roadmap.get("recommended_practices", [])
        yield_effect = roadmap.get(
            "expected_yield_loss_reduction",
            "Биологиялық өнімнің 35–50%-ын сақтау" if kk else "Сохранение 35-50% биологического урожая")
        c_prof = roadmap.get("crop_profile", {})
        s_prof = roadmap.get("soil_profile", {})

        # Display Selected Pair Summary Card
        card_col1, card_col2, card_col3 = st.columns([2, 2, 3])
        with card_col1:
            st.markdown(f"**🌾 {'Дақыл' if kk else 'Культура'}:** `{c_prof.get('name', '')}`")
            st.caption(
                f"{'Түрі' if kk else 'Тип'}: {c_prof.get('crop_type', '')} · "
                f"{'Тамыры' if kk else 'Корень'}: {c_prof.get('root_system', '')} · "
                f"{'Солу шегі' if kk else 'Порог увядания'}: {c_prof.get('wilting_threshold', '')}"
            )
        with card_col2:
            st.markdown(f"**🌍 {'Топырақ' if kk else 'Почва'}:** `{s_prof.get('name', '')}`")
            st.caption(
                f"WRB: {s_prof.get('wrb_classification', '')} · "
                f"{'Ылғал сыйымдылығы' if kk else 'Влагоемкость'}: {s_prof.get('available_water_capacity', '')} · "
                f"{'Қабыршақтану қаупі' if kk else 'Риск корки'}: {s_prof.get('crust_formation_risk', '')}"
            )
        with card_col3:
            st.markdown(f"**📊 {'Күтілетін нәтиже' if kk else 'Ожидаемый эффект'}:**")
            st.success(f"🛡️ {yield_effect}")

        if roadmap.get("stage_warning"):
            st.warning(f"⚠️ {roadmap['stage_warning']}")

        st.info(f"🔬 **{'Тәуекел механизмі' if kk else 'Механизм риска'}:** {roadmap.get('drought_risk_mechanism', '')}")

        # 3 Action Tiers
        col_act1, col_act2, col_act3 = st.columns(3)

        with col_act1:
            st.markdown("#### 🚨 Шұғыл шаралар (0–48 сағ)" if kk else "#### 🚨 Экстренные меры (0–48 ч)")
            if emergency_acts:
                for i, act in enumerate(emergency_acts, 1):
                    st.markdown(f"- **{i}.** {act}")
            else:
                st.caption("Шұғыл шаралар жоқ" if kk else "Нет экстренных мер")

        with col_act2:
            st.markdown("#### 📅 Маусымдық шаралар" if kk else "#### 📅 Сезонные меры")
            if tactical_acts:
                for i, act in enumerate(tactical_acts, 1):
                    st.markdown(f"- **{i}.** {act}")
            else:
                st.caption("Маусымдық шаралар жоқ" if kk else "Нет сезонных мер")

        with col_act3:
            st.markdown("#### 🏛️ Стратегиялық шаралар" if kk else "#### 🏛️ Стратегические меры")
            if strategic_acts:
                for i, act in enumerate(strategic_acts, 1):
                    st.markdown(f"- **{i}.** {act}")
            else:
                st.caption("Стратегиялық шаралар жоқ" if kk else "Нет стратегических мер")

        # Detailed Recommended Practices
        if detailed_practices:
            st.markdown(
                "#### 🛠️ Ұсынылатын агротехнологиялар регламенті" if kk
                else "#### 🛠️ Технологические регламенты рекомендованных практик")
            cost_map = {"low": "🟢 Төмен", "medium": "🟡 Орташа", "high": "🔴 Жоғары"} if kk else {
                "low": "🟢 Низкие", "medium": "🟡 Средние", "high": "🔴 Высокие"}
            for p in detailed_practices:
                water_eff = p.get("efficiency", "")
                with st.expander(f"🔹 {p.get('name', '')} ({p.get('category', '')}) | 💧 {water_eff}"):
                    st.markdown(f"**{'Реттеу' if kk else 'Регламент'}:** {p.get('protocol', '')}")
                    if p.get("limitations"):
                        st.markdown(f"**{'Шектеулер' if kk else 'Ограничения'}:** {p.get('limitations')}")
                    st.caption(
                        f"{'Шығындар' if kk else 'Затраты'}: {cost_map.get(p.get('cost', 'medium'), p.get('cost', ''))} · "
                        f"{'Кезең' if kk else 'Горизонт'}: {p.get('horizon', '')}")

        # Download Markdown Roadmap for Farmer
        if kk:
            md_export = f"""# Quareo Drought Advisor — Агрономиялық жол картасы
## Дақыл: {c_prof.get('name', '')} | Топырақ: {s_prof.get('name', '')}
**Күтілетін тиімділік:** {yield_effect}

### 1. Тәуекел механизмі
{roadmap.get('drought_risk_mechanism', '')}

### 2. Шұғыл шаралар (0-48 сағат)
""" + "\n".join(f"- {a}" for a in emergency_acts) + """

### 3. Маусымдық тактикалық шаралар
""" + "\n".join(f"- {a}" for a in tactical_acts) + """

### 4. Стратегиялық шаралар
""" + "\n".join(f"- {a}" for a in strategic_acts) + """

### 5. Ұсынылатын агротехнологиялар
""" + "\n".join(
                f"• **{p.get('name', '')}** ({p.get('category', '')}, 💧 {p.get('efficiency', '')}): {p.get('protocol', '')}"
                for p in detailed_practices) + f"""

---
*Дереккөз: Gale OneFile: Agriculture, FAO Land & Water, ICARDA, ҚазЕжӨҒЗИ*
"""
        else:
            md_export = f"""# Quareo Drought Advisor — Дорожная карта защиты урожая
## Культура: {c_prof.get('name', '')} | Почва: {s_prof.get('name', '')}
**Ожидаемый эффект:** {yield_effect}

### 1. Механизм риска
{roadmap.get('drought_risk_mechanism', '')}

### 2. Экстренные меры (0–48 часов)
""" + "\n".join(f"- {a}" for a in emergency_acts) + """

### 3. Сезонные тактические меры
""" + "\n".join(f"- {a}" for a in tactical_acts) + """

### 4. Стратегические меры
""" + "\n".join(f"- {a}" for a in strategic_acts) + """

### 5. Рекомендованные агротехнологии
""" + "\n".join(
                f"• **{p.get('name', '')}** ({p.get('category', '')}, 💧 {p.get('efficiency', '')}): {p.get('protocol', '')}"
                for p in detailed_practices) + """

---
*Источники: Gale OneFile: Agriculture, FAO Land & Water, ICARDA, КазНИИ земледелия и растениеводства*
"""
        st.download_button(
            label="📥 Жол картасын жүктеу (.md)" if kk else "📥 Скачать дорожную карту (.md)",
            data=md_export.encode("utf-8"),
            file_name=f"Roadmap_{selected_crop.get('id', 'crop')}_{selected_soil.get('id', 'soil')}_{lang}.md",
            mime="text/markdown",
            key="dl_advisor_roadmap"
        )

    # ──────────────────────────────────────────────────────────
    # Sub-tab 2: Dataset Catalog & Tables
    # ──────────────────────────────────────────────────────────
    with subtab2:
        st.markdown("### 📊 Агрономиялық деректер қоры" if kk else "### 📊 Каталог датасета")

        if kk:
            radio_options = ["🌾 Дақылдар", "🌍 Топырақтар", "🛡️ Практикалар"]
        else:
            radio_options = ["🌾 Культуры", "🌍 Почвы", "🛡️ Агротехнологии"]

        tbl_idx = st.radio(
            "Кестені таңдаңыз:" if kk else "Выберите сущность датасета:",
            options=range(3),
            format_func=lambda i: radio_options[i],
            horizontal=True
        )

        if tbl_idx == 0:
            crops_list = []
            for c in ds.crops.values():
                crops_list.append({
                    ("Дақыл" if kk else "Культура"): _T(c, "name", lang),
                    ("Латыны" if kk else "Латынь"): c.get("latin_name", ""),
                    ("Түрі" if kk else "Тип"): _T(c, "crop_type", lang),
                    ("Құрғақшылыққа төзімділік (1-5)" if kk else "Засухоустойчивость (1-5)"): c.get("drought_tolerance_score", ""),
                    ("Тамыр тереңдігі (м)" if kk else "Глубина корня (м)"): c.get("root_depth_meters", ""),
                    ("Тамыр түрі" if kk else "Тип корня"): _T(c, "root_type", lang),
                    ("Увядание шегі (%)" if kk else "Порог увядания (%)"): c.get("wilting_threshold_swc_percent", ""),
                    ("Сыни фазалар" if kk else "Критические фазы"): ", ".join(_T(c, "critical_water_phases", lang)),
                })
            df_crops = pd.DataFrame(crops_list)
            st.dataframe(df_crops, width='stretch', hide_index=True)
            csv_crops = df_crops.to_csv(index=False).encode("utf-8")
            st.download_button("📥 crops.csv жүктеу" if kk else "📥 Скачать crops.csv",
                               csv_crops, f"crops_{lang}.csv", "text/csv", key="dl_csv_crops")

        elif tbl_idx == 1:
            soils_list = []
            for s in ds.soils.values():
                soils_list.append({
                    ("Топырақ" if kk else "Почва"): _T(s, "name", lang),
                    ("WRB жіктемесі" if kk else "WRB Классификация"): s.get("wrb_classification", ""),
                    ("ҚР аймақтары" if kk else "Регионы РК"): s.get("region_kk", "") if kk else s.get("region_kz", ""),
                    ("Ылғал сыйымдылығы (мм/м)" if kk else "Влагоемкость (мм/м)"): s.get("available_water_capacity_mm_per_m", ""),
                    ("Инфильтрация (мм/сағ)" if kk else "Инфильтрация (мм/ч)"): s.get("infiltration_rate_mm_hr", ""),
                    ("Қабыршақтану қаупі" if kk else "Риск корки"): _T(s, "crust_formation_risk", lang),
                    ("Негізгі әлсіз тұсы" if kk else "Главная уязвимость"): _T(s, "primary_vulnerability", lang),
                })
            df_soils = pd.DataFrame(soils_list)
            st.dataframe(df_soils, width='stretch', hide_index=True)
            csv_soils = df_soils.to_csv(index=False).encode("utf-8")
            st.download_button("📥 soils.csv жүктеу" if kk else "📥 Скачать soils.csv",
                               csv_soils, f"soils_{lang}.csv", "text/csv", key="dl_csv_soils")

        else:
            practices_list = []
            for p in ds.practices.values():
                practices_list.append({
                    ("Технология" if kk else "Технология"): _T(p, "name", lang),
                    ("Санаты" if kk else "Категория"): _T(p, "category", lang),
                    ("Су үнемдеу эффектісі" if kk else "Эффект водосбережения"): p.get("estimated_water_saving_percent", ""),
                    ("Горизонты" if kk else "Горизонт"): _T(p, "action_horizon", lang),
                    ("Затраттар" if kk else "Затраты"): p.get("cost_level", ""),
                    ("Реттеу" if kk else "Регламент"): _T(p, "implementation_protocol", lang),
                })
            df_practices = pd.DataFrame(practices_list)
            st.dataframe(df_practices, width='stretch', hide_index=True)
            csv_practices = df_practices.to_csv(index=False).encode("utf-8")
            st.download_button("📥 drought_practices.csv жүктеу" if kk else "📥 Скачать drought_practices.csv",
                               csv_practices, f"drought_practices_{lang}.csv", "text/csv", key="dl_csv_practices")

    # ──────────────────────────────────────────────────────────
    # Sub-tab 3: Academic References (Gale & FAO)
    # ──────────────────────────────────────────────────────────
    with subtab3:
        st.markdown(
            "### 📚 Ғылыми негіздеме: Gale OneFile, FAO және ICARDA мақалалары" if kk
            else "### 📚 Научная база: Рецензируемые публикации Gale, FAO и ICARDA")
        for rid, ref in ds.references.items():
            title = ref.get("title", "")
            authors = ref.get("authors", "")
            journal = ref.get("journal_or_publisher", ref.get("source", ""))
            year = ref.get("year", "")
            findings = _T(ref, "key_findings", lang)
            source = ref.get("source", "Gale / FAO")
            doc_id = ref.get("doi_or_gale_doc_id", "")

            st.markdown(f"**📖 [{title}]**")
            st.markdown(
                (f"*Авторлар:* {authors} · *Басылым:* **{journal}** ({year})") if kk
                else (f"*Авторы:* {authors} · *Издание:* **{journal}** ({year})"))
            st.markdown(
                f"> *Негізгі қорытынды:* {findings}" if kk
                else f"> *Ключевой вывод:* {findings}")
            st.caption(
                f"Дереккөз: {source} · Идентификатор: {doc_id}" if kk
                else f"База: {source} · Идентификатор: {doc_id}")
            st.divider()
