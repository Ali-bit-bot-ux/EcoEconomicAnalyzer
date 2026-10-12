# -*- coding: utf-8 -*-
"""
AgroAdvisor component for AgriCascade Streamlit Dashboard.
Connects the scientific Quareo Drought Dataset and AI Recommendation Engine
(16 crops, 7 soils, 17 practices, scientific citations from Gale/FAO/ICARDA/КазНИИЗиР).
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


def render_agroadvisor_tab(lang: str = "kk"):
    """Renders the comprehensive AgroAdvisor tab in Streamlit."""
    try:
        ds, advisor = get_advisor_engine()
    except Exception as e:
        st.error(f"Қате: Датасетті жүктеу мүмкін болмады: {e}" if lang == "kk" else f"Ошибка загрузки датасета: {e}")
        return

    # Header
    if lang == "kk":
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
            label="🌾 Дақылдар / Культуры" if lang == "kk" else "🌾 Сельхозкультуры",
            value=f"{len(ds.crops)} түрі",
            delta="Дәнді, майлы, бұршақ" if lang == "kk" else "Зерновые, масличные, бобовые",
            delta_color="off"
        )
    with kpi2:
        st.metric(
            label="🌍 Топырақ түрлері / Почвы" if lang == "kk" else "🌍 Почвенные зоны",
            value=f"{len(ds.soils)} аймақ",
            delta="Қазақстанның астық белдеуі" if lang == "kk" else "Север, сухие степи, юг",
            delta_color="off"
        )
    with kpi3:
        st.metric(
            label="🛡️ Технологиялар / Практики" if lang == "kk" else "🛡️ Агротехнологии",
            value=f"{len(ds.practices)} әдіс",
            delta="Су үнемдеу 30-70%" if lang == "kk" else "Водосбережение 30-70%",
            delta_color="normal"
        )
    with kpi4:
        st.metric(
            label="📚 Ғылыми мақалалар" if lang == "kk" else "📚 Научные статьи",
            value=f"{len(ds.references)} дереккөз",
            delta="Gale OneFile / FAO / ICARDA",
            delta_color="off"
        )

    st.divider()

    # Sub-tabs within Tab 7
    subtab1_title = "🎯 Құрғақшылыққа қарсы кеңесші" if lang == "kk" else "🎯 Интерактивный советник фермера"
    subtab2_title = "📂 Датасет каталогы" if lang == "kk" else "📂 Каталог датасета (Таблицы)"
    subtab3_title = "📖 Ғылыми дереккөздер (Gale & FAO)" if lang == "kk" else "📖 Научные источники (Gale & FAO)"

    subtab1, subtab2, subtab3 = st.tabs([subtab1_title, subtab2_title, subtab3_title])

    # ──────────────────────────────────────────────────────────
    # Sub-tab 1: Interactive Recommendation Engine
    # ──────────────────────────────────────────────────────────
    with subtab1:
        st.markdown(
            "### 🚜 Дақыл мен топырақ бойынша шешім қабылдау"
            if lang == "kk" else
            "### 🚜 Формирование дорожной карты защиты урожая"
        )

        col_sel1, col_sel2, col_sel3 = st.columns(3)

        # 1. Crops Selector
        crop_options = list(ds.crops.values())
        crop_labels = [
            f"{c.get('name_ru', c.get('id', ''))} ({c.get('latin_name', '')}) — засухоустойчивость: {c.get('drought_tolerance_score', '-')}/5"
            for c in crop_options
        ]
        with col_sel1:
            crop_idx = st.selectbox(
                "Дақылды таңдаңыз" if lang == "kk" else "Сельскохозяйственная культура",
                options=range(len(crop_options)),
                format_func=lambda i: crop_labels[i],
                key="agroadvisor_crop_select"
            )
            selected_crop = crop_options[crop_idx]

        # 2. Soils Selector
        soil_options = list(ds.soils.values())
        soil_labels = [
            f"{s.get('name_ru', s.get('id', ''))} ({s.get('wrb_classification', '')}) — {str(s.get('region_kz', '')).split(';')[0][:30]}"
            for s in soil_options
        ]
        with col_sel2:
            soil_idx = st.selectbox(
                "Топырақ типін таңдаңыз" if lang == "kk" else "Тип почвы региона",
                options=range(len(soil_options)),
                format_func=lambda i: soil_labels[i],
                key="agroadvisor_soil_select"
            )
            selected_soil = soil_options[soil_idx]

        # 3. Growth Stage
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
                "Вегетация кезеңі" if lang == "kk" else "Фаза вегетации",
                options=range(len(stage_options_ru)),
                format_func=lambda i: stage_options_kk[i] if lang == "kk" else stage_options_ru[i],
                key="agroadvisor_stage_select"
            )

        # Generate roadmap
        roadmap = advisor.generate_roadmap(
            crop_query=selected_crop.get("name_ru", ""),
            soil_query=selected_soil.get("name_ru", ""),
            growth_stage=stage_options_ru[stage_idx] if stage_idx > 0 else None
        )

        st.markdown("---")

        action_map = roadmap.get("action_roadmap", {})
        emergency_acts = action_map.get("emergency_0_48h") or roadmap.get("emergency_actions", [])
        tactical_acts = action_map.get("seasonal_tactical") or roadmap.get("tactical_actions", [])
        strategic_acts = action_map.get("strategic") or roadmap.get("strategic_actions", [])
        detailed_practices = action_map.get("detailed_practices") or roadmap.get("detailed_practices", [])
        yield_effect = action_map.get("expected_yield_saving") or roadmap.get("expected_yield_loss_reduction", "35-50% биологического урожая")

        # Display Selected Pair Summary Card
        card_col1, card_col2, card_col3 = st.columns([2, 2, 3])
        with card_col1:
            st.markdown(f"**🌾 Дақыл:** `{selected_crop.get('name_ru', '')}`")
            st.caption(
                f"Түрі: {selected_crop.get('crop_type', '')} · Тамыры: {selected_crop.get('root_type', '')} ({selected_crop.get('root_depth_meters', '')} м) · "
                f"Увядание шегі: {selected_crop.get('wilting_threshold_swc_percent', '')}%"
            )
        with card_col2:
            st.markdown(f"**🌍 Топырақ:** `{selected_soil.get('name_ru', '')}`")
            st.caption(
                f"WRB: {selected_soil.get('wrb_classification', '')} · Влагоемкость: {selected_soil.get('available_water_capacity_mm_per_m', '')} мм/м · "
                f"Қабыршақтану қаупі: {selected_soil.get('crust_formation_risk', '')}"
            )
        with card_col3:
            st.markdown(f"**📊 Күтілетін нәтиже / Эффект:**")
            st.success(f"🛡️ {yield_effect}")

        st.info(f"🔬 **Механизм риска:** {roadmap.get('drought_risk_mechanism', '')}")

        # 3 Action Tiers
        col_act1, col_act2, col_act3 = st.columns(3)

        with col_act1:
            st.markdown("#### 🚨 Шұғыл шаралар (0–48 сағ)" if lang == "kk" else "#### 🚨 Экстренные меры (0–48 ч)")
            if emergency_acts:
                for i, act in enumerate(emergency_acts, 1):
                    st.markdown(f"- **{i}.** {act}")
            else:
                st.caption("Шұғыл шаралар жоқ" if lang == "kk" else "Нет экстренных мер")

        with col_act2:
            st.markdown("#### 📅 Маусымдық шаралар" if lang == "kk" else "#### 📅 Сезонные меры")
            if tactical_acts:
                for i, act in enumerate(tactical_acts, 1):
                    st.markdown(f"- **{i}.** {act}")
            else:
                st.caption("Маусымдық шаралар жоқ" if lang == "kk" else "Нет сезонных мер")

        with col_act3:
            st.markdown("#### 🏛️ Стратегиялық шаралар" if lang == "kk" else "#### 🏛️ Стратегические меры")
            if strategic_acts:
                for i, act in enumerate(strategic_acts, 1):
                    st.markdown(f"- **{i}.** {act}")
            else:
                st.caption("Стратегиялық шаралар жоқ" if lang == "kk" else "Нет стратегических мер")

        # Detailed Recommended Practices
        if detailed_practices:
            st.markdown("#### 🛠️ Ұсынылатын агротехнологиялар регламенті" if lang == "kk" else "#### 🛠️ Технологические регламенты рекомендованных практик")
            for p in detailed_practices:
                water_eff = p.get('estimated_water_saving_percent', '')
                with st.expander(f"🔹 {p.get('name_ru', p.get('id', ''))} ({p.get('category', '')}) | 💧 {water_eff}"):
                    st.markdown(f"**Регламент:** {p.get('implementation_protocol', '')}")
                    if p.get('practical_limitations'):
                        st.markdown(f"**Шектеулері / Ограничения:** {p.get('practical_limitations')}")
                    st.caption(f"Өзіндік құн / Затраты: {p.get('cost_level', 'medium')} · Горизонт: {p.get('action_horizon', '')}")

        # Download Markdown Roadmap for Farmer
        md_export = f"""# Quareo Drought Advisor — Агрономиялық жол картасы
## Дақыл: {selected_crop.get('name_ru', '')} | Топырақ: {selected_soil.get('name_ru', '')}
**Күтілетін тиімділік:** {yield_effect}

### 1. Тәуекел механизмі
{roadmap.get('drought_risk_mechanism', '')}

### 2. Шұғыл шаралар (0-48 сағат)
""" + "\n".join(f"- {a}" for a in emergency_acts) + """

### 3. Маусымдық тактикалық шаралар
""" + "\n".join(f"- {a}" for a in tactical_acts) + """

### 4. Стратегиялық шаралар
""" + "\n".join(f"- {a}" for a in strategic_acts) + f"""

---
*Дереккөз: Gale OneFile: Agriculture, FAO Land & Water, ICARDA, ҚазЕжӨҒЗИ*
"""
        st.download_button(
            label="📥 Жол картасын жүктеу (.md)" if lang == "kk" else "📥 Скачать дорожную карту (.md)",
            data=md_export.encode("utf-8"),
            file_name=f"Roadmap_{selected_crop.get('id', 'crop')}_{selected_soil.get('id', 'soil')}.md",
            mime="text/markdown",
            key="dl_advisor_roadmap"
        )

    # ──────────────────────────────────────────────────────────
    # Sub-tab 2: Dataset Catalog & Tables
    # ──────────────────────────────────────────────────────────
    with subtab2:
        st.markdown("### 📊 Агрономиялық деректер қоры" if lang == "kk" else "### 📊 Каталог датасета")

        tbl_choice = st.radio(
            "Кестені таңдаңыз:" if lang == "kk" else "Выберите сущность датасета:",
            options=["🌾 Дақылдар (Crops)", "🌍 Топырақтар (Soils)", "🛡️ Практикалар (Drought Practices)"],
            horizontal=True
        )

        if "Дақылдар" in tbl_choice or "Crops" in tbl_choice:
            crops_list = []
            for c in ds.crops.values():
                crops_list.append({
                    "Культура": c.get("name_ru", ""),
                    "Латынь": c.get("latin_name", ""),
                    "Тип": c.get("crop_type", ""),
                    "Засухоустойчивость (1-5)": c.get("drought_tolerance_score", ""),
                    "Глубина корня (м)": c.get("root_depth_meters", ""),
                    "Тип корня": c.get("root_type", ""),
                    "Критический порог увядания (%)": c.get("wilting_threshold_swc_percent", ""),
                    "Критические фазы": ", ".join(c.get("critical_water_phases", []))
                })
            df_crops = pd.DataFrame(crops_list)
            st.dataframe(df_crops, width='stretch', hide_index=True)
            csv_crops = df_crops.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Скачать crops.csv", csv_crops, "crops.csv", "text/csv", key="dl_csv_crops")

        elif "Топырақтар" in tbl_choice or "Soils" in tbl_choice:
            soils_list = []
            for s in ds.soils.values():
                soils_list.append({
                    "Почва": s.get("name_ru", ""),
                    "WRB Классификация": s.get("wrb_classification", ""),
                    "Регионы РК": s.get("region_kz", ""),
                    "Влагоемкость (мм/м)": s.get("available_water_capacity_mm_per_m", ""),
                    "Инфильтрация (мм/ч)": s.get("infiltration_rate_mm_hr", ""),
                    "Риск корки": s.get("crust_formation_risk", ""),
                    "Главная уязвимость": s.get("primary_vulnerability", "")
                })
            df_soils = pd.DataFrame(soils_list)
            st.dataframe(df_soils, width='stretch', hide_index=True)
            csv_soils = df_soils.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Скачать soils.csv", csv_soils, "soils.csv", "text/csv", key="dl_csv_soils")

        else:
            practices_list = []
            for p in ds.practices.values():
                practices_list.append({
                    "Технология": p.get("name_ru", ""),
                    "Категория": p.get("category", ""),
                    "Эффект водосбережения": p.get("estimated_water_saving_percent", ""),
                    "Горизонт": p.get("action_horizon", ""),
                    "Затраты": p.get("cost_level", ""),
                    "Регламент": p.get("implementation_protocol", "")
                })
            df_practices = pd.DataFrame(practices_list)
            st.dataframe(df_practices, width='stretch', hide_index=True)
            csv_practices = df_practices.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Скачать drought_practices.csv", csv_practices, "drought_practices.csv", "text/csv", key="dl_csv_practices")

    # ──────────────────────────────────────────────────────────
    # Sub-tab 3: Academic References (Gale & FAO)
    # ──────────────────────────────────────────────────────────
    with subtab3:
        st.markdown(
            "### 📚 Ғылыми негіздеме: Gale OneFile, FAO және ICARDA мақалалары"
            if lang == "kk" else
            "### 📚 Научная база: Рецензируемые публикации Gale, FAO и ICARDA"
        )
        for rid, ref in ds.references.items():
            title = ref.get("title", "")
            authors = ref.get("authors", "")
            journal = ref.get("journal_or_publisher", ref.get("source", ""))
            year = ref.get("year", "")
            findings = ref.get("key_findings", "")
            source = ref.get("source", "Gale / FAO")
            doc_id = ref.get("doi_or_gale_doc_id", "")

            st.markdown(f"**📖 [{title}]**")
            st.markdown(f"*Авторлар:* {authors} · *Басылым:* **{journal}** ({year})")
            st.markdown(f"> *Аннотация / Негізгі қорытынды:* {findings}")
            st.caption(f"Дерекқор: {source} · Идентификатор: {doc_id}")
            st.divider()
