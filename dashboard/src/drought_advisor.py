"""
Drought Advisor AI Engine: Generates agronomic decision roadmaps
for farmers based on crop species, soil characteristics, and scientific research.
"""

from typing import Dict, List, Optional, Any
from .dataset_loader import DroughtDataset


class DroughtAdvisor:
    def __init__(self, dataset: Optional[DroughtDataset] = None):
        self.dataset = dataset or DroughtDataset()

    def generate_roadmap(
        self,
        crop_query: str,
        soil_query: Optional[str] = None,
        growth_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main recommendation engine for farmers:
        Takes farmer crop query and soil query, resolves scientific entities,
        and constructs an actionable, tiered drought mitigation roadmap.
        """
        # 1. Resolve crop
        crop = self.dataset.search_crop(crop_query)
        if not crop:
            available_crops = [c["name_ru"] for c in self.dataset.crops.values()]
            return {
                "status": "error",
                "message": f"Культура '{crop_query}' не найдена в базе данных.",
                "available_crops": available_crops
            }

        # 2. Resolve soil (if not provided, choose default optimal soil for the crop)
        soil = None
        if soil_query:
            soil = self.dataset.search_soil(soil_query)
            if not soil:
                return {
                    "status": "error",
                    "message": f"Тип почвы '{soil_query}' не распознан.",
                    "available_soils": [s["name_ru"] for s in self.dataset.soils.values()]
                }
        else:
            # Fallback to first optimal soil or chestnut
            default_soil_id = crop.get("optimal_soil_types", ["chestnut_dark_light"])[0]
            soil = self.dataset.soils.get(default_soil_id, list(self.dataset.soils.values())[0])

        # 3. Match or synthesize matrix solution
        matrix_entry = self.dataset.get_matrix_entry(crop["id"], soil["id"])

        if matrix_entry:
            risk_mechanism = matrix_entry["drought_risk_mechanism"]
            emergency_actions = list(matrix_entry["emergency_actions_0_48h"])
            tactical_actions = list(matrix_entry["seasonal_tactical_actions"])
            strategic_actions = list(matrix_entry["long_term_strategic_actions"])
            practice_ids = matrix_entry.get("recommended_practice_ids", [])
            yield_effect = matrix_entry.get("expected_yield_loss_reduction", "Сохранение 35-50% урожая")
            scientific_basis = matrix_entry.get("scientific_basis", "")
        else:
            # Synthesize dynamic rule
            risk_mechanism = (
                f"На почве '{soil['name_ru']}' культура '{crop['name_ru']}' сталкивается с дефицитом влаги: "
                f"{soil['primary_vulnerability']}. Корневая система ({crop['root_type']}, глубина {crop['root_depth_meters']} м) "
                f"испытывает стресс при снижении влажности ниже {crop['wilting_threshold_swc_percent']}% от наименьшей влагоемкости."
            )
            emergency_actions = [
                "Провести внекорневую обработку антитранспирантом за 24-48 часов до наступления волны максимальной температуры (>+32°C).",
                "Применить баковую смесь солей калия (K2SO4) и антистрессовых аминокислот в вечерние часы для поддержания тургора клеток.",
                "При наличии поверхностной корки провести мелкое рыхление («сухой полив») на 3-4 см для разрушения капилляров."
            ]
            tactical_actions = [
                f"Контролировать влажность почвы в критические фазы развития культуры: {', '.join(crop['critical_water_phases'])}.",
                "Ограничить засоренность полей гербицидной прополкой, исключив конкуренцию за остаточную почвенную влагу.",
                "При севе вносить влагоудерживающие гидрогели или применять мульчирование растительными остатками."
            ]
            strategic_actions = [
                "Внедрение влагосберегающей безотвальной обработки или прямого посева No-Till со снегозадержанием.",
                f"Глубокое щелевание подпахотного слоя с учетом специфики почвы: {soil['recommended_tillage_focus']}."
            ]
            practice_ids = ["foliar_osmoprotectants", "antitranspirant_spraying", "no_till_stubble"]
            yield_effect = "Сохранение 30-45% урожая при соблюдении агротехнических регламентов"
            scientific_basis = "Адаптивные рекомендации на базе стандартов FAO-56 и исследований Gale OneFile: Agriculture."

        # Collect detailed practice objects
        detailed_practices = []
        for pid in practice_ids:
            if pid in self.dataset.practices:
                detailed_practices.append(self.dataset.practices[pid])

        # Collect scientific literature citations
        citations = []
        all_ref_ids = set()
        for p in detailed_practices:
            for ref_id in p.get("gale_reference_ids", []):
                all_ref_ids.add(ref_id)
        for ref_id in all_ref_ids:
            if ref_id in self.dataset.references:
                citations.append(self.dataset.references[ref_id])

        # If specific growth stage is specified, highlight relevant critical phase
        stage_warning = None
        if growth_stage:
            matched_phases = [
                p for p in crop["critical_water_phases"] if growth_stage.lower() in p.lower()
            ]
            if matched_phases:
                stage_warning = (
                    f"ВНИМАНИЕ: Фаза '{growth_stage}' является критической фазой водопотребления для культуры {crop['name_ru']}! "
                    f"Дефицит влаги в этот период наносит необратимый ущерб формированию репродуктивных органов."
                )

        return {
            "status": "success",
            "query": {
                "crop": crop_query,
                "soil": soil_query,
                "growth_stage": growth_stage
            },
            "crop_profile": {
                "id": crop["id"],
                "name": crop["name_ru"],
                "latin_name": crop["latin_name"],
                "crop_type": crop["crop_type"],
                "drought_tolerance_score": f"{crop['drought_tolerance_score']} / 5.0",
                "drought_mechanism": crop["drought_mechanism"],
                "root_system": f"{crop['root_type']}, глубина {crop['root_depth_meters']} м",
                "transpiration_rate": crop["transpiration_coefficient"],
                "critical_water_phases": crop["critical_water_phases"],
                "wilting_threshold": f"{crop['wilting_threshold_swc_percent']}% от НВ",
                "visual_stress_symptoms": crop["drought_symptoms"]
            },
            "soil_profile": {
                "id": soil["id"],
                "name": soil["name_ru"],
                "wrb_classification": soil["wrb_classification"],
                "regions": soil["region_kz"],
                "texture": soil["texture"],
                "available_water_capacity": f"{soil['available_water_capacity_mm_per_m']} мм/метр",
                "cracking_tendency": soil["cracking_tendency"],
                "crust_formation_risk": soil["crust_formation_risk"],
                "primary_vulnerability": soil["primary_vulnerability"],
                "tillage_recommendation": soil["recommended_tillage_focus"]
            },
            "drought_risk_mechanism": risk_mechanism,
            "stage_warning": stage_warning,
            "action_roadmap": {
                "emergency_0_48h": emergency_actions,
                "seasonal_tactical": tactical_actions,
                "long_term_strategic": strategic_actions
            },
            "recommended_practices": [
                {
                    "id": p["id"],
                    "name": p["name_ru"],
                    "category": p["category"],
                    "horizon": p["action_horizon"],
                    "cost": p["cost_level"],
                    "efficiency": p["estimated_water_saving_percent"],
                    "protocol": p["implementation_protocol"],
                    "limitations": p.get("practical_limitations", "")
                }
                for p in detailed_practices
            ],
            "expected_yield_loss_reduction": yield_effect,
            "scientific_basis": scientific_basis,
            "academic_citations": citations
        }

    def format_roadmap_markdown(self, roadmap: Dict[str, Any]) -> str:
        """Renders roadmap dictionary as clean, beautiful Markdown report."""
        if roadmap.get("status") == "error":
            return f"❌ **Ошибка**: {roadmap.get('message')}"

        c = roadmap["crop_profile"]
        s = roadmap["soil_profile"]
        r = roadmap["action_roadmap"]

        lines = [
            f"# 🌾 План защиты посевов от засухи: {c['name']} на почве «{s['name']}»",
            "",
            f"**Латинское название:** *{c['latin_name']}* | **Тип культуры:** {c['crop_type']} | **Индекс засухоустойчивости:** {c['drought_tolerance_score']}",
            f"**Почвенная классификация WRB:** {s['wrb_classification']} | **Регионы:** {s['regions']}",
            ""
        ]

        if roadmap.get("stage_warning"):
            lines.extend([
                f"> [!WARNING]",
                f"> {roadmap['stage_warning']}",
                ""
            ])

        lines.extend([
            "## 1. Анализ рисков и физика взаимодействия почвы и растения",
            f"**Уязвимость почвы:** {s['primary_vulnerability']}",
            f"**Корневая система:** {c['root_system']}",
            f"**Критические фазы влагопотребления:** {', '.join(c['critical_water_phases'])}",
            f"**Визуальные симптомы стресса:** {c['visual_stress_symptoms']}",
            f"**Механизм кризиса:** {roadmap['drought_risk_mechanism']}",
            "",
            "---",
            "## 2. ⚡ Пошаговая дорожная карта спасения растения (Action Roadmap)",
            "",
            "### 🚨 Фаза 1: Неотложные экстренные меры (0 – 48 часов)",
            "*(Действия при наступлении волны жары / суховея для спасения вегетации)*"
        ])

        for act in r["emergency_0_48h"]:
            lines.append(f"- [ ] {act}")

        lines.extend([
            "",
            "### 🛠️ Фаза 2: Тактические меры текущего сезона",
            "*(Агроприемы в период вегетации для максимального удержания влаги)*"
        ])

        for act in r["seasonal_tactical"]:
            lines.append(f"- [ ] {act}")

        lines.extend([
            "",
            "### 🛡️ Фаза 3: Долгосрочные стратегические меры",
            "*(Подготовка почвы, севооборот и инфраструктура на следующие сезоны)*"
        ])

        for act in r["long_term_strategic"]:
            lines.append(f"- [ ] {act}")

        lines.extend([
            "",
            "---",
            "## 3. 📋 Регламент рекомендованных агротехнических практик",
            ""
        ])

        for p in roadmap["recommended_practices"]:
            cost_badge = {"low": "🟢 Низкие", "medium": "🟡 Средние", "high": "🔴 Высокие"}.get(p["cost"], p["cost"])
            lines.extend([
                f"### • {p['name']}",
                f"- **Категория:** `{p['category']}` | **Горизонт:** `{p['horizon']}` | **Затраты:** {cost_badge}",
                f"- **Эффективность влагосбережения:** **{p['efficiency']}**",
                f"- **Технологический регламент:** {p['protocol']}",
                f"- **Ограничения:** *{p['limitations']}*",
                ""
            ])

        lines.extend([
            "---",
            "## 4. 🔬 Научное обоснование и академические источники (Gale / FAO / ICARDA)",
            f"**Оценка эффекта:** {roadmap['expected_yield_loss_reduction']}",
            f"**Научный базис:** {roadmap['scientific_basis']}",
            ""
        ])

        for cite in roadmap.get("academic_citations", []):
            lines.extend([
                f"- **[{cite['source']}]** *{cite['title']}* ({cite['authors']}, {cite['year']}).",
                f"  - *Журнал / Ресурс:* {cite['journal_or_publisher']} (`{cite['doi_or_gale_doc_id']}`)",
                f"  - *Ключевой вывод:* {cite['key_findings']}",
                ""
            ])

        return "\n".join(lines)
