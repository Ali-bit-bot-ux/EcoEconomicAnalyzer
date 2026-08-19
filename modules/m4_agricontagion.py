"""
Daryn — modules/m4_agricontagion.py
======================================
Модуль 4: AgriContagion Graph — Граф распространения агро-рисков

СТРУКТУРА ГРАФА (NetworkX DiGraph):
  Узлы:
    - Region (СКО, Акмолинская, Костанайская, Павлодарская)
    - Product (пшеница, мука, хлеб)
    - Infrastructure (элеваторы из SiloSentry)
    - WaterSource (р. Или, Сырдарья, Тобол)

  Рёбра (направленные, с весами):
    WaterSource → Region:     вес = SMAI-аномалия (водный риск)
    Region → Infrastructure:  вес = расстояние (км) / пропускная способность
    Infrastructure → Product:  вес = коэффициент передачи
    Product → Product:         вес = ценовой transfer (пшеница→мука→хлеб)
    Region → Region:           вес = trade_flow (торговые потоки)

ЭКОНОМЕТРИКА (Granger Causality):
  Гипотеза: PSI(t) → wheat_price(t+2..3 месяца)
  Тест: grangercausalitytests + VAR (Vector Autoregression)

РЕЗУЛЬТАТ:
  - Матрица причинности (p-values)
  - Идентификация "конtagion paths" (путей распространения риска)
  - Mathematically backed: связь космоса и инфляции
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger
from typing import Optional

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings("ignore")

from config.settings import (
    GRANGER_MAX_LAG,
    GRANGER_SIGNIFICANCE,
    VAR_MAX_LAGS,
    BASELINE_YEARS,
    CURRENT_YEAR,
    PROCESSED_DIR,
)

# ─────────────────────────────────────────────
# Определение структуры графа
# ─────────────────────────────────────────────

GRAPH_NODES = {
    # Водные источники
    "water_ili":       {"type": "WaterSource", "name": "Река Или",      "country": "KZ/CN"},
    "water_syrdarya":  {"type": "WaterSource", "name": "Сырдарья",      "country": "KZ/KG"},
    "water_tobol":     {"type": "WaterSource", "name": "Тобол",         "country": "KZ/RU"},

    # Регионы производства
    "region_sko":      {"type": "Region", "name": "СКО",                "area_ha": 2500000},
    "region_kostanay": {"type": "Region", "name": "Костанайская",       "area_ha": 1800000},
    "region_akmola":   {"type": "Region", "name": "Акмолинская",        "area_ha": 2000000},
    "region_pavlodar": {"type": "Region", "name": "Павлодарская",       "area_ha": 500000},

    # Инфраструктура
    "elev_petro":      {"type": "Infrastructure", "name": "Петропавловский ХПП", "cap_t": 120000},
    "elev_kokshe":     {"type": "Infrastructure", "name": "Кокшетауский элеватор", "cap_t": 95000},
    "elev_kostanay":   {"type": "Infrastructure", "name": "Костанайский ХПП",   "cap_t": 200000},
    "elev_astana":     {"type": "Infrastructure", "name": "Астанинский элеватор", "cap_t": 150000},

    # Продуктовая цепочка
    "product_wheat":   {"type": "Product", "name": "Пшеница", "unit": "т"},
    "product_flour":   {"type": "Product", "name": "Мука",    "unit": "т"},
    "product_bread":   {"type": "Product", "name": "Хлеб",    "unit": "шт"},

    # Конечные рынки
    "market_domestic": {"type": "Market", "name": "Внутренний рынок РК"},
    "market_export":   {"type": "Market", "name": "Экспорт (Средняя Азия, Китай)"},
}

GRAPH_EDGES = [
    # Вода → Регионы (трансграничный риск)
    ("water_tobol",    "region_sko",       {"weight": 0.5,  "type": "water_supply"}),
    ("water_ili",      "region_akmola",    {"weight": 0.3,  "type": "water_supply"}),
    ("water_syrdarya", "region_kostanay",  {"weight": 0.2,  "type": "water_supply"}),

    # Регионы → Элеваторы (логистика)
    ("region_sko",      "elev_petro",      {"weight": 1.0,  "type": "logistics", "dist_km": 50}),
    ("region_akmola",   "elev_kokshe",     {"weight": 1.0,  "type": "logistics", "dist_km": 80}),
    ("region_akmola",   "elev_astana",     {"weight": 0.8,  "type": "logistics", "dist_km": 120}),
    ("region_kostanay", "elev_kostanay",   {"weight": 1.0,  "type": "logistics", "dist_km": 40}),

    # Регионы → межрегиональная торговля
    ("region_sko",      "region_akmola",   {"weight": 0.4,  "type": "trade_flow"}),
    ("region_kostanay", "region_akmola",   {"weight": 0.5,  "type": "trade_flow"}),

    # Элеваторы → Продукт
    ("elev_petro",     "product_wheat",    {"weight": 1.0,  "type": "storage"}),
    ("elev_kokshe",    "product_wheat",    {"weight": 1.0,  "type": "storage"}),
    ("elev_kostanay",  "product_wheat",    {"weight": 1.0,  "type": "storage"}),
    ("elev_astana",    "product_wheat",    {"weight": 1.0,  "type": "storage"}),

    # Продуктовая цепочка (price transfer)
    ("product_wheat",  "product_flour",    {"weight": 0.65, "type": "price_transfer", "markup": 1.15}),
    ("product_flour",  "product_bread",    {"weight": 0.80, "type": "price_transfer", "markup": 1.40}),

    # Рынки сбыта
    ("product_wheat",  "market_export",    {"weight": 0.45, "type": "market"}),
    ("product_wheat",  "market_domestic",  {"weight": 0.55, "type": "market"}),
    ("product_bread",  "market_domestic",  {"weight": 1.0,  "type": "market"}),
]


class AgriContagionModule:
    """
    Модуль 4: Граф агро-рисков + эконометрическая верификация причинности.
    """

    OUTPUT_GRAPH_PATH = PROCESSED_DIR / "agri_graph.gpickle"
    OUTPUT_CAUSALITY_PATH = PROCESSED_DIR / "causality_report.json"
    OUTPUT_PATH = PROCESSED_DIR / "agricontagion.parquet"

    # ─────────────────────────────────────────────
    # Построение графа
    # ─────────────────────────────────────────────

    def build_graph(
        self,
        df_psi: Optional[pd.DataFrame] = None,
        df_hydro: Optional[pd.DataFrame] = None,
        df_silos: Optional[pd.DataFrame] = None,
    ) -> nx.DiGraph:
        """
        Строит ориентированный граф NetworkX с узлами и рёбрами.
        Если переданы данные из модулей 1-3 — обновляет веса рёбер.

        Returns:
            nx.DiGraph: граф AgriContagion
        """
        G = nx.DiGraph()

        # Добавляем узлы
        for node_id, attrs in GRAPH_NODES.items():
            G.add_node(node_id, **attrs)

        # Добавляем рёбра
        for src, dst, attrs in GRAPH_EDGES:
            G.add_edge(src, dst, **attrs)

        # Динамическое обновление весов на основе реальных данных
        if df_psi is not None:
            current_psi = df_psi[df_psi["year"] == CURRENT_YEAR]
            if not current_psi.empty:
                psi_val = float(current_psi["psi"].values[0] or 0)
                psi_risk = min(1.0, max(0.0, -psi_val / 3))  # нормализуем [-3..0] → [0..1]

                # Увеличиваем веса рёбер региона с высоким PSI-риском
                for node in ["region_sko", "region_akmola", "region_kostanay"]:
                    if G.has_node(node):
                        G.nodes[node]["psi_risk"] = psi_risk
                        G.nodes[node]["psi_value"] = psi_val

        if df_hydro is not None:
            # Обновляем веса водных рёбер на основе SMAI
            df_hydro["year"] = pd.to_datetime(df_hydro["date"]).dt.year
            current_hydro = df_hydro[df_hydro["year"] == CURRENT_YEAR]

            for basin_id, edge_info in [
                ("tobol",    ("water_tobol",   "region_sko")),
                ("ili",      ("water_ili",     "region_akmola")),
                ("syrdarya", ("water_syrdarya","region_kostanay")),
            ]:
                basin_data = current_hydro[current_hydro["basin"] == basin_id]
                if not basin_data.empty:
                    smai = float(basin_data["smai"].mean())
                    risk = min(1.0, max(0.0, -smai))
                    if G.has_edge(*edge_info):
                        G[edge_info[0]][edge_info[1]]["smai_risk"] = smai
                        G[edge_info[0]][edge_info[1]]["weight"] = max(0.1, 1 - risk)

        if df_silos is not None and not df_silos.empty:
            # Обновляем заполненность элеваторов
            for _, row in df_silos.iterrows():
                fill = row.get("estimated_fill_pct", None)
                if fill is not None:
                    # Находим ближайший узел элеватора
                    name = str(row.get("name", ""))
                    if "Петропавл" in name and G.has_node("elev_petro"):
                        G.nodes["elev_petro"]["fill_pct"] = fill
                    elif "Кокшет" in name and G.has_node("elev_kokshe"):
                        G.nodes["elev_kokshe"]["fill_pct"] = fill
                    elif "Костанай" in name and G.has_node("elev_kostanay"):
                        G.nodes["elev_kostanay"]["fill_pct"] = fill

        logger.success(
            f"Граф AgriContagion построен: {G.number_of_nodes()} узлов, "
            f"{G.number_of_edges()} рёбер"
        )
        return G

    def graph_stats(self, G: nx.DiGraph) -> dict:
        """Вычисляет ключевые метрики графа."""
        # PageRank — узлы с наибольшим влиянием
        pagerank = nx.pagerank(G, weight="weight")
        # Centrality — узлы-посредники
        betweenness = nx.betweenness_centrality(G, weight="weight")

        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "top_influential": [
                {"node": n, "pagerank": round(v, 4)} for n, v in top_pagerank
            ],
            "top_bottlenecks": [
                {"node": n, "betweenness": round(v, 4)} for n, v in top_betweenness
            ],
        }

    # ─────────────────────────────────────────────
    # Granger Causality
    # ─────────────────────────────────────────────

    def test_stationarity(self, series: pd.Series, name: str) -> tuple[bool, float]:
        """ADF-тест на стационарность ряда."""
        result = adfuller(series.dropna(), autolag="AIC")
        p_value = float(result[1])
        is_stationary = p_value < 0.05
        logger.info(f"ADF [{name}]: p={p_value:.4f} → {'стационарный' if is_stationary else 'НЕ стационарный'}")
        return is_stationary, p_value

    def make_stationary(self, series: pd.Series, max_diff: int = 2) -> pd.Series:
        """Дифференцирует ряд до достижения стационарности."""
        s = series.copy()
        for d in range(max_diff):
            is_stat, _ = self.test_stationarity(s, f"diff_{d}")
            if is_stat:
                return s
            s = s.diff().dropna()
        return s

    def run_granger_causality(
        self,
        df_psi: pd.DataFrame,
        df_economics: pd.DataFrame,
    ) -> dict:
        """
        Тест Грейнджера: PSI(t) → wheat_price(t+lag)

        Нулевая гипотеза H0: PSI НЕ помогает предсказать цену.
        Если p-value < 0.05 → отвергаем H0 → PSI GRANGER-причиняет цену.

        Returns:
            dict с результатами по каждому лагу
        """
        logger.info("─" * 52)
        logger.info("Granger Causality: PSI → wheat_price")
        logger.info("─" * 52)

        # Объединяем PSI и цены по году
        df = df_psi[["year", "psi"]].merge(
            df_economics[["year", "wheat_price_usd_t", "wheat_price_yoy_pct"]],
            on="year",
            how="inner",
        ).dropna()

        if len(df) < 8:
            logger.warning("Недостаточно данных для Granger causality (нужно ≥8 лет)")
            return {"error": "insufficient_data"}

        # Тест стационарности
        psi_stat = self.make_stationary(df["psi"], max_diff=1)
        price_stat = self.make_stationary(df["wheat_price_yoy_pct"], max_diff=1)

        # Выравниваем индексы после дифференцирования
        common_idx = psi_stat.index.intersection(price_stat.index)
        test_df = pd.DataFrame({
            "price": price_stat.loc[common_idx].values,
            "psi": psi_stat.loc[common_idx].values,
        })

        if len(test_df) < 6:
            return {"error": "insufficient_data_after_diff"}

        # Granger test
        max_lag = max(1, min(GRANGER_MAX_LAG, len(test_df) // 3 - 1))
        try:
            gc_results = grangercausalitytests(
                test_df[["price", "psi"]],
                maxlag=max_lag,
                verbose=False,
            )

            causality_report = {}
            for lag in range(1, max_lag + 1):
                tests = gc_results[lag][0]
                # F-тест результат
                f_stat, p_val, df_denom, df_num = tests["ssr_ftest"]
                causality_report[f"lag_{lag}"] = {
                    "f_statistic": round(float(f_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "significant": bool(p_val < GRANGER_SIGNIFICANCE),
                    "interpretation": (
                        f"✅ PSI Granger-причиняет цену с лагом {lag} лет (p={p_val:.3f})"
                        if p_val < GRANGER_SIGNIFICANCE
                        else f"❌ Нет значимой связи с лагом {lag} лет (p={p_val:.3f})"
                    ),
                }

            # Находим наиболее значимый лаг
            significant_lags = [
                (k, v) for k, v in causality_report.items() if v["significant"]
            ]
            best_lag = None
            if significant_lags:
                best_lag = min(
                    significant_lags,
                    key=lambda x: x[1]["p_value"],
                )

            return {
                "hypothesis": "PSI → wheat_price_change",
                "method": "Granger Causality (F-test)",
                "n_observations": len(test_df),
                "max_lag_tested": max_lag,
                "significant_lags": [k for k, v in causality_report.items() if v["significant"]],
                "best_lag": best_lag[0] if best_lag else None,
                "best_p_value": best_lag[1]["p_value"] if best_lag else None,
                "conclusion": (
                    f"✅ Гипотеза подтверждена: PSI Granger-причиняет цену пшеницы"
                    if significant_lags
                    else "❌ Статистически значимой связи не обнаружено"
                ),
                "per_lag": causality_report,
            }

        except Exception as e:
            logger.error(f"Granger causality ошибка: {e}")
            return {"error": str(e)}

    def run_var_model(
        self,
        df_psi: pd.DataFrame,
        df_economics: pd.DataFrame,
        df_hydro_annual: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        VAR (Vector Autoregression) — мультивариатная модель.
        Переменные: [PSI, SMAI_tobol, wheat_price_yoy_pct]
        Позволяет учесть взаимное влияние всех факторов.

        Returns:
            dict с коэффициентами модели и прогнозом
        """
        logger.info("VAR Model: PSI + SMAI + wheat_price")

        df = df_psi[["year", "psi"]].merge(
            df_economics[["year", "wheat_price_yoy_pct"]],
            on="year",
        ).dropna()

        if df_hydro_annual is not None:
            df = df.merge(df_hydro_annual, on="year", how="left")

        # Убираем строки с NaN
        df_clean = df.drop(columns=["year"]).dropna()

        if len(df_clean) < 8:
            return {"error": "insufficient_data"}

        try:
            model = VAR(df_clean)
            n = len(df_clean)
            k = df_clean.shape[1]  # число переменных
            # Безопасный максимум лагов: оставляем достаточно df для оценки
            safe_maxlags = max(1, min(VAR_MAX_LAGS, (n - k - 1) // (k * 2 + 1)))
            lag_selection = model.select_order(maxlags=safe_maxlags)
            optimal_lag = lag_selection.aic
            if optimal_lag == 0:
                optimal_lag = 1  # VAR(0) бессмысленен, берём минимум 1

            results = model.fit(optimal_lag)

            # Прогноз на 2 шага вперёд
            forecast = results.forecast(df_clean.values[-optimal_lag:], steps=2)

            return {
                "method": "VAR",
                "optimal_lag": optimal_lag,
                "n_obs": n,
                "aic": round(float(results.aic), 4),
                "bic": round(float(results.bic), 4),
                "variables": list(df_clean.columns),
                "forecast_1step": dict(zip(df_clean.columns, forecast[0].tolist())),
                "forecast_2step": dict(zip(df_clean.columns, forecast[1].tolist())),
            }

        except Exception as e:
            logger.error(f"VAR ошибка: {e}")
            return {"error": str(e)}

    # ─────────────────────────────────────────────
    # Основной запуск
    # ─────────────────────────────────────────────

    def run(self) -> dict:
        """Полный пайплайн AgriContagion."""
        logger.info("=" * 52)
        logger.info("МОДУЛЬ 4: AgriContagion Graph")
        logger.info("=" * 52)

        # Загружаем результаты предыдущих модулей
        df_psi, df_hydro, df_silos, df_economics = None, None, None, None

        psi_path = PROCESSED_DIR / "phenoshift.parquet"
        hydro_path = PROCESSED_DIR / "hydroborder.parquet"
        silos_path = PROCESSED_DIR / "silos.parquet"
        econ_path = PROCESSED_DIR / "economics.parquet"

        if psi_path.exists():
            df_psi = pd.read_parquet(psi_path)
            logger.info("✓ PSI данные загружены")
        if hydro_path.exists():
            df_hydro = pd.read_parquet(hydro_path)
            logger.info("✓ HydroBorder данные загружены")
        if silos_path.exists():
            df_silos = pd.read_parquet(silos_path)
            logger.info("✓ SiloSentry данные загружены")
        if econ_path.exists():
            df_economics = pd.read_parquet(econ_path)
            logger.info("✓ Economics данные загружены")

        # 1. Строим граф
        G = self.build_graph(df_psi, df_hydro, df_silos)
        stats = self.graph_stats(G)
        logger.info(f"Graph stats: {stats}")

        # 2. Granger Causality
        causality_result = {}
        var_result = {}
        if df_psi is not None and df_economics is not None:
            causality_result = self.run_granger_causality(df_psi, df_economics)

            # Ежегодный SMAI для VAR
            df_hydro_annual = None
            if df_hydro is not None:
                df_hydro["year"] = pd.to_datetime(df_hydro["date"]).dt.year
                df_hydro_annual = (
                    df_hydro[df_hydro["basin"] == "tobol"]
                    .groupby("year")["smai"]
                    .mean()
                    .reset_index()
                    .rename(columns={"smai": "smai_tobol"})
                )
            var_result = self.run_var_model(df_psi, df_economics, df_hydro_annual)

        # 3. Итоговый отчёт
        causality_full = {
            "graph_stats": stats,
            "granger_causality": causality_result,
            "var_model": var_result,
            "node_data": {
                n: {k: v for k, v in d.items() if isinstance(v, (str, int, float))}
                for n, d in G.nodes(data=True)
            },
        }

        # Вывод ключевого вывода
        conclusion = causality_result.get("conclusion", "—")
        logger.info(f"\n{'═' * 52}")
        logger.info(f"ВЫВОД: {conclusion}")
        logger.info(f"{'═' * 52}\n")

        # 4. Сохраняем
        with open(self.OUTPUT_GRAPH_PATH, "wb") as f:
            pickle.dump(G, f)
        with open(self.OUTPUT_CAUSALITY_PATH, "w", encoding="utf-8") as f:
            json.dump(causality_full, f, ensure_ascii=False, indent=2)

        # Парquet с данными по узлам
        node_rows = [
            {"node_id": n, **{k: v for k, v in d.items() if isinstance(v, (str, int, float))}}
            for n, d in G.nodes(data=True)
        ]
        df_nodes = pd.DataFrame(node_rows)
        df_nodes.to_parquet(self.OUTPUT_PATH)

        logger.success(f"AgriContagion граф сохранён → {self.OUTPUT_GRAPH_PATH}")
        logger.success(f"Отчёт причинности → {self.OUTPUT_CAUSALITY_PATH}")

        return causality_full


if __name__ == "__main__":
    module = AgriContagionModule()
    result = module.run()
    print(json.dumps(result.get("granger_causality", {}), ensure_ascii=False, indent=2))
