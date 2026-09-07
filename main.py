"""
Daryn — main.py
================
Главная точка входа. Запускает все модули (1–5) по цепочке.

Использование:
  python main.py                 # Полный пайплайн
  python main.py --module 1      # Только Модуль 1
  python main.py --dry-run       # Проверка без API-вызовов
  python main.py --year 2021     # Анализ конкретного года
  python main.py --dashboard     # Запуск дашборда после пайплайна
"""

import sys
import argparse
import time
from pathlib import Path

from loguru import logger


def setup_logging():
    """Настройка логгера с чистым стандартизированным выводом."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level: <8} | {name} - {message}",
        level="INFO",
    )
    logger.add(
        "logs/daryn_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )


def print_banner():
    """Стартовый баннер проекта."""
    banner = """
================================================================
  DARYN -- Agro-Intelligence
  Early Warning System for Agricultural & Food Security Risks
  Google Earth Engine + Remote Sensing + Econometric Modeling
================================================================
    """
    print(banner.strip())


def run_pipeline(args):
    """Запуск полного пайплайна или отдельного модуля."""
    target_modules = None
    if args.module:
        target_modules = [int(m) for m in args.module.split(",")]

    total_start = time.time()
    results = {}

    # ─── Инициализация GEE ───────────────────────────────────────
    if not args.dry_run and (target_modules is None or any(m in [1, 2, 3] for m in (target_modules or []))):
        logger.info("Initializing Google Earth Engine...")
        from pipeline.gee_client import gee
        if not gee.initialize():
            logger.error("Failed to initialize GEE. Run: earthengine authenticate")
            sys.exit(1)
        hc = gee.health_check()
        logger.info(f"GEE Health: {hc['status']} - {hc['message']}")
    else:
        logger.info("Dry-run mode: GEE initialization skipped")

    # ─── Спринт 1: Экономические данные ────────────────────────
    if target_modules is None or 0 in target_modules:
        logger.info("\n" + "-" * 60)
        logger.info("SPRINT 1: Economic Data Loading")
        logger.info("-" * 60)
        t0 = time.time()
        from pipeline.economic_fetcher import EconomicFetcher
        fetcher = EconomicFetcher()
        results["economics"] = fetcher.run() if not args.dry_run else None
        logger.success(f"Economics loaded in {time.time() - t0:.1f}s")

    # ─── Спринт 2: Модуль 1 — PhenoShift ───────────────────────
    if target_modules is None or 1 in target_modules:
        logger.info("\n" + "-" * 60)
        logger.info("SPRINT 2: Module 1 - PhenoShift Index")
        logger.info("-" * 60)
        t0 = time.time()
        from modules.m1_phenoshift import PhenoShiftModule
        module1 = PhenoShiftModule()
        target_year = args.year if args.year else None
        results["phenoshift"] = module1.run(target_year=target_year) if not args.dry_run else None
        logger.success(f"PhenoShift completed in {time.time() - t0:.1f}s")

    # ─── Спринт 3: Модуль 2 — HydroBorder ──────────────────────
    if target_modules is None or 2 in target_modules:
        logger.info("\n" + "-" * 60)
        logger.info("SPRINT 3: Module 2 - HydroBorder (SMAP)")
        logger.info("-" * 60)
        t0 = time.time()
        from modules.m2_hydroborder import HydroBorderModule
        module2 = HydroBorderModule()
        results["hydroborder"] = module2.run() if not args.dry_run else None
        logger.success(f"HydroBorder completed in {time.time() - t0:.1f}s")

    # ─── Спринт 3: Модуль 3 — SiloSentry ───────────────────────
    if target_modules is None or 3 in target_modules:
        logger.info("\n" + "-" * 60)
        logger.info("SPRINT 3: Module 3 - SiloSentry (Elevators)")
        logger.info("-" * 60)
        t0 = time.time()
        from modules.m3_silosentry import SiloSentryModule
        module3 = SiloSentryModule()
        results["silosentry"] = module3.run() if not args.dry_run else None
        logger.success(f"SiloSentry completed in {time.time() - t0:.1f}s")

    # ─── Спринт 4: Модуль 4 — AgriContagion ────────────────────
    if target_modules is None or 4 in target_modules:
        logger.info("\n" + "-" * 60)
        logger.info("SPRINT 4: Module 4 - AgriContagion Graph + Granger")
        logger.info("-" * 60)
        t0 = time.time()
        from modules.m4_agricontagion import AgriContagionModule
        module4 = AgriContagionModule()
        results["agricontagion"] = module4.run() if not args.dry_run else None
        logger.success(f"AgriContagion completed in {time.time() - t0:.1f}s")

    # ─── Спринт 5: Модуль 5 — MicroVulnerability ───────────────
    if target_modules is None or 5 in target_modules:
        logger.info("\n" + "-" * 60)
        logger.info("SPRINT 5: Module 5 - MicroVulnerability")
        logger.info("-" * 60)
        t0 = time.time()
        from modules.m5_microvulnerability import MicroVulnerabilityModule
        module5 = MicroVulnerabilityModule()
        results["microvulnerability"] = module5.run() if not args.dry_run else None
        logger.success(f"MicroVulnerability completed in {time.time() - t0:.1f}s")

    # ─── Итог ────────────────────────────────────────────────────
    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.success(f"PIPELINE FINISHED in {total_time:.1f}s")
    logger.info("Data saved to: data/processed/")
    logger.info("=" * 60)

    # ─── Telegram Оповещение ─────────────────────────────────────
    if not args.dry_run and "phenoshift" in results and results["phenoshift"] is not None:
        try:
            df_psi = results["phenoshift"]
            # Находим данные текущего года
            from config.settings import CURRENT_YEAR
            current_psi_row = df_psi[df_psi["year"] == CURRENT_YEAR]
            
            if not current_psi_row.empty:
                current_psi = current_psi_row["psi"].values[0]
                if current_psi < -1.5:
                    logger.info("🚨 Обнаружен критический уровень PSI, попытка отправки Telegram-уведомления...")
                    
                    smai_val = -1.0 # Дефолтное значение
                    if "hydroborder" in results and results["hydroborder"] is not None:
                        df_hydro = results["hydroborder"]
                        if not df_hydro.empty:
                            y_col = "smai" if "smai" in df_hydro.columns else "soil_moisture"
                            smai_val = float(df_hydro[y_col].iloc[-1])
                            
                    from modules.telegram_bot import DarynTelegramBot
                    import datetime
                    bot = DarynTelegramBot()
                    bot.send_drought_alert(
                        region="СКО (Тестовый полигон)", 
                        psi_val=current_psi, 
                        smai_val=smai_val, 
                        date=datetime.date.today().strftime("%Y-%m-%d")
                    )
        except Exception as e:
            logger.error(f"Ошибка при проверке триггера Telegram: {e}")

    # ─── Дашборд ─────────────────────────────────────────────────
    if args.dashboard:
        logger.info("\n🖥️  Запуск Streamlit дашборда...")
        import subprocess
        subprocess.Popen(["streamlit", "run", "dashboard/app.py"])
        logger.info("Дашборд запущен: http://localhost:8501")

    return results


def main():
    """Главная функция."""
    setup_logging()
    print_banner()

    parser = argparse.ArgumentParser(
        description="Daryn — Агро-Разведчик: пайплайн анализа продовольственных рисков"
    )
    parser.add_argument(
        "--module", "-m",
        type=str,
        default=None,
        help="Запустить конкретный модуль(ы): '1', '2', '1,2,3'. По умолчанию — все.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Проверка конфигурации без реальных API-вызовов.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Год анализа для PhenoShift (по умолчанию: из settings.py)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Запустить Streamlit дашборд после выполнения пайплайна.",
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("🔍 Режим DRY-RUN: только проверка импортов и конфигурации")

    # Создаём директорию для логов
    Path("logs").mkdir(exist_ok=True)

    run_pipeline(args)


if __name__ == "__main__":
    main()
