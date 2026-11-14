"""
AgriCascade — tests/run_all_daryn_tests.py
==========================================
Мастер-раннер приемочных тестов для финала конкурса «Дарын».
Запускает верификационные тесты Phase 3–7.
"""

import unittest
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from tests.test_multiregion_multicrop import TestMultiRegionMultiCrop
from tests.test_gis_choropleth import TestGISChoropleth
from tests.test_step5_1_elevators import TestElevatorLogisticsClustering
from tests.test_step5_2_rerouting import TestGrainLogisticsRerouting
from tests.test_step5_3_price_chain import TestPriceTransmissionChain
from tests.test_step6_1_bilingual import TestStep61Bilingual
from tests.test_step6_2_executive_audit import TestExecutiveAudit
from tests.test_step6_3_soak import TestStep63SoakStability
from tests.test_phase7_space_segment import TestPhase7SpaceSegment
from tests.test_phase7_offline_cache import TestPhase7OfflineCache
from tests.test_real_2026_data import TestReal2026Data


def run_full_verification_suite():
    print("=" * 70)
    print("🌾 AGRICASCADE — ПОЛНЫЙ ВЕРИФИКАЦИОННЫЙ ПАКЕТ ТЕСТОВ (ФИНАЛ «ДАРЫН»)")
    print("=" * 70)
    start_time = time.time()

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    # Добавляем все тест-кейсы
    suite.addTests(loader.loadTestsFromTestCase(TestMultiRegionMultiCrop))
    suite.addTests(loader.loadTestsFromTestCase(TestGISChoropleth))
    suite.addTests(loader.loadTestsFromTestCase(TestElevatorLogisticsClustering))
    suite.addTests(loader.loadTestsFromTestCase(TestGrainLogisticsRerouting))
    suite.addTests(loader.loadTestsFromTestCase(TestPriceTransmissionChain))
    suite.addTests(loader.loadTestsFromTestCase(TestStep61Bilingual))
    suite.addTests(loader.loadTestsFromTestCase(TestExecutiveAudit))
    suite.addTests(loader.loadTestsFromTestCase(TestStep63SoakStability))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase7SpaceSegment))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase7OfflineCache))
    suite.addTests(loader.loadTestsFromTestCase(TestReal2026Data))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"ИТОГ: Тестов выполнено: {result.testsRun}")
    print(f"Ошибок: {len(result.errors)}, Сбоев: {len(result.failures)}")
    print(f"Время выполнения: {elapsed:.2f} сек.")
    print("СТАТУС: " + ("✅ ВСЕ ТЕСТЫ ЗЕЛЕНЫЕ (ГОТОВО К ФИНАЛУ)" if result.wasSuccessful() else "❌ ЕСТЬ ОШИБКИ"))
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_full_verification_suite()
    sys.exit(0 if success else 1)
