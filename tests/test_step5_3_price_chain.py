import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.elevator_logistics import (
    calculate_price_transmission_chain,
    BASE_WHEAT_PRICE_KZT_T,
    BASE_FLOUR_PRICE_KZT_KG,
    BASE_BREAD_PRICE_KZT_LOAF,
    ELASTICITY_YIELD_TO_WHEAT,
    ELASTICITY_WHEAT_TO_FLOUR,
    ELASTICITY_FLOUR_TO_BREAD,
)


class TestPriceTransmissionChain(unittest.TestCase):
    """Тестирование эконометрического калькулятора передачи цен (Step 5.3)."""

    def test_no_shock_baseline_stable(self):
        """При нулевом шоке урожая все цены должны оставаться на базовом уровне."""
        result = calculate_price_transmission_chain(yield_shock_pct=0.0)
        self.assertEqual(result["wheat_jump_pct"], 0.0)
        self.assertEqual(result["flour_jump_pct"], 0.0)
        self.assertEqual(result["bread_jump_pct"], 0.0)
        self.assertEqual(result["risk_tier"], "LOW")
        self.assertEqual(result["wheat_price_projected"], BASE_WHEAT_PRICE_KZT_T)
        self.assertEqual(result["bread_price_projected"], BASE_BREAD_PRICE_KZT_LOAF)

    def test_severe_shock_direction(self):
        """Падение урожая на 40% должно вызвать рост всей цепочки цен."""
        result = calculate_price_transmission_chain(yield_shock_pct=-40.0)

        # Пшеница должна подорожать
        self.assertGreater(result["wheat_price_projected"], BASE_WHEAT_PRICE_KZT_T)
        # Мука должна подорожать
        self.assertGreater(result["flour_price_projected"], BASE_FLOUR_PRICE_KZT_KG)
        # Хлеб должен подорожать
        self.assertGreater(result["bread_price_projected"], BASE_BREAD_PRICE_KZT_LOAF)

        # Пшеница растет больше, чем мука (выше elasticity, ближе к шоку)
        self.assertGreater(result["wheat_jump_pct"], result["flour_jump_pct"])
        # Мука растет больше, чем хлеб (субсидии СПК демпфируют)
        self.assertGreater(result["flour_jump_pct"], result["bread_jump_pct"])

    def test_granger_passthrough_coefficients(self):
        """Проверяем точную математику коэффициентов Granger Pass-Through."""
        shock = -30.0  # -30%
        result = calculate_price_transmission_chain(yield_shock_pct=shock)

        # wheat_jump = 30% * |elasticity_yield_to_wheat| = 30 * 0.61 = 18.3%
        expected_wheat_jump = 30.0 * abs(ELASTICITY_YIELD_TO_WHEAT)
        expected_flour_jump = expected_wheat_jump * ELASTICITY_WHEAT_TO_FLOUR
        expected_bread_jump = expected_flour_jump * ELASTICITY_FLOUR_TO_BREAD

        self.assertAlmostEqual(result["wheat_jump_pct"], round(expected_wheat_jump, 1), places=0)
        self.assertAlmostEqual(result["flour_jump_pct"], round(expected_flour_jump, 1), places=0)
        self.assertAlmostEqual(result["bread_jump_pct"], round(expected_bread_jump, 1), places=0)

    def test_timeline_structure(self):
        """Проверяем структуру временных рядов цен (6 временных точек)."""
        result = calculate_price_transmission_chain(yield_shock_pct=-25.0)

        self.assertEqual(len(result["timeline_days"]), 6)
        self.assertEqual(len(result["wheat_curve"]), 6)
        self.assertEqual(len(result["flour_curve"]), 6)
        self.assertEqual(len(result["bread_curve"]), 6)

        # Временные точки должны идти по возрастанию
        timeline = result["timeline_days"]
        for i in range(1, len(timeline)):
            self.assertGreater(timeline[i], timeline[i - 1])

        # Пшеница растет быстро (к 60-му дню уже выше базовой)
        self.assertGreater(result["wheat_curve"][2], result["wheat_curve"][0])
        # Хлеб растет медленно (к 30-му дню практически на базовом уровне)
        self.assertAlmostEqual(result["bread_curve"][1], result["bread_curve"][0], delta=1.0)

    def test_risk_tiers(self):
        """Проверяем корректное присвоение уровней риска."""
        low = calculate_price_transmission_chain(yield_shock_pct=-5.0)
        medium = calculate_price_transmission_chain(yield_shock_pct=-25.0)
        high = calculate_price_transmission_chain(yield_shock_pct=-45.0)

        self.assertEqual(low["risk_tier"], "LOW")
        self.assertEqual(medium["risk_tier"], "MEDIUM")
        self.assertEqual(high["risk_tier"], "HIGH")

    def test_bilingual_recommendations(self):
        """Проверяем наличие двуязычных рекомендаций."""
        result = calculate_price_transmission_chain(yield_shock_pct=-35.0)
        self.assertIn("recommendation_ru", result)
        self.assertIn("recommendation_kk", result)
        self.assertGreater(len(result["recommendation_ru"]), 10)
        self.assertGreater(len(result["recommendation_kk"]), 10)


if __name__ == "__main__":
    unittest.main()

