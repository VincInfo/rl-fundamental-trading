from __future__ import annotations
"""Tests für Analyse-Zusammenfassungen der Pipeline-Datensätze"""

import unittest

import pandas as pd

from data_pipeline.analysis.features import summarize_features
from data_pipeline.analysis.fundamentals import summarize_fundamentals
from data_pipeline.analysis.prices import summarize_prices


class TestAnalysis(unittest.TestCase):
    def test_analysis_summaries_return_expected_core_metrics(self) -> None:
        # Arrange: Kleine, kontrollierte In-Memory-Datensätze mit einer Aktie
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "symbol": ["AAPL", "AAPL"],
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
                "volume": [1000, 1100],
            }
        )
        fundamentals = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "period_end": pd.to_datetime(["2023-12-31"]),
                "report_date": pd.to_datetime(["2024-01-02"]),
                "report_type": ["quarterly"],
                "filing_lag_days": [2],
                "revenue": [100],
                "net_income": [10],
                "operating_cashflow": [12],
                "debt_to_equity": [1.2],
                "gross_margin": [0.4],
                "roe": [0.1],
            }
        )
        features = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "symbol": ["AAPL", "AAPL"],
                "report_date": [pd.NaT, pd.Timestamp("2024-01-02")],
                "has_fundamentals": [0, 1],
                "is_report_day": [0, 1],
                "days_since_report": [999, 0],
            }
        )

        # Act: Alle drei Analysefunktionen ausführen
        prices_summary = summarize_prices(prices)
        fundamentals_summary = summarize_fundamentals(fundamentals)
        features_summary, coverage = summarize_features(features)

        # Assert: Kernmetriken sollen die erwarteten Kontrollwerte liefern
        self.assertEqual(prices_summary["rows"], 2)
        self.assertEqual(fundamentals_summary["symbols"], 1)
        self.assertEqual(features_summary["lookahead_violations"], 0)
        self.assertEqual(int(coverage.loc[coverage["symbol"] == "AAPL", "rows_with_fundamentals"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
