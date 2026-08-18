from __future__ import annotations
"""Tests für Validierungsregeln und Fehlerfälle der Datenpipeline"""

import unittest

import pandas as pd

from data_pipeline.transforms.validate import ValidationError, validate_features, validate_fundamentals, validate_prices


class TestValidate(unittest.TestCase):
    def test_validate_prices_rejects_duplicates(self) -> None:
        # Arrange: Zwei identische symbol/date-Zeilen simulieren Duplikate
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "symbol": ["AAPL", "AAPL"],
                "open": [1, 1],
                "high": [1, 1],
                "low": [1, 1],
                "close": [1, 1],
                "volume": [1, 1],
            }
        )
        # Assert: Duplikate müssen mit ValidationError abgewiesen werden
        with self.assertRaises(ValidationError):
            validate_prices(prices)

    def test_validate_fundamentals_rejects_bad_dates(self) -> None:
        # Arrange: report_date liegt fälschlich vor period_end
        fundamentals = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "period_end": pd.to_datetime(["2024-03-31"]),
                "report_date": pd.to_datetime(["2024-03-01"]),
                "report_type": ["quarterly"],
                "filing_lag_days": [10],
            }
        )
        # Assert: Zeitlich inkonsistente Fundamentaldaten werden abgewiesen
        with self.assertRaises(ValidationError):
            validate_fundamentals(fundamentals)

    def test_validate_features_rejects_lookahead(self) -> None:
        # Arrange: report_date nach Handelstag entspricht Look-Ahead
        features = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "symbol": ["AAPL"],
                "report_date": pd.to_datetime(["2024-01-03"]),
                "days_since_report": [0],
                "is_report_day": [0],
            }
        )
        # Assert: Leakage-Szenario muss erkannt und geblockt werden
        with self.assertRaises(ValidationError):
            validate_features(features)


if __name__ == "__main__":
    unittest.main()
