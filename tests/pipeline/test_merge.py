from __future__ import annotations
"""Tests für den as-of-Join von Kurs- und Fundamentaldaten"""

import unittest

import pandas as pd

from data_pipeline.transforms.merge import build_features_daily


class TestMerge(unittest.TestCase):
    def test_asof_merge_prevents_lookahead(self) -> None:
        # Arrange: Handelstage liegen teilweise vor, auf und nach dem report_date
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
                "open": [100, 101, 102, 103],
                "high": [101, 102, 103, 104],
                "low": [99, 100, 101, 102],
                "close": [100, 101, 102, 103],
                "volume": [1000, 1100, 1200, 1300],
            }
        )

        fundamentals = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "period_end": pd.to_datetime(["2023-12-31"]),
                "report_date": pd.to_datetime(["2024-01-04"]),
                "report_type": ["quarterly"],
                "filing_lag_days": [4],
                "revenue": [100.0],
                "net_income": [10.0],
                "operating_cashflow": [12.0],
                "debt_to_equity": [1.0],
                "gross_margin": [0.5],
                "roe": [0.2],
            }
        )

        # Act: Features über as-of-Join erzeugen
        out = build_features_daily(prices=prices, fundamentals=fundamentals)

        # Assert: Vor report_date keine Fundamentals, am report_date Berichtstag, danach korrekte Differenz
        self.assertEqual(int(out.loc[out["date"] == pd.Timestamp("2024-01-02"), "has_fundamentals"].iloc[0]), 0)
        self.assertEqual(int(out.loc[out["date"] == pd.Timestamp("2024-01-03"), "has_fundamentals"].iloc[0]), 0)
        self.assertEqual(int(out.loc[out["date"] == pd.Timestamp("2024-01-04"), "is_report_day"].iloc[0]), 1)
        self.assertEqual(int(out.loc[out["date"] == pd.Timestamp("2024-01-05"), "days_since_report"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
