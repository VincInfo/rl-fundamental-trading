from __future__ import annotations
"""Tests für strikte Zeit-Splits und Walk-Forward-Fenster"""

import unittest

import pandas as pd

from data_pipeline.api import build_walk_forward_feature_splits, split_features_by_time
from data_pipeline.transforms.validate import ValidationError


class TestSplit(unittest.TestCase):
    def _make_features(self) -> pd.DataFrame:
        # Hilfsdatensatz: Zwei Symbole über identische Handelstage
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        rows: list[dict[str, object]] = []
        for symbol in ["AAPL", "MSFT"]:
            for date in dates:
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "close": 100.0,
                        "report_date": pd.Timestamp("2023-12-31"),
                        "is_report_day": 0,
                        "days_since_report": 1,
                        "has_fundamentals": 1,
                    }
                )
        return pd.DataFrame(rows)

    def test_split_features_by_time_is_strict(self) -> None:
        # Arrange: Datensatz mit eindeutigen Zeitfenstern
        features = self._make_features()

        split = split_features_by_time(
            features=features,
            train_end="2024-01-03",
            val_end="2024-01-05",
            test_end="2024-01-07",
        )

        # Assert: Train/Val/Test dürfen nur die jeweils definierten Datumsbereiche enthalten.
        self.assertTrue((split.train["date"] <= pd.Timestamp("2024-01-03")).all())
        self.assertTrue(((split.val["date"] > pd.Timestamp("2024-01-03")) & (split.val["date"] <= pd.Timestamp("2024-01-05"))).all())
        self.assertTrue(((split.test["date"] > pd.Timestamp("2024-01-05")) & (split.test["date"] <= pd.Timestamp("2024-01-07"))).all())

        # Zwei Symbole pro Tag: 3 Tage Train, 2 Tage Val, 2 Tage Test
        self.assertEqual(len(split.train), 6)
        self.assertEqual(len(split.val), 4)
        self.assertEqual(len(split.test), 4)

    def test_split_features_by_time_rejects_invalid_boundaries(self) -> None:
        # Arrange: Vertauschte Grenzen, damit train_end nicht vor val_end liegt
        features = self._make_features()

        # Assert: Ungültige Grenzen müssen als ValidationError enden
        with self.assertRaises(ValidationError):
            split_features_by_time(features=features, train_end="2024-01-05", val_end="2024-01-03")

    def test_build_walk_forward_feature_splits(self) -> None:
        # Arrange: Kleine Serie, damit die Anzahl der Fenster eindeutig kontrollierbar ist
        features = self._make_features()

        windows = build_walk_forward_feature_splits(
            features=features,
            min_train_periods=3,
            val_periods=2,
            test_periods=2,
            step_periods=1,
        )

        # Assert: Fenstergrenzen und Teilmengen müssen den Parametern exakt folgen
        # Bei 8 Tagen und (3,2,2) entstehen 2 Fenster: train_end an Tag 3 und 4
        self.assertEqual(len(windows), 2)

        first = windows[0]
        self.assertEqual(first.train_end, pd.Timestamp("2024-01-03"))
        self.assertEqual(first.val_start, pd.Timestamp("2024-01-04"))
        self.assertEqual(first.val_end, pd.Timestamp("2024-01-05"))
        self.assertEqual(first.test_start, pd.Timestamp("2024-01-06"))
        self.assertEqual(first.test_end, pd.Timestamp("2024-01-07"))

        self.assertEqual(len(first.train), 6)
        self.assertEqual(len(first.val), 4)
        self.assertEqual(len(first.test), 4)


if __name__ == "__main__":
    unittest.main()
