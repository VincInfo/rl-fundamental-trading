from __future__ import annotations
"""Tests für die Aufbereitung der yfinance-Spaltenstruktur"""

import unittest

import pandas as pd

from data_pipeline.sources.prices_yfinance import _flatten_if_multiindex


class TestPricesLoader(unittest.TestCase):
    def test_flatten_multiindex_columns(self) -> None:
        # Arrange: Beispiel mit MultiIndex-Spalten wie von yfinance
        cols = pd.MultiIndex.from_tuples(
            [
                ("Close", "AAPL"),
                ("High", "AAPL"),
                ("Low", "AAPL"),
                ("Open", "AAPL"),
                ("Volume", "AAPL"),
            ],
            names=["Price", "Ticker"],
        )
        data = pd.DataFrame([[100.0, 101.0, 99.0, 100.5, 1_000_000]], columns=cols)

        # Act: MultiIndex auf flache Spaltennamen reduzieren
        out = _flatten_if_multiindex(data)

        # Assert: Ergebnis enthält keine MultiIndex-Struktur mehr
        self.assertFalse(isinstance(out.columns, pd.MultiIndex))
        self.assertEqual(list(out.columns), ["Close", "High", "Low", "Open", "Volume"])


if __name__ == "__main__":
    unittest.main()
