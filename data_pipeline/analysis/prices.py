from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_prices(prices: pd.DataFrame) -> dict[str, Any]:
    """Gibt kompakte Qualitäts- und Abdeckungskennzahlen für Tageskurse zurück."""

    # Leerer Input wird mit Null-/None-Werten beantwortet
    if prices.empty:
        return {
            "rows": 0,
            "symbols": 0,
            "date_min": None,
            "date_max": None,
            "duplicate_symbol_date_rows": 0,
            "missing_close_rows": 0,
            "missing_volume_rows": 0,
        }

    # Kernmetriken für Datenqualität und zeitliche Abdeckung
    return {
        "rows": int(len(prices)),
        "symbols": int(prices["symbol"].nunique()),
        "date_min": str(pd.to_datetime(prices["date"]).min().date()),
        "date_max": str(pd.to_datetime(prices["date"]).max().date()),
        "duplicate_symbol_date_rows": int(prices.duplicated(subset=["symbol", "date"]).sum()),
        "missing_close_rows": int(prices["close"].isna().sum()),
        "missing_volume_rows": int(prices["volume"].isna().sum()),
    }
