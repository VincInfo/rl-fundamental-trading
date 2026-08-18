from __future__ import annotations

from typing import Any

import pandas as pd


KEY_COLUMNS = [
    "revenue",
    "net_income",
    "operating_cashflow",
    "debt_to_equity",
    "gross_margin",
    "roe",
]


def summarize_fundamentals(fundamentals: pd.DataFrame) -> dict[str, Any]:
    """Gibt kompakte Kennzahlen zur Filing-Abdeckung und Vollständigkeit der Fundamentaldaten zurück."""

    # Leerer Input liefert ein vollständiges, aber neutrales Ergebnisobjekt
    if fundamentals.empty:
        return {
            "rows": 0,
            "symbols": 0,
            "period_min": None,
            "period_max": None,
            "report_type_counts": {},
            "filing_lag_days_median": None,
            "filing_lag_days_p90": None,
            "missing_ratio": {},
        }

    # Fehlquoten der zentralen Finanzkennzahlen je Spalte
    missing_ratio = {
        col: float(fundamentals[col].isna().mean())
        for col in KEY_COLUMNS
        if col in fundamentals.columns
    }

    # Filing-Lag robust als numerische Serie für Quantile aufbereiten
    lag = pd.to_numeric(fundamentals["filing_lag_days"], errors="coerce").dropna()
    return {
        "rows": int(len(fundamentals)),
        "symbols": int(fundamentals["symbol"].nunique()),
        "period_min": str(pd.to_datetime(fundamentals["period_end"]).min().date()),
        "period_max": str(pd.to_datetime(fundamentals["period_end"]).max().date()),
        "report_type_counts": {
            str(key): int(value) for key, value in fundamentals["report_type"].value_counts(dropna=False).to_dict().items()
        },
        "filing_lag_days_median": float(lag.median()) if not lag.empty else None,
        "filing_lag_days_p90": float(lag.quantile(0.9)) if not lag.empty else None,
        "missing_ratio": missing_ratio,
    }
