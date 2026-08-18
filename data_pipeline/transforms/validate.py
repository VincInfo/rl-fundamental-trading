from __future__ import annotations

"""Validierungsregeln für Kurs-, Fundamental- und Feature-Datensätze.

Checks sollen frühzeitig sicherstellen, dass Datensätze
schema-konform sind und keine zeitlichen Inkonsistenzen enthalten
"""

import pandas as pd


class ValidationError(ValueError):
    """Wird ausgelöst, wenn Datenvalidierungen fehlschlagen"""


def validate_prices(prices: pd.DataFrame) -> None:
    # Pflichtspalten des normalisierten Kurs-Schemas
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValidationError(f"Kursdaten ohne erforderliche Spalten: {sorted(missing)}")

    # Jede Aktie darf pro Tag genau eine Zeile haben
    if prices.duplicated(subset=["symbol", "date"]).any():
        raise ValidationError("Kursdaten enthalten doppelte Symbol/Datum-Zeilen")


def validate_fundamentals(fundamentals: pd.DataFrame) -> None:
    # Mindestschema der Fundamentalquelle inklusive zeitlicher Felder
    required = {
        "symbol",
        "period_end",
        "report_date",
        "report_type",
        "filing_lag_days",
    }
    missing = required - set(fundamentals.columns)
    if missing:
        raise ValidationError(f"Fundamentaldaten ohne erforderliche Spalten: {sorted(missing)}")

    # Bericht kann nicht vor Ende seiner Berichtsperiode veröffentlicht sein
    if (pd.to_datetime(fundamentals["report_date"]) < pd.to_datetime(fundamentals["period_end"])).any():
        raise ValidationError("Zeilen gefunden, bei denen report_date vor period_end liegt")


def validate_features(features: pd.DataFrame) -> None:
    # Kernfelder für die zeitliche Modellnutzung und Analyse
    required = {"date", "symbol", "report_date", "days_since_report", "is_report_day"}
    missing = required - set(features.columns)
    if missing:
        raise ValidationError(f"Features ohne erforderliche Spalten: {sorted(missing)}")

    # Harte Leakage-Prüfung: report_date darf niemals in der Zukunft des Handelstags liegen
    with_fund = features[features["report_date"].notna()].copy()
    if not with_fund.empty:
        if (pd.to_datetime(with_fund["report_date"]) > pd.to_datetime(with_fund["date"])).any():
            raise ValidationError("Look-Ahead erkannt: report_date liegt nach dem Handelstag")
