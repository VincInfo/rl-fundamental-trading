from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_features(features: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fasst zusammengeführte Features zusammen und liefert eine Abdeckungstabelle je Symbol zurück."""

    # Für leere Inputs ein konsistentes Summary + leere Coverage-Tabelle liefern.
    if features.empty:
        summary = {
            "rows": 0,
            "symbols": 0,
            "date_min": None,
            "date_max": None,
            "lookahead_violations": 0,
            "report_day_rows": 0,
            "with_fundamentals_ratio": 0.0,
        }
        coverage = pd.DataFrame(
            columns=["symbol", "rows", "rows_with_fundamentals", "fundamentals_coverage_ratio", "report_day_rows"]
        )
        return summary, coverage

    with_fund = features["report_date"].notna()
    with_fund_frame = features[with_fund].copy()
    lookahead = 0
    if not with_fund_frame.empty:
        # Sicherheitscheck: report_date darf niemals nach dem Handelstag liegen.
        lookahead = int((pd.to_datetime(with_fund_frame["report_date"]) > pd.to_datetime(with_fund_frame["date"])).sum())

    # Abdeckung je Symbol berechnen: wie oft Fundamentals verfügbar waren.
    coverage = (
        features.groupby("symbol", as_index=False)
        .agg(
            rows=("symbol", "size"),
            rows_with_fundamentals=("has_fundamentals", "sum"),
            report_day_rows=("is_report_day", "sum"),
        )
        .sort_values("symbol")
    )
    coverage["fundamentals_coverage_ratio"] = coverage["rows_with_fundamentals"] / coverage["rows"]

    # Kompakte Gesamtmetriken für schnellen Qualitätsüberblick.
    summary = {
        "rows": int(len(features)),
        "symbols": int(features["symbol"].nunique()),
        "date_min": str(pd.to_datetime(features["date"]).min().date()),
        "date_max": str(pd.to_datetime(features["date"]).max().date()),
        "lookahead_violations": lookahead,
        "report_day_rows": int(features["is_report_day"].sum()),
        "with_fundamentals_ratio": float(features["has_fundamentals"].mean()),
    }
    return summary, coverage
