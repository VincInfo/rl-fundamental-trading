from __future__ import annotations

"""Zusammenführen von Kurs- und Fundamentaldaten zu Tages-Features.

Die zentrale Regel dieses Moduls lautet: Für jeden Handelstag dürfen nur
Fundamentaldaten verwendet werden, die bis zu diesem Tag veröffentlicht wurden.
Das wird über einen as-of-Join mit Richtung "backward" umgesetzt.
"""

import pandas as pd


def build_features_daily(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Führt Tageskurse per as-of-Join mit den zuletzt veröffentlichten Fundamentaldaten je Symbol zusammen"""

    # Stabiler Rückgabepfad: Leere Kursdaten erzeugen direkt ein leeres Ergebnis
    if prices.empty:
        return prices.copy()

    frames: list[pd.DataFrame] = []
    for symbol, p in prices.groupby("symbol", sort=False):
        # Kurs- und Fundamentalteildaten je Symbol getrennt verarbeiten ->
        # damit Join nur innerhalb derselben Aktie stattfindet
        p = p.sort_values("date").copy()
        p["date"] = pd.to_datetime(p["date"])
        f = fundamentals[fundamentals["symbol"] == symbol].sort_values("report_date").copy()
        if not f.empty:
            f["report_date"] = pd.to_datetime(f["report_date"])

        if f.empty:
            # Wenn keine Fundamentals vorliegen, bleiben Kursdaten erhalten +
            # fundamentale Spalten werden explizit als fehlend markiert
            merged = p.copy()
            merged["period_end"] = pd.NaT
            merged["report_date"] = pd.NaT
            merged["report_type"] = pd.NA
            merged["filing_lag_days"] = pd.NA
            merged["revenue"] = pd.NA
            merged["net_income"] = pd.NA
            merged["operating_cashflow"] = pd.NA
            merged["debt_to_equity"] = pd.NA
            merged["gross_margin"] = pd.NA
            merged["roe"] = pd.NA
        else:
            # Symbolspalte aus dem rechten Frame entfernen, weil das Symbol
            # bereits aus den Kursdaten kommt -> sonst doppelt
            f_daily = f.drop(columns=["symbol"]).copy()
            # As-of-Join mit "backward": nimmt den letzten report_date,
            # der kleiner/gleich dem Handelstag ist (kein Look-Ahead)!
            merged = pd.merge_asof(
                p,
                f_daily,
                left_on="date",
                right_on="report_date",
                direction="backward",
            )

        # Zusatzfeatures für Analyse und Modellnutzung:
        # - is_report_day: Tag mit neuer Fundamentalveröffentlichung
        # - days_since_report: Alter der zuletzt bekannten Fundamentals
        # - has_fundamentals: Verfügbarkeitssignal für Missingness
        merged["is_report_day"] = (merged["date"] == merged["report_date"]).astype(int)
        merged["days_since_report"] = (
            (pd.to_datetime(merged["date"]) - pd.to_datetime(merged["report_date"])).dt.days.fillna(999).astype(int)
        )
        merged["has_fundamentals"] = merged["report_date"].notna().astype(int)

        frames.append(merged)

    # Alle Symbolergebnisse zusammenführen und global sortieren
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out
