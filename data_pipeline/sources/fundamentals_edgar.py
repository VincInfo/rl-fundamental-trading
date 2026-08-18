from __future__ import annotations

"""Laden und Aufbereiten von Fundamentaldaten aus SEC EDGAR Companyfacts.

Das Modul transformiert heterogene XBRL-Facts in ein kompaktes, einheitliches
Schema für die Pipeline. Dabei wird besonders auf zeitliche Korrektheit geachtet,
damit nur bereits veröffentlichte Informationen in späteren Schritten genutzt werden.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ALLOWED_FORMS = {"10-Q": "quarterly", "10-K": "annual"}


@dataclass(frozen=True)
class EdgarTagMap:
    # Manche Unternehmen melden Umsatz unter unterschiedlichen us-gaap-Tags.
    # Deshalb werden Primär- und Alternativtag gemeinsam unterstützt.
    revenue: tuple[str, ...] = (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    )
    net_income: tuple[str, ...] = ("NetIncomeLoss",)
    operating_cashflow: tuple[str, ...] = ("NetCashProvidedByUsedInOperatingActivities",)
    gross_profit: tuple[str, ...] = ("GrossProfit",)
    liabilities: tuple[str, ...] = ("Liabilities",)
    stockholders_equity: tuple[str, ...] = ("StockholdersEquity",)


def _http_get_json(url: str, user_agent: str) -> dict[str, Any]:
    """Führt einen SEC-Request mit User-Agent aus und liefert JSON."""

    # SEC erwartet einen aussagekräftigen User-Agent; ohne diesen Header drohen
    # Rate-Limits oder Blockaden.
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    response.raise_for_status()
    return response.json()


def _ticker_to_cik(user_agent: str) -> dict[str, str]:
    """Mappt Ticker-Symbole auf CIK-IDs aus dem SEC-Tickerverzeichnis."""

    raw = _http_get_json(SEC_TICKER_URL, user_agent=user_agent)
    mapping: dict[str, str] = {}
    for _, entry in raw.items():
        # Einheitliche Großschreibung + 10-stellige CIK für Companyfacts-URLs.
        ticker = str(entry.get("ticker", "")).upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if ticker and cik:
            mapping[ticker] = cik
    return mapping


def _extract_tag_values(company_facts: dict[str, Any], tag_names: tuple[str, ...]) -> pd.DataFrame:
    """Extrahiert gewünschte us-gaap-Tags aus der Companyfacts-Antwort."""

    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    frames: list[pd.DataFrame] = []

    for tag in tag_names:
        fact_info = us_gaap.get(tag)
        if not fact_info:
            continue

        units = fact_info.get("units", {})
        for unit_name, entries in units.items():
            # Nur wirtschaftlich sinnvolle Einheitentypen aufnehmen.
            if unit_name not in {"USD", "USD/shares", "pure"}:
                continue
            frame = pd.DataFrame(entries)
            if frame.empty:
                continue
            # Ursprungs-Tag behalten, damit wir später sauber pivotieren können.
            frame["tag"] = tag
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["end", "filed", "form", "val", "fy", "fp", "tag"])

    merged = pd.concat(frames, ignore_index=True)
    cols = ["end", "filed", "form", "val", "fy", "fp", "tag"]
    available = [c for c in cols if c in merged.columns]
    return merged[available].copy()


def _first_available(series: pd.Series) -> float:
    """Gibt den ersten nicht-leeren Wert einer Serie zurück."""

    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.iloc[0])


def load_fundamentals(symbols: list[str], start: str, end: str, user_agent: str) -> pd.DataFrame:
    """Lädt quartalsweise und jährliche Fundamentaldaten aus SEC-EDGAR-Companyfacts."""

    # Einmalige Vorbereitung: Ticker->CIK-Mapping und angefragte Tags.
    ticker_to_cik = _ticker_to_cik(user_agent=user_agent)
    tags = EdgarTagMap()
    requested_tags = (
        *tags.revenue,
        *tags.net_income,
        *tags.operating_cashflow,
        *tags.gross_profit,
        *tags.liabilities,
        *tags.stockholders_equity,
    )

    all_rows: list[pd.DataFrame] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    for symbol in symbols:
        # Ohne CIK kann kein Companyfacts-Endpunkt aufgerufen werden.
        cik = ticker_to_cik.get(symbol.upper())
        if not cik:
            continue

        # Rohdaten laden und auf benötigte Facts reduzieren.
        facts_url = SEC_COMPANY_FACTS_URL.format(cik=cik)
        company_facts = _http_get_json(facts_url, user_agent=user_agent)
        raw = _extract_tag_values(company_facts, requested_tags)
        if raw.empty:
            continue

        # Zeit- und Qualitätsfilter:
        # - nur Datensätze mit den Kernfeldern,
        # - nur erlaubte Form-Typen,
        # - period_end im Zielhorizont,
        # - report_date maximal leicht nach dem Ende des Analysehorizonts.
        raw = raw.dropna(subset=["end", "filed", "form", "val"])
        raw["period_end"] = pd.to_datetime(raw["end"], errors="coerce").astype("datetime64[ns]")
        raw["report_date"] = pd.to_datetime(raw["filed"], errors="coerce").astype("datetime64[ns]")
        raw = raw[raw["form"].isin(ALLOWED_FORMS)].copy()
        raw = raw[raw["period_end"].between(start_ts, end_ts)].copy()
        raw = raw[raw["report_date"] <= end_ts + pd.Timedelta(days=120)].copy()
        if raw.empty:
            continue

        # Facts in Wide-Format drehen, damit pro Bericht eine Zeile entsteht.
        pivot = (
            raw.pivot_table(
                index=["period_end", "report_date", "form", "fy", "fp"],
                columns="tag",
                values="val",
                aggfunc=_first_available,
            )
            .reset_index()
            .sort_values(["period_end", "report_date"])
        )

        pivot["symbol"] = symbol.upper()
        pivot["report_type"] = pivot["form"].map(ALLOWED_FORMS)

        # Erstes Filing je Periode und Berichtstyp behalten, um die erste Marktverfügbarkeit abzubilden.
        pivot = pivot.sort_values("report_date").drop_duplicates(
            subset=["symbol", "period_end", "report_type"],
            keep="first",
        )

        # Umsatz robust bilden: bevorzugter Tag, falls leer auf Alternativ-Tag fallen.
        revenue = pivot.get("RevenueFromContractWithCustomerExcludingAssessedTax")
        if revenue is None:
            revenue = pd.Series(index=pivot.index, dtype=float)
        revenues_alt = pivot.get("Revenues")
        if revenues_alt is None:
            revenues_alt = pd.Series(index=pivot.index, dtype=float)

        pivot["revenue"] = revenue.combine_first(revenues_alt)
        pivot["net_income"] = pivot.get("NetIncomeLoss")
        pivot["operating_cashflow"] = pivot.get("NetCashProvidedByUsedInOperatingActivities")
        pivot["gross_profit"] = pivot.get("GrossProfit")
        pivot["liabilities"] = pivot.get("Liabilities")
        pivot["stockholders_equity"] = pivot.get("StockholdersEquity")

        # Abgeleitete Kennzahlen für Modell-Features.
        # Divisionen gegen Null werden durch replace(0, NA) abgesichert.
        pivot["debt_to_equity"] = pivot["liabilities"] / pivot["stockholders_equity"].replace(0, pd.NA)
        pivot["gross_margin"] = pivot["gross_profit"] / pivot["revenue"].replace(0, pd.NA)
        pivot["roe"] = pivot["net_income"] / pivot["stockholders_equity"].replace(0, pd.NA)
        # Filing-Lag ist zentral für zeitliche Verfügbarkeit und Analyse.
        pivot["filing_lag_days"] = (pivot["report_date"] - pivot["period_end"]).dt.days

        keep_cols = [
            "symbol",
            "period_end",
            "report_date",
            "report_type",
            "filing_lag_days",
            "revenue",
            "net_income",
            "operating_cashflow",
            "debt_to_equity",
            "gross_margin",
            "roe",
        ]
        all_rows.append(pivot[keep_cols].copy())

    if not all_rows:
        # Stabiler Rückgabevertrag mit festen Spalten auch bei leerem Ergebnis.
        return pd.DataFrame(
            columns=[
                "symbol",
                "period_end",
                "report_date",
                "report_type",
                "filing_lag_days",
                "revenue",
                "net_income",
                "operating_cashflow",
                "debt_to_equity",
                "gross_margin",
                "roe",
            ]
        )

    # Alle Symbole zusammenführen und zeitlich konsistent sortieren.
    merged = pd.concat(all_rows, ignore_index=True)
    merged = merged.sort_values(["symbol", "period_end", "report_date"]).reset_index(drop=True)
    return merged
