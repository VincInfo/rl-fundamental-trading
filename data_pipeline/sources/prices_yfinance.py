from __future__ import annotations

"""Laden und Normalisieren von täglichen Kursdaten über yfinance.

Ziel des Moduls: stabiler, einheitlicher Kursdatensatz für die
nachgelagerten Transformationsschritte. Die Funktionen kümmern sich darum,
kleine Schemaunterschiede der Quelle abzufangen und auf feste Spaltennamen
zu mappen
"""

from typing import Iterable

import pandas as pd
import yfinance as yf


PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _flatten_if_multiindex(frame: pd.DataFrame) -> pd.DataFrame:
    """Wandelt yfinance-MultiIndex-Spalten in einfache Zeichenketten um"""

    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    flattened = frame.copy()
    # yfinance liefert je nach Aufruf einen MultiIndex (z. B. bei mehreren Tickern)
    # Für die Pipeline werden flache Spaltennamen verwendet, damit Mapping und Validierung
    # unabhängig vom Rückgabeformat stabil bleiben
    flattened.columns = [
        str(first) if str(second).strip() == "" else str(first)
        for first, second in flattened.columns.to_flat_index()
    ]
    return flattened


def load_prices(symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Lädt tägliche OHLCV-Kurse über yfinance und normalisiert sie auf ein einheitliches Schema"""

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        # Daten pro Symbol getrennt laden. macht Fehlerbilder pro Ticker
        # nachvollziehbar + hält die Nachbearbeitung einfach
        data = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data.empty:
            # Leere Antworten werden übersprungen -> einzelne Ausfälle sollen
            # gesamte Pipeline nicht stoppen
            continue

        # Quellspalten auf das interne, feste Schema abbilden.
        frame = _flatten_if_multiindex(data).reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        expected = {"date", "open", "high", "low", "close", "volume"}
        missing = expected - set(frame.columns)
        if missing:
            # harte Schema-Prüfung: Wenn yfinance Format ändert:
            # früh und eindeutig fehlschlagen
            raise ValueError(f"yfinance schema changed, missing columns for {symbol}: {sorted(missing)}")

        # Interne Standardreihenfolge und Basissäuberung je Symbol
        frame["symbol"] = symbol
        frame = frame[PRICE_COLUMNS]
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
        # Falls die Quelle doppelte Tageszeilen liefert, bleibt die letzte Zeile bestehen
        frame = frame.sort_values("date").drop_duplicates(subset=["date", "symbol"], keep="last")
        frames.append(frame)

    if not frames:
        # Rückgabevertrag: auch ohne Daten immer dieselben Spalten
        return pd.DataFrame(columns=PRICE_COLUMNS)

    # Alle Symbolrahmen zusammenführen und global sortieren
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).astype("datetime64[ns]")
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    return merged
