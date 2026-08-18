from __future__ import annotations

"""Zeitliche Aufteilung von Feature-Datensätzen & Persistenz der Split-Metadaten.

Modul stellt zwei Split-Strategien bereit:
1) Einmaliger, strikter Train/Val/Test-Split entlang der Zeitachse
2) Walk-Forward-Fenster mit expandierendem Train-Anteil

Zusätzlich können Metadaten (Grenzen, Zeilenzahlen, Datumsabdeckung)
für reproduzierbare Experimente als JSON geschrieben werden
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.transforms.validate import ValidationError


@dataclass(frozen=True)
class TemporalSplit:
    # Ergebniscontainer für einen festen, einmaligen Zeit-Split
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp | None


@dataclass(frozen=True)
class WalkForwardSplit:
    # Ergebniscontainer für ein einzelnes Walk-Forward-Fenster
    window_index: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _ts_to_iso(value: pd.Timestamp | None) -> str | None:
    """Konvertiert Timestamps robust in ISO-Strings für JSON-Ausgaben"""

    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _subset_metadata(frame: pd.DataFrame, date_col: str) -> dict[str, Any]:
    """Erzeugt kompakte Metadaten für eine Teilmenge (Rows, Datumsgrenzen, Symbole)"""

    if frame.empty:
        return {
            "rows": 0,
            "date_min": None,
            "date_max": None,
            "symbols": 0,
        }

    date_series = pd.to_datetime(frame[date_col], errors="coerce")
    return {
        "rows": int(len(frame)),
        "date_min": _ts_to_iso(date_series.min()),
        "date_max": _ts_to_iso(date_series.max()),
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else None,
    }


def split_features_strict_time(
    features: pd.DataFrame,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
    date_col: str = "date",
) -> TemporalSplit:
    """Teilt Features strikt zeitlich in Train/Validierung/Test ohne Überlappung"""

    # Ohne Datumsspalte kann kein zeitlicher Split berechnet werden
    if date_col not in features.columns:
        raise ValidationError(f"Features ohne Datumsspalte: {date_col}")

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end) if test_end is not None else None

    # Grundkonsistenz der Grenzen prüfen, um leere/überlappende Fenster zu vermeiden
    if not train_end_ts < val_end_ts:
        raise ValidationError("Ungültige Split-Grenzen: train_end muss vor val_end liegen")
    if test_end_ts is not None and not val_end_ts < test_end_ts:
        raise ValidationError("Ungültige Split-Grenzen: val_end muss vor test_end liegen")

    # Daten zeitlich normalisieren und in reproduzierbarer Reihenfolge sortieren
    data = features.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(["symbol", date_col]).reset_index(drop=True)

    # Teilmengen ohne Überlappung entlang der Zeitachse
    train = data[data[date_col] <= train_end_ts].copy()
    val = data[(data[date_col] > train_end_ts) & (data[date_col] <= val_end_ts)].copy()

    if test_end_ts is None:
        test = data[data[date_col] > val_end_ts].copy()
    else:
        test = data[(data[date_col] > val_end_ts) & (data[date_col] <= test_end_ts)].copy()

    return TemporalSplit(
        train=train,
        val=val,
        test=test,
        train_end=train_end_ts,
        val_end=val_end_ts,
        test_end=test_end_ts,
    )


def build_walk_forward_splits(
    features: pd.DataFrame,
    min_train_periods: int,
    val_periods: int,
    test_periods: int,
    step_periods: int = 1,
    date_col: str = "date",
) -> list[WalkForwardSplit]:
    """Erzeugt Walk-Forward-Splits mit expandierendem Train-Fenster auf Basis eindeutiger Handelstage"""

    # Voraussetzung: Zeitspalte vorhanden und Fenstergrößen sinnvoll (>0)
    if date_col not in features.columns:
        raise ValidationError(f"Features ohne Datumsspalte: {date_col}")

    if min_train_periods < 1 or val_periods < 1 or test_periods < 1 or step_periods < 1:
        raise ValidationError("Fensterparameter müssen jeweils mindestens 1 sein")

    # Zeitachse je Handelstag bestimmen -> darauf basieren Fenstergrenzen
    data = features.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(["symbol", date_col]).reset_index(drop=True)

    # müssen genug Handelstage für Train+Val+Test vorhanden sein
    unique_dates = pd.Index(sorted(data[date_col].dropna().unique()))
    min_total = min_train_periods + val_periods + test_periods
    if len(unique_dates) < min_total:
        raise ValidationError(
            "Zu wenige Zeitpunkte für Walk-Forward-Splits: "
            f"benötigt mindestens {min_total}, vorhanden {len(unique_dates)}"
        )

    windows: list[WalkForwardSplit] = []
    max_train_end_idx = len(unique_dates) - (val_periods + test_periods) - 1

    window_idx = 0
    # Train-Ende schrittweise nach vorne schieben -> expandierender Train-Split
    for train_end_idx in range(min_train_periods - 1, max_train_end_idx + 1, step_periods):
        val_start_idx = train_end_idx + 1
        val_end_idx = val_start_idx + val_periods - 1
        test_start_idx = val_end_idx + 1
        test_end_idx = test_start_idx + test_periods - 1

        train_end_ts = unique_dates[train_end_idx]
        val_start_ts = unique_dates[val_start_idx]
        val_end_ts = unique_dates[val_end_idx]
        test_start_ts = unique_dates[test_start_idx]
        test_end_ts = unique_dates[test_end_idx]

        # Je Fenster Teilmengen anhand Datumsgrenzen schneiden
        train = data[data[date_col] <= train_end_ts].copy()
        val = data[(data[date_col] >= val_start_ts) & (data[date_col] <= val_end_ts)].copy()
        test = data[(data[date_col] >= test_start_ts) & (data[date_col] <= test_end_ts)].copy()

        windows.append(
            WalkForwardSplit(
                window_index=window_idx,
                train=train,
                val=val,
                test=test,
                train_end=train_end_ts,
                val_start=val_start_ts,
                val_end=val_end_ts,
                test_start=test_start_ts,
                test_end=test_end_ts,
            )
        )
        window_idx += 1

    return windows


def temporal_split_metadata(
    split: TemporalSplit,
    date_col: str = "date",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Erzeugt serialisierbare Metadaten für einen zeitlichen Train/Val/Test-Split"""

    # Struktur enthält Grenzen und Datensatz-Stats -> Experimente nachvollziehbar bleiben
    payload: dict[str, Any] = {
        "split_type": "temporal",
        "boundaries": {
            "train_end": _ts_to_iso(split.train_end),
            "val_end": _ts_to_iso(split.val_end),
            "test_end": _ts_to_iso(split.test_end),
        },
        "datasets": {
            "train": _subset_metadata(split.train, date_col=date_col),
            "val": _subset_metadata(split.val, date_col=date_col),
            "test": _subset_metadata(split.test, date_col=date_col),
        },
    }
    if extra:
        payload["extra"] = extra
    return payload


def walk_forward_splits_metadata(
    splits: list[WalkForwardSplit],
    date_col: str = "date",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Erzeugt serialisierbare Metadaten für mehrere Walk-Forward-Fenster"""

    # Metadaten je Fenster sammeln, inkl. Grenzen und Teilmengen-Statistiken
    windows: list[dict[str, Any]] = []
    for split in splits:
        windows.append(
            {
                "window_index": int(split.window_index),
                "boundaries": {
                    "train_end": _ts_to_iso(split.train_end),
                    "val_start": _ts_to_iso(split.val_start),
                    "val_end": _ts_to_iso(split.val_end),
                    "test_start": _ts_to_iso(split.test_start),
                    "test_end": _ts_to_iso(split.test_end),
                },
                "datasets": {
                    "train": _subset_metadata(split.train, date_col=date_col),
                    "val": _subset_metadata(split.val, date_col=date_col),
                    "test": _subset_metadata(split.test, date_col=date_col),
                },
            }
        )

    payload: dict[str, Any] = {
        "split_type": "walk_forward",
        "windows": windows,
        "window_count": len(windows),
    }
    if extra:
        payload["extra"] = extra
    return payload


def write_temporal_split_metadata(
    split: TemporalSplit,
    output_path: str | Path,
    date_col: str = "date",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Schreibt Metadaten für einen zeitlichen Split als JSON-Datei"""

    # Zielordner bei Bedarf erzeugen und JSON in UTF-8 schreiben
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = temporal_split_metadata(split=split, date_col=date_col, extra=extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_walk_forward_splits_metadata(
    splits: list[WalkForwardSplit],
    output_path: str | Path,
    date_col: str = "date",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Schreibt Metadaten für Walk-Forward-Splits als JSON-Datei"""

    # Zielordner bei Bedarf erzeugen und JSON in UTF-8 schreiben
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = walk_forward_splits_metadata(splits=splits, date_col=date_col, extra=extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
