from __future__ import annotations

"""Vorverarbeitung für RL-nahe Feature-Nutzung (insbesondere Skalierung).

Wichtige Regel: Der Scaler wird ausschließlich auf dem Train-Split gefittet.
Danach werden dieselben Parameter unverändert auf Validierung und Test angewendet,
um Leakage über Statistikparameter zu vermeiden.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from data_pipeline.transforms.split import TemporalSplit
from data_pipeline.transforms.validate import ValidationError


@dataclass(frozen=True)
class TrainFittedScaler:
    # Kompakte Repräsentation eines auf Train gefitteten Standard-Scalers
    feature_columns: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    fitted_rows: int


def infer_numeric_feature_columns(
    frame: pd.DataFrame,
    exclude_columns: set[str] | None = None,
) -> list[str]:
    """Leitet numerische Feature-Spalten ab und schließt Zeit-/Identifikationsspalten aus."""

    # Nicht skalieren: Zeit- und Identifikationsspalten sowie kategorische Report-Typen
    excluded = {"date", "symbol", "period_end", "report_date", "report_type"}
    if exclude_columns:
        excluded = excluded | set(exclude_columns)

    numeric_cols = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [col for col in numeric_cols if col not in excluded]


def fit_standard_scaler_from_train(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> TrainFittedScaler:
    """Fittet Mittelwert und Standardabweichung ausschließlich auf dem Train-Split."""

    # Ohne Train-Daten Parameterschätzung bedeutungslos
    if train.empty:
        raise ValidationError("Train-Split ist leer; Skalierung kann nicht gefittet werden")

    # jede angeforderte Feature-Spalte muss im Train-Frame existieren
    missing = [col for col in feature_columns if col not in train.columns]
    if missing:
        raise ValidationError(f"Nicht gefundene Feature-Spalten im Train-Split: {missing}")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for col in feature_columns:
        # Robust numerisch casten ->nicht konvertierbare Werte werden NaN
        values = pd.to_numeric(train[col], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0))

        # Fallbacks: rein fehlende Spalten -> mean 0; konstante Spalten -> std 1
        if pd.isna(mean):
            mean = 0.0
        if pd.isna(std) or std == 0.0:
            std = 1.0

        means[col] = mean
        stds[col] = std

    return TrainFittedScaler(
        feature_columns=list(feature_columns),
        means=means,
        stds=stds,
        fitted_rows=len(train),
    )


def apply_standard_scaler(
    frame: pd.DataFrame,
    scaler: TrainFittedScaler,
) -> pd.DataFrame:
    """Wendet einen auf Train gefitteten Standard-Scaler auf einen Datensatz an."""

    out = frame.copy()
    # Ziel-Frame muss dieselben Feature-Spalten enthalten wie Fit-Frame
    missing = [col for col in scaler.feature_columns if col not in out.columns]
    if missing:
        raise ValidationError(f"Nicht gefundene Feature-Spalten im Ziel-Datensatz: {missing}")

    for col in scaler.feature_columns:
        # Transform: (x - mean_train) / std_train
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = (values - scaler.means[col]) / scaler.stds[col]

    return out


def scale_temporal_split_with_train_fit(
    split: TemporalSplit,
    feature_columns: list[str] | None = None,
    exclude_columns: set[str] | None = None,
) -> tuple[TemporalSplit, TrainFittedScaler]:
    """Fittet Skalierung auf Train und wendet sie konsistent auf Train/Val/Test an."""

    # ohne explizite Spaltenliste automatisch geeignete numerische Features wählen
    cols = feature_columns
    if cols is None:
        cols = infer_numeric_feature_columns(split.train, exclude_columns=exclude_columns)
    if not cols:
        raise ValidationError("Keine skalierbaren numerischen Feature-Spalten gefunden")

    # Fit nur auf Train -> zentrale Anti-Leakage-Regel
    scaler = fit_standard_scaler_from_train(split.train, feature_columns=cols)

    # derselbe Scaler wird auf alle drei Teilmengen angewendet
    scaled = TemporalSplit(
        train=apply_standard_scaler(split.train, scaler),
        val=apply_standard_scaler(split.val, scaler),
        test=apply_standard_scaler(split.test, scaler),
        train_end=split.train_end,
        val_end=split.val_end,
        test_end=split.test_end,
    )
    return scaled, scaler


def scaler_metadata(scaler: TrainFittedScaler) -> dict[str, Any]:
    """Erzeugt serialisierbare Metadaten für einen auf Train gefitteten Scaler."""

    # Metadaten helfen bei Reproduzierbarkeit und Debugging von Experimenten
    return {
        "feature_columns": scaler.feature_columns,
        "means": scaler.means,
        "stds": scaler.stds,
        "fitted_rows": scaler.fitted_rows,
    }
