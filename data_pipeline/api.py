from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.sources.fundamentals_edgar import load_fundamentals
from data_pipeline.sources.prices_yfinance import load_prices
from data_pipeline.transforms.merge import build_features_daily
from data_pipeline.transforms.preprocess import (
    TrainFittedScaler,
    scaler_metadata,
    scale_temporal_split_with_train_fit,
)
from data_pipeline.transforms.split import (
    TemporalSplit,
    WalkForwardSplit,
    build_walk_forward_splits,
    split_features_strict_time,
    write_temporal_split_metadata,
    write_walk_forward_splits_metadata,
)
from data_pipeline.transforms.validate import validate_features, validate_fundamentals, validate_prices


@dataclass
class PipelineResult:
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    features: pd.DataFrame


def build_dataset(symbols: list[str], start: str, end: str, sec_user_agent: str) -> PipelineResult:
    """Erzeugt Kursdaten, Fundamentaldaten und zusammengeführte Tages-Features für ein MVP-Universum"""

    prices = load_prices(symbols=symbols, start=start, end=end)
    fundamentals = load_fundamentals(symbols=symbols, start=start, end=end, user_agent=sec_user_agent)

    validate_prices(prices)
    validate_fundamentals(fundamentals)

    features = build_features_daily(prices=prices, fundamentals=fundamentals)
    validate_features(features)

    return PipelineResult(prices=prices, fundamentals=fundamentals, features=features)


def split_features_by_time(
    features: pd.DataFrame,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
) -> TemporalSplit:
    """Teilt einen Feature-Datensatz strikt zeitlich in Train, Validierung und Test"""

    return split_features_strict_time(
        features=features,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        date_col="date",
    )


def build_walk_forward_feature_splits(
    features: pd.DataFrame,
    min_train_periods: int,
    val_periods: int,
    test_periods: int,
    step_periods: int = 1,
) -> list[WalkForwardSplit]:
    """Erzeugt Walk-Forward-Splits mit expandierendem Train-Fenster für RL-Experimente"""

    return build_walk_forward_splits(
        features=features,
        min_train_periods=min_train_periods,
        val_periods=val_periods,
        test_periods=test_periods,
        step_periods=step_periods,
        date_col="date",
    )


def scale_split_features_train_only(
    split: TemporalSplit,
    feature_columns: list[str] | None = None,
    exclude_columns: set[str] | None = None,
) -> tuple[TemporalSplit, TrainFittedScaler]:
    """Skaliert Features mit Fit auf Train und Anwendung auf Validierung/Test."""

    return scale_temporal_split_with_train_fit(
        split=split,
        feature_columns=feature_columns,
        exclude_columns=exclude_columns,
    )


def save_temporal_split_metadata(
    split: TemporalSplit,
    output_path: str | Path,
    scaler: TrainFittedScaler | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Speichert Metadaten für einen zeitlichen Split als JSON für reproduzierbare Runs."""

    payload_extra: dict[str, Any] = {}
    if extra:
        payload_extra.update(extra)
    if scaler is not None:
        payload_extra["scaler"] = scaler_metadata(scaler)

    return write_temporal_split_metadata(
        split=split,
        output_path=output_path,
        date_col="date",
        extra=payload_extra if payload_extra else None,
    )


def save_walk_forward_split_metadata(
    splits: list[WalkForwardSplit],
    output_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Speichert Metadaten für Walk-Forward-Splits als JSON für reproduzierbare Runs."""

    return write_walk_forward_splits_metadata(
        splits=splits,
        output_path=output_path,
        date_col="date",
        extra=extra,
    )
