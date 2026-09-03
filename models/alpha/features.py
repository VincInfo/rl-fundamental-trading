from __future__ import annotations

import pandas as pd
from data_pipeline import DataSplit, DataVariant

from eval.data_splits.loader import load_evaluation_splits

from models.alpha.config import (
    ENGINEERED_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
    FUNDAMENTAL_LEVEL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TrainingConfig,
)


def load_splits(
    variant: DataVariant = DataVariant.WITH_FUNDAMENTALS,
) -> dict[DataSplit, pd.DataFrame]:
    """Load chronologically split market data from the shared pipeline."""
    return load_evaluation_splits(variant)


def daily_rebalance_mask(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First hourly bar per calendar day (rebalance time)."""
    normalized_dates = index.normalize()
    return index[~normalized_dates.duplicated(keep="first")]


def _wide_market_features(
    close: pd.DataFrame,
    config: TrainingConfig,
) -> pd.DataFrame:
    bars_per_day = config.bars_per_trading_day
    one_bar_return = close.pct_change()

    computations = {
        "return_1d": close / close.shift(bars_per_day) - 1,
        "return_5d": close / close.shift(5 * bars_per_day) - 1,
        "momentum_20d": close / close.shift(20 * bars_per_day) - 1,
        "vol_20d": one_bar_return.rolling(20 * bars_per_day).std(),
    }
    missing_engineered = set(ENGINEERED_FEATURE_COLUMNS) - set(computations)
    if missing_engineered:
        raise ValueError(
            f"Missing engineered feature definitions: {sorted(missing_engineered)}"
        )
    engineered = {
        name: computations[name] for name in ENGINEERED_FEATURE_COLUMNS
    }

    frames = []
    for feature_name, values in engineered.items():
        feature_frame = values.copy()
        feature_frame.columns = pd.MultiIndex.from_product(
            [[feature_name], feature_frame.columns],
            names=["Feature", "Ticker"],
        )
        frames.append(feature_frame)

    return pd.concat(frames, axis="columns", sort=False)


def _wide_forward_return(close: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    horizon_bars = config.horizon_trading_days * config.bars_per_trading_day
    target = close.shift(-horizon_bars) / close - 1
    target.columns = pd.MultiIndex.from_product(
        [[config.target_column], target.columns],
        names=["Feature", "Ticker"],
    )
    return target


def _last_step_change(values: pd.Series) -> pd.Series:
    """Propagate the last level jump until the next filing changes the series."""
    is_new_level = values.ne(values.shift())
    step = (values - values.shift()).where(is_new_level)
    return step.ffill().fillna(0.0)


def _wide_fundamental_changes(wide_frame: pd.DataFrame) -> pd.DataFrame:
    available = set(wide_frame.columns.get_level_values("Feature"))
    missing = [name for name in FUNDAMENTAL_LEVEL_COLUMNS if name not in available]
    if missing:
        raise ValueError(f"Missing fundamental level columns: {missing}")

    frames = []
    for column in FUNDAMENTAL_LEVEL_COLUMNS:
        changed = wide_frame[column].apply(_last_step_change, axis=0)
        changed.columns = pd.MultiIndex.from_product(
            [[f"delta_{column}"], changed.columns],
            names=["Feature", "Ticker"],
        )
        frames.append(changed)
    return pd.concat(frames, axis="columns", sort=False)


def _combined_wide_frame(
    wide_frame: pd.DataFrame,
    config: TrainingConfig,
    include_target: bool,
) -> pd.DataFrame:
    close = wide_frame["Close"]
    market_features = _wide_market_features(close, config)

    fundamental_mask = wide_frame.columns.get_level_values("Feature").isin(
        FUNDAMENTAL_FEATURE_COLUMNS
    )
    fundamentals = wide_frame.loc[:, fundamental_mask]
    frames = [market_features, fundamentals, _wide_fundamental_changes(wide_frame)]
    if include_target:
        frames.append(_wide_forward_return(close, config))

    combined_wide = pd.concat(frames, axis="columns", sort=False)
    if config.sample_daily:
        combined_wide = combined_wide.loc[daily_rebalance_mask(combined_wide.index)]
    return combined_wide


def _to_long_frame(combined_wide: pd.DataFrame) -> pd.DataFrame:
    long_frame = combined_wide.stack("Ticker", future_stack=True)
    long_frame.index.names = ["Datetime", "Ticker"]
    return long_frame


def build_inference_features(
    wide_frame: pd.DataFrame,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Long feature table for scoring, without the supervised target.

    Drops rows only when model inputs are missing, so the last horizon days
    remain available as PPO state inputs.
    """
    long_frame = _to_long_frame(_combined_wide_frame(wide_frame, config, include_target=False))
    feature_columns = list(MODEL_FEATURE_COLUMNS)
    return long_frame.dropna(subset=feature_columns)[feature_columns]


def build_dataset(
    wide_frame: pd.DataFrame,
    config: TrainingConfig,
) -> pd.DataFrame:
    """
    Transform wide panel data into a long ML table with features and target.

    One row per (rebalance timestamp, ticker) when sample_daily=True.
    """
    long_frame = _to_long_frame(_combined_wide_frame(wide_frame, config, include_target=True))
    feature_columns = list(MODEL_FEATURE_COLUMNS)
    required_columns = feature_columns + [config.target_column]
    long_frame = long_frame.dropna(subset=required_columns)
    return long_frame[required_columns]


def get_feature_names(
    dataset: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> list[str]:
    """Return model input columns, excluding the supervised target."""
    return [column for column in dataset.columns if column != target_column]
