from __future__ import annotations

import pandas as pd
import numpy as np
from data_pipeline import DataSplit, DataVariant

from eval.data_splits.loader import load_evaluation_splits

from models.alpha.config import (
    CONTEXT_FEATURE_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    EVENT_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
    FUNDAMENTAL_FLOW_COLUMNS,
    FUNDAMENTAL_LEVEL_COLUMNS,
    TARGET_COLUMN,
    TrainingConfig,
    VANILLA_PRICE_COLUMNS,
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


def _require_vanilla_ohlcv(wide_frame: pd.DataFrame) -> None:
    available = set(wide_frame.columns.get_level_values("Feature"))
    missing = [name for name in VANILLA_PRICE_COLUMNS if name not in available]
    if missing:
        raise ValueError(f"Missing VANILLA OHLCV columns: {missing}")


def _as_feature_frame(name: str, values: pd.DataFrame) -> pd.DataFrame:
    feature_frame = values.copy()
    feature_frame.columns = pd.MultiIndex.from_product(
        [[name], feature_frame.columns],
        names=["Feature", "Ticker"],
    )
    return feature_frame


def _wide_market_features(
    wide_frame: pd.DataFrame,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Build cross-sectionally usable features from all VANILLA OHLCV fields."""
    _require_vanilla_ohlcv(wide_frame)

    bars_per_day = config.bars_per_trading_day
    close = wide_frame["Close"]
    open_ = wide_frame["Open"]
    high = wide_frame["High"]
    low = wide_frame["Low"]
    volume = wide_frame["Volume"]

    one_bar_return = close.pct_change()
    avg_volume_20d = volume.rolling(20 * bars_per_day, min_periods=1).mean()

    computations = {
        "return_1d": close / close.shift(bars_per_day) - 1,
        "return_5d": close / close.shift(5 * bars_per_day) - 1,
        "momentum_20d": close / close.shift(20 * bars_per_day) - 1,
        "vol_20d": one_bar_return.rolling(20 * bars_per_day).std(),
        "oc_return": close / open_ - 1,
        "hl_range": (high - low) / close,
        "volume_change_1d": volume / volume.shift(bars_per_day) - 1,
        "rel_volume_20d": volume / avg_volume_20d - 1,
    }
    computations = {
        name: values.replace([np.inf, -np.inf], np.nan)
        for name, values in computations.items()
    }
    missing_engineered = set(ENGINEERED_FEATURE_COLUMNS) - set(computations)
    if missing_engineered:
        raise ValueError(
            f"Missing engineered feature definitions: {sorted(missing_engineered)}"
        )

    frames = [
        _as_feature_frame(name, computations[name])
        for name in ENGINEERED_FEATURE_COLUMNS
    ]
    return pd.concat(frames, axis="columns", sort=False)


def _wide_forward_return(close: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    horizon_bars = config.horizon_trading_days * config.bars_per_trading_day
    target = close.shift(-horizon_bars) / close - 1
    if config.active_return:
        # Market-neutral label: stock return minus equal-weight cross-section.
        target = target.sub(target.mean(axis=1), axis=0)
    return _as_feature_frame(config.target_column, target)


def _last_step_change(values: pd.Series) -> pd.Series:
    """Propagate the last level jump until the next filing changes the series."""
    is_new_level = values.ne(values.shift())
    step = (values - values.shift()).where(is_new_level)
    return step.ffill().fillna(0.0)


def _wide_fundamental_changes(
    wide_frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    available = set(wide_frame.columns.get_level_values("Feature"))
    missing = [name for name in columns if name not in available]
    if missing:
        raise ValueError(f"Missing fundamental columns: {missing}")

    frames = []
    for column in columns:
        changed = wide_frame[column].apply(_last_step_change, axis=0)
        frames.append(_as_feature_frame(f"delta_{column}", changed))
    return pd.concat(frames, axis="columns", sort=False)


def _wide_cross_sectional_ranks(values: pd.DataFrame, name: str) -> pd.DataFrame:
    ranked = values.rank(axis=1, pct=True)
    return _as_feature_frame(name, ranked)


def _wide_event_features(wide_frame: pd.DataFrame) -> pd.DataFrame:
    lag = wide_frame["filing_lag_days"]
    computations = {
        "filing_recency": 1.0 / (1.0 + lag),
        "post_filing_5d": (lag <= 5).astype(float),
    }
    missing = set(EVENT_FEATURE_COLUMNS) - set(computations)
    if missing:
        raise ValueError(f"Missing event feature definitions: {sorted(missing)}")
    return pd.concat(
        [_as_feature_frame(name, computations[name]) for name in EVENT_FEATURE_COLUMNS],
        axis="columns",
        sort=False,
    )


def _wide_fundamental_model_features(wide_frame: pd.DataFrame) -> pd.DataFrame:
    """Levels, filing context, event flags, deltas, and cross-sectional ranks."""
    available = set(wide_frame.columns.get_level_values("Feature"))
    required = FUNDAMENTAL_FEATURE_COLUMNS
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"Missing fundamental feature columns: {missing}")

    level_mask = wide_frame.columns.get_level_values("Feature").isin(
        FUNDAMENTAL_LEVEL_COLUMNS + CONTEXT_FEATURE_COLUMNS
    )
    levels = wide_frame.loc[:, level_mask]
    ratio_deltas = _wide_fundamental_changes(wide_frame, FUNDAMENTAL_LEVEL_COLUMNS)
    flow_deltas = _wide_fundamental_changes(wide_frame, FUNDAMENTAL_FLOW_COLUMNS)

    rank_frames = [
        _wide_cross_sectional_ranks(wide_frame[column], f"rank_{column}")
        for column in FUNDAMENTAL_LEVEL_COLUMNS
    ]
    for column in FUNDAMENTAL_FLOW_COLUMNS:
        delta_name = f"delta_{column}"
        rank_frames.append(
            _wide_cross_sectional_ranks(flow_deltas[delta_name], f"rank_{delta_name}")
        )

    return pd.concat(
        [
            levels,
            _wide_event_features(wide_frame),
            ratio_deltas,
            flow_deltas,
            *rank_frames,
        ],
        axis="columns",
        sort=False,
    )


def _combined_wide_frame(
    wide_frame: pd.DataFrame,
    config: TrainingConfig,
    include_target: bool,
) -> pd.DataFrame:
    close = wide_frame["Close"]
    market_features = _wide_market_features(wide_frame, config)
    frames = [market_features]

    if config.include_fundamentals:
        frames.append(_wide_fundamental_model_features(wide_frame))

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
    feature_columns = list(config.feature_columns)
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
    feature_columns = list(config.feature_columns)
    required_columns = feature_columns + [config.target_column]
    long_frame = long_frame.dropna(subset=required_columns)
    return long_frame[required_columns]


def get_feature_names(
    dataset: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> list[str]:
    """Return model input columns, excluding the supervised target."""
    return [column for column in dataset.columns if column != target_column]
