from __future__ import annotations

import pandas as pd
from data_pipeline import DataSplit

from models.alpha.features import daily_rebalance_mask, load_splits

MARKET_FEATURE_COLUMNS = ("date", "symbol", "close", "return_1d")


def build_market_features(wide_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pipeline Close data into daily long-format prices and returns.

    Samples the first hourly bar per calendar day so PPO rebalances at the
    same time as the alpha model. ``return_1d`` is the return between two
    consecutive rebalance timestamps.
    """
    feature_names = set(wide_frame.columns.get_level_values("Feature"))
    if "Close" not in feature_names:
        raise ValueError("Wide frame must contain a 'Close' feature.")

    close = wide_frame["Close"].sort_index()
    close = close.loc[daily_rebalance_mask(close.index)].dropna(how="any")
    daily_return = close.pct_change()
    complete_days = daily_return.notna().all(axis=1)
    close = close.loc[complete_days]
    daily_return = daily_return.loc[complete_days]

    close_long = close.stack(future_stack=True).rename("close")
    return_long = daily_return.stack(future_stack=True).rename("return_1d")
    long_frame = pd.concat([close_long, return_long], axis="columns")
    long_frame.index.names = ["date", "symbol"]
    return long_frame.reset_index()[list(MARKET_FEATURE_COLUMNS)]


def load_train_features() -> pd.DataFrame:
    """Load the pipeline train split as PPO market features."""
    return build_market_features(load_splits()[DataSplit.TRAIN])
