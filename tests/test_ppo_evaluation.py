import math

import numpy as np
import pandas as pd

from models.ppo.evaluation import (
    action_shares,
    episode_length,
    equal_weight_mean_log_return,
)
from models.ppo.panel import build_panel
from models.ppo.synthetic import make_synthetic_features
from models.ppo.trading_env import BUY, HOLD, SELL


def _random_alpha_wide(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    dates = np.sort(features["date"].unique())
    symbols = sorted(str(symbol) for symbol in features["symbol"].unique())
    scores = np.random.default_rng(seed).normal(0.0, 0.02, size=(len(dates), len(symbols)))
    return pd.DataFrame(scores, index=pd.Index(dates, name="date"), columns=symbols)


def test_episode_length_matches_env_index_logic():
    assert episode_length(284) == 282
    assert episode_length(2) == 0
    assert episode_length(3) == 1


def test_action_shares_counts_buy_hold_sell():
    actions = np.array(
        [
            [SELL, HOLD, BUY],
            [BUY, BUY, HOLD],
        ]
    )
    shares = action_shares(actions)
    assert shares["sell"] == 1 / 6
    assert shares["hold"] == 2 / 6
    assert shares["buy"] == 3 / 6


def test_equal_weight_mean_log_return_is_finite():
    features = make_synthetic_features(n_stocks=4, n_days=80, seed=3)
    alpha = _random_alpha_wide(features, seed=3)
    panel = build_panel(features, alpha, vol_window=10)
    value = equal_weight_mean_log_return(panel)
    assert math.isfinite(value)
