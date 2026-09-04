import math

import numpy as np
import pandas as pd

from eval.ppo import (
    action_shares,
    alpha_quantile_actions,
    episode_length,
    equal_weight_mean_log_return,
    make_alpha_quantile_policy,
    rollout_fixed_policy,
)
from models.ppo.panel import build_panel
from models.ppo.synthetic import make_synthetic_features
from models.ppo.trading_env import BUY, HOLD, SELL, MultiStockTradingEnv


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


def test_alpha_quantile_actions_buys_top_and_sells_bottom():
    alpha = np.array([-0.3, 0.1, 0.4, -0.1, 0.0])
    actions = alpha_quantile_actions(alpha, buy_fraction=0.4, sell_fraction=0.4)
    assert actions[2] == BUY
    assert actions[0] == SELL
    assert HOLD in set(actions.tolist())


def test_rule_policy_rollout_is_finite():
    features = make_synthetic_features(n_stocks=5, n_days=60, seed=4)
    alpha = _random_alpha_wide(features, seed=4)
    panel = build_panel(features, alpha, vol_window=10)
    env = MultiStockTradingEnv(panel, min_holding_days=2)
    metrics = rollout_fixed_policy(env, make_alpha_quantile_policy())
    assert metrics["n_steps"] > 0
    assert math.isfinite(metrics["mean_reward"])
