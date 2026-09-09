from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from models.rl.panel import build_panel
from models.rl.synthetic import make_synthetic_features
from models.rl.envs import MultiStockTradingEnvContinuous


def _random_alpha_wide(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    dates = np.sort(features["date"].unique())
    symbols = sorted(str(symbol) for symbol in features["symbol"].unique())
    scores = np.random.default_rng(seed).normal(0.0, 0.02, size=(len(dates), len(symbols)))
    return pd.DataFrame(scores, index=pd.Index(dates, name="date"), columns=symbols)


def _make_env(n_stocks: int = 4, n_days: int = 200, seed: int = 7) -> MultiStockTradingEnvContinuous:
    features = make_synthetic_features(n_stocks=n_stocks, n_days=n_days, seed=seed)
    alpha_wide = _random_alpha_wide(features, seed=seed)
    panel = build_panel(features, alpha_wide, vol_window=20)
    return MultiStockTradingEnvContinuous(panel, min_holding_days=3)


def test_env_passes_gymnasium_checker() -> None:
    check_env(_make_env())


def test_snapshot_rebalance_mode_is_rejected() -> None:
    features = make_synthetic_features(n_stocks=4, n_days=200, seed=7)
    alpha_wide = _random_alpha_wide(features, seed=7)
    panel = build_panel(features, alpha_wide, vol_window=20)
    with pytest.raises(ValueError, match="snapshot"):
        MultiStockTradingEnvContinuous(panel, rebalance_mode="snapshot")


def test_observation_and_action_shapes() -> None:
    env = _make_env(n_stocks=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6 * env.n_stocks + 1,)
    assert env.action_space.shape == (env.n_stocks,)
    assert float(env.action_space.low.min()) == -1.0
    assert float(env.action_space.high.max()) == 1.0


def test_episode_rewards_are_finite() -> None:
    env = _make_env(n_stocks=4, n_days=150)
    env.reset(seed=0)
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(reward)
        steps += 1
    assert steps > 0


def test_min_holding_days_blocks_early_sell() -> None:
    env = _make_env(n_stocks=3, n_days=60)
    env.reset(seed=0)

    buy_all = np.ones(env.n_stocks, dtype=np.float32)
    env.step(buy_all)

    sell_all = -np.ones(env.n_stocks, dtype=np.float32)
    _, _, _, _, info = env.step(sell_all)
    assert info["n_blocked"] > 0


def test_out_of_bounds_action_is_clipped() -> None:
    env = _make_env(n_stocks=3, n_days=60)
    env.reset(seed=0)

    huge_action = np.full(env.n_stocks, 5.0, dtype=np.float32)
    _, reward, _, _, _ = env.step(huge_action)
    assert np.isfinite(reward)

    weights = env.holdings / max(env._portfolio_value(), env.eps)
    assert weights.max() <= env.w_max + 1e-9
    assert weights.sum() <= 1.0 + 1e-9
