from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from models.alpha_model.base import RandomAlphaModel
from models.data.panel import build_panel
from models.data.synthetic import make_synthetic_features
from models.env.trading_env import MultiStockTradingEnv


def _make_env(n_stocks: int = 4, n_days: int = 200, seed: int = 7) -> MultiStockTradingEnv:
    features = make_synthetic_features(n_stocks=n_stocks, n_days=n_days, seed=seed)
    alpha = RandomAlphaModel(seed=seed).predict(features)
    panel = build_panel(features, alpha, vol_window=20)
    return MultiStockTradingEnv(panel, min_holding_days=3)


def test_env_passes_gymnasium_checker() -> None:
    check_env(_make_env())


def test_observation_and_action_shapes() -> None:
    env = _make_env(n_stocks=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4 * env.n_stocks + 1,)
    assert env.action_space.nvec.tolist() == [3] * env.n_stocks


def test_episode_rewards_are_finite() -> None:
    env = _make_env(n_stocks=4, n_days=150)
    env.reset(seed=0)
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        steps += 1
    assert steps > 0


def test_min_holding_days_blocks_early_sell() -> None:
    env = _make_env(n_stocks=3, n_days=60)
    env.reset(seed=0)

    buy_all = np.full(env.n_stocks, 2, dtype=np.int64)
    env.step(buy_all)  # Positionen eröffnen

    sell_all = np.zeros(env.n_stocks, dtype=np.int64)
    _, _, _, _, info = env.step(sell_all)  # unmittelbarer Verkauf muss blockiert werden
    assert info["n_blocked"] > 0
