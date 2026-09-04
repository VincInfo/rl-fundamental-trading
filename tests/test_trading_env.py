from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env

from models.ppo.panel import build_panel
from models.ppo.synthetic import make_synthetic_features
from models.ppo.trading_env import MultiStockTradingEnv


def _random_alpha_wide(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Environment-Tests laufen bewusst ohne XGBoost; getestet wird der Datenkontrakt.
    dates = np.sort(features["date"].unique())
    symbols = sorted(str(symbol) for symbol in features["symbol"].unique())
    scores = np.random.default_rng(seed).normal(0.0, 0.02, size=(len(dates), len(symbols)))
    return pd.DataFrame(scores, index=pd.Index(dates, name="date"), columns=symbols)


def _make_env(n_stocks: int = 4, n_days: int = 200, seed: int = 7) -> MultiStockTradingEnv:
    features = make_synthetic_features(n_stocks=n_stocks, n_days=n_days, seed=seed)
    alpha_wide = _random_alpha_wide(features, seed=seed)
    panel = build_panel(features, alpha_wide, vol_window=20)
    return MultiStockTradingEnv(panel, min_holding_days=3)


def test_env_passes_gymnasium_checker() -> None:
    check_env(_make_env())


def test_observation_and_action_shapes() -> None:
    env = _make_env(n_stocks=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5 * env.n_stocks + 1,)
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


def test_transaction_cost_info_contains_absolute_cost() -> None:
    env = _make_env(n_stocks=3, n_days=60)
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.full(env.n_stocks, 2, dtype=np.int64))

    assert info["transaction_cost"] == env.initial_cash * info["cost_rate"]
