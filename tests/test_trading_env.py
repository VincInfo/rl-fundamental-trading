from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from models.rl.panel import build_panel
from models.rl.synthetic import make_synthetic_features
from models.rl.envs import BUY, HOLD, SELL, MultiStockTradingEnv


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
    assert obs.shape == (6 * env.n_stocks + 1,)
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


def test_log_return_is_unshaped_and_reported() -> None:
    env = _make_env(n_stocks=3, n_days=60)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.full(env.n_stocks, 2, dtype=np.int64))
    assert "log_return" in info
    assert np.isfinite(info["log_return"])
    assert np.isfinite(reward)


def test_random_episode_window_has_fixed_length_and_varies_start() -> None:
    env = _make_env(n_stocks=3, n_days=120, seed=8)
    env.episode_window = 30
    env.randomize_start = True
    starts: list[int] = []
    for seed in range(6):
        env.reset(seed=seed)
        starts.append(env.current_index)
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(
                np.full(env.n_stocks, HOLD, dtype=np.int64)
            )
            steps += 1
        assert steps == 30
    assert len(set(starts)) > 1


def test_rebalance_every_holds_between_rebalance_days() -> None:
    env = _make_env(n_stocks=3, n_days=80, seed=9)
    env.rebalance_every = 5
    env.rebalance_mode = "snapshot"
    env.min_holding_days = 0
    env.reset(seed=0)
    buy = np.full(env.n_stocks, BUY, dtype=np.int64)
    _, _, _, _, info0 = env.step(buy)
    assert info0["turnover"] > 0.0
    for _ in range(4):
        _, _, _, _, info = env.step(buy)
        assert info["turnover"] == pytest.approx(0.0, abs=1e-12)


def test_snapshot_sell_can_open_short_when_allowed() -> None:
    env = _make_env(n_stocks=4, n_days=60, seed=10)
    env.allow_short = True
    env.max_gross_exposure = 2.0
    env.rebalance_mode = "snapshot"
    env.min_holding_days = 0
    env.reset(seed=0)
    sell = np.full(env.n_stocks, SELL, dtype=np.int64)
    env.step(sell)
    assert np.any(env.holdings < 0.0)


def test_long_only_snapshot_does_not_go_short() -> None:
    env = _make_env(n_stocks=4, n_days=60, seed=11)
    env.allow_short = False
    env.rebalance_mode = "snapshot"
    env.min_holding_days = 0
    env.reset(seed=0)
    sell = np.full(env.n_stocks, SELL, dtype=np.int64)
    env.step(sell)
    assert np.all(env.holdings >= -1e-9)


def test_incremental_sell_can_open_short_when_allowed() -> None:
    env = _make_env(n_stocks=4, n_days=60, seed=12)
    env.allow_short = True
    env.max_gross_exposure = 2.0
    env.rebalance_mode = "incremental"
    env.min_holding_days = 0
    env.reset(seed=0)
    sell = np.full(env.n_stocks, SELL, dtype=np.int64)
    env.step(sell)
    assert np.any(env.holdings < 0.0)


def test_snapshot_mixed_book_has_longs_and_shorts() -> None:
    env = _make_env(n_stocks=4, n_days=60, seed=13)
    env.allow_short = True
    env.max_gross_exposure = 2.0
    env.rebalance_mode = "snapshot"
    env.min_holding_days = 0
    env.reset(seed=0)
    action = np.array([BUY, BUY, SELL, SELL], dtype=np.int64)
    env.step(action)
    assert np.any(env.holdings > 0.0)
    assert np.any(env.holdings < 0.0)
