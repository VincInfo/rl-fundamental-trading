from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from src.env.trading_env import FundamentalTradingEnv, load_features_daily

def run_smoke_test() -> None:
    """Laeuft eine kurze Zufalls-Policy als Plausibilitaetstest."""

    project_root = Path(__file__).resolve().parents[2]
    features_path = project_root / "mock_data" / "features_daily.csv"
    frame = load_features_daily(str(features_path))

    env = FundamentalTradingEnv(frame, initial_cash=10_000.0, transaction_cost_bps=10.0)
    observation, info = env.reset()

    print(f"Startsymbol: {info['symbol']} | Beobachtungsdimension: {observation.shape[0]}")

    action_counter: Counter[int] = Counter()
    rewards: list[float] = []
    episode_returns: list[float] = []

    done = False
    while not done:
        action = int(env.action_space.sample())
        action_counter[action] += 1
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        done = terminated or truncated
        if done:
            episode_returns.append(info["portfolio_value"] / env.initial_cash - 1.0)

    final_portfolio = info["portfolio_value"]
    print(f"Finales Portfolio: {final_portfolio:.2f}")
    print(f"Gesamtrendite: {(final_portfolio / env.initial_cash - 1.0):.4%}")
    print(f"Mittlerer Reward: {np.mean(rewards):.6f}")
    print(f"Aktionen: {dict(action_counter)}")
    print(f"Episode Returns: {episode_returns}")


if __name__ == "__main__":
    run_smoke_test()