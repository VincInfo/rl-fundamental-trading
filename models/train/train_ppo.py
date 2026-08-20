from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from models.alpha_model.base import AlphaModel, RandomAlphaModel
from models.data.panel import build_panel
from models.data.synthetic import make_synthetic_features
from models.env.trading_env import MultiStockTradingEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:  # pragma: no cover - klare Meldung falls RL-Extra fehlt
    raise RuntimeError(
        "stable-baselines3 fehlt. Installiere es mit: uv add stable-baselines3"
    ) from exc


def split_by_time(features: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splittet den Datensatz entlang der Zeitachse, um Data Leakage zu vermeiden."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio muss zwischen 0 und 1 liegen.")

    unique_dates = np.sort(features["date"].unique())
    cutoff_index = min(max(1, int(len(unique_dates) * train_ratio)), len(unique_dates) - 1)
    cutoff_date = unique_dates[cutoff_index - 1]

    train = features[features["date"] <= cutoff_date].reset_index(drop=True)
    test = features[features["date"] > cutoff_date].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Zeitlicher Split hat eine leere Train- oder Test-Menge erzeugt.")
    return train, test


def build_env(
    features: pd.DataFrame,
    alpha_model: AlphaModel,
    vol_window: int,
    min_holding_days: int,
) -> MultiStockTradingEnv:
    """Kombiniert Alpha-Scores und Kursdaten zu einer Trading-Umgebung."""

    alpha = alpha_model.predict(features)
    panel = build_panel(features, alpha, vol_window=vol_window)
    return MultiStockTradingEnv(panel, min_holding_days=min_holding_days)


def rollout_mean_reward(model: PPO, env: VecNormalize) -> float:
    """Führt eine deterministische Episode aus und gibt den mittleren Reward zurück."""

    obs = env.reset()
    rewards: list[float] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, _ = env.step(action)
        rewards.append(float(reward[0]))
        done = bool(dones[0])
    return float(np.mean(rewards)) if rewards else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO-Training auf der Multi-Stock-Trading-Umgebung.")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n-stocks", type=int, default=5)
    parser.add_argument("--n-days", type=int, default=750)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--min-holding-days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = make_synthetic_features(n_stocks=args.n_stocks, n_days=args.n_days, seed=args.seed)
    train_features, _ = split_by_time(features)

    alpha_model = RandomAlphaModel(seed=args.seed)

    # Gymnasium-Kompatibilität einmalig auf einer rohen Umgebung prüfen.
    check_env(build_env(train_features, alpha_model, args.vol_window, args.min_holding_days))

    def _factory() -> MultiStockTradingEnv:
        return build_env(train_features, alpha_model, args.vol_window, args.min_holding_days)

    env = VecNormalize(DummyVecEnv([_factory]), norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    env.training = False
    env.norm_reward = False
    print(f"Mittlerer Reward (deterministische Episode): {rollout_mean_reward(model, env):.6f}")


if __name__ == "__main__":
    main()
