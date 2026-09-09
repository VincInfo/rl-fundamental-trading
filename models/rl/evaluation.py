from __future__ import annotations

import math

import numpy as np

from models.rl.panel import MarketPanel
from models.rl.envs import BUY, HOLD, SELL

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 fehlt. Installiere es mit: uv add stable-baselines3"
    ) from exc

ACTION_NAMES = {SELL: "sell", HOLD: "hold", BUY: "buy"}
CONTINUOUS_ACTION_THRESHOLD = 0.1


def episode_length(n_days: int) -> int:
    """Number of env steps in one episode (see MultiStockTradingEnv index logic)."""
    return max(int(n_days) - 2, 0)


def equal_weight_mean_log_return(panel: MarketPanel, eps: float = 1e-8) -> float:
    """Mean daily log return of a buy-and-hold equal-weight portfolio, no costs."""
    if panel.n_days <= 1:
        return 0.0
    portfolio_return = panel.returns[1:].mean(axis=1)
    log_return = np.log(np.maximum(1.0 + portfolio_return, eps))
    return float(log_return.mean())


def action_shares(actions: np.ndarray) -> dict[str, float]:
    """Fraction of Sell/Hold/Buy over a ``(steps, n_stocks)`` action array."""
    if actions.size == 0:
        return {name: 0.0 for name in ACTION_NAMES.values()}
    return {
        name: float((actions == code).mean())
        for code, name in ACTION_NAMES.items()
    }


def mean_policy_entropy(model: PPO, obs: np.ndarray) -> float:
    """Entropy of the current policy distribution (sum over stocks)."""
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(obs_tensor)
    return float(distribution.entropy().mean().cpu().item())


def rollout_diagnostics(model: PPO, env: VecNormalize) -> dict[str, float]:
    """Deterministic episode: reward, action mix, and policy entropy."""
    obs = env.reset()
    rewards: list[float] = []
    collected_actions: list[np.ndarray] = []
    entropies: list[float] = []
    done = False
    while not done:
        entropies.append(mean_policy_entropy(model, obs))
        action, _ = model.predict(obs, deterministic=True)
        collected_actions.append(np.asarray(action).reshape(-1))
        obs, reward, dones, _ = env.step(action)
        rewards.append(float(reward[0]))
        done = bool(dones[0])

    stacked = (
        np.stack(collected_actions, axis=0)
        if collected_actions
        else np.zeros((0, 1), dtype=np.int64)
    )
    shares = action_shares(stacked)
    n_stocks = int(stacked.shape[1]) if stacked.size else 0
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "n_steps": len(rewards),
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "max_entropy": float(n_stocks * math.log(3.0)) if n_stocks else 0.0,
        "action_share_sell": shares["sell"],
        "action_share_hold": shares["hold"],
        "action_share_buy": shares["buy"],
    }


def continuous_action_shares(
    actions: np.ndarray,
    threshold: float = CONTINUOUS_ACTION_THRESHOLD,
) -> dict[str, float]:
    """Interpret continuous actions in [-1, 1] as buy/hold/sell buckets."""
    if actions.size == 0:
        return {"sell": 0.0, "hold": 0.0, "buy": 0.0}
    return {
        "sell": float((actions < -threshold).mean()),
        "hold": float((np.abs(actions) <= threshold).mean()),
        "buy": float((actions > threshold).mean()),
    }


def continuous_rollout_diagnostics(model: SAC, env: VecNormalize) -> dict[str, float]:
    """Deterministic episode diagnostics for a continuous-action agent (SAC)."""
    obs = env.reset()
    rewards: list[float] = []
    collected_actions: list[np.ndarray] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        collected_actions.append(np.asarray(action, dtype=np.float64).reshape(-1))
        obs, reward, dones, _ = env.step(action)
        rewards.append(float(reward[0]))
        done = bool(dones[0])

    stacked = (
        np.stack(collected_actions, axis=0)
        if collected_actions
        else np.zeros((0, 1), dtype=np.float64)
    )
    shares = continuous_action_shares(stacked)
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "n_steps": len(rewards),
        "action_mean_abs": float(np.mean(np.abs(stacked))) if stacked.size else 0.0,
        "action_std": float(np.std(stacked)) if stacked.size else 0.0,
        "action_share_sell": shares["sell"],
        "action_share_hold": shares["hold"],
        "action_share_buy": shares["buy"],
    }


def diagnostics_for(model: BaseAlgorithm, env: VecNormalize) -> dict[str, float]:
    """Dispatch to the appropriate rollout diagnostics based on model type."""
    if isinstance(model, SAC):
        return continuous_rollout_diagnostics(model, env)
    return rollout_diagnostics(model, env)
