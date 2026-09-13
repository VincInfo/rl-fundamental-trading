from __future__ import annotations

import math

import numpy as np
import pandas as pd

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
COMPARISON_METRICS = (
    "cumulative_return", "annualized_volatility", "sharpe_ratio", "max_drawdown",
    "mean_turnover", "total_transaction_cost", "mean_reward", "n_steps",
    "action_share_sell", "action_share_hold", "action_share_buy",
    "residual_action_share_down", "residual_action_share_keep",
    "residual_action_share_up",
)


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


def ppo_rollout_metrics(model: PPO, env: VecNormalize) -> dict[str, float]:
    """Evaluate a deterministic PPO rollout using environment accounting."""
    obs = env.reset()
    rewards: list[float] = []
    returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    actions: list[np.ndarray] = []
    residual_actions: list[np.ndarray] = []
    done = False
    previous_value = None
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        info = infos[0]
        executed_action = info.get("final_action", info.get("action", action))
        actions.append(np.asarray(executed_action).reshape(-1))
        residual_actions.append(
            np.asarray(info.get("residual_action", action)).reshape(-1)
        )
        value = float(info["portfolio_value"])
        if previous_value is not None:
            returns.append(float(np.log(value / previous_value)))
        previous_value = value
        rewards.append(float(reward[0]))
        turnovers.append(float(info["turnover"]))
        costs.append(float(info["transaction_cost"]))
        done = bool(dones[0])

    values = np.asarray(returns, dtype=float)
    volatility = float(values.std(ddof=1) * math.sqrt(252)) if len(values) > 1 else 0.0
    mean_return = float(values.mean()) if len(values) else 0.0
    sharpe = float(mean_return / values.std(ddof=1) * math.sqrt(252)) if len(values) > 1 and values.std(ddof=1) > 0 else 0.0
    equity = np.exp(np.cumsum(values)) if len(values) else np.ones(1)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    stacked = np.stack(actions) if actions else np.zeros((0, 1), dtype=np.int64)
    shares = action_shares(stacked)
    stacked_residual = (
        np.stack(residual_actions)
        if residual_actions
        else np.zeros((0, 1), dtype=np.int64)
    )
    residual_shares = {
        "down": float((stacked_residual == 0).mean()) if stacked_residual.size else 0.0,
        "keep": float((stacked_residual == 1).mean()) if stacked_residual.size else 0.0,
        "up": float((stacked_residual == 2).mean()) if stacked_residual.size else 0.0,
    }
    return {
        "cumulative_return": float(equity[-1] - 1.0),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "total_transaction_cost": float(np.sum(costs)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "n_steps": len(rewards),
        **{f"action_share_{name}": value for name, value in shares.items()},
        **{
            f"residual_action_share_{name}": value
            for name, value in residual_shares.items()
        },
    }


def reference_portfolio_metrics(
    features: pd.DataFrame,
    *,
    start_index: int = 0,
) -> dict[str, dict[str, float]]:
    """Evaluate references on the selected daily close return window.

    ``start_index`` aligns the first included return with the portfolio state
    used by the PPO environment. The default preserves the full-frame behavior.
    """
    returns = features.pivot(index="date", columns="symbol", values="return_1d")
    returns = returns.sort_index().dropna(how="all").fillna(0.0)
    if start_index < 0 or start_index >= len(returns):
        raise ValueError(
            f"start_index={start_index} outside return window of length {len(returns)}"
        )
    returns = returns.iloc[start_index:]
    equal_weight = returns.mean(axis=1).to_numpy(dtype=float)
    buy_and_hold = (
        returns.add(1.0).cumprod(axis=0).mean(axis=1).pct_change().fillna(0.0)
        .to_numpy(dtype=float)
    )

    def summarize(daily_returns: np.ndarray) -> dict[str, float]:
        log_returns = np.log(np.maximum(1.0 + daily_returns, 1e-8))
        volatility = float(log_returns.std(ddof=1) * math.sqrt(252)) if len(log_returns) > 1 else 0.0
        mean_return = float(log_returns.mean()) if len(log_returns) else 0.0
        sharpe = (
            float(mean_return / log_returns.std(ddof=1) * math.sqrt(252))
            if len(log_returns) > 1 and log_returns.std(ddof=1) > 0
            else 0.0
        )
        equity = np.exp(np.cumsum(log_returns)) if len(log_returns) else np.ones(1)
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        return {
            "cumulative_return": float(equity[-1] - 1.0),
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": float(drawdown.min()),
            "n_steps": int(len(log_returns)),
        }

    return {
        "buy_and_hold": summarize(buy_and_hold),
        "equal_weight": summarize(equal_weight),
    }


def comparison_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep the stable, comparable fields from a PPO evaluation result."""
    return {name: float(metrics.get(name, 0.0)) for name in COMPARISON_METRICS}


def compare_metric_sets(
    market_only: dict[str, float], full_alpha: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Return comparable metrics and the full-alpha minus market-only delta."""
    market = comparison_metrics(market_only)
    full = comparison_metrics(full_alpha)
    return {
        "market_only": market,
        "full_alpha": full,
        "delta_full_minus_market": {
            name: full[name] - market[name] for name in COMPARISON_METRICS
        },
    }
