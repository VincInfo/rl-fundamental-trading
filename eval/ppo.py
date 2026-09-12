"""Trading policies, portfolio rollouts, and performance metrics for evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from models.rl.panel import MarketPanel
from models.rl.envs import BUY, HOLD, SELL, MultiStockTradingEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "stable-baselines3 is missing. Install it with: uv add stable-baselines3"
    ) from exc

ACTION_NAMES = {SELL: "sell", HOLD: "hold", BUY: "buy"}
TRADING_DAYS_PER_YEAR = 252


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


def alpha_quantile_actions(
    alpha: np.ndarray,
    *,
    buy_fraction: float = 0.3,
    sell_fraction: float = 0.3,
) -> np.ndarray:
    """Buy top-alpha names, sell bottom-alpha names, hold the middle."""
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    n_stocks = alpha.shape[0]
    if n_stocks == 0:
        return np.zeros(0, dtype=np.int64)

    n_buy = max(1, int(round(n_stocks * buy_fraction)))
    n_sell = max(1, int(round(n_stocks * sell_fraction)))
    n_buy = min(n_buy, n_stocks)
    n_sell = min(n_sell, n_stocks)
    if n_buy + n_sell > n_stocks:
        n_sell = n_stocks - n_buy

    order = np.argsort(alpha)
    actions = np.full(n_stocks, HOLD, dtype=np.int64)
    if n_sell > 0:
        actions[order[:n_sell]] = SELL
    if n_buy > 0:
        actions[order[-n_buy:]] = BUY
    return actions


def alpha_zscore_actions(
    alpha: np.ndarray,
    *,
    z_threshold: float = 0.5,
    min_abs_alpha: float = 0.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Long/short by cross-sectional alpha z-score with a symmetric dead zone.

    More adaptive than a fixed top/bottom count: the number of buys/sells
    depends on how dispersed today's alpha signal is. Names inside
    ``[-z_threshold, z_threshold]`` or below the cost floor are held.
    """
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    n_stocks = alpha.shape[0]
    if n_stocks == 0:
        return np.zeros(0, dtype=np.int64)

    mean = float(alpha.mean())
    std = float(alpha.std())
    z = (alpha - mean) / (std + eps)
    actions = np.full(n_stocks, HOLD, dtype=np.int64)
    tradable = np.abs(alpha) >= float(min_abs_alpha)
    actions[(z >= z_threshold) & tradable] = BUY
    actions[(z <= -z_threshold) & tradable] = SELL
    return actions


def apply_cost_floor(actions: np.ndarray, alpha: np.ndarray, min_abs_alpha: float) -> np.ndarray:
    """Demote trades to Hold when |alpha| does not clear the cost floor."""
    out = np.asarray(actions, dtype=np.int64).copy()
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    too_small = np.abs(alpha) < float(min_abs_alpha)
    out[too_small] = HOLD
    return out


def rule_actions_from_alpha(
    alpha: np.ndarray,
    *,
    rule_mode: str = "zscore",
    z_threshold: float = 0.5,
    buy_fraction: float = 0.3,
    sell_fraction: float = 0.3,
    min_abs_alpha: float = 0.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Dispatch the configured alpha rule (z-score default, quantile optional)."""
    mode = rule_mode.lower().strip()
    if mode == "quantile":
        actions = alpha_quantile_actions(
            alpha,
            buy_fraction=buy_fraction,
            sell_fraction=sell_fraction,
        )
        return apply_cost_floor(actions, alpha, min_abs_alpha)
    if mode != "zscore":
        raise ValueError(f"Unknown rule_mode={rule_mode!r}; expected 'zscore' or 'quantile'.")
    return alpha_zscore_actions(
        alpha,
        z_threshold=z_threshold,
        min_abs_alpha=min_abs_alpha,
        eps=eps,
    )


def cost_floor_from_bps(transaction_cost_bps: float, cost_multiple: float) -> float:
    """Minimum |alpha| required to justify a trade under a simple cost hurdle."""
    return (float(transaction_cost_bps) / 10_000.0) * float(cost_multiple)


def portfolio_metrics(ledger: pd.DataFrame) -> dict[str, float]:
    """Calculate portfolio performance metrics from a daily value ledger."""
    if ledger.empty:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "mean_turnover": 0.0,
            "total_transaction_cost": 0.0,
        }

    values = ledger["portfolio_value"].astype(float)
    daily_returns = values.pct_change().dropna()
    n_periods = len(daily_returns)
    cumulative_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    annualized_return = (
        float((1.0 + cumulative_return) ** (TRADING_DAYS_PER_YEAR / n_periods) - 1.0)
        if n_periods
        else 0.0
    )
    annualized_volatility = (
        float(daily_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
        if n_periods > 1
        else 0.0
    )
    daily_std = daily_returns.std(ddof=1)
    sharpe_ratio = (
        float(daily_returns.mean() / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR))
        if n_periods > 1 and daily_std > 0.0
        else 0.0
    )
    drawdown = values / values.cummax() - 1.0
    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": float(ledger["turnover"].iloc[1:].mean())
        if len(ledger) > 1
        else 0.0,
        "total_transaction_cost": float(ledger["transaction_cost"].sum())
        if "transaction_cost" in ledger
        else 0.0,
    }


def mean_policy_entropy(model: PPO, obs: np.ndarray) -> float:
    """Entropy of the current policy distribution (sum over stocks)."""
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(obs_tensor)
    return float(distribution.entropy().mean().cpu().item())


def _summarize_rollout(
    rewards: list[float],
    collected_actions: list[np.ndarray],
    entropies: list[float] | None = None,
) -> dict[str, float]:
    stacked = (
        np.stack(collected_actions, axis=0)
        if collected_actions
        else np.zeros((0, 1), dtype=np.int64)
    )
    shares = action_shares(stacked)
    n_stocks = int(stacked.shape[1]) if stacked.size else 0
    metrics = {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "n_steps": len(rewards),
        "action_share_sell": shares["sell"],
        "action_share_hold": shares["hold"],
        "action_share_buy": shares["buy"],
    }
    if entropies is not None:
        metrics["mean_entropy"] = float(np.mean(entropies)) if entropies else 0.0
        metrics["max_entropy"] = float(n_stocks * math.log(3.0)) if n_stocks else 0.0
    return metrics


def _run_rollout(
    model: PPO,
    env: VecNormalize,
    *,
    residual: bool = False,
) -> tuple[pd.DataFrame, list[float], list[np.ndarray], list[float], list[np.ndarray], list[float]]:
    """Run one deterministic episode and retain portfolio-level observations."""
    from models.rl.residual import compose_residual_actions

    obs = env.reset()
    initial_cash = float(env.get_attr("initial_cash")[0])
    panel = env.get_attr("panel")[0]
    n_stocks = int(env.get_attr("n_stocks")[0])
    start_index = int(env.get_attr("current_index")[0])
    records: list[dict[str, object]] = [
        {
            "date": panel.dates[start_index],
            "portfolio_value": initial_cash,
            "cash": initial_cash,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "reward": 0.0,
            "log_return": 0.0,
            "action": [HOLD] * n_stocks,
        }
    ]
    rewards: list[float] = []
    log_returns: list[float] = []
    collected_actions: list[np.ndarray] = []
    collected_residuals: list[np.ndarray] = []
    entropies: list[float] = []
    done = False
    while not done:
        entropies.append(mean_policy_entropy(model, obs))
        action, _ = model.predict(obs, deterministic=True)
        residual_action = np.asarray(action).reshape(-1)
        if residual:
            wrapped = env.venv.envs[0]
            rule = wrapped.rule_actions()
            final = compose_residual_actions(rule, residual_action)
            collected_actions.append(final)
            collected_residuals.append(residual_action)
        else:
            collected_actions.append(residual_action)
        obs, reward, dones, infos = env.step(action)
        rewards.append(float(reward[0]))
        info = infos[0]
        log_returns.append(float(info.get("log_return", 0.0)))
        records.append(
            {
                "date": info["date"],
                "portfolio_value": info["portfolio_value"],
                "cash": info["cash"],
                "turnover": info["turnover"],
                "transaction_cost": info["transaction_cost"],
                "reward": rewards[-1],
                "log_return": log_returns[-1],
                "action": info["action"],
            }
        )
        done = bool(dones[0])

    ledger = pd.DataFrame(records)
    ledger["daily_return"] = ledger["portfolio_value"].pct_change().fillna(0.0)
    return ledger, rewards, collected_actions, entropies, collected_residuals, log_returns


def collect_rollout(
    model: PPO,
    env: VecNormalize,
    *,
    residual: bool = False,
) -> pd.DataFrame:
    """Return the daily portfolio ledger of one deterministic episode."""
    ledger, *_ = _run_rollout(model, env, residual=residual)
    return ledger


def rollout_diagnostics(
    model: PPO,
    env: VecNormalize,
    ledger_path: str | Path | None = None,
    *,
    residual: bool = False,
) -> dict[str, float]:
    """Deterministic episode: reward, action mix, policy entropy, and portfolio metrics.

    When ``residual`` is True, action shares are reported for the executed
    Buy/Hold/Sell trades (after composing residuals with the alpha rule), and
    residual keep/up/down shares are added separately.
    """
    ledger, rewards, collected_actions, entropies, collected_residuals, log_returns = (
        _run_rollout(model, env, residual=residual)
    )
    if ledger_path is not None:
        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(path, index=False)

    metrics = _summarize_rollout(rewards, collected_actions, entropies)
    metrics["mean_log_return"] = float(np.mean(log_returns)) if log_returns else 0.0
    metrics.update(portfolio_metrics(ledger))
    if residual and collected_residuals:
        residual_shares = action_shares(np.stack(collected_residuals, axis=0))
        metrics["residual_share_down"] = residual_shares["sell"]
        metrics["residual_share_keep"] = residual_shares["hold"]
        metrics["residual_share_up"] = residual_shares["buy"]
    return metrics


def rollout_fixed_policy(
    env: MultiStockTradingEnv,
    action_fn: Callable[[MultiStockTradingEnv], np.ndarray],
) -> dict[str, float]:
    """Run one episode with a deterministic action function on a raw env."""
    metrics, _ = rollout_fixed_policy_with_ledger(env, action_fn)
    return metrics


def rollout_fixed_policy_with_ledger(
    env: MultiStockTradingEnv,
    action_fn: Callable[[MultiStockTradingEnv], np.ndarray],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run a fixed policy and return both summary metrics and daily ledger."""
    _, info = env.reset()
    n_stocks = env.n_stocks
    records: list[dict[str, object]] = [
        {
            "date": env.panel.dates[env.current_index],
            "portfolio_value": env.initial_cash,
            "cash": env.initial_cash,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "reward": 0.0,
            "log_return": 0.0,
            "action": [HOLD] * n_stocks,
        }
    ]
    rewards: list[float] = []
    log_returns: list[float] = []
    collected_actions: list[np.ndarray] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = np.asarray(action_fn(env), dtype=np.int64).reshape(-1)
        collected_actions.append(action)
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        log_returns.append(float(info.get("log_return", 0.0)))
        records.append(
            {
                "date": info["date"],
                "portfolio_value": info["portfolio_value"],
                "cash": info["cash"],
                "turnover": info["turnover"],
                "transaction_cost": info["transaction_cost"],
                "reward": rewards[-1],
                "log_return": log_returns[-1],
                "action": info.get("action", action.tolist()),
            }
        )
    ledger = pd.DataFrame(records)
    ledger["daily_return"] = ledger["portfolio_value"].pct_change().fillna(0.0)
    metrics = _summarize_rollout(rewards, collected_actions)
    metrics["mean_log_return"] = float(np.mean(log_returns)) if log_returns else 0.0
    metrics.update(portfolio_metrics(ledger))
    return metrics, ledger


def make_alpha_quantile_policy(
    *,
    buy_fraction: float = 0.3,
    sell_fraction: float = 0.3,
) -> Callable[[MultiStockTradingEnv], np.ndarray]:
    """Legacy quantile policy (kept for tests / explicit comparisons)."""

    def _policy(env: MultiStockTradingEnv) -> np.ndarray:
        alpha = env.panel.alpha[env.current_index]
        return alpha_quantile_actions(
            alpha,
            buy_fraction=buy_fraction,
            sell_fraction=sell_fraction,
        )

    return _policy


def make_alpha_rule_policy(
    *,
    rule_mode: str = "zscore",
    z_threshold: float = 0.5,
    buy_fraction: float = 0.3,
    sell_fraction: float = 0.3,
    transaction_cost_bps: float = 10.0,
    rule_cost_multiple: float = 1.0,
    eps: float = 1e-8,
) -> Callable[[MultiStockTradingEnv], np.ndarray]:
    """Configured alpha rule used as residual baseline / evaluation policy."""
    min_abs_alpha = cost_floor_from_bps(transaction_cost_bps, rule_cost_multiple)

    def _policy(env: MultiStockTradingEnv) -> np.ndarray:
        alpha = env.panel.alpha[env.current_index]
        return rule_actions_from_alpha(
            alpha,
            rule_mode=rule_mode,
            z_threshold=z_threshold,
            buy_fraction=buy_fraction,
            sell_fraction=sell_fraction,
            min_abs_alpha=min_abs_alpha,
            eps=eps,
        )

    return _policy


def make_hybrid_rule_policy(
    long_alpha_wide: pd.DataFrame,
    short_alpha_wide: pd.DataFrame,
    *,
    rule_mode: str = "zscore",
    z_threshold: float = 0.5,
    buy_fraction: float = 0.3,
    sell_fraction: float = 0.3,
    transaction_cost_bps: float = 10.0,
    rule_cost_multiple: float = 1.0,
    eps: float = 1e-8,
) -> Callable[[MultiStockTradingEnv], np.ndarray]:
    """Buy from one alpha book, sell from another; conflicts become Hold."""
    min_abs_alpha = cost_floor_from_bps(transaction_cost_bps, rule_cost_multiple)
    aligned: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _matrices(env: MultiStockTradingEnv) -> tuple[np.ndarray, np.ndarray]:
        key = id(env.panel)
        if key not in aligned:
            dates = pd.Index(env.panel.dates)
            symbols = list(env.panel.symbols)
            long_mat = (
                long_alpha_wide.reindex(index=dates, columns=symbols).fillna(0.0).to_numpy()
            )
            short_mat = (
                short_alpha_wide.reindex(index=dates, columns=symbols).fillna(0.0).to_numpy()
            )
            aligned[key] = (long_mat, short_mat)
        return aligned[key]

    def _policy(env: MultiStockTradingEnv) -> np.ndarray:
        long_mat, short_mat = _matrices(env)
        t = env.current_index
        buy_side = rule_actions_from_alpha(
            long_mat[t],
            rule_mode=rule_mode,
            z_threshold=z_threshold,
            buy_fraction=buy_fraction,
            sell_fraction=sell_fraction,
            min_abs_alpha=min_abs_alpha,
            eps=eps,
        )
        sell_side = rule_actions_from_alpha(
            short_mat[t],
            rule_mode=rule_mode,
            z_threshold=z_threshold,
            buy_fraction=buy_fraction,
            sell_fraction=sell_fraction,
            min_abs_alpha=min_abs_alpha,
            eps=eps,
        )
        actions = np.full(env.n_stocks, HOLD, dtype=np.int64)
        actions[buy_side == BUY] = BUY
        actions[sell_side == SELL] = SELL
        conflict = (buy_side == BUY) & (sell_side == SELL)
        actions[conflict] = HOLD
        return actions

    return _policy
