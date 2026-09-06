from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models.ppo.config import EnvConfig
from eval.ppo import cost_floor_from_bps, rule_actions_from_alpha
from models.ppo.trading_env import BUY, HOLD, SELL, MultiStockTradingEnv

# Residual codes relative to the rule action: decrease / keep / increase.
RESIDUAL_DOWN, RESIDUAL_KEEP, RESIDUAL_UP = 0, 1, 2


def compose_residual_actions(rule_actions: np.ndarray, residual_actions: np.ndarray) -> np.ndarray:
    """Map residual {-1,0,+1} onto rule Buy/Hold/Sell actions."""
    rule = np.asarray(rule_actions, dtype=np.int64).reshape(-1)
    residual = np.asarray(residual_actions, dtype=np.int64).reshape(-1)
    if rule.shape != residual.shape:
        raise ValueError(
            f"Rule/residual shape mismatch: {rule.shape} vs {residual.shape}"
        )
    delta = residual - RESIDUAL_KEEP
    return np.clip(rule + delta, SELL, BUY).astype(np.int64)


class ResidualAlphaEnv(gym.Wrapper):
    """PPO outputs residuals around the configured alpha rule policy.

    Action semantics per stock:
    - 0: move one step toward Sell relative to the rule
    - 1: follow the rule
    - 2: move one step toward Buy relative to the rule
    """

    def __init__(
        self,
        env: MultiStockTradingEnv,
        env_config: EnvConfig | None = None,
        *,
        buy_fraction: float | None = None,
        sell_fraction: float | None = None,
    ) -> None:
        super().__init__(env)
        config = env_config or EnvConfig()
        # Backward-compatible overrides from older call sites.
        if buy_fraction is not None:
            config.rule_buy_fraction = float(buy_fraction)
        if sell_fraction is not None:
            config.rule_sell_fraction = float(sell_fraction)
        self.env_config = config
        self.action_space = spaces.MultiDiscrete([3] * env.n_stocks)

    @property
    def trading_env(self) -> MultiStockTradingEnv:
        return self.env  # type: ignore[return-value]

    def rule_actions(self) -> np.ndarray:
        alpha = self.trading_env.panel.alpha[self.trading_env.current_index]
        min_abs_alpha = cost_floor_from_bps(
            self.env_config.transaction_cost_bps,
            self.env_config.rule_cost_multiple,
        )
        return rule_actions_from_alpha(
            alpha,
            rule_mode=self.env_config.rule_mode,
            z_threshold=self.env_config.rule_z_threshold,
            buy_fraction=self.env_config.rule_buy_fraction,
            sell_fraction=self.env_config.rule_sell_fraction,
            min_abs_alpha=min_abs_alpha,
            eps=self.env_config.eps,
        )

    def step(self, action: np.ndarray):
        residual = np.asarray(action, dtype=np.int64).reshape(-1)
        rule = self.rule_actions()
        final = compose_residual_actions(rule, residual)
        obs, reward, terminated, truncated, info = self.env.step(final)
        info = dict(info)
        info["rule_action"] = rule.tolist()
        info["residual_action"] = residual.tolist()
        info["final_action"] = final.tolist()
        return obs, reward, terminated, truncated, info
