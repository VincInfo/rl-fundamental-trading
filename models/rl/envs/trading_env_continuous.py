from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from models.rl.envs.trading_env import MultiStockTradingEnv


class MultiStockTradingEnvContinuous(MultiStockTradingEnv):
    """Long-only Multi-Stock-Trading-Umgebung mit kontinuierlichem Action-Space (SAC).

    Aktionen liegen je Aktie in [-1, +1] und werden direkt als Trade-Richtung
    und -stärke interpretiert. Position-Sizing, Constraints und Reward-Logik
    sind identisch mit der diskreten Variante; nur die Aktions-Interpretation
    unterscheidet sich.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # snapshot rebalancing maps discrete BUY/SELL codes; float actions never match those.
        if self.rebalance_mode == "snapshot":
            raise ValueError(
                "rebalance_mode='snapshot' is not supported for continuous actions; "
                "use 'incremental'."
            )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_stocks,),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != self.n_stocks:
            raise ValueError(f"Aktion hat Länge {action.shape[0]}, erwartet {self.n_stocks}.")

        direction = np.clip(action, -1.0, 1.0)
        return self._apply_direction(direction, action)
