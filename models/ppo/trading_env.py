from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models.ppo.panel import MarketPanel

SELL, HOLD, BUY = 0, 1, 2
# Bildet die diskreten Aktionen auf Handelsrichtungen {-1, 0, +1} ab.
_ACTION_TO_DIRECTION = np.array([-1.0, 0.0, 1.0], dtype=np.float64)


class MultiStockTradingEnv(gym.Env):
    """Long-only Multi-Stock-Trading-Umgebung für einen PPO-Agenten.

    Der Agent entscheidet je Aktie und Rebalancing-Zeitpunkt zwischen Sell,
    Hold und Buy. Die Signalstärke aus Alpha und Risiko bestimmt über das
    risikoadjustierte Position Sizing, wie groß die resultierende Positions-
    änderung ausfällt. Portfolio-Constraints begrenzen die Zielgewichte.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        panel: MarketPanel,
        initial_cash: float = 1_000_000.0,
        transaction_cost_bps: float = 10.0,
        trade_penalty_bps: float = 2.0,
        min_holding_days: int = 3,
        w_max: float = 0.2,
        rebalance_budget: float = 0.2,
        eps: float = 1e-8,
    ) -> None:
        """Initialisiert die Trading-Umgebung.

        Args:
            panel: Ausgerichtete Marktdaten (Kurse, Returns, Alpha-Scores,
                Volatilität) für alle Aktien; das Environment liest hieraus nur.
            initial_cash: Startkapital in Geldeinheiten. Da intern in Gewichten
                gerechnet wird, dient der Betrag nur als Basis des Portfoliowerts.
            transaction_cost_bps: Reale Transaktionskosten pro Trade in
                Basispunkten (1 bp = 0,01 %); senken den Portfoliowert je Turnover.
            trade_penalty_bps: Zusätzlicher Reward-Abzug pro Turnover-Einheit als
                Overtrading-Bremse; wirkt nur auf das Lernsignal, nicht auf den Wert.
            min_holding_days: Mindesthaltedauer nach einem Kauf, bevor ein Verkauf
                derselben Position erlaubt ist.
            w_max: Maximales Portfolio-Gewicht je Einzelaktie; erzwingt Diversifikation.
            rebalance_budget: Rebalancing-Budget B_t; begrenzt den pro Schritt
                umgeschichteten Portfolioanteil (Position Sizing dw = a * B_t * q).
            eps: Kleiner Sicherheitswert gegen Division durch null.
        """
        super().__init__()
        if panel.n_days <= 1:
            raise ValueError("Das Panel muss mindestens zwei Handelstage enthalten.")
        if not 0.0 < w_max <= 1.0:
            raise ValueError("w_max muss im Intervall (0, 1] liegen.")

        self.panel = panel
        self.n_stocks = panel.n_stocks
        self.initial_cash = float(initial_cash)
        self.transaction_cost_rate = float(transaction_cost_bps) / 10_000.0
        self.trade_penalty_rate = float(trade_penalty_bps) / 10_000.0
        self.min_holding_days = int(min_holding_days)
        self.w_max = float(w_max)
        self.rebalance_budget = float(rebalance_budget)
        self.eps = float(eps)

        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)
        # State: Alpha (N) + Volatilität (N) + Gewichte inkl. Cash (N+1) + Returns (N)
        observation_dim = 4 * self.n_stocks + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )

        # Erster Entscheidungstag; Index 0 hat noch keinen vorherigen Return.
        self._start_index = 1
        self.current_index = self._start_index
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float64)
        self.holding_days = np.zeros(self.n_stocks, dtype=np.int64)

    def _portfolio_value(self) -> float:
        return float(self.cash + self.holdings.sum())

    def _weights(self) -> tuple[np.ndarray, float]:
        portfolio_value = max(self._portfolio_value(), self.eps)
        stock_weights = self.holdings / portfolio_value
        cash_weight = self.cash / portfolio_value
        return stock_weights, cash_weight

    def _risk_adjusted_opportunity(self, alpha: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        # z_i = |alpha_i| / (sigma_i + eps), anschließend auf Summe 1 normiert.
        z = np.abs(alpha) / (sigma + self.eps)
        z_sum = z.sum()
        if z_sum <= 0.0:
            return np.zeros_like(z)
        return z / z_sum

    def _build_observation(self) -> np.ndarray:
        t = self.current_index
        stock_weights, cash_weight = self._weights()
        observation = np.concatenate(
            [
                self.panel.alpha[t],
                self.panel.volatility[t],
                stock_weights,
                np.array([cash_weight], dtype=np.float64),
                self.panel.returns[t],
            ]
        )
        return observation.astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_index = self._start_index
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float64)
        self.holding_days = np.zeros(self.n_stocks, dtype=np.int64)
        return self._build_observation(), self._build_info(
            turnover=0.0, cost_rate=0.0, n_blocked=0, portfolio_value=self.initial_cash
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != self.n_stocks:
            raise ValueError(f"Aktion hat Länge {action.shape[0]}, erwartet {self.n_stocks}.")

        t = self.current_index
        prev_value = self._portfolio_value()
        prev_weights = self.holdings / max(prev_value, self.eps)

        q = self._risk_adjusted_opportunity(self.panel.alpha[t], self.panel.volatility[t])
        direction = _ACTION_TO_DIRECTION[action]

        # Overtrading-Bremse: frühe Sells auf gehaltenen Positionen blockieren.
        blocked = (direction < 0.0) & (self.holding_days < self.min_holding_days) & (prev_weights > 0.0)
        direction = np.where(blocked, 0.0, direction)

        # Risikoadjustiertes Position Sizing und Long-only-Constraints.
        delta_weights = direction * self.rebalance_budget * q
        target_weights = np.clip(prev_weights + delta_weights, 0.0, self.w_max)
        invested = target_weights.sum()
        if invested > 1.0:
            target_weights = target_weights / invested

        turnover = float(np.abs(target_weights - prev_weights).sum())
        cost_rate = self.transaction_cost_rate * turnover
        value_after_cost = prev_value * (1.0 - cost_rate)

        self.holdings = target_weights * value_after_cost
        self.cash = value_after_cost - float(self.holdings.sum())

        # Übergang auf den nächsten Handelstag und Marktbewegung anwenden.
        self.current_index += 1
        self.holdings = self.holdings * (1.0 + self.panel.returns[self.current_index])
        new_value = self._portfolio_value()

        reward = float(
            np.log(max(new_value, self.eps) / max(prev_value, self.eps))
            - self.trade_penalty_rate * turnover
        )

        held = self.holdings > self.eps
        self.holding_days = np.where(held, self.holding_days + 1, 0)

        terminated = self.current_index >= self.panel.n_days - 1
        truncated = False
        info = self._build_info(
            turnover=turnover,
            cost_rate=cost_rate,
            n_blocked=int(np.sum(blocked)),
            portfolio_value=new_value,
            action=action,
        )
        return self._build_observation(), reward, terminated, truncated, info

    def _build_info(
        self,
        *,
        turnover: float,
        cost_rate: float,
        n_blocked: int,
        portfolio_value: float,
        action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        return {
            "date": self.panel.dates[self.current_index],
            "portfolio_value": float(portfolio_value),
            "cash": float(self.cash),
            "turnover": float(turnover),
            "cost_rate": float(cost_rate),
            "n_blocked": int(n_blocked),
            "action": action.tolist() if action is not None else [HOLD] * self.n_stocks,
        }

    def render(self) -> None:
        date = self.panel.dates[self.current_index]
        print(
            f"{date} | value={self._portfolio_value():.2f} | "
            f"cash={self.cash:.2f} | positions={np.count_nonzero(self.holdings > self.eps)}"
        )
