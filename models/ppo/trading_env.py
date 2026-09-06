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
    """Multi-Stock-Trading-Umgebung für einen PPO-Agenten.

    Der Agent entscheidet je Aktie und Rebalancing-Zeitpunkt zwischen Sell,
    Hold und Buy. Standard ist long-only und täglich; Shorts und seltenere
    Rebalances (z. B. 20 Tage = Alpha-Horizont) sind optional.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        panel: MarketPanel,
        initial_cash: float = 1_000_000.0,
        transaction_cost_bps: float = 10.0,
        trade_penalty_bps: float = 10.0,
        alpha_alignment_bps: float = 5.0,
        min_holding_days: int = 5,
        w_max: float = 0.2,
        rebalance_budget: float = 0.2,
        eps: float = 1e-8,
        episode_window: int | None = None,
        randomize_start: bool = False,
        rebalance_every: int = 1,
        allow_short: bool = False,
        max_gross_exposure: float | None = None,
        rebalance_mode: str = "incremental",
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
            alpha_alignment_bps: Reward-Bonus, wenn Aktionen mit dem Alpha-Vorzeichen
                übereinstimmen (gewichtet mit risikoadjustiertem q).
            min_holding_days: Mindesthaltedauer nach einem Kauf, bevor ein Verkauf
                derselben Position erlaubt ist.
            w_max: Maximales Portfolio-Gewicht je Einzelaktie; erzwingt Diversifikation.
            rebalance_budget: Rebalancing-Budget B_t; begrenzt den pro Schritt
                umgeschichteten Portfolioanteil (Position Sizing dw = a * B_t * q).
            eps: Kleiner Sicherheitswert gegen Division durch null.
            episode_window: Episode length in steps. ``None`` uses the full panel.
            randomize_start: If True, sample a random start index on ``reset``.
            rebalance_every: Apply actions every N days; otherwise hold.
            allow_short: If True, SELL can take negative weights.
            max_gross_exposure: Cap on sum(|weights|). Defaults to 1.0, or 2.0 with shorts.
            rebalance_mode: ``incremental`` deltas or ``snapshot`` target book from actions.
        """
        super().__init__()
        if panel.n_days <= 1:
            raise ValueError("Das Panel muss mindestens zwei Handelstage enthalten.")
        if not 0.0 < w_max <= 1.0:
            raise ValueError("w_max muss im Intervall (0, 1] liegen.")
        mode = rebalance_mode.lower().strip()
        if mode not in {"incremental", "snapshot"}:
            raise ValueError(f"Unknown rebalance_mode={rebalance_mode!r}.")

        self.panel = panel
        self.n_stocks = panel.n_stocks
        self.initial_cash = float(initial_cash)
        self.transaction_cost_rate = float(transaction_cost_bps) / 10_000.0
        self.trade_penalty_rate = float(trade_penalty_bps) / 10_000.0
        self.alpha_alignment_rate = float(alpha_alignment_bps) / 10_000.0
        self.min_holding_days = int(min_holding_days)
        self.w_max = float(w_max)
        self.rebalance_budget = float(rebalance_budget)
        self.eps = float(eps)
        self.episode_window = None if episode_window is None else int(episode_window)
        self.randomize_start = bool(randomize_start)
        self.rebalance_every = max(int(rebalance_every), 1)
        self.allow_short = bool(allow_short)
        self.rebalance_mode = mode
        if max_gross_exposure is None:
            max_gross_exposure = 2.0 if self.allow_short else 1.0
        self.max_gross_exposure = float(max_gross_exposure)

        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)
        # State: Alpha, q, signed_z, weights, cash, returns, holding_days.
        observation_dim = self.observation_dim(self.n_stocks)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )

        # Erster Entscheidungstag; Index 0 hat noch keinen vorherigen Return.
        self._start_index = 1
        self._end_index = panel.n_days - 1
        self.current_index = self._start_index
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float64)
        self.holding_days = np.zeros(self.n_stocks, dtype=np.int64)

    @staticmethod
    def observation_dim(n_stocks: int) -> int:
        return 6 * int(n_stocks) + 1

    def _portfolio_value(self) -> float:
        return float(self.cash + self.holdings.sum())

    def _weights(self) -> tuple[np.ndarray, float]:
        portfolio_value = max(self._portfolio_value(), self.eps)
        stock_weights = self.holdings / portfolio_value
        cash_weight = self.cash / portfolio_value
        return stock_weights, cash_weight

    def _signed_opportunity(self, alpha: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return alpha / (sigma + self.eps)

    def _risk_adjusted_opportunity(self, alpha: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        # z_i = |alpha_i| / (sigma_i + eps), anschließend auf Summe 1 normiert.
        z = np.abs(alpha) / (sigma + self.eps)
        z_sum = z.sum()
        if z_sum <= 0.0:
            return np.zeros_like(z)
        return z / z_sum

    def _build_observation(self) -> np.ndarray:
        t = self.current_index
        alpha = self.panel.alpha[t]
        sigma = self.panel.volatility[t]
        q = self._risk_adjusted_opportunity(alpha, sigma)
        signed_z = self._signed_opportunity(alpha, sigma)
        stock_weights, cash_weight = self._weights()
        holding_frac = self.holding_days.astype(np.float64)
        observation = np.concatenate(
            [
                alpha,
                q,
                signed_z,
                stock_weights,
                np.array([cash_weight], dtype=np.float64),
                self.panel.returns[t],
                holding_frac,
            ]
        )
        return observation.astype(np.float32)

    def _choose_episode_bounds(self) -> tuple[int, int]:
        last_index = self.panel.n_days - 1
        if last_index <= 1:
            return 1, last_index
        window = self.episode_window
        if window is None or window <= 0:
            return 1, last_index
        window = min(int(window), last_index - 1)
        if self.randomize_start and last_index - 1 > window:
            max_start = last_index - window
            start = int(self.np_random.integers(1, max_start + 1))
        else:
            start = 1
        return start, min(start + window, last_index)

    def _is_rebalance_day(self) -> bool:
        return (self.current_index - self._start_index) % self.rebalance_every == 0

    def _weight_bounds(self) -> tuple[float, float]:
        lower = -self.w_max if self.allow_short else 0.0
        return lower, self.w_max

    def _scale_gross(self, weights: np.ndarray) -> np.ndarray:
        gross = float(np.abs(weights).sum())
        if gross > self.max_gross_exposure + self.eps:
            weights = weights * (self.max_gross_exposure / gross)
        if not self.allow_short:
            invested = float(weights.sum())
            if invested > 1.0:
                weights = weights / invested
        return weights

    def _snapshot_target_weights(self, action: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Map Buy/Hold/Sell into a full target book (alpha-horizon rebalance)."""
        buy = action == BUY
        sell = action == SELL
        weights = np.zeros(self.n_stocks, dtype=np.float64)
        long_gross = self.max_gross_exposure if not self.allow_short else 0.5 * self.max_gross_exposure
        short_gross = 0.0 if not self.allow_short else 0.5 * self.max_gross_exposure

        if np.any(buy):
            q_buy = np.maximum(q[buy], self.eps)
            weights[buy] = q_buy / q_buy.sum() * long_gross
        if self.allow_short and np.any(sell):
            q_sell = np.maximum(q[sell], self.eps)
            weights[sell] = -q_sell / q_sell.sum() * short_gross

        lower, upper = self._weight_bounds()
        weights = np.clip(weights, lower, upper)
        return self._scale_gross(weights)

    def _incremental_target_weights(
        self,
        prev_weights: np.ndarray,
        direction: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        lower, upper = self._weight_bounds()
        delta_weights = direction * self.rebalance_budget * q
        target = np.clip(prev_weights + delta_weights, lower, upper)
        return self._scale_gross(target)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._start_index, self._end_index = self._choose_episode_bounds()
        self.current_index = self._start_index
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float64)
        self.holding_days = np.zeros(self.n_stocks, dtype=np.int64)
        return self._build_observation(), self._build_info(
            turnover=0.0,
            cost_rate=0.0,
            transaction_cost=0.0,
            n_blocked=0,
            portfolio_value=self.initial_cash,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != self.n_stocks:
            raise ValueError(f"Aktion hat Länge {action.shape[0]}, erwartet {self.n_stocks}.")

        t = self.current_index
        prev_value = self._portfolio_value()
        prev_weights = self.holdings / max(prev_value, self.eps)

        alpha = self.panel.alpha[t]
        q = self._risk_adjusted_opportunity(alpha, self.panel.volatility[t])
        rebalanced = self._is_rebalance_day()
        direction = _ACTION_TO_DIRECTION[action]
        blocked = np.zeros(self.n_stocks, dtype=bool)

        if not rebalanced:
            direction = np.zeros_like(direction)
            target_weights = prev_weights
        else:
            if self.min_holding_days > 0 and self.rebalance_mode == "incremental":
                cover_long = (direction < 0.0) & (self.holding_days < self.min_holding_days) & (
                    prev_weights > 0.0
                )
                cover_short = (
                    self.allow_short
                    & (direction > 0.0)
                    & (self.holding_days < self.min_holding_days)
                    & (prev_weights < 0.0)
                )
                blocked = cover_long | cover_short
                direction = np.where(blocked, 0.0, direction)
            if self.rebalance_mode == "snapshot":
                target_weights = self._snapshot_target_weights(action, q)
            else:
                target_weights = self._incremental_target_weights(prev_weights, direction, q)

        turnover = float(np.abs(target_weights - prev_weights).sum())
        cost_rate = self.transaction_cost_rate * turnover
        transaction_cost = prev_value * cost_rate
        value_after_cost = prev_value * (1.0 - cost_rate)

        self.holdings = target_weights * value_after_cost
        self.cash = value_after_cost - float(self.holdings.sum())

        # Übergang auf den nächsten Handelstag und Marktbewegung anwenden.
        self.current_index += 1
        self.holdings = self.holdings * (1.0 + self.panel.returns[self.current_index])
        new_value = self._portfolio_value()

        log_return = float(np.log(max(new_value, self.eps) / max(prev_value, self.eps)))
        # Encourage trades that agree with alpha direction, weighted by opportunity q.
        alpha_sign = np.sign(alpha)
        alignment = float(np.sum(direction * alpha_sign * q))
        reward = float(
            log_return
            - self.trade_penalty_rate * turnover
            + self.alpha_alignment_rate * alignment
        )

        held = np.abs(self.holdings) > self.eps
        self.holding_days = np.where(held, self.holding_days + 1, 0)

        terminated = self.current_index >= self._end_index
        truncated = False
        info = self._build_info(
            turnover=turnover,
            cost_rate=cost_rate,
            transaction_cost=transaction_cost,
            n_blocked=int(np.sum(blocked)),
            portfolio_value=new_value,
            action=action,
            alignment=alignment,
            log_return=log_return,
        )
        return self._build_observation(), reward, terminated, truncated, info

    def _build_info(
        self,
        *,
        turnover: float,
        cost_rate: float,
        transaction_cost: float,
        n_blocked: int,
        portfolio_value: float,
        action: np.ndarray | None = None,
        alignment: float = 0.0,
        log_return: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "date": self.panel.dates[self.current_index],
            "portfolio_value": float(portfolio_value),
            "cash": float(self.cash),
            "turnover": float(turnover),
            "cost_rate": float(cost_rate),
            "transaction_cost": float(transaction_cost),
            "n_blocked": int(n_blocked),
            "alignment": float(alignment),
            "log_return": float(log_return),
            "action": action.tolist() if action is not None else [HOLD] * self.n_stocks,
        }

    def render(self) -> None:
        date = self.panel.dates[self.current_index]
        print(
            f"{date} | value={self._portfolio_value():.2f} | "
            f"cash={self.cash:.2f} | positions={np.count_nonzero(np.abs(self.holdings) > self.eps)}"
        )
