from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

DEFAULT_FEATURE_COLUMNS = (
    "return_1d",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "revenue",
    "net_income",
    "operating_cashflow",
    "debt_to_equity",
    "gross_margin",
    "roe",
    "eps_estimate",
    "eps_actual",
    "eps_surprise",
    "is_report_day",
    "days_since_report",
)

@dataclass
class TradingState:
    """Kompakte Zustandsbeschreibung für Debugging und Analyse"""

    symbol: str
    date: pd.Timestamp
    cash: float
    shares: float
    portfolio_value: float

def load_features_daily(path: str) -> pd.DataFrame:
    """Liest den erzeugten Feature-Datensatz ein und parst das Datum"""

    frame = pd.read_csv(path, parse_dates=["date"])
    required_columns = {"date", "symbol", *DEFAULT_FEATURE_COLUMNS}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Fehlende Spalten in features_daily.csv: {sorted(missing_columns)}")

    fundamental_columns = (
        "revenue",
        "net_income",
        "operating_cashflow",
        "debt_to_equity",
        "gross_margin",
        "roe",
        "eps_estimate",
        "eps_actual",
        "eps_surprise",
    )
    # Vor dem ersten Report sind Fundamentals leer und werden hier neutralisiert
    frame[list(fundamental_columns)] = frame[list(fundamental_columns)].fillna(0.0)
    frame["is_report_day"] = frame["is_report_day"].fillna(0).astype(int)
    frame["days_since_report"] = frame["days_since_report"].fillna(999).astype(float)
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)

class FundamentalTradingEnv(gym.Env):
    """Minimale Long-only Trading-Umgebung auf Basis von Fundamentaldaten

    Der Agent sieht pro Zeitschritt Tagesfeatures für genau ein Symbol
    Aktionen sind Hold Buy Sell
    Das Portfolio besteht aus Cash und Aktienposition
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: tuple[str, ...] | None = None,
        initial_cash: float = 10_000.0,
        transaction_cost_bps: float = 10.0,
        trade_penalty_bps: float = 2.0,
        min_holding_days: int = 3,
        random_symbol_on_reset: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.data = data.copy()
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
        self.initial_cash = float(initial_cash)
        self.transaction_cost_rate = float(transaction_cost_bps) / 10_000.0
        self.trade_penalty_rate = float(trade_penalty_bps) / 10_000.0
        self.min_holding_days = int(min_holding_days)
        self.random_symbol_on_reset = random_symbol_on_reset
        self._rng = np.random.default_rng(seed)

        self._symbol_frames = {
            symbol: frame.sort_values("date").reset_index(drop=True)
            for symbol, frame in self.data.groupby("symbol", sort=True)
        }
        if not self._symbol_frames:
            raise ValueError("Die uebergebenen Daten enthalten keine Symbole.")

        self.symbols = list(self._symbol_frames)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._observation_feature_names()),),
            dtype=np.float32,
        )

        self.current_symbol: str | None = None
        self.current_frame: pd.DataFrame | None = None
        self.current_index = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.holding_days = 0
        self._start_close = 1.0

    def _observation_feature_names(self) -> tuple[str, ...]:
        # Reihenfolge hier definiert exakt die Eingabestruktur des Modells
        return (
            "price_rel_to_start",
            "intraday_range_pct",
            "open_close_pct",
            "log_volume",
            "log_revenue",
            "log_net_income",
            "log_operating_cashflow",
            "profit_margin",
            "cashflow_margin",
            "eps_surprise_pct",
            "debt_to_equity",
            "gross_margin",
            "roe",
            "eps_estimate",
            "eps_actual",
            "eps_surprise",
            "return_1d",
            "is_report_day",
            "days_since_report_norm",
            "cash_ratio",
            "position_ratio",
            "portfolio_return",
        )

    def _build_observation(self, row: pd.Series) -> np.ndarray:
        # Preisnahe Kurzfristfeatures
        price_rel_to_start = row["close"] / self._start_close - 1.0
        intraday_range_pct = (row["high"] - row["low"]) / max(row["close"], 1e-8)
        open_close_pct = (row["close"] - row["open"]) / max(row["open"], 1e-8)
        log_volume = np.log1p(max(float(row["volume"]), 0.0))

        # Fundamentaldaten werden robust skaliert damit Extremwerte weniger dominieren
        revenue_safe = max(abs(float(row["revenue"])), 1e-8)
        log_revenue = np.log1p(max(float(row["revenue"]), 0.0))
        log_net_income = np.sign(float(row["net_income"])) * np.log1p(abs(float(row["net_income"])))
        log_operating_cashflow = np.sign(float(row["operating_cashflow"])) * np.log1p(abs(float(row["operating_cashflow"])))
        profit_margin = float(row["net_income"]) / revenue_safe
        cashflow_margin = float(row["operating_cashflow"]) / revenue_safe
        eps_surprise_pct = float(row["eps_surprise"]) / max(abs(float(row["eps_estimate"])), 1.0)

        # Portfoliofeatures machen den Zustand vom aktuellen Exposure abhängig
        portfolio_value = self._portfolio_value(float(row["close"]))
        cash_ratio = self.cash / max(portfolio_value, 1e-8)
        position_ratio = (self.shares * float(row["close"])) / max(portfolio_value, 1e-8)
        portfolio_return = portfolio_value / self.initial_cash - 1.0

        features = np.array(
            [
                price_rel_to_start,
                intraday_range_pct,
                open_close_pct,
                log_volume,
                log_revenue,
                log_net_income,
                log_operating_cashflow,
                profit_margin,
                cashflow_margin,
                eps_surprise_pct,
                float(row["debt_to_equity"]),
                float(row["gross_margin"]),
                float(row["roe"]),
                float(row["eps_estimate"]),
                float(row["eps_actual"]),
                float(row["eps_surprise"]),
                float(row["return_1d"]),
                float(row["is_report_day"]),
                float(row["days_since_report"]) / 252.0,
                cash_ratio,
                position_ratio,
                portfolio_return,
            ],
            dtype=np.float32,
        )
        return features

    def _portfolio_value(self, close_price: float) -> float:
        return self.cash + self.shares * close_price

    def _get_current_row(self) -> pd.Series:
        if self.current_frame is None:
            raise RuntimeError("Die Environment wurde noch nicht resettet")
        return self.current_frame.iloc[self.current_index]

    def _select_symbol(self, options: dict[str, Any] | None = None) -> str:
        # Für Evaluation kann ein festes Symbol übergeben werden
        if options and "symbol" in options:
            symbol = str(options["symbol"])
            if symbol not in self._symbol_frames:
                raise ValueError(f"Unbekanntes Symbol: {symbol}")
            return symbol
        if self.random_symbol_on_reset:
            return str(self._rng.choice(self.symbols))
        return self.symbols[0]

    def _reset_episode_state(self, symbol: str) -> None:
        self.current_symbol = symbol
        self.current_frame = self._symbol_frames[symbol].copy().reset_index(drop=True)
        self.current_index = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.holding_days = 0
        self._start_close = float(self.current_frame.iloc[0]["close"])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Reset setzt immer Portfolio und Zeitindex auf Anfang einer Episode
        super().reset(seed=seed)
        symbol = self._select_symbol(options)
        self._reset_episode_state(symbol)

        observation = self._build_observation(self._get_current_row())
        info = self._build_info(
            action=None,
            trade_notional=0.0,
            trade_cost=0.0,
            action_blocked=False,
            requested_action=None,
            prev_value=self.initial_cash,
        )
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_frame is None:
            raise RuntimeError("Die Environment wurde noch nicht resettet")
        if self.current_index >= len(self.current_frame) - 1:
            raise RuntimeError("Episode bereits beendet Bitte reset aufrufen")

        current_row = self._get_current_row()
        current_close = float(current_row["close"])

        prev_value = self._portfolio_value(current_close)
        trade_notional = 0.0
        trade_cost = 0.0
        extra_trade_penalty = 0.0
        action_blocked = False
        executed_action = 0

        # Buy investiert das gesamte verfügbare Cash
        if action == 1 and self.cash > 0.0:
            trade_notional = self.cash
            trade_cost = trade_notional * self.transaction_cost_rate
            investable_cash = self.cash - trade_cost
            self.shares += investable_cash / max(current_close, 1e-8)
            self.cash = 0.0
            executed_action = 1
            extra_trade_penalty = self.trade_penalty_rate
        elif action == 2 and self.shares > 0.0:
            # Sell darf nur ausgeführt werden wenn die Mindesthaltedauer erreicht ist
            if self.holding_days >= self.min_holding_days:
                trade_notional = self.shares * current_close
                trade_cost = trade_notional * self.transaction_cost_rate
                self.cash += trade_notional - trade_cost
                self.shares = 0.0
                executed_action = 2
                extra_trade_penalty = self.trade_penalty_rate
            else:
                # Für Analyse wichtig damit blockierte Aktionen später sichtbar sind
                action_blocked = True

        # Haltedauer zählt nur solange wirklich eine Position offen ist
        if self.shares > 0.0:
            self.holding_days += 1
        else:
            self.holding_days = 0

        self.current_index += 1
        next_row = self._get_current_row()
        next_close = float(next_row["close"])
        next_value = self._portfolio_value(next_close)

        # Reward basiert auf Log-Return und bestraft zusätzlich aktives Handeln leicht
        reward = float(np.log(max(next_value, 1e-8) / max(prev_value, 1e-8)) - extra_trade_penalty)
        terminated = self.current_index >= len(self.current_frame) - 1
        truncated = False

        observation = self._build_observation(next_row)
        info = self._build_info(
            action=executed_action,
            trade_notional=trade_notional,
            trade_cost=trade_cost,
            action_blocked=action_blocked,
            requested_action=action,
            prev_value=prev_value,
            next_value=next_value,
        )
        return observation, float(reward), terminated, truncated, info

    def _build_info(
        self,
        *,
        action: int | None,
        trade_notional: float,
        trade_cost: float,
        action_blocked: bool,
        requested_action: int | None,
        prev_value: float,
        next_value: float | None = None,
    ) -> dict[str, Any]:
        # Info transportiert Debug und Analysefelder für spätere Auswertung
        row = self._get_current_row()
        portfolio_value = self._portfolio_value(float(row["close"]))
        return {
            "symbol": self.current_symbol,
            "date": row["date"],
            "action": action,
            "requested_action": requested_action,
            "action_blocked": action_blocked,
            "cash": self.cash,
            "shares": self.shares,
            "holding_days": self.holding_days,
            "portfolio_value": portfolio_value if next_value is None else next_value,
            "trade_notional": trade_notional,
            "trade_cost": trade_cost,
            "prev_value": prev_value,
        }

    def render(self) -> None:
        if self.current_frame is None:
            return
        # Render ist bewusst simpel gehalten und zeigt den aktuellen Handelszustand
        row = self._get_current_row()
        portfolio_value = self._portfolio_value(float(row["close"]))
        print(
            f"{self.current_symbol} | {row['date'].date()} | cash={self.cash:.2f} | "
            f"shares={self.shares:.4f} | portfolio={portfolio_value:.2f}"
        )
