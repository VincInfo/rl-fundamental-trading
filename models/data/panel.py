from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketPanel:
    """Ausgerichtete Marktdaten für alle Symbole über sämtliche Handelstage.

    Alle Matrizen haben die Form ``(n_days, n_stocks)`` mit identischer Zeilen-
    (Datum) und Spaltenreihenfolge (Symbol), sodass das Environment rein
    positionsbasiert darauf zugreifen kann.
    """

    dates: np.ndarray
    symbols: tuple[str, ...]
    close: np.ndarray
    returns: np.ndarray
    alpha: np.ndarray
    volatility: np.ndarray

    @property
    def n_days(self) -> int:
        return int(self.close.shape[0])

    @property
    def n_stocks(self) -> int:
        return int(self.close.shape[1])

    def __post_init__(self) -> None:
        expected = (self.n_days, self.n_stocks)
        for name in ("returns", "alpha", "volatility"):
            matrix = getattr(self, name)
            if matrix.shape != expected:
                raise ValueError(f"Matrix '{name}' hat Shape {matrix.shape}, erwartet {expected}.")
        if len(self.symbols) != self.n_stocks:
            raise ValueError("Anzahl Symbole passt nicht zur Spaltenanzahl der Matrizen.")
        if len(self.dates) != self.n_days:
            raise ValueError("Anzahl Datumswerte passt nicht zur Zeilenanzahl der Matrizen.")


def build_panel(
    features: pd.DataFrame,
    alpha: pd.DataFrame,
    vol_window: int = 20,
    price_column: str = "close",
    return_column: str = "return_1d",
) -> MarketPanel:
    """Baut ein ausgerichtetes ``MarketPanel`` aus Long-Format-Features und Alpha-Scores.

    ``features`` ist im Long-Format mit den Spalten ``date`` und ``symbol``.
    ``alpha`` ist im Wide-Format (Index: Datum, Spalten: Symbole) und stammt
    aus dem Alpha-Modell. Beide werden auf denselben Datums-/Symbolraster
    ausgerichtet, damit das Environment rein positionsbasiert zugreifen kann.
    """

    required = {"date", "symbol", price_column, return_column}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten im Feature-Frame: {sorted(missing)}")

    close_wide = features.pivot(index="date", columns="symbol", values=price_column).sort_index()
    return_wide = features.pivot(index="date", columns="symbol", values=return_column).sort_index()

    if close_wide.isna().any().any():
        raise ValueError("Kursdaten enthalten Lücken; alle Symbole müssen über den Zeitraum ausgerichtet sein.")

    symbols = tuple(str(symbol) for symbol in close_wide.columns)
    alpha_wide = alpha.reindex(index=close_wide.index, columns=list(close_wide.columns)).fillna(0.0)
    return_wide = return_wide.fillna(0.0)

    # Rollierende Volatilität als Point-in-Time-Risikoschätzung je Aktie.
    volatility = return_wide.rolling(window=vol_window, min_periods=1).std().fillna(0.0)

    return MarketPanel(
        dates=close_wide.index.to_numpy(),
        symbols=symbols,
        close=close_wide.to_numpy(dtype=np.float64),
        returns=return_wide.to_numpy(dtype=np.float64),
        alpha=alpha_wide.to_numpy(dtype=np.float64),
        volatility=volatility.to_numpy(dtype=np.float64),
    )
