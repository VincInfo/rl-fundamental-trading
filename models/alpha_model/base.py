from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class AlphaModel(Protocol):
    """Schnittstelle zwischen Alpha-Modell und RL-Environment.

    Implementierungen prognostizieren je Aktie und Handelstag einen
    Point-in-Time-Alpha-Score. Der RL-Agent konsumiert diese Scores als
    Bestandteil seines States, trainiert sie aber nicht selbst.
    """

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Gibt ein Wide-Format zurück: Index ``date``, eine Spalte je Symbol."""
        ...


class RandomAlphaModel:
    """Platzhalter-Alpha-Modell mit reproduzierbaren Zufalls-Scores.

    Dient als stabile Schnittstelle, bis das überwachte Alpha-Modell (XGBoost)
    verfügbar ist, damit Environment und PPO-Training unabhängig entwickelt
    werden können.
    """

    def __init__(self, scale: float = 0.02, seed: int | None = None) -> None:
        self._scale = float(scale)
        self._rng = np.random.default_rng(seed)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = {"date", "symbol"} - set(features.columns)
        if missing:
            raise ValueError(f"Fehlende Spalten im Feature-Frame: {sorted(missing)}")

        dates = np.sort(features["date"].unique())
        symbols = sorted(str(symbol) for symbol in features["symbol"].unique())
        scores = self._rng.normal(0.0, self._scale, size=(len(dates), len(symbols)))
        return pd.DataFrame(scores, index=pd.Index(dates, name="date"), columns=symbols)
