from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_features(
    n_stocks: int = 5,
    n_days: int = 750,
    seed: int = 42,
    start: str = "2019-01-01",
) -> pd.DataFrame:
    """Erzeugt einen ausgerichteten Multi-Stock-Kursdatensatz im Long-Format.

    Dient als Platzhalter, bis die echte Datenpipeline (EDGAR / yfinance) steht.
    Jede Aktie kombiniert einen gemeinsamen Marktfaktor mit idiosynkratischem
    Rauschen, sodass Korrelationen und titelspezifische Bewegungen entstehen.
    """

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    symbols = [f"STK{i:02d}" for i in range(n_stocks)]

    market_return = rng.normal(0.0003, 0.008, size=n_days)

    rows: list[dict] = []
    for symbol in symbols:
        beta = rng.uniform(0.6, 1.3)
        idiosyncratic = rng.normal(0.0, 0.012, size=n_days)
        daily_return = np.clip(beta * market_return + idiosyncratic, -0.15, 0.15)
        close = rng.uniform(20.0, 200.0) * np.cumprod(1.0 + daily_return)
        for day, price, ret in zip(dates, close, daily_return, strict=True):
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "close": round(float(price), 4),
                    "return_1d": round(float(ret), 6),
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
