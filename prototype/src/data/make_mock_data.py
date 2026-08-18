from __future__ import annotations
"""Mockdaten-Generator für einen RL-Prototyp im Aktienhandel.

Pipeline erzeugt drei zusammenhängende Tabellen:
1) Quartals-Fundamentaldaten je Aktie
2) Tägliche Kursdaten je Aktie
3) Tägliche Features durch as-of Join auf die zuletzt verfügbaren Fundamentals

! Fundamentaldaten werden nur ab report_date sichtbar gemacht,
damit kein Look-Ahead-Bias entsteht !
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MockDataConfig:
    """Konfiguration für Zeitraum, Aktienuniversum & Reproduzierbarkeit"""

    start: str = "2019-01-01"
    end: str = "2024-12-31"
    symbols: tuple[str, ...] = ("ALFA", "BETA", "GAMM", "DELT", "EPSI")
    seed: int = 42


def _business_days(start: str, end: str) -> pd.DatetimeIndex:
    """Erzeugt alle Handelstage (Business Days) im Zeitraum"""

    return pd.date_range(start=start, end=end, freq="B")


def _quarterly_period_ends(start: str, end: str) -> pd.DatetimeIndex:
    """Liefert Quartalsperiodenenden (wirtschaftliche Periode, nicht Filing-Datum)."""

    quarter_ends = pd.date_range(start=start, end=end, freq="QE")
    return pd.DatetimeIndex([d if d.weekday() < 5 else d - pd.offsets.BDay(1) for d in quarter_ends])


def _next_business_day(day: pd.Timestamp) -> pd.Timestamp:
    """Verschiebt ein Datum auf den naechsten Handelstag, falls noetig."""

    if day.weekday() < 5:
        return day
    return day + pd.offsets.BDay(1)


def _simulate_fundamentals(
    symbols: tuple[str, ...], period_ends: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Simuliert quartalsweise Fundamentaldaten pro Aktie.

    Pro Symbol werden stabile Charakteristika (Qualitaet/Verschuldungsneigung)
    gezogen. Wirken sich auf Margen, ROE und EPS-Surprises aus
    """

    rows: list[dict] = []
    for symbol in symbols:
        base_revenue = rng.uniform(3_000, 25_000)
        quality = rng.normal(0.0, 1.0)
        debt_bias = rng.normal(0.0, 1.0)

        revenue = base_revenue
        revenue_history: list[float] = []
        for idx, period_end in enumerate(period_ends):
            yoy_growth = np.clip(rng.normal(0.04 + 0.015 * quality, 0.03), -0.15, 0.25)
            margin = np.clip(rng.normal(0.18 + 0.03 * quality, 0.04), 0.05, 0.45)
            roe = np.clip(rng.normal(0.10 + 0.04 * quality, 0.03), -0.05, 0.35)
            debt_to_equity = np.clip(rng.normal(1.0 + 0.3 * debt_bias, 0.25), 0.1, 3.5)

            revenue = revenue * (1.0 + yoy_growth / 4.0)
            net_income = revenue * np.clip(margin - 0.06 + rng.normal(0.0, 0.01), -0.15, 0.30)
            operating_cashflow = net_income * np.clip(rng.normal(1.15, 0.15), 0.7, 1.8)

            eps_estimate = np.clip(rng.normal(1.0 + quality * 0.25, 0.15), 0.15, 3.0)
            eps_surprise = np.clip(rng.normal(0.0 + quality * 0.03, 0.08), -0.30, 0.30)
            eps_actual = eps_estimate + eps_surprise

            # Filing-Lag imitiert den realen Abstand zwischen period_end und report_date.
            filing_lag_days = int(rng.integers(20, 46))
            report_date = _next_business_day(period_end + pd.Timedelta(days=filing_lag_days))
            fiscal_quarter = int((period_end.month - 1) / 3 + 1)
            report_type = "annual" if fiscal_quarter == 4 else "quarterly"
            revenue_yoy = (revenue / max(revenue_history[idx - 4], 1e-8)) - 1.0 if idx >= 4 else np.nan
            eps_surprise_pct = eps_surprise / max(abs(eps_estimate), 1e-8)

            rows.append(
                {
                    "symbol": symbol,
                    "period_end": period_end,
                    "report_date": report_date,
                    "fiscal_quarter": fiscal_quarter,
                    "report_type": report_type,
                    "filing_lag_days": filing_lag_days,
                    "revenue": round(revenue, 2),
                    "net_income": round(net_income, 2),
                    "operating_cashflow": round(operating_cashflow, 2),
                    "debt_to_equity": round(float(debt_to_equity), 3),
                    "gross_margin": round(float(margin), 3),
                    "net_margin": round(float(net_income / max(revenue, 1e-8)), 3),
                    "roe": round(float(roe), 3),
                    "revenue_yoy": round(float(revenue_yoy), 3) if not np.isnan(revenue_yoy) else np.nan,
                    "eps_estimate": round(float(eps_estimate), 3),
                    "eps_actual": round(float(eps_actual), 3),
                    "eps_surprise": round(float(eps_surprise), 3),
                    "eps_surprise_pct": round(float(eps_surprise_pct), 3),
                }
            )

            revenue_history.append(revenue)

    return pd.DataFrame(rows).sort_values(["symbol", "report_date"]).reset_index(drop=True)


def _event_alpha_series(
    dates: pd.DatetimeIndex, report_days: pd.Series, surprises: pd.Series, decay_days: int = 20
) -> np.ndarray:
    """Berechnet zeitlich abklingenden Eventeffekt nach Reports.

    Positive/negative EPS-Surprises beeinflussen Rendite für einige Tage
    nach report_date & nehmen dann exponentiell ab
    """

    alpha = np.zeros(len(dates), dtype=float)
    idx_by_date = {d: i for i, d in enumerate(dates)}

    for rep_day, surprise in zip(report_days, surprises, strict=True):
        if rep_day not in idx_by_date:
            continue
        start_idx = idx_by_date[rep_day]
        end_idx = min(start_idx + decay_days, len(dates))
        horizon = np.arange(end_idx - start_idx)
        alpha[start_idx:end_idx] += float(surprise) * np.exp(-horizon / 7.0)
    return alpha


def _simulate_prices(
    symbols: tuple[str, ...],
    dates: pd.DatetimeIndex,
    fundamentals_q: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simuliert tägliche OHLCV-Daten mit Markt-, Idiosynkrasie- und Eventanteil."""

    market_noise = rng.normal(0.0002, 0.008, size=len(dates))
    market_trend = np.cumsum(market_noise)
    market_factor = np.diff(np.insert(market_trend, 0, 0.0))

    rows: list[dict] = []
    for symbol in symbols:
        f = fundamentals_q[fundamentals_q["symbol"] == symbol].copy()
        quality_proxy = float(f["roe"].mean() - f["debt_to_equity"].mean() * 0.02)
        idio_noise = rng.normal(0.0001, 0.012, size=len(dates))
        event_alpha = _event_alpha_series(dates, f["report_date"], f["eps_surprise"], decay_days=20)

        # Renditekomponenten: Markt + idiosynkratisch + Unternehmensqualitaet + Eventeffekt.
        daily_ret = 0.12 * market_factor + idio_noise + quality_proxy * 0.0007 + event_alpha * 0.002
        daily_ret = np.clip(daily_ret, -0.15, 0.15)

        start_price = rng.uniform(20, 200)
        close = start_price * np.cumprod(1.0 + daily_ret)
        open_ = close / (1.0 + rng.normal(0.0, 0.002, size=len(dates)))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0005, 0.02, size=len(dates)))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0005, 0.02, size=len(dates)))
        volume = rng.integers(200_000, 2_500_000, size=len(dates))

        for i, day in enumerate(dates):
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "open": round(float(open_[i]), 2),
                    "high": round(float(high[i]), 2),
                    "low": round(float(low[i]), 2),
                    "close": round(float(close[i]), 2),
                    "volume": int(volume[i]),
                    "return_1d": round(float(daily_ret[i]), 6),
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


def _build_daily_features(prices: pd.DataFrame, fundamentals_q: pd.DataFrame) -> pd.DataFrame:
    """Baut den RL-Feature-Datensatz auf Tagesebene.

    As-of Join (direction='backward') stellt sicher, dass pro Tag nur die
    zuletzt veroeffentlichten Fundamentaldaten verfuegbar sind!
    """

    daily_frames: list[pd.DataFrame] = []
    for symbol, p in prices.groupby("symbol", sort=False):
        p = p.copy().sort_values("date")
        f = fundamentals_q[fundamentals_q["symbol"] == symbol].copy().sort_values("report_date")

        f_daily = f.drop(columns=["symbol"]).rename(columns={"report_date": "date"})
        merged = pd.merge_asof(
            p,
            f_daily,
            on="date",
            by=None,
            direction="backward",
        )

        merged["symbol"] = symbol
        merged = merged.rename(columns={"date": "trade_date", "report_date": "latest_report_date"})

        rep_dates = set(f["report_date"].tolist())
        merged["is_report_day"] = merged["trade_date"].isin(rep_dates).astype(int)

        # Tage seit letzter Veroeffentlichung als Event-Abstandsfeature.
        report_map = pd.Series(f["report_date"].values, index=f["report_date"].values)
        latest_report_dates = pd.Series(merged["trade_date"]).map(
            lambda d: report_map.index[report_map.index <= d].max() if (report_map.index <= d).any() else pd.NaT
        )
        latest_report_dates_dt = pd.to_datetime(latest_report_dates)
        trade_dates_dt = pd.to_datetime(merged["trade_date"])
        has_fundamentals = latest_report_dates_dt.notna().to_numpy()
        day_deltas = (trade_dates_dt - latest_report_dates_dt).dt.days.to_numpy()
        day_deltas = np.where(has_fundamentals, day_deltas, 999).astype(int)

        merged["days_since_report"] = day_deltas
        merged["has_fundamentals"] = has_fundamentals.astype(int)
        merged["report_date"] = latest_report_dates_dt.values

        daily_frames.append(merged)

    cols = [
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "period_end",
        "report_date",
        "fiscal_quarter",
        "report_type",
        "filing_lag_days",
        "revenue",
        "revenue_yoy",
        "net_income",
        "net_margin",
        "operating_cashflow",
        "debt_to_equity",
        "gross_margin",
        "roe",
        "eps_estimate",
        "eps_actual",
        "eps_surprise",
        "eps_surprise_pct",
        "is_report_day",
        "days_since_report",
        "has_fundamentals",
    ]
    out = pd.concat(daily_frames, ignore_index=True)
    out = out[cols].rename(columns={"trade_date": "date"})
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def generate_mock_dataset(config: MockDataConfig | None = None) -> dict[str, pd.DataFrame]:
    """Führt die gesamte Mockdaten-Pipeline aus und gibt alle Tabellen zurück"""

    config = config or MockDataConfig()
    rng = np.random.default_rng(config.seed)

    dates = _business_days(config.start, config.end)
    period_ends = _quarterly_period_ends(config.start, config.end)

    fundamentals_q = _simulate_fundamentals(config.symbols, period_ends, rng)
    prices = _simulate_prices(config.symbols, dates, fundamentals_q, rng)
    features_daily = _build_daily_features(prices, fundamentals_q)

    return {
        "prices": prices,
        "fundamentals_quarterly": fundamentals_q,
        "features_daily": features_daily,
    }


def save_mock_dataset(data: dict[str, pd.DataFrame], out_dir: str | Path = "mock_data") -> None:
    """Speichert alle erzeugten Tabellen als CSV in out_dir"""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, frame in data.items():
        frame.to_csv(out_path / f"{name}.csv", index=False)


def main() -> None:
    """CLI-Einstiegspunkt für lokale Generierung der Mockdaten"""

    dataset = generate_mock_dataset()
    save_mock_dataset(dataset)
    print("Mockdaten erstellt in ./mock_data")
    for key, df in dataset.items():
        print(f"{key}: {df.shape}")


if __name__ == "__main__":
    main()
