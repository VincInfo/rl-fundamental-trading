import pandas as pd
import pytest

from models.alpha.features import daily_rebalance_mask
from models.rl.features import MARKET_FEATURE_COLUMNS, build_market_features
from models.rl.panel import build_panel


def _hourly_wide_frame(n_days: int, tickers: list[str]) -> pd.DataFrame:
    days = pd.bdate_range("2024-01-02", periods=n_days, tz="America/New_York")
    timestamps = [
        day + pd.Timedelta(hours=hour, minutes=30)
        for day in days
        for hour in range(9, 16)
    ]
    index = pd.DatetimeIndex(timestamps, name="Datetime")
    columns = pd.MultiIndex.from_product(
        [["Close"], tickers],
        names=["Feature", "Ticker"],
    )
    values = pd.DataFrame(index=index, columns=columns, dtype=float)
    for ticker_index, ticker in enumerate(tickers):
        for day_index, day in enumerate(days):
            open_price = 100.0 + day_index + ticker_index * 10.0
            for hour in range(9, 16):
                timestamp = day + pd.Timedelta(hours=hour, minutes=30)
                values.loc[timestamp, ("Close", ticker)] = open_price + (hour - 9)
    return values


def test_build_market_features_samples_first_bar_and_daily_return():
    tickers = ["AAPL", "MSFT"]
    wide_frame = _hourly_wide_frame(n_days=5, tickers=tickers)

    features = build_market_features(wide_frame)

    assert list(features.columns) == list(MARKET_FEATURE_COLUMNS)
    assert set(features["symbol"]) == set(tickers)
    assert features["date"].nunique() == 4
    assert len(features) == 8

    first_rebalance = daily_rebalance_mask(wide_frame.index)[1]
    aapl = features[(features["symbol"] == "AAPL") & (features["date"] == first_rebalance)]
    assert float(aapl["close"].iloc[0]) == 101.0
    assert float(aapl["return_1d"].iloc[0]) == pytest.approx(0.01)


def test_build_market_features_requires_close():
    tickers = ["AAPL"]
    wide_frame = _hourly_wide_frame(n_days=3, tickers=tickers)
    wide_frame = wide_frame.rename(columns={"Close": "Open"}, level="Feature")

    with pytest.raises(ValueError, match="Close"):
        build_market_features(wide_frame)


def test_build_market_features_aligns_for_panel():
    wide_frame = _hourly_wide_frame(n_days=8, tickers=["AAPL", "MSFT"])
    features = build_market_features(wide_frame)
    dates = sorted(features["date"].unique())
    symbols = sorted(features["symbol"].unique())
    alpha = pd.DataFrame(0.0, index=pd.Index(dates, name="date"), columns=symbols)

    panel = build_panel(features, alpha, vol_window=3)

    assert panel.n_days == 7
    assert panel.n_stocks == 2
    assert panel.symbols == ("AAPL", "MSFT")
