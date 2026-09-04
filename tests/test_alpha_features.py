import pandas as pd
import pytest

from models.alpha.config import (
    MARKET_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TrainingConfig,
)
from models.alpha.features import build_dataset, build_inference_features


def _synthetic_wide_frame(
    rows: int,
    tickers: list[str],
    *,
    roe_jump_at: int | None = None,
    close_scale: dict[str, float] | None = None,
) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01 09:30",
        periods=rows,
        freq="h",
        tz="America/New_York",
    )
    columns = pd.MultiIndex.from_product(
        [
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "roe",
                "gross_margin",
                "debt_to_equity",
                "revenue",
                "net_income",
                "operating_cashflow",
                "filing_lag_days",
            ],
            tickers,
        ],
        names=["Feature", "Ticker"],
    )
    values = pd.DataFrame(index=index, columns=columns, dtype=float)
    scales = close_scale or {ticker: 0.1 for ticker in tickers}
    for offset, ticker in enumerate(tickers):
        close = 100 + pd.Series(range(rows), index=index) * scales[ticker]
        values[("Close", ticker)] = close
        values[("Open", ticker)] = close * 0.999
        values[("High", ticker)] = close * 1.002
        values[("Low", ticker)] = close * 0.998
        values[("Volume", ticker)] = 1_000_000 + pd.Series(range(rows), index=index) * 10
        values[("roe", ticker)] = 0.20 + 0.01 * offset
        values[("gross_margin", ticker)] = 0.4 + 0.05 * offset
        values[("debt_to_equity", ticker)] = 0.5 + 0.1 * offset
        values[("revenue", ticker)] = 1_000.0 + 100 * offset
        values[("net_income", ticker)] = 100.0 + 10 * offset
        values[("operating_cashflow", ticker)] = 120.0 + 10 * offset
        values[("filing_lag_days", ticker)] = 40
        if roe_jump_at is not None:
            values.loc[index[roe_jump_at]:, ("roe", ticker)] = 0.25
            values.loc[index[roe_jump_at]:, ("filing_lag_days", ticker)] = 0
            values.loc[index[roe_jump_at]:, ("revenue", ticker)] = (
                values[("revenue", ticker)].iloc[0] * 1.1
            )
    return values


def _training_config(*, include_fundamentals: bool = True) -> TrainingConfig:
    return TrainingConfig(
        horizon_trading_days=5,
        bars_per_trading_day=7,
        sample_daily=True,
        include_fundamentals=include_fundamentals,
        active_return=True,
    )


def test_build_dataset_produces_model_features_and_target():
    tickers = ["AAPL", "MSFT"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers)
    dataset = build_dataset(wide_frame, _training_config())

    assert list(dataset.columns) == list(MODEL_FEATURE_COLUMNS) + ["target_return"]
    assert not dataset.empty
    assert set(dataset.index.get_level_values("Ticker")) == set(tickers)
    assert {"filing_recency", "post_filing_5d", "rank_roe", "delta_revenue"} <= set(
        dataset.columns
    )


def test_build_dataset_market_only_excludes_fundamentals():
    tickers = ["AAPL", "MSFT"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers)
    dataset = build_dataset(wide_frame, _training_config(include_fundamentals=False))

    assert list(dataset.columns) == list(MARKET_FEATURE_COLUMNS) + ["target_return"]
    assert "roe" not in dataset.columns
    assert "delta_roe" not in dataset.columns
    assert {"oc_return", "hl_range", "volume_change_1d", "rel_volume_20d"} <= set(
        dataset.columns
    )
    assert not dataset.empty
    assert dataset[["oc_return", "hl_range", "rel_volume_20d"]].notna().all().all()


def test_build_dataset_requires_vanilla_ohlcv():
    tickers = ["AAPL"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers).drop(
        columns="Volume", level="Feature"
    )
    with pytest.raises(ValueError, match="Missing VANILLA OHLCV"):
        build_dataset(wide_frame, _training_config(include_fundamentals=False))


def test_build_inference_features_keeps_rows_without_target():
    tickers = ["AAPL", "MSFT"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers)
    config = _training_config()

    dataset = build_dataset(wide_frame, config)
    inference = build_inference_features(wide_frame, config)

    assert list(inference.columns) == list(MODEL_FEATURE_COLUMNS)
    assert TARGET_COLUMN not in inference.columns
    assert len(inference) > len(dataset)
    assert set(inference.index.get_level_values("Ticker")) == set(tickers)


def test_delta_roe_persists_after_filing_jump():
    tickers = ["AAPL"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers, roe_jump_at=250)
    dataset = build_dataset(wide_frame, _training_config())
    delta = dataset.xs("AAPL", level="Ticker")["delta_roe"]

    assert 0.0 in set(delta.round(10))
    assert any(delta.round(10) == 0.05)
    assert dataset.xs("AAPL", level="Ticker")["filing_lag_days"].eq(0).any()
    assert dataset.xs("AAPL", level="Ticker")["post_filing_5d"].eq(1.0).any()


def test_active_return_is_cross_sectionally_centered():
    tickers = ["AAPL", "MSFT"]
    wide_frame = _synthetic_wide_frame(
        rows=400,
        tickers=tickers,
        close_scale={"AAPL": 0.05, "MSFT": 0.20},
    )
    dataset = build_dataset(wide_frame, _training_config(include_fundamentals=False))
    means = dataset["target_return"].groupby(level="Datetime").mean()
    assert means.abs().max() < 1e-10
