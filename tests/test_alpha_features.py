import pandas as pd

from models.alpha.config import (
    ENGINEERED_FEATURE_COLUMNS,
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
) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01 09:30",
        periods=rows,
        freq="h",
        tz="America/New_York",
    )
    columns = pd.MultiIndex.from_product(
        [
            ["Close", "roe", "gross_margin", "debt_to_equity", "filing_lag_days"],
            tickers,
        ],
        names=["Feature", "Ticker"],
    )
    values = pd.DataFrame(index=index, columns=columns, dtype=float)
    for ticker in tickers:
        values[("Close", ticker)] = 100 + pd.Series(range(rows), index=index) * 0.1
        values[("roe", ticker)] = 0.20
        values[("gross_margin", ticker)] = 0.4
        values[("debt_to_equity", ticker)] = 0.5
        values[("filing_lag_days", ticker)] = 40
        if roe_jump_at is not None:
            values.loc[index[roe_jump_at]:, ("roe", ticker)] = 0.25
            values.loc[index[roe_jump_at]:, ("filing_lag_days", ticker)] = 0
    return values


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        horizon_trading_days=5,
        bars_per_trading_day=7,
        sample_daily=True,
    )


def test_build_dataset_produces_model_features_and_target():
    tickers = ["AAPL", "MSFT"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers)
    dataset = build_dataset(wide_frame, _training_config())

    assert list(dataset.columns) == list(MODEL_FEATURE_COLUMNS) + ["target_return"]
    assert not dataset.empty
    assert set(dataset.index.get_level_values("Ticker")) == set(tickers)


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


def test_vanilla_features_exclude_fundamentals():
    tickers = ["AAPL", "MSFT"]
    fundamental_frame = _synthetic_wide_frame(rows=300, tickers=tickers)
    close_columns = [("Close", ticker) for ticker in tickers]
    vanilla_frame = fundamental_frame.loc[:, close_columns]
    vanilla_frame.columns = pd.MultiIndex.from_tuples(
        close_columns,
        names=["Feature", "Ticker"],
    )

    dataset = build_dataset(vanilla_frame, _training_config())
    inference = build_inference_features(vanilla_frame, _training_config())

    assert list(dataset.columns) == list(ENGINEERED_FEATURE_COLUMNS) + [TARGET_COLUMN]
    assert list(inference.columns) == list(ENGINEERED_FEATURE_COLUMNS)
    assert not any(column.startswith("delta_") for column in dataset.columns)


def test_delta_roe_persists_after_filing_jump():
    tickers = ["AAPL"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers, roe_jump_at=250)
    dataset = build_dataset(wide_frame, _training_config())
    delta = dataset.xs("AAPL", level="Ticker")["delta_roe"]

    assert 0.0 in set(delta.round(10))
    assert any(delta.round(10) == 0.05)
    assert dataset.xs("AAPL", level="Ticker")["filing_lag_days"].eq(0).any()
