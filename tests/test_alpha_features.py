import pandas as pd

from models.alpha.config import MODEL_FEATURE_COLUMNS, TARGET_COLUMN, TrainingConfig
from models.alpha.features import build_dataset, build_inference_features


def _synthetic_wide_frame(rows: int, tickers: list[str]) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01 09:30",
        periods=rows,
        freq="h",
        tz="America/New_York",
    )
    columns = pd.MultiIndex.from_product(
        [
            ["Close", "roe", "gross_margin", "debt_to_equity"],
            tickers,
        ],
        names=["Feature", "Ticker"],
    )
    values = pd.DataFrame(index=index, columns=columns, dtype=float)
    for ticker in tickers:
        values[("Close", ticker)] = 100 + pd.Series(range(rows), index=index) * 0.1
        values[("roe", ticker)] = 0.2
        values[("gross_margin", ticker)] = 0.4
        values[("debt_to_equity", ticker)] = 0.5

    return values


def test_build_dataset_produces_model_features_and_target():
    tickers = ["AAPL", "MSFT"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers)
    config = TrainingConfig(
        horizon_trading_days=5,
        bars_per_trading_day=7,
        sample_daily=True,
    )

    dataset = build_dataset(wide_frame, config)

    assert list(dataset.columns) == list(MODEL_FEATURE_COLUMNS) + [config.target_column]
    assert not dataset.empty
    assert set(dataset.index.get_level_values("Ticker")) == set(tickers)


def test_build_inference_features_keeps_rows_without_target():
    tickers = ["AAPL", "MSFT"]
    wide_frame = _synthetic_wide_frame(rows=300, tickers=tickers)
    config = TrainingConfig(
        horizon_trading_days=5,
        bars_per_trading_day=7,
        sample_daily=True,
    )

    dataset = build_dataset(wide_frame, config)
    inference = build_inference_features(wide_frame, config)

    assert list(inference.columns) == list(MODEL_FEATURE_COLUMNS)
    assert TARGET_COLUMN not in inference.columns
    assert len(inference) > len(dataset)
    assert set(inference.index.get_level_values("Ticker")) == set(tickers)
