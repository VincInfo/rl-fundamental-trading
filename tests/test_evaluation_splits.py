import pandas as pd
import pytest
from data_pipeline import DataSplit

from eval.data_splits.loader import summarize_splits, validate_splits


def _splits(
    *,
    validation_start: str = "2024-01-04",
    test_start: str = "2024-01-07",
) -> dict[DataSplit, pd.DataFrame]:
    columns = pd.Index(["Close", "roe"])
    ranges = {
        DataSplit.TRAIN: pd.date_range("2024-01-01", periods=3, freq="D"),
        DataSplit.VALIDATION: pd.date_range(validation_start, periods=3, freq="D"),
        DataSplit.TEST: pd.date_range(test_start, periods=3, freq="D"),
    }
    return {
        split: pd.DataFrame(1.0, index=index, columns=columns)
        for split, index in ranges.items()
    }


def test_validate_splits_accepts_chronological_provider_splits():
    validate_splits(_splits())


def test_validate_splits_rejects_overlapping_periods():
    with pytest.raises(ValueError, match="overlaps"):
        validate_splits(_splits(test_start="2024-01-06"))


def test_validate_splits_rejects_missing_split():
    splits = _splits()
    del splits[DataSplit.TEST]

    with pytest.raises(ValueError, match="missing"):
        validate_splits(splits)


def test_summarize_splits_returns_date_and_shape_metadata():
    summaries = summarize_splits(_splits())
    train = summaries[DataSplit.TRAIN]

    assert train.start == pd.Timestamp("2024-01-01")
    assert train.end == pd.Timestamp("2024-01-03")
    assert train.n_rows == 3
    assert train.n_columns == 2