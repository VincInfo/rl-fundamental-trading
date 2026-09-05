from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from data_pipeline import DataSplit, DataVariant, get_data


@dataclass(frozen=True)
class SplitSummary:
    """Compact metadata for one chronologically split data frame"""

    split: DataSplit
    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int
    n_columns: int


def load_evaluation_splits(
    variant: DataVariant = DataVariant.WITH_FUNDAMENTALS,
) -> dict[DataSplit, pd.DataFrame]:
    """Load and validate all chronological splits from the data provider"""
    splits = get_data(variant)
    validate_splits(splits)
    return {
        DataSplit.TRAIN: splits[DataSplit.TRAIN],
        DataSplit.VALIDATION: splits[DataSplit.VALIDATION],
        DataSplit.TEST: splits[DataSplit.TEST],
    }


def validate_splits(splits: dict[DataSplit, pd.DataFrame]) -> None:
    """Validate split presence, temporal ordering, and compatible schemas"""
    expected_splits = set(DataSplit)
    if set(splits) != expected_splits:
        missing = expected_splits - set(splits)
        unexpected = set(splits) - expected_splits
        raise ValueError(
            f"Expected splits {sorted(expected_splits, key=str)}; "
            f"missing={sorted(missing, key=str)}, "
            f"unexpected={sorted(unexpected, key=str)}"
        )

    previous_split: DataSplit | None = None
    previous_end: pd.Timestamp | None = None
    expected_columns: pd.Index | None = None

    for split in (DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST):
        frame = splits[split]
        if frame.empty:
            raise ValueError(f"Split {split} is empty.")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"Split {split} must use a DatetimeIndex.")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"Split {split} index must be sorted ascending.")
        if frame.index.has_duplicates:
            raise ValueError(f"Split {split} contains duplicate timestamps.")
        if expected_columns is None:
            expected_columns = frame.columns
        elif not frame.columns.equals(expected_columns):
            raise ValueError(f"Split {split} has a different column schema.")

        start = frame.index.min()
        end = frame.index.max()
        if previous_end is not None and start <= previous_end:
            raise ValueError(
                f"Split {split} overlaps with {previous_split}: "
                f"{start} <= {previous_end}."
            )
        previous_split = split
        previous_end = end


def summarize_splits(
    splits: dict[DataSplit, pd.DataFrame],
) -> dict[DataSplit, SplitSummary]:
    """Return reproducible date and shape metadata for validated splits"""
    validate_splits(splits)
    return {
        split: SplitSummary(
            split=split,
            start=frame.index.min(),
            end=frame.index.max(),
            n_rows=len(frame),
            n_columns=len(frame.columns),
        )
        for split, frame in splits.items()
    }