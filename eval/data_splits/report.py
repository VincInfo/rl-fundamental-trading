"""Print a concise report of the configured chronological evaluation splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.data_splits.loader import load_evaluation_splits, summarize_splits


def main() -> None:
    summaries = summarize_splits(load_evaluation_splits())
    for summary in summaries.values():
        print(
            f"{summary.split.value}: {summary.start} -> {summary.end} | "
            f"rows={summary.n_rows} | columns={summary.n_columns}"
        )


if __name__ == "__main__":
    main()