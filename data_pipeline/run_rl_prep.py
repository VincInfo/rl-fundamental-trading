from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from data_pipeline.api import (
    build_dataset,
    build_walk_forward_feature_splits,
    save_temporal_split_metadata,
    save_walk_forward_split_metadata,
    scale_split_features_train_only,
    split_features_by_time,
)
from data_pipeline.run_mvp import resolve_sec_user_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt RL-fertige Datensplits mit Train-only-Skalierung und "
            "persistenten Split-Metadaten"
        )
    )
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"], help="Ticker-Symbole")
    parser.add_argument("--start", default="2022-01-01", help="Startdatum (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="Enddatum (YYYY-MM-DD)")

    parser.add_argument("--train-end", default="2023-06-30", help="Ende Train-Fenster (YYYY-MM-DD)")
    parser.add_argument("--val-end", default="2023-12-31", help="Ende Validierungsfenster (YYYY-MM-DD)")
    parser.add_argument("--test-end", default="2024-12-31", help="Ende Testfenster (YYYY-MM-DD)")

    parser.add_argument("--wf-min-train-periods", type=int, default=252, help="Minimale Handelstage im Train-Fenster")
    parser.add_argument("--wf-val-periods", type=int, default=63, help="Handelstage im Validierungsfenster")
    parser.add_argument("--wf-test-periods", type=int, default=63, help="Handelstage im Testfenster")
    parser.add_argument("--wf-step-periods", type=int, default=21, help="Schrittweite zwischen Walk-Forward-Fenstern")

    parser.add_argument(
        "--scale-feature-columns",
        nargs="*",
        default=None,
        help="Optionale Liste numerischer Feature-Spalten für Skalierung; ohne Angabe automatische Ableitung",
    )
    parser.add_argument(
        "--exclude-scale-columns",
        nargs="*",
        default=[],
        help="Spalten, die bei automatischer Skalierung ausgeschlossen werden",
    )

    parser.add_argument("--output-dir", default="artifacts/rl_prepared", help="Basis-Ausgabeverzeichnis")
    parser.add_argument("--run-name", default="rl_baseline", help="Präfix für den Laufordner")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    sec_user_agent = resolve_sec_user_agent(project_root)
    if not sec_user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT ist nicht gesetzt. Nutze eine der folgenden Optionen:\n"
            "1) export SEC_USER_AGENT='Ihr Name ihre.email@example.com'\n"
            "2) .env.example nach .env kopieren und SEC_USER_AGENT dort setzen"
        )

    result = build_dataset(
        symbols=[s.upper() for s in args.symbols],
        start=args.start,
        end=args.end,
        sec_user_agent=sec_user_agent,
    )

    split = split_features_by_time(
        features=result.features,
        train_end=args.train_end,
        val_end=args.val_end,
        test_end=args.test_end,
    )

    scaled_split, scaler = scale_split_features_train_only(
        split=split,
        feature_columns=args.scale_feature_columns,
        exclude_columns=set(args.exclude_scale_columns),
    )

    walk_forward = build_walk_forward_feature_splits(
        features=result.features,
        min_train_periods=args.wf_min_train_periods,
        val_periods=args.wf_val_periods,
        test_periods=args.wf_test_periods,
        step_periods=args.wf_step_periods,
    )

    run_dir = Path(args.output_dir) / f"{args.run_name}_{_timestamp()}"
    splits_dir = run_dir / "splits"
    raw_dir = run_dir / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Rohdaten und skalierte Splits getrennt speichern, damit Training reproduzierbar bleibt.
    result.prices.to_csv(raw_dir / "prices_daily.csv", index=False)
    result.fundamentals.to_csv(raw_dir / "fundamentals_quarterly.csv", index=False)
    result.features.to_csv(raw_dir / "features_daily.csv", index=False)

    scaled_split.train.to_csv(splits_dir / "features_train_scaled.csv", index=False)
    scaled_split.val.to_csv(splits_dir / "features_val_scaled.csv", index=False)
    scaled_split.test.to_csv(splits_dir / "features_test_scaled.csv", index=False)

    save_temporal_split_metadata(
        split=scaled_split,
        output_path=splits_dir / "temporal_split_metadata.json",
        scaler=scaler,
        extra={
            "run_name": args.run_name,
            "symbols": [s.upper() for s in args.symbols],
            "start": args.start,
            "end": args.end,
            "train_end": args.train_end,
            "val_end": args.val_end,
            "test_end": args.test_end,
        },
    )
    save_walk_forward_split_metadata(
        splits=walk_forward,
        output_path=splits_dir / "walk_forward_metadata.json",
        extra={
            "run_name": args.run_name,
            "wf_min_train_periods": args.wf_min_train_periods,
            "wf_val_periods": args.wf_val_periods,
            "wf_test_periods": args.wf_test_periods,
            "wf_step_periods": args.wf_step_periods,
        },
    )

    print(f"RL-Run-Verzeichnis: {run_dir}")
    print(f"Gespeicherte Rohdaten: {raw_dir}")
    print(f"Gespeicherte Splits: {splits_dir}")
    print(
        "Zeilen Train/Val/Test (skaliert): "
        f"{len(scaled_split.train)}/{len(scaled_split.val)}/{len(scaled_split.test)}"
    )
    print(f"Anzahl Walk-Forward-Fenster: {len(walk_forward)}")


if __name__ == "__main__":
    main()
