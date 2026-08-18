from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_pipeline.analysis.features import summarize_features
from data_pipeline.analysis.fundamentals import summarize_fundamentals
from data_pipeline.analysis.prices import summarize_prices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysiert Pipeline-Ausgaben für Kurse, Fundamentals und zusammengeführte Features.")
    parser.add_argument("--input-dir", default="artifacts/real_data_mvp", help="Verzeichnis mit prices_daily.csv, fundamentals_quarterly.csv und features_daily.csv")
    parser.add_argument("--output-dir", default="artifacts/real_data_mvp/analysis", help="Ausgabeverzeichnis für Analyseartefakte")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Standardisierte Pipeline-Artefakte laden
    prices = pd.read_csv(in_dir / "prices_daily.csv", parse_dates=["date"])
    fundamentals = pd.read_csv(in_dir / "fundamentals_quarterly.csv", parse_dates=["period_end", "report_date"])
    features = pd.read_csv(in_dir / "features_daily.csv", parse_dates=["date", "report_date"])

    # Teilanalysen je Datensatz berechnen
    prices_summary = summarize_prices(prices)
    fundamentals_summary = summarize_fundamentals(fundamentals)
    features_summary, coverage = summarize_features(features)

    # Gesamtzusammenfassung in ein JSON-Objekt überführen
    summary = {
        "prices": prices_summary,
        "fundamentals": fundamentals_summary,
        "features": features_summary,
    }

    summary_path = out_dir / "dataset_summary.json"
    coverage_path = out_dir / "features_coverage_by_symbol.csv"

    # Ergebnisse als JSON und CSV persistieren
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    coverage.to_csv(coverage_path, index=False)

    print(f"Gespeicherte Analyse-Zusammenfassung: {summary_path}")
    print(f"Gespeicherte Feature-Abdeckung: {coverage_path}")


if __name__ == "__main__":
    main()
