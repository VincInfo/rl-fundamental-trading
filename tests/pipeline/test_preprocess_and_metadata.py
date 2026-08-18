from __future__ import annotations
"""Tests für Train-only-Skalierung und persistente Split-Metadaten"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.api import (
    save_temporal_split_metadata,
    save_walk_forward_split_metadata,
    scale_split_features_train_only,
    split_features_by_time,
    build_walk_forward_feature_splits,
)


class TestPreprocessAndMetadata(unittest.TestCase):
    def _make_features(self) -> pd.DataFrame:
        # Hilfsdatensatz mit monotonen Zahlen, damit Skalierung reproduzierbar prüfbar ist
        rows = []
        for day, date in enumerate(pd.date_range("2024-01-01", periods=9, freq="D"), start=1):
            rows.append(
                {
                    "date": date,
                    "symbol": "AAPL",
                    "close": float(day),
                    "volume": float(day * 10),
                    "is_report_day": 1 if day == 5 else 0,
                }
            )
        return pd.DataFrame(rows)

    def test_scale_split_features_train_only(self) -> None:
        # Arrange: Erst strikt zeitlich splitten, dann Skalierung nur auf Train fitten
        features = self._make_features()
        split = split_features_by_time(
            features=features,
            train_end="2024-01-04",
            val_end="2024-01-06",
            test_end="2024-01-09",
        )

        scaled, scaler = scale_split_features_train_only(split, feature_columns=["close", "volume"])

        train_close = scaled.train["close"]
        train_volume = scaled.train["volume"]

        # Assert: Train muss auf Mittelwert 0 und Standardabweichung 1 normiert sein
        self.assertAlmostEqual(float(train_close.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(train_volume.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(train_close.std(ddof=0)), 1.0, places=7)
        self.assertAlmostEqual(float(train_volume.std(ddof=0)), 1.0, places=7)

        # Für Val/Test wird derselbe Train-Fit verwendet (kein Refit)
        self.assertAlmostEqual(scaler.means["close"], 2.5)
        self.assertAlmostEqual(scaler.stds["close"], (1.25) ** 0.5)

        expected_val_first_close = (5.0 - scaler.means["close"]) / scaler.stds["close"]
        self.assertAlmostEqual(float(scaled.val.iloc[0]["close"]), expected_val_first_close, places=7)

    def test_save_split_metadata_files(self) -> None:
        # Arrange: Zeit-Split und Walk-Forward-Fenster erzeugen
        features = self._make_features()
        split = split_features_by_time(
            features=features,
            train_end="2024-01-04",
            val_end="2024-01-06",
            test_end="2024-01-09",
        )
        windows = build_walk_forward_feature_splits(
            features=features,
            min_train_periods=4,
            val_periods=2,
            test_periods=2,
            step_periods=1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            temporal_path = save_temporal_split_metadata(
                split=split,
                output_path=tmp_path / "temporal_split_metadata.json",
                extra={"run_id": "demo"},
            )
            walk_path = save_walk_forward_split_metadata(
                splits=windows,
                output_path=tmp_path / "walk_forward_metadata.json",
            )

            temporal_payload = json.loads(Path(temporal_path).read_text(encoding="utf-8"))
            walk_payload = json.loads(Path(walk_path).read_text(encoding="utf-8"))

            # Assert: JSON enthält Typ, Zeilenstatistik und optionale Zusatzinfos
            self.assertEqual(temporal_payload["split_type"], "temporal")
            self.assertEqual(temporal_payload["datasets"]["train"]["rows"], 4)
            self.assertEqual(temporal_payload["extra"]["run_id"], "demo")

            self.assertEqual(walk_payload["split_type"], "walk_forward")
            self.assertGreaterEqual(walk_payload["window_count"], 1)
            self.assertIn("boundaries", walk_payload["windows"][0])


if __name__ == "__main__":
    unittest.main()
