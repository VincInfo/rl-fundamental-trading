# Datenpipeline: Aktueller Stand und Mechaniken

Stand: 2026-08-18

## 1) Ziel der Pipeline

Die Pipeline bereitet reale Kurs- und Fundamentaldaten so auf, dass sie:

- zeitlich korrekt sind (kein Look-Ahead)
- für RL-Training direkt nutzbar sind
- reproduzierbar und überprüfbar gespeichert werden

## 2) Architektur (grob)

1. Datenquellen laden
- Kurse über yfinance
- Fundamentals über SEC EDGAR Companyfacts

2. Normalisieren und validieren
- Einheitliches Schema für Kurse und Fundamentals
- Pflichtspalten, Duplikate und zeitliche Konsistenz prüfen

3. Feature Engineering
- As-of-Join (backward) je Symbol
- Ableitung von Report- und Verfügbarkeitsfeatures

4. RL-Vorbereitung
- Strikter Zeit-Split oder Walk-Forward-Splits
- Skalierung nur mit Train-Fit
- Metadaten pro Run speichern

5. Analyse
- Zusammenfassungen und Coverage-Tabellen exportieren

## 3) Datenfluss von Ende zu Ende

### A) Basislauf (run_mvp)

- Einstieg: data_pipeline/run_mvp.py
- Kern-API: data_pipeline/api.py -> build_dataset(...)

Ablauf:

1. load_prices(...)
2. load_fundamentals(...)
3. validate_prices(...), validate_fundamentals(...)
4. build_features_daily(...)
5. validate_features(...)
6. CSV-Ausgabe

Typische Outputs:

- prices_daily.csv
- fundamentals_quarterly.csv
- features_daily.csv

### B) RL-Lauf (run_rl_prep)

- Einstieg: data_pipeline/run_rl_prep.py

Zusätzlicher Ablauf nach build_dataset(...):

1. split_features_by_time(...) für Train/Val/Test
2. scale_split_features_train_only(...) für Train-only-Scaling
3. build_walk_forward_feature_splits(...)
4. save_temporal_split_metadata(...)
5. save_walk_forward_split_metadata(...)
6. Speicherung in datiertem Run-Ordner

Typische Outputs:

- raw/prices_daily.csv
- raw/fundamentals_quarterly.csv
- raw/features_daily.csv
- splits/features_train_scaled.csv
- splits/features_val_scaled.csv
- splits/features_test_scaled.csv
- splits/temporal_split_metadata.json
- splits/walk_forward_metadata.json

## 4) Quellen und Aufbereitung im Detail

### 4.1 Kurse (prices_yfinance)

Eingang:

- tägliche OHLCV-Daten

Aufbereitung:

- MultiIndex-Spalten flatten (falls nötig)
- auf internes Schema mappen: date, symbol, open, high, low, close, volume
- Datumsformat normalisieren
- Duplikate pro symbol/date entfernen (keep=last)

### 4.2 Fundamentals (fundamentals_edgar)

Eingang:

- SEC-Tickerliste (Ticker -> CIK)
- Companyfacts pro CIK

Aufbereitung:

- nur relevante Tags und sinnvolle Units aufnehmen
- nur 10-Q und 10-K berücksichtigen
- period_end auf Zielhorizont filtern
- report_date mit tolerierter Nachlaufgrenze filtern
- pro symbol/period_end/report_type nur erstes Filing behalten

## 5) Feature Engineering: Was ist enthalten?

### 5.1 Fundamentale Kernspalten

- revenue
- net_income
- operating_cashflow
- debt_to_equity
- gross_margin
- roe
- filing_lag_days

### 5.2 Merge-Mechanik

- Join-Typ: merge_asof
- Richtung: backward
- Schlüssel: Kursdatum gegen report_date

Bedeutung:

- pro Handelstag werden nur bereits veröffentlichte Fundamentals verwendet

### 5.3 Zusätzliche Tages-Features

- is_report_day: 1 wenn date == report_date, sonst 0
- days_since_report: Tage seit letztem verfügbaren Report (Fallback 999 bei fehlendem Report)
- has_fundamentals: 1 wenn report_date vorhanden, sonst 0

## 6) Anti-Leakage-Mechaniken

Die Pipeline enthält mehrere Schutzschichten:

1. As-of-Join mit backward
- verhindert Nutzung zukünftiger Reports im Merge

2. Validierung der Fundamentals
- report_date darf nicht vor period_end liegen

3. Validierung der Features
- report_date darf nicht nach date liegen

4. Skalierung
- Fit ausschließlich auf Train
- identische Parameter für Val und Test

## 7) Split-Strategien

### 7.1 Strikter Zeit-Split

- Train: bis train_end
- Val: (train_end, val_end]
- Test: (val_end, test_end] oder alles nach val_end

### 7.2 Walk-Forward

- expandierendes Train-Fenster
- feste Fenster für Val/Test
- Schrittweite über step_periods

Ziel:

- robustere Bewertung über mehrere Marktphasen

## 8) Reproduzierbarkeit

Reproduzierbarkeit wird über Metadaten abgesichert:

- Split-Grenzen
- Datumsabdeckung je Teilmenge
- Zeilenzahlen je Teilmenge
- Parameter des Runs
- (bei Temporal-Split) Scaler-Metadaten

## 9) Analyse-Outputs

Über data_pipeline/analysis/run_analysis.py:

- dataset_summary.json
- features_coverage_by_symbol.csv

Diese Artefakte geben schnelle Antworten auf:

- Datenabdeckung und Zeitraum
- fehlende Werte in Kernkennzahlen
- potenzielle Look-Ahead-Verletzungen
- Fundamentals-Abdeckung je Symbol

## 10) Relevante API-Funktionen

Zentrale Funktionen in data_pipeline/api.py:

- build_dataset(...)
- split_features_by_time(...)
- build_walk_forward_feature_splits(...)
- scale_split_features_train_only(...)
- save_temporal_split_metadata(...)
- save_walk_forward_split_metadata(...)

## 11) Aktueller Reifegrad

Der aktuelle Stand ist ein belastbares MVP+ für Forschungszwecke:

- zeitlich korrekte Datenaufbereitung vorhanden
- RL-orientierte Splits und Skalierung vorhanden
- reproduzierbare Metadaten vorhanden
- automatisierte Tests vorhanden

## 12) Bekannte Grenzen / nächste sinnvolle Schritte

1. Universumshistorie erweitern
- Survivorship-Bias weiter reduzieren (z. B. Delistings)

2. Qualitätschecks vertiefen
- zusätzliche Plausibilitätschecks für Extremwerte und Ausreißer

3. Produktionsnähe erhöhen
- optional robustere Preis-/Fundamentaldatenquellen

4. Enge Kopplung zur RL-Umgebung
- standardisierte Übergabeformate für Observation/Action/Reward-Pipeline

Wichtig:

- Trade-Kosten, Slippage, Positionsgrenzen und Risiko-Penalties gehören in die RL-Umgebung bzw. Reward-Logik, nicht in die Rohdatenaufbereitung.
