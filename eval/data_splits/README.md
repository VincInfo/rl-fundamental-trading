# Evaluationsdaten und Data Splits

Die Daten werden aus dem Repository
[`rl-ss26-market-data-pipeline`](https://gitlab.lrz.de/daniel.maier/rl-ss26-market-data-pipeline)
geladen. Der Datenzulieferer stellt die aufeinanderfolgenden Splits
`Training`, `Validation` und `Test` bereit.

## Verwendete Daten

Aktuell wird die Datenvariante `WITH_FUNDAMENTALS` verwendet. Sie enthält
Marktdaten, technische Indikatoren und Fundamentaldaten für das konfigurierte
Aktienuniversum.

Das Aktienuniversum umfasst 20 Unternehmen:

```text
AAPL, MSFT, NVDA, IBM, CSCO, TXN, AMAT, CRM, MU, LRCX,
TSLA, LOW, TGT, PEP, COST, MDLZ, JNJ, LLY, ISRG, ADP
```

Die Daten werden als Wide-Format mit einem `DatetimeIndex` und MultiIndex-
Spalten geliefert. Im zuletzt überprüften Datenstand enthält jeder Split
320 Spalten.

Der im Projekt festgelegte Commit des Datenzulieferers ist in der
`pyproject.toml` hinterlegt. Dadurch kann nachvollzogen werden, aus welchem
Stand die verwendeten Daten stammen.

## Aufteilung der Daten

Die Daten werden chronologisch und ohne zufälliges Durchmischen aufgeteilt:

```text
Vergangenheit --------------------------------------------------> Zukunft

        Training              Validation              Test
    Modelltraining       Modellwahl/Checks       finale Evaluation
```

### Training

Der Trainingssplit wird zum Trainieren des Alpha-Modells und des PPO-Agenten
verwendet. Informationen aus Validation oder Test dürfen nicht in das Training
eingehen.

### Validation

Der Validierungssplit wird verwendet, um Modellvarianten, Hyperparameter und
Implementierungsdetails zu prüfen. Ergebnisse auf diesem Split dürfen zur
Modellauswahl verwendet werden.

### Test

Der Testsplit wird erst nach Abschluss der Modellwahl für die finale Evaluation
verwendet. Er darf nicht verwendet werden, um nachträglich Hyperparameter,
Modellvarianten oder Strategieparameter auszuwählen.

## Aktuell verifizierter Datenstand

Die folgenden Werte wurden mit `DataVariant.WITH_FUNDAMENTALS` aus dem
Datenzulieferer ausgelesen:

| Split | Start | Ende | Zeilen | Spalten |
|---|---|---|---:|---:|
| Training | 2024-09-19 14:30:00 -04:00 | 2025-11-06 11:30:00 -05:00 | 1974 | 320 |
| Validation | 2025-11-06 12:30:00 -05:00 | 2026-03-27 15:30:00 -04:00 | 658 | 320 |
| Test | 2026-03-30 09:30:00 -04:00 | 2026-08-12 15:30:00 -04:00 | 658 | 320 |

Die angegebenen Zeiträume und Größen beziehen sich auf den zuletzt überprüften
Datenstand. Sie müssen nach einer Aktualisierung des Datenzulieferers erneut
geprüft werden und dürfen nicht dauerhaft als unveränderliche Werte verstanden
werden.

## Technische Validierung

Die zentrale Lade- und Validierungslogik befindet sich in
[`loader.py`](loader.py).

Beim Laden werden folgende Eigenschaften geprüft:

- alle drei erwarteten Splits sind vorhanden,
- kein Split ist leer,
- jeder Split besitzt einen aufsteigend sortierten `DatetimeIndex`,
- innerhalb eines Splits gibt es keine doppelten Zeitstempel,
- die Spaltenschemata aller Splits sind identisch,
- die Splits überschneiden sich zeitlich nicht,
- die Reihenfolge ist `Training` vor `Validation` vor `Test`

Die Split-Übersicht kann mit folgendem Befehl ausgegeben werden:

```bash
uv run python eval/data_splits/report.py
```

Die Ausgabe enthält pro Split den Anfangs- und Endzeitpunkt sowie die Anzahl
von Zeilen und Spalten.

Die automatisierten Tests befinden sich in
[`tests/test_evaluation_splits.py`](../../tests/test_evaluation_splits.py).
Sie prüfen insbesondere gültige chronologische Splits, fehlende Splits und
zeitliche Überschneidungen.

## Point-in-Time-Korrektheit

Die chronologische Aufteilung allein garantiert noch keine vollständige
Point-in-Time-Korrektheit. Für die spätere fachliche Evaluation muss zusätzlich sichergestellt werden, dass für einen Entscheidungspunkt nur Informationen verwendet werden, die zu diesem Zeitpunkt bekannt waren.

Das betrifft insbesondere:

- den Veröffentlichungszeitpunkt von Fundamentaldaten,
- die zeitliche Verwendung von Preisen und Returns,
- die Berechnung von Features ohne zukünftige Informationen,
- die Normalisierung von Features ohne Testdaten,
- die Auswahl von Modellvarianten ausschließlich anhand von Training und
  Validation

Diese Prüfung gehört zur späteren Modell- und Portfolioevaluation. Dieser
Ordner dokumentiert zunächst die vom Datenzulieferer übernommenen Splits und
ihre technische Validierung.

## Aktualisierung

Nach einer Änderung des Datenzulieferer-Commits müssen die Split-Informationen neu ausgelesen werden:

```bash
uv run python eval/data_splits/report.py
```
