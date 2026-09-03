# Evaluation des hybriden Trading-Systems

Die Evaluation des Projekts wird in zwei miteinander verbundene Teile aufgeteilt.

Dabei wird zum einen die fachliche Trading-Performance betrachtet und zum anderen,
ob das Modell technisch sinnvoll trainiert und eingesetzt wird.

1. **Technische Modellevaluation:**

   Wie gut lernt das Alpha-Modell und wie verhält sich der PPO-Agent während und
   nach dem Training?

2. **Fachliche Trading Evaluation:**

   Führt das gelernte Verhalten auf ungesehenen Daten zu einer guten
   risikoadjustierten Portfolioentwicklung?

Die Evaluation betrachtet drei Ebenen getrennt:

1. **Prognosequalität:**

   Liefert das Alpha-Modell brauchbare Informationen über zukünftige Renditen
   beziehungsweise deren relative Rangordnung?

2. **Handels- und Portfolioqualität:**

   Wie entwickelt sich ein mit dem Agenten verwaltetes Portfolio unter
   realistischen Kosten- und Positionsannahmen?

3. **Erklärbarkeit und Robustheit:**

   Ist das beobachtete Ergebnis nachvollziehbar und stabil über mehrere
   Trainingsläufe?


## Ziel der Evaluation

Zentrale Forschungsfrage für die Evaluation:

> Liefert die Kombination aus Fundamentaldaten, Alpha-Modell und
> Reinforcement Learning einen zusätzlichen Nutzen gegenüber einfacheren
> Handelsstrategien?

Daraus ergeben sich folgende Teilfragen:

* Kann das Alpha-Modell zukünftige Aktienrenditen oder die relative
  Outperformance besser als eine triviale Baseline vorhersagen?
* Führt der PPO-Agent zu einer besseren risikoadjustierten Portfolioentwicklung
  als direkte Alpha- und passive Strategien?
* Welchen zusätzlichen Beitrag liefert die RL-Komponente gegenüber dem
  Alpha-Signal allein?
* Wie stabil sind die Ergebnisse über unterschiedliche Trainingsläufe?
* Sind die Handelsentscheidungen und die daraus resultierenden Positionen
  nachvollziehbar?

Ein positives Ergebnis bedeutet dabei nicht zwingend, dass der PPO-Agent die
höchste absolute Rendite erzielt. Für ein Handelssystem sind Rendite, Risiko,
Drawdown, Turnover und Transaktionskosten gemeinsam zu beurteilen.


## Zweiteilige Evaluationsstrategie

### Teil 1: Technische Evaluation des Modells

Dieser Teil fokussiert auf die Eigenschaften des vorgeschalteten Alpha-Modells
und des PPO-Agenten. Er beantwortet, ob die Modellkomponenten entsprechend ihrer
Aufgabe funktionieren.

Beim Alpha-Modell wird geprüft, ob die vorhergesagten Werte tatsächlich
Informationen über die zukünftige Rendite beziehungsweise deren relative
Rangordnung enthalten.

Beim PPO-Agenten hingegen wird geprüft, ob sich während des Trainings ein
sinnvolles Lernverhalten zeigt und ob die gelernte Policy auf einen neuen
Zeitraum übertragen werden kann.

Für die technische Evaluation werden insbesondere betrachtet:

* Entwicklung des Trainings-Rewards,
* Vergleich von Trainings- und Validation-Reward,
* Policy-Entropie während des Trainings,
* Verteilung der Aktionen `Buy`, `Hold` und `Sell`,
* Verhalten über mehrere Random Seeds.

**Noch nicht implementiert:** Momentan: zusammengefasste
Trainings- und Validation-Metriken gespeichert -> vollständiger zeitlicher
Verlauf des Rewards + der Policy-Entropie muss für die technische Evaluation
noch protokolliert werden!

Besonderes Augenmerk liegt auf der Beobachtung von Overfitting. Ein deutlich
besserer Reward im Training als in der Validation kann darauf hindeuten, dass
der Agent das begrenzte Trainingsfenster auswendig gelernt hat oder dass seine
Policy nicht auf einen neuen Zeitraum generalisiert.

Die technische Evaluation soll nicht versuchen, möglichst viele interne
PPO-Werte zu dokumentieren. Für das Projekt reichen die Werte aus, die für das
Verständnis des Lernverhaltens relevant sind. Dazu gehören vor allem der
Reward-Verlauf, die Entropie, die Aktionsverteilung und der Vergleich zwischen
Trainings- und Evaluationszeitraum.

Der gleiche Random Seed wird zusätzlich für die Reproduzierbarkeit verwendet.
Für eine einfache Robustheitsanalyse werden dagegen mehrere unterschiedliche
Seeds verwendet. Dafür reichen zunächst drei vorab festgelegte Seeds aus.

**Noch nicht implementiert:** Die Ausführung und Zusammenfassung mehrerer
Trainingsläufe muss noch ergänzt werden.


### Teil 2: Fachliche Evaluation der Trading-Strategie

Dieser Teil fokussiert auf das Ergebnis, das für einen Anleger oder eine
Trading-Strategie relevant ist. Der trainierte Agent wird auf bisher ungesehenen
Daten ausgeführt und mit einfachen Vergleichsstrategien verglichen.

Dabei werden insbesondere folgende Fragen beantwortet:

* Erwirtschaftet der Agent eine positive beziehungsweise bessere Rendite?
* Ist die Rendite im Verhältnis zum Risiko überzeugend?
* Wie hoch sind Drawdown, Turnover und Transaktionskosten?
* Ist die Strategie besser als eine passive oder direkte Alpha-Strategie?
* Bleibt das Verhalten über den gesamten Evaluationszeitraum plausibel?

Die fachliche Evaluation basiert auf täglichen Portfolio-Werten und nicht nur
auf dem PPO-Reward. Der Reward ist ein internes Lernsignal und kann wegen
Trade-Penalty, Transaktionskosten und der konkreten Skalierung von der
realisierten Portfolio-Rendite abweichen.

Für die finale Bewertung wird daher insbesondere die **Nettoperformance nach
Transaktionskosten** betrachtet.



### Verbindung der beiden Teile

Die finale Interpretation verbindet beide Evaluationsperspektiven:

```text
Alpha-Modell liefert Signal
          |
          v
PPO verarbeitet Signal und Portfoliozustand
          |
          v
Agent lernt eine Handels-Policy
          |
          v
Portfolio erzielt Rendite unter Kosten und Risiko
```

Ein zusätzlicher Mehrwert des PPO-Agenten ist nur dann plausibel, wenn die
Verbindung zwischen Signal, Policy und Portfolioergebnis nachvollziehbar ist.

Deshalb wird neben der reinen Performance auch untersucht, ob die Alpha-Scores
mit nachvollziehbaren Änderungen der Aktionen beziehungsweise Positionen
zusammenhängen.


## Aktueller Systemstand

Das Repository implementiert eine Pipeline mit folgenden Komponenten:

```text
Markt- und Fundamentaldaten
                |
                v
Feature Engineering
                |
                v
XGBoost-Alpha-Modell: Prognose zukünftiger Renditen
                |
                v
Alpha-Scores und Volatilität
                |
                v
PPO-Agent: Buy, Hold oder Sell je Aktie
                |
                v
Risikoadjustiertes Position Sizing
                |
                v
Portfolio-Constraints und Transaktionskosten
                |
                v
Portfolio-Return und Reward
```

Die Aktienauswahl umfasst 20 Unternehmen aus mehreren Sektoren. Das
Alpha-Modell wird mit XGBoost trainiert. Der PPO-Agent verwendet pro Aktie
diskrete Aktionen aus `Buy`, `Hold` und `Sell`.

Die Trading-Umgebung berücksichtigt unter anderem:

* ein maximales Gewicht pro Aktie,
* ein begrenztes Rebalancing-Budget,
* risikoadjustierte Positionsgrößen,
* Transaktionskosten,
* eine Mindesthaltedauer,
* eine zusätzliche Penalty für Turnover.


## Evaluationsdesign

### Zeitliche Aufteilung

Finanzdaten sind zeitabhängig. Deshalb dürfen Trainings- und Evaluationsdaten
nicht zufällig durchmischt werden. Die zeitliche Reihenfolge muss erhalten
bleiben:

```text
Vergangenheit --------------------------------------------------> Zukunft

             Training              Validation              Test

         Modelltraining       Modellwahl/Checks       finale Auswertung
```

Die Rollen der Splits sind:

* **Training:**

  Der Alpha- und PPO-Ansatz wird auf diesem Zeitraum trainiert.

* **Validation:**

  Hyperparameter, Modellvarianten und Implementierungsdetails werden auf diesem
  Zeitraum geprüft.

* **Test:**

  Dieser Zeitraum wird erst für die finale Auswertung verwendet und darf nicht
  zur Auswahl der besten Variante herangezogen werden.

Alle Strategien müssen auf exakt denselben Zeitpunkten, denselben Aktien und
demselben Datenstand ausgewertet werden. Ein Vergleich gilt nur als fair, wenn
keine Strategie einen längeren Zeitraum oder zusätzliche Informationen erhält.

**Aktueller Stand:** derzeit sind Training und Validation
vorhanden -> separater Test-Split + die finale Testauswertung sind noch
nicht implementiert!

Die finale Testauswertung wird erst durchgeführt, nachdem Modellvarianten und
Hyperparameter anhand von Training und Validation festgelegt wurden.


### Point-in-Time-Korrektheit

Für einen Entscheidungstag `t` dürfen nur Informationen verwendet werden, die
zu diesem Zeitpunkt tatsächlich bekannt waren.

Das ist bei Fundamentaldaten besonders wichtig: Ein Quartalsbericht darf erst
ab seinem Veröffentlichungszeitpunkt in Features eingehen, nicht bereits ab
dem Ende des Berichtsquartals.

Geprüft und dokumentiert werden daher:

* Veröffentlichungszeitpunkte der Fundamentaldaten,
* Verwendung von Preisen und Returns erst nach dem jeweiligen
  Entscheidungspunkt,
* keine Nutzung zukünftiger Zielwerte in den Features,
* keine Skalierung oder Normalisierung mit Informationen aus dem Testzeitraum,
* unveränderte Datenaufbereitung zwischen Baselines und PPO-Agent.

Ein Verstoß gegen diese Regeln führt zu Look-Ahead Bias und kann die
Performance massiv überschätzen.


### Einheitliche Portfolioannahmen

Alle Strategien werden mit denselben Annahmen simuliert:

* identisches Startkapital,
* identische Aktienauswahl,
* identische Handelstage,
* identische Transaktionskosten,
* identische Positionslimits,
* identische Behandlung von Cash,
* identische Rebalancing-Frequenz,
* keine Short-Positionen, sofern die Trading-Umgebung long-only bleibt.

Die Transaktionskosten werden sowohl für den PPO-Agenten als auch für alle
Baselines berücksichtigt.

Eine Strategie darf hierbei nicht mit Kosten und eine andere ohne Kosten
verglichen werden.


## Vergleichsstrategien

Der PPO-Agent wird gegen mehrere einfache Vergleichsstrategien evaluiert.
Dadurch kann unterschieden werden, ob ein möglicher Mehrwert bereits durch
Diversifikation oder das Alpha-Signal entsteht oder tatsächlich durch die
RL-Komponente.
### Equal Weight

Das verfügbare Kapital wird gleichmäßig auf alle Aktien verteilt.

Diese Strategie ist einfach, diversifiziert und ein wichtiger Referenzpunkt für
die Frage, ob der komplexe Modellansatz überhaupt einen Mehrwert liefert.

### Buy and Hold

Zu Beginn des Evaluationszeitraums wird ein Portfolio aufgebaut und
anschließend nicht mehr aktiv umgeschichtet.

Diese Strategie zeigt, ob aktives Handeln den zusätzlichen Turnover rechtfertigt.

### Alpha-Ranking

Die Aktien werden nach dem vorhergesagten Alpha sortiert. Ein festgelegter
Anteil der Aktien mit den höchsten Scores wird gekauft, die übrigen Aktien
werden nicht gehalten oder erhalten ein festgelegtes Basisgewicht.

Die Regeln werden vor der Auswertung definiert und dürfen nicht nachträglich
anhand des Testzeitraums angepasst werden.

Diese Strategie ist besonders wichtig, weil sie direkt prüft, ob der PPO-Agent
gegenüber einer einfachen Nutzung seines Alpha-Inputs einen zusätzlichen
Mehrwert erzeugt.

### PPO-Agent

Der trainierte PPO-Agent entscheidet je Aktie zwischen `Buy`, `Hold` und `Sell`.

Die Aktion wird durch das bestehende risikoadjustierte Position Sizing in eine
Positionsänderung übersetzt.

Für die finale Evaluation wird der Agent deterministisch ausgeführt, damit der
Testlauf reproduzierbar bleibt.

**Noch nicht implementiert:** Die bestehende Auswertung erfasst bereits
Aktionen und Rewards -> noch keine vollständige Equity-Kurve + nicht
alle Portfolio-Metriken in einer mit den Baselines vergleichbaren Form


## Einfluss des Alpha-Modells auf PPO

Das Alpha-Modell ist dem PPO-Agenten vorgeschaltet und liefert einen zentralen
Teil der Beobachtung. Daher muss separat untersucht werden, ob der Agent die
Alpha-Scores tatsächlich nutzt und ob diese Information einen Mehrwert liefert.

Dafür werden drei Varianten miteinander verglichen:

1. **PPO mit trainierten Alpha-Scores:**

   Der vollständige Ansatz verwendet die vom Alpha-Modell prognostizierten
   Scores.

2. **PPO mit neutralisierten Alpha-Scores:**

   Die Alpha-Komponente wird durch konstante beziehungsweise neutrale Werte
   ersetzt.

   Dadurch wird geprüft, wie sich der Agent ohne nutzbare Alpha-Information
   verhält.

3. **Direkte Alpha-Ranking-Strategie:**

   Die Alpha-Scores werden ohne PPO unmittelbar für die Portfolioauswahl
   verwendet.

Die zweite Variante ist als Ablation und nicht als realistische
Handelsstrategie zu verstehen. Sie dient dazu, den Beitrag der
Alpha-Information innerhalb des PPO-Ansatzes einzuordnen.

Für einen fairen Vergleich wird jede PPO-Variante separat trainiert. Wird ein
bereits trainierter Agent lediglich mit neutralisierten Alpha-Scores ausgeführt,
handelt es sich nur um eine zusätzliche Sensitivitätsanalyse, nicht um eine
vollständige Ablation.

Die direkte Alpha-Ranking-Strategie beantwortet zusätzlich die Frage, ob PPO
überhaupt einen Mehrwert gegenüber einer einfachen Nutzung des Alpha-Signals
erzeugt.

Für diese Analyse werden möglichst dieselben Portfolioannahmen verwendet.

Der Einfluss des Alpha-Modells wird damit in drei Stufen betrachtet:

```text
Qualität des Alpha-Signals
          |
          v
Nutzung des Signals durch PPO
          |
          v
Einfluss auf das Portfolioergebnis
```

Damit wird verhindert, dass ein positiver Effekt des Alpha-Modells automatisch
dem PPO-Agenten zugeschrieben wird oder dass eine schwache Trading-Performance
ohne Analyse als reines PPO-Problem interpretiert wird.

**Noch nicht implementiert:** neutrale PPO-Variante + die direkte
Alpha-Ranking-Strategie müssen noch umgesetzt + separat ausgewertet werden


## Zu berichtende Metriken

Die Ergebnisse jeder Strategie werden aus der täglichen
Portfolio-Value-Zeitreihe berechnet.

### Rendite

Die tägliche Portfoliorendite ist:

```text
r_t = V_t / V_(t-1) - 1
```

mit dem Portfolio-Wert `V_t` am Tag `t`.

Die kumulierte Rendite über den gesamten Zeitraum lautet:

```text
R_kumuliert = V_T / V_0 - 1
```

Für Vergleiche über unterschiedlich lange Zeiträume wird zusätzlich die
annualisierte Rendite berichtet.

Bei `N` Handelstagen ist eine übliche Definition:

```text
R_annualisiert = (V_T / V_0) ** (252 / N) - 1
```

Die Rendite wird klar als brutto oder netto gekennzeichnet. Für die
Hauptauswertung wird grundsätzlich die Nettorendite nach Transaktionskosten
verwendet.


### Risiko

Die annualisierte Volatilität wird aus den täglichen Renditen berechnet:

```text
Vol_annualisiert = std(r_t) * sqrt(252)
```

Die Sharpe Ratio setzt die annualisierte Überschussrendite ins Verhältnis zur
annualisierten Volatilität.

Falls kein risikofreier Zinssatz modelliert wird, wird `0` als Näherung
verwendet. Diese Annahme wird angegeben:

```text
Sharpe = mean(r_t) / std(r_t) * sqrt(252)
```

Die Sharpe Ratio ist bei sehr kurzen Zeiträumen oder nahezu null Volatilität
nicht stabil und wird dann entsprechend gekennzeichnet.


### Drawdown

Der Drawdown misst den Verlust gegenüber dem bisher erreichten Höchststand:

```text
Drawdown_t = V_t / max(V_0, ..., V_t) - 1
```

Der maximale Drawdown ist der kleinste Wert dieser Zeitreihe.

Er zeigt, wie stark ein Anleger zwischenzeitlich im Verlust gewesen wäre und
ist für die Präsentation meist aussagekräftiger als die Volatilität allein.


### Handelsaktivität

Zusätzlich zur Performance werden folgende Größen erfasst:

* durchschnittlicher täglicher Turnover,
* kumulierte Transaktionskosten,
* Anteil der Handelstage mit Rebalancing,
* durchschnittliche Anzahl gehaltener Positionen,
* durchschnittlicher Cash-Anteil,
* Buy-, Hold- und Sell-Anteile des PPO-Agenten.

Ein hoher Turnover kann eine gute Bruttorendite vollständig durch Kosten
aufzehren. Die Handelsaktivität ist daher Teil des Ergebnisses und nicht nur
ein rein technisches Detail.


## Ergebnistabelle

Die zentrale Evaluationstabelle enthält pro Strategie mindestens folgende
Spalten:

| Strategie     | Kumulierte Rendite | Annualisierte Rendite |  Volatilität | Sharpe Ratio | Max. Drawdown |     Turnover |
| ------------- | -----------------: | --------------------: | -----------: | -----------: | ------------: | -----------: |
| Equal Weight  |       aus Backtest |          aus Backtest | aus Backtest | aus Backtest |  aus Backtest | aus Backtest |
| Buy and Hold  |       aus Backtest |          aus Backtest | aus Backtest | aus Backtest |  aus Backtest | aus Backtest |
| Alpha-Ranking |       aus Backtest |          aus Backtest | aus Backtest | aus Backtest |  aus Backtest | aus Backtest |
| PPO-Agent     |       aus Backtest |          aus Backtest | aus Backtest | aus Backtest |  aus Backtest | aus Backtest |

Die Tabelle wird immer gemeinsam mit dem betrachteten Zeitraum, der Anzahl
der Aktien, den Kostenannahmen und dem Datenstand gezeigt.

Gerundete Prozentwerte ohne diese Angaben sind nicht ausreichend
reproduzierbar.


## Visualisierungen

Die finale Evaluation verwendet insbesondere folgende Visualisierungen:

1. **Equity-Kurve:**

   Entwicklung eines normalisierten Startwerts, zum Beispiel `100`, für alle
   Strategien im selben Diagramm.

2. **Drawdown-Kurve:**

   Zwischenzeitliche Verluste der Strategien über die Zeit.

3. **Metrik-Tabelle:**

   Rendite, Sharpe Ratio, maximaler Drawdown und Turnover.

4. **Aktionsverteilung:**

   Anteil von `Buy`, `Hold` und `Sell` beim PPO-Agenten.

Die Equity-Kurve ist hierbei die wichtigste Grafik, da sie unmittelbar zeigt,
ob und wann sich die Strategien auseinanderentwickeln.

Dabei wird nicht nur die beste Periode gezeigt, sondern der vollständige
Evaluationszeitraum.


## Alpha-Modell separat evaluieren

Die Qualität der Renditeprognose wird getrennt von der Qualität des
Portfolio-Managements betrachtet.

Ein gutes Alpha-Modell garantiert nicht automatisch einen guten PPO-Agenten,
und ein schwaches Alpha-Modell kann durch eine andere Strategiekomponente
teilweise kompensiert werden.

Für das Alpha-Modell werden insbesondere ausgewertet:

* Mean Squared Error (MSE),
* Information Coefficient (IC),
* Mittelwert und Streuung des IC über die Zeit,
* Qualität des Aktien-Rankings.

Der **IC ist dabei die zentrale Metrik**, da für die spätere Verwendung im
Trading insbesondere relevant ist, ob Aktien mit höheren Alpha-Scores
tatsächlich tendenziell höhere zukünftige Renditen erzielen.

Der IC soll pro Handelstag als Querschnittskorrelation über die Aktien berechnet
und anschließend über die Tage aggregiert werden.

Dadurch wird vermieden, dass ein einzelner Zeitraum oder eine ungleichmäßige
Anzahl von Beobachtungen das Ergebnis dominiert.

Der MSE wird ergänzend betrachtet, da er die absolute Prognosequalität
beschreibt. Für das anschließende Ranking ist jedoch die relative Ordnung der
Aktien wichtiger als eine perfekt kalibrierte Renditeprognose.

**Aktueller Stand:** bisherige Alpha-Evaluation berechnet Korrelation
über alle Validation-Samples -> tägliche querschnittliche IC muss noch implementiert werden


## Robustheit und Ablation

PPO ist ein stochastisches Lernverfahren. Ein einzelner Trainingslauf reicht
daher nur für eine erste Demonstration.

Für die Robustheitsanalyse werden mehrere Random Seeds verwendet. Für jede
zentrale Performance-Metrik werden anschließend Mittelwert und Streuung
berichtet.

Zum Beispiel:

```text
Sharpe Ratio: 0,62 +/- 0,18 über fünf Seeds
```

Dabei werden die Seeds vor der finalen Testauswertung festgelegt und nicht
nachträglich anhand der Testperformance ausgewählt.

Die wichtigsten Vergleiche und Ablationen konzentrieren sich auf die Frage,
welchen zusätzlichen Beitrag das Alpha-Signal und die RL-Komponente liefern.

Dazu werden insbesondere verglichen:

* PPO mit trainierten Alpha-Scores gegenüber PPO mit neutralisierten
   Alpha-Scores als Ablation,
* Alpha-Ranking gegenüber PPO als Vergleich der RL-Komponente mit der direkten
   Nutzung des Alpha-Signals.

Die Variante ohne nutzbare Alpha-Information ist die zentrale Ablation für
dieses Projekt. Die direkte Alpha-Ranking-Strategie ist die wichtigste
Vergleichsstrategie für den zusätzlichen Beitrag von PPO. Weitere Varianten wie
unterschiedliche Position-Sizing-Regeln oder eine Auswertung ohne
Transaktionskosten werden nicht als notwendiger Bestandteil der Evaluation
betrachtet.

**Noch nicht implementiert:** Mehrere Seeds und die beiden genannten
Ablationsvergleiche


## Reproduzierbarkeit

Jede finale Auswertung wird mit folgenden Informationen gespeichert:

* Git-Commit oder Versionsstand des Codes,
* verwendeter Datenstand und Pipeline-Commit,
* Trainings-, Validation- und Testzeitraum,
* Tickerliste,
* Alpha-Modellartefakt,
* PPO-Modellartefakt,
* Random Seed,
* PPO-Hyperparameter,
* Portfolio- und Kostenparameter,
* verwendeter Evaluationsbefehl.

Die Ergebnisse werden als strukturiertes Datenformat, in diesem Fall CSV,
gespeichert.

**Noch nicht implementiert:** einheitlicher Evaluationsbefehl, der die
Ergebnistabelle, täglichen Portfolio-Werte, Equity-Kurve, Drawdown-Kurve und
Aktionsstatistiken erzeugt, ergänzen

Zusätzlich werden die Equity-Kurve, die Drawdown-Kurve und die wichtigsten
Statistiken als Dateien abgelegt.

Ein idealer Evaluationslauf ist mit einem einzigen dokumentierten Befehl
reproduzierbar und erzeugt:

```text
1. die Performance-Tabelle,
2. die täglichen Portfolio-Werte,
3. die Equity-Kurve,
4. die Drawdown-Kurve,
5. die Aktions- und Turnover-Statistiken
```

Damit lässt sich sowohl die technische Funktionsweise des Systems als auch
der fachliche Mehrwert des hybriden Trading-Ansatzes nachvollziehbar
evaluieren.
