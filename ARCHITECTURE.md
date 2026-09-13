# Hybrid RL Architecture for Quantitative Trading

## 1. Overview

Das System trennt **Prognose**, **Handelsentscheidung** und **Positionsgröße**:

- Das **Alpha-Modell** prognostiziert die Attraktivität einer Aktie.
- Das **Risk Adjustment** setzt das Signal ins Verhältnis zum Risiko.
- Das **RL-Environment** kombiniert eine Alpha-Regel mit den Residualaktionen
        des PPO-Agenten.
- Das **Position Sizing** übersetzt Aktion und Signalstärke in eine Trade-Größe.
- **Portfolio Constraints** begrenzen die resultierenden Positionen und Trades.

```text
Market + Fundamental Data → Feature Engineering → Alpha Model
→ Alpha Scores → Alpha Rule → PPO Residual Policy (Down/Keep/Up)
→ finale Aktion (Sell/Hold/Buy)
→ Position Sizing → Portfolio Constraints → Rebalancing
→ New Portfolio → Return − Transaction Costs → Reward
```

## 2. Data & Features

Verwendet werden Marktdaten wie Preise, Returns, Volumen, Volatilität, Momentum und Markt-/Indexinformationen sowie Fundamentaldaten wie Wachstum, Margen, ROE/ROIC, Verschuldung, P/E, EV/EBITDA und weitere Accounting- oder Bewertungskennzahlen.

Alle Daten und daraus berechneten Features müssen Point-in-Time korrekt sein: Zum Zeitpunkt `t` dürfen nur tatsächlich veröffentlichte und verfügbare Informationen eingehen. Das gilt besonders für Fundamentaldaten und verhindert Look-Ahead Bias. Das Feature Engineering berechnet und normalisiert daraus unter anderem Returns verschiedener Horizonte, Momentum, Volatilität und fundamentale Kennzahlen (je nach dem was Edgar hergibt).

## 3. Alpha Model

Als erste Implementierung wird XGBoost als überwachtes Alpha-Modell verwendet. Das Modell prognostiziert für jede Aktie eine zukünftige Rendite, die anschließend als Alpha-Score dient. LightGBM kann später als Vergleichsmodell untersucht werden.

### Input & Output

Das Alpha-Modell wird für jede Aktie `i` und jeden Entscheidungszeitpunkt `t` angewendet. Sein Input ist ein Feature-Vektor:

```math
\mathbf{x}_{i,t}
=
[
\text{Market Features}_{i,t},
\text{Fundamental Features}_{i,t},
\text{Context Features}_t
]
```

Dabei steht $\mathbf{x}_{i,t}$ für alle zum Zeitpunkt `t` verfügbaren Informationen zur Aktie `i`. Dazu gehören beispielsweise vergangene Returns, Momentum, Volatilität, Volumen, relative Performance, Wachstum, Margen, ROE/ROIC, Verschuldung und Bewertungskennzahlen. Context Features können zusätzlich Markt-, Sektor- oder Regimeinformationen enthalten.

Alle Features müssen **Point-in-Time korrekt** sein: Es dürfen nur Informationen verwendet werden, die zum Zeitpunkt `t` tatsächlich bekannt waren.

Das Trainingsziel $y_{i,t}$ ist beispielsweise die nach `H` Handelstagen realisierte Rendite:

```math
y_{i,t}
=
r_{i,t\rightarrow t+H}
=
\frac{P_{i,t+H}}{P_{i,t}}-1
```

Dabei bezeichnet:

- $P_{i,t}$ den Preis der Aktie `i` zum Entscheidungszeitpunkt `t`
- $P_{i,t+H}$ den Preis nach `H` Handelstagen
- `H` den Prognosehorizont
- $y_{i,t}$ die tatsächliche zukünftige Rendite, die im Training als Zielwert dient

Alternativ kann eine markt- oder sektorbereinigte Rendite verwendet werden:

```math
y_{i,t}^{active}
=
r_{i,t\rightarrow t+H}
-
r_{benchmark,t\rightarrow t+H}
```


Hier beschreibt $y_{i,t}^{active}$ die Rendite der Aktie abzüglich der Rendite eines Benchmarks im gleichen Zeitraum. Das Modell prognostiziert dann nicht die absolute Entwicklung, sondern die erwartete Out- oder Underperformance.


XGBoost lernt aus den historischen Trainingsdaten die Abbildung:

```math
f_\theta(\mathbf{x}_{i,t})
\rightarrow
\hat r_{i,t+H}
```

Dabei ist $f_\theta$ das Alpha-Modell mit den gelernten Modellparametern $\theta$. Es verarbeitet den Feature-Vektor $x_{i,t}$ und prognostiziert mit $\hat r_{i,t+H}$ die Rendite der Aktie über die nächsten `H` Handelstage.

Der Modelloutput wird als Alpha-Score verwendet:

```math
\alpha_{i,t}
=
\hat r_{i,t+H}
```

Damit gilt:

- $\alpha_{i,t}>0$: erwartete positive Entwicklung
- $\alpha_{i,t}<0$: erwartete negative Entwicklung
- $|\alpha_{i,t}|$: Stärke der Prognose

Für einen Rebalancing-Zeitpunkt werden die Prognosen aller `N` betrachteten Aktien zu einem Alpha-Vektor zusammengefasst:

```math
\boldsymbol{\alpha}_t
=
[\alpha_{1,t},\alpha_{2,t},\ldots,\alpha_{N,t}]
```

Dabei bezeichnet `N` die Anzahl der Aktien im aktuellen Universum. Der Alpha-Vektor enthält somit für jede Aktie genau einen prognostizierten Alpha-Score. Er wird anschließend für das Risk Adjustment und als Bestandteil des States des RL-Agenten verwendet.

Ein positives bzw. negatives Alpha signalisiert eine erwartete positive bzw. negative Entwicklung; der Betrag beschreibt die Signalstärke.

## 4. Risk Estimation & Risk-Adjusted Alpha

Als erste Risikoschätzung dient die historische rollierende Volatilität $\sigma_{i,t}$. Sie wird für jede Aktie `i` zum Zeitpunkt `t` aus den vergangenen Returns eines festgelegten Zeitfensters berechnet, beispielsweise aus den letzten 20 oder 60 Handelstagen.

Alpha und Risiko werden zu einem signierten risikoadjustierten Signal
kombiniert:

```math
z_{i,t}
=
\frac{\alpha_{i,t}}{\sigma_{i,t}}
```

Dabei bezeichnet:

- $\alpha_{i,t}$ den Alpha-Score der Aktie `i` zum Zeitpunkt `t`
- $\sigma_{i,t}$ die geschätzte Volatilität bzw. das Risiko der Aktie
- $z_{i,t}$ das signierte risikoadjustierte Signal

Für das Position Sizing wird zusätzlich die richtungsunabhängige
risikoadjustierte Opportunity verwendet:

```math
q_{i,t}
=
\frac{|\alpha_{i,t}|/(\sigma_{i,t}+\varepsilon)}
{\sum_{j=1}^{N}|\alpha_{j,t}|/(\sigma_{j,t}+\varepsilon)}
```

Ein starkes Alpha bei geringer Volatilität erzeugt somit eine höhere
Opportunity als ein gleich starkes Alpha bei hohem Risiko.

Anschließend wird $z_{i,t}$ über alle `N` Aktien des zum Zeitpunkt `t` betrachteten Universums normalisiert:

```math
q_{i,t}
=
\frac{|z_{i,t}|}{\sum_{j=1}^{N}|z_{j,t}|}
```

Der Index `j` iteriert dabei über alle Aktien des aktuellen Universums. Der Nenner ist die Summe ihrer risikoadjustierten Signalstärken. Dadurch gilt:

```math
\sum_{i=1}^{N}q_{i,t}=1
```

$q_{i,t}$ beschreibt den relativen Anteil der risikoadjustierten Opportunity
einer Aktie und dient dem Position Sizing. In der Implementierung wird
$\varepsilon$ direkt zur Volatilität addiert, damit eine Volatilität von null
nicht zu einer Division durch null führt.

## 5. RL Environment

Als aktuelle Implementierung wird PPO mit einem Multi-Discrete Action Space
verwendet. Der Agent verarbeitet die von XGBoost erzeugten Alpha-Scores,
Risiko-, Markt- und Portfolioinformationen sowie die aktuelle Alpha-Regel.
Die Fundamentaldaten werden dem PPO-Agenten nicht direkt übergeben, sondern
wirken über die Alpha-Scores des vorgelagerten Alpha-Modells.

Das RL Environment verbindet das mit XGBoost implementierte Alpha-Modell mit der Portfolio-Simulation. Zu jedem Rebalancing-Zeitpunkt `t` erhält der Agent einen State, wählt Aktionen und bekommt nach deren Ausführung einen Reward.

### State Space

Der implementierte Basis-State enthält pro Aktie Alpha, die normalisierte
risikoadjustierte Opportunity $q$, das signierte risikoadjustierte Alpha, das
aktuelle Aktiengewicht, den Return und die Haltedauer. Zusätzlich werden Cash
und bei der Residual-Umgebung die aktuelle Regelaktion angehängt:

```math
s_t=
[
\boldsymbol{\alpha}_t,
\mathbf{q}_t,
\mathbf{z}_t,
\mathbf{w}_{t-1},
cash_t,
\mathbf{r}_t,
\mathbf{h}_t,
\mathbf{rule}_t
]
```

Dabei bezeichnet:

- $\boldsymbol{\alpha}_t$: von XGBoost erzeugte Alpha-Scores für alle `N` Aktien
- $\mathbf{q}_t$: normalisierte risikoadjustierte Opportunity je Aktie
- $\mathbf{z}_t$: signiertes Alpha geteilt durch die Volatilität
- $\mathbf{w}_{t-1}$: aktuelle Portfolio-Gewichte einschließlich Cash
- $cash_t$: aktueller Cash-Anteil
- $\mathbf{r}_t$: aktueller Return je Aktie
- $\mathbf{h}_t$: Haltedauer je Aktie
- $\mathbf{rule}_t$: aktuelle Alpha-Regelaktion im Residual-Setup

Der Basis-State hat die Dimension $6N+1$. Im Residual-Setup wird die
Regelaktion angehängt, sodass die Beobachtung die Dimension $7N+1$ besitzt.
Die Beobachtungen werden mit `VecNormalize` normalisiert.

### Action Space

Im direkten Environment entspricht die diskrete Aktion für jede Aktie `i`:

```math
a_{i,t}\in\{-1,0,+1\},
\qquad
-1=SELL,\;0=HOLD,\;+1=BUY
```

In der aktuell für Training und Evaluation verwendeten Residual-Umgebung
gibt PPO stattdessen pro Aktie eine Residualaktion aus:

```text
0 = Down, 1 = Keep, 2 = Up
```

`Keep` übernimmt die Alpha-Regel. `Down` bzw. `Up` verschieben die
Regelaktion um einen Schritt in Richtung Sell bzw. Buy. Erst daraus entsteht
die tatsächlich ausgeführte finale Aktion `Sell`, `Hold` oder `Buy`.

Über alle `N` Aktien entsteht jeweils ein Aktionsvektor:

```math
\mathbf{a}_t=
[a_{1,t},a_{2,t},\ldots,a_{N,t}]
```

Der Agent entscheidet damit über die Handelsrichtung, nicht über die
Positionsgröße. Diese wird anschließend durch die separate Position-Sizing-
Regel berechnet.

### Implementation Approach

XGBoost und der PPO-Agent werden getrennt trainiert. Zuerst wird XGBoost auf historischen Features und zukünftigen Renditen trainiert. Seine Point-in-Time-Prognosen werden anschließend zusammen mit Risiko-, Markt- und Portfolioinformationen als Input des RL-Agenten verwendet.

Die PPO-Policy erzeugt für jede Aktie drei Wahrscheinlichkeiten für Down,
Keep und Up. Die Residual-Umgebung kombiniert die gewählte Residualaktion mit
der Alpha-Regel und übergibt die finale Sell/Hold/Buy-Aktion an das Trading-
Environment. Dieses berechnet die Positionsänderungen, wendet die Portfolio
Constraints an und bestimmt neue Portfolio-Gewichte, Transaktionskosten,
Portfoliorendite und Reward.

Eine gemeinsame Klassifikation aller Aktionskombinationen wird vermieden, da bei `N` Aktien bereits $3^N$ Kombinationen entstehen. Stattdessen verwendet PPO eine separate diskrete Aktionsverteilung pro Aktie. Geeignete DQN-Varianten können später als Vergleich untersucht werden.

## 6. Risk-Adjusted Position Sizing

Die Position-Sizing-Regel übersetzt die ausgeführte finale Handelsrichtung in
eine konkrete Veränderung des Portfolio-Gewichts:

```math
\Delta w_{i,t}
=
a_{i,t}\cdot B_t\cdot q_{i,t}
```

Dabei bezeichnet:

- $\Delta w_{i,t}$ die Veränderung des Portfolio-Gewichts der Aktie `i` zum Zeitpunkt `t`
- $a_{i,t}\in\{-1,0,+1\}$ die finale Aktion Sell, Hold oder Buy
- $B_t$ das maximal verfügbare Rebalancing-Budget
- $q_{i,t}$ die in **Abschnitt 4 „Risk Estimation & Risk-Adjusted Alpha“** berechnete relative risikoadjustierte Signalstärke

Ein Buy bei einem hohen $q_{i,t}$ führt zu einem größeren Kauf als bei einem schwachen Signal. Ein Sell erzeugt entsprechend eine negative Positionsänderung. Für Hold gilt $\Delta w_{i,t}=0$; die bestehende Position bleibt unverändert.

## 7. Portfolio Constraints

Nach dem Position Sizing werden zunächst die vorgeschlagenen Zielgewichte berechnet:

```math
w_{i,t}^{target}
=
w_{i,t-1}+\Delta w_{i,t}
```

Dabei ist $w_{i,t-1}$ das bisherige Portfolio-Gewicht und $\Delta w_{i,t}$ die in Abschnitt 6 berechnete Positionsänderung. Vor der Ausführung werden diese Zielgewichte auf einen zulässigen Portfolioraum begrenzt.

Für eine erste **Long-only-Implementierung** gilt beispielsweise:

```math
0\leq w_{i,t}\leq w_{max},
\qquad
\sum_i w_{i,t}\leq1
```

Damit kann keine Aktie negativ gewichtet werden, das Gewicht einer einzelnen Aktie ist auf $w_{max}$ begrenzt und das Portfolio investiert insgesamt nicht mehr als 100 % des verfügbaren Kapitals. Weitere Regeln können das Rebalancing-Budget begrenzen und Leverage ausschließen. Nicht investiertes Kapital wird als Cash gehalten.

Die Constraints bilden somit die Sicherheitsschicht zwischen den vom Modell vorgeschlagenen Positionsänderungen und den tatsächlich ausgeführten Trades.

## 8. Rebalancing, Costs & Reward

Nach Anwendung der Portfolio Constraints werden die zulässigen Zielgewichte durch Käufe und Verkäufe umgesetzt. Der dabei entstehende Turnover misst die gesamte tatsächlich ausgeführte Portfolioveränderung:

```math
Turnover_t
=
\sum_{i=1}^{N}
|w_{i,t}-w_{i,t-1}|
```

Dabei sind $w_{i,t-1}$ und $w_{i,t}$ die Portfolio-Gewichte vor und nach dem Rebalancing. Über `i` wird dabei über alle `N` Aktien des Portfolios iteriert. Die Transaktionskosten $C_t$ können zunächst proportional zum Turnover modelliert werden:

```math
C_t=c_{TC}\cdot Turnover_t
```

Dabei bezeichnet $c_{TC}$ den angenommenen Kostensatz. Der Reward ergibt sich anschließend aus der Portfoliorendite nach Berücksichtigung dieser Kosten:

```math
R_{t+1}
=
r^{portfolio}_{t+1}
-
\lambda_{TC}C_t
```

$r^{portfolio}_{t+1}$ ist die nach dem Rebalancing erzielte Rendite und $\lambda_{TC}$ steuert, wie stark Transaktionskosten im Reward gewichtet werden. Dadurch wird der Agent für Rendite belohnt und gleichzeitig von häufigem oder unnötigem Trading abgehalten.

Weitere Risikoterme können später experimentell ergänzt werden. Sharpe Ratio, Maximum Drawdown, Volatilität, Turnover, Transaktionskosten sowie Gesamt- und annualisierte Rendite dienen zunächst primär der Evaluation.

## 9. Baselines & Evaluation

Die primäre Evaluation vergleicht zwei separat trainierte Residual-PPO-
Systeme:

1. **`market_only`**: Alpha-Modell mit marktbezogenen Merkmalen
2. **`full_alpha`**: Alpha-Modell mit Markt- und Fundamentaldaten

Beide PPO-Systeme verwenden denselben chronologischen Testsplit, dieselbe
Environment-Konfiguration und einen deterministischen Rollout. Die PPO-
Rollouts und die Referenzportfolios werden auf demselben Return-Fenster mit
91 Perioden ausgewertet. Als wirtschaftliche Referenzen werden **Buy-and-
Hold** und **Equal Weight** berichtet. Sie verwenden zwar dasselbe
Return-Fenster, aber nicht die PPO-konsistente Cash-Ausgangslage und keine
PPO-Environment-Transaktionskosten.

Alpha-Ranking, direkte Alpha-Regelportfolios und historische Regel-Baselines
sind ergänzende Diagnostik und nicht der primäre PPO-Vergleich.

Die zentrale Forschungsfrage lautet:

> Welchen zusätzlichen Nutzen liefern Fundamentaldaten gegenüber reinen
> Marktdaten für datengetriebene Handelsentscheidungen in einem hybriden
> System aus Alpha-Prognose und Reinforcement Learning?

## 10. Component Responsibilities

```text
Data / Features      Informationen über Unternehmen und Markt
        ↓
Alpha Model          Wie attraktiv ist die Aktie?
        ↓
Risk Adjustment      Wie stark ist das Signal relativ zum Risiko?
        ↓
Alpha Rule           Baseline-Handelsrichtung
        ↓
RL Agent             Residual Down, Keep oder Up
        ↓
Final Action         Sell, Hold oder Buy
        ↓
Position Sizing      Wie groß soll der Trade sein?
        ↓
Portfolio Layer      Ist der Trade innerhalb der Regeln möglich?
        ↓
Rebalancing          Portfolio Return − Costs → Reward
```

Die Kernidee ist die klare Trennung der Verantwortlichkeiten:

- **Alpha Model:** prognostiziert die zukünftige Attraktivität eines Assets.
- **Risk Adjustment:** berücksichtigt das aktuelle Risiko.
- **RL Agent:** modifiziert die Alpha-Regel über Residualaktionen.
- **Final Action:** ergibt die tatsächlich ausgeführte Sell/Hold/Buy-Richtung.
- **Position Sizing:** berechnet daraus konkrete Positionsänderungen.
- **Portfolio Layer:** setzt Constraints durch und bestimmt die tatsächlich ausführbaren Trades.

## 11. Open Design Decisions

Vor oder während der Implementierung sind noch festzulegen bzw. experimentell zu untersuchen:

- Aktienuniversum und Prognosehorizont `H`
- Rebalancing-Frequenz
- Features und Alpha-Modell, insbesondere XGBoost vs. LightGBM
- Volatilitätsberechnung und Normalisierung des Risk-Adjusted Alpha
- maximales Aktiengewicht `w_max` und Rebalancing-Budget `B_t`
- Modellierung des Multi-Asset-Action-Spaces
- PPO vs. DQN bzw. weitere RL-Algorithmen
- genaue Reward Function und Transaktionskosten
- Umgang mit Cash
- Long-only vs. Short Selling