# Hybrid RL Architecture for Quantitative Trading

## 1. Overview

Das System trennt **Prognose**, **Handelsentscheidung** und **Positionsgröße**:

- Das **Alpha-Modell** prognostiziert die Attraktivität einer Aktie.
- Das **Risk Adjustment** setzt das Signal ins Verhältnis zum Risiko.
- Der **RL-Agent** entscheidet je Aktie zwischen **Buy, Hold und Sell**.
- Das **Position Sizing** übersetzt Aktion und Signalstärke in eine Trade-Größe.
- **Portfolio Constraints** begrenzen die resultierenden Positionen und Trades.

```text
Market + Fundamental Data → Feature Engineering → Alpha Model
→ Alpha Scores → Risk Adjustment → RL Agent (Buy/Hold/Sell)
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

Dabei ist $f_\theta$ das Alpha-Modell mit den gelernten Modellparametern $\theta$. Es verarbeitet den Feature-Vektor $\mathbf{x}_{i,t}$ und prognostiziert mit $\hat r_{i,t+H}$ die Rendite der Aktie über die nächsten `H` Handelstage.

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

Alpha und Risiko werden zu einem risikoadjustierten Signal kombiniert:

```math
z_{i,t}
=
\frac{|\alpha_{i,t}|}{\sigma_{i,t}}
```

Dabei bezeichnet:

- $\alpha_{i,t}$ den Alpha-Score der Aktie `i` zum Zeitpunkt `t`
- $|\alpha_{i,t}|$ die Stärke des Signals unabhängig von seiner Richtung
- $\sigma_{i,t}$ die geschätzte Volatilität bzw. das Risiko der Aktie
- $z_{i,t}$ die risikoadjustierte Signalstärke

Ein starkes Alpha bei geringer Volatilität erzeugt somit einen höheren Wert als ein gleich starkes Alpha bei hohem Risiko.

Anschließend wird $z_{i,t}$ über alle `N` Aktien des zum Zeitpunkt `t` betrachteten Universums normalisiert:

```math
q_{i,t}
=
\frac{z_{i,t}}{\sum_{j=1}^{N}z_{j,t}}
```

Der Index `j` iteriert dabei über alle Aktien des aktuellen Universums. Der Nenner ist die Summe ihrer risikoadjustierten Signalstärken. Dadurch gilt:

```math
\sum_{i=1}^{N}q_{i,t}=1
```

$q_{i,t}$ beschreibt den relativen Anteil der risikoadjustierten Opportunity einer Aktie und dient später dem Position Sizing. Für die Implementierung sollte im Nenner von $z_{i,t}$ zusätzlich ein kleiner Wert $\varepsilon$ berücksichtigt werden, damit eine Volatilität von null nicht zu einer Division durch null führt.

## 5. RL Environment

Als erste Implementierung wird PPO mit einem Multi-Discrete Action Space verwendet. Der Agent verarbeitet die von XGBoost erzeugten Alpha-Scores zusammen mit Risiko-, Markt- und Portfolioinformationen und erzeugt für jede Aktie eine Buy-, Hold- oder Sell-Aktion.

Das RL Environment verbindet das mit XGBoost implementierte Alpha-Modell mit der Portfolio-Simulation. Zu jedem Rebalancing-Zeitpunkt `t` erhält der Agent einen State, wählt Aktionen und bekommt nach deren Ausführung einen Reward.

### State Space

Ein möglicher State ist:

```math
s_t=
[
\boldsymbol{\alpha}_t,
\boldsymbol{\sigma}_t,
\mathbf{w}_{t-1},
\mathbf{r}_t,
\mathbf{m}_t
]
```

Dabei bezeichnet:

- $\boldsymbol{\alpha}_t$: von XGBoost erzeugte Alpha-Scores für alle `N` Aktien
- $\boldsymbol{\sigma}_t$: geschätzte Volatilitäten dieser Aktien
- $\mathbf{w}_{t-1}$: aktuelle Portfolio-Gewichte einschließlich Cash
- $\mathbf{r}_t$: vergangene bzw. aktuelle Returns
- $\mathbf{m}_t$: zusätzliche Markt- oder Regimeinformationen

Der State $s_t$ ist somit der Input des RL-Agenten. Die Bestandteile werden typischerweise zu einem numerischen Vektor zusammengeführt und vor dem Training skaliert.

### Action Space

Der Output des Agenten ist für jede Aktie `i` eine diskrete Aktion:

```math
a_{i,t}\in\{-1,0,+1\},
\qquad
-1=SELL,\;0=HOLD,\;+1=BUY
```

Über alle `N` Aktien entsteht der Aktionsvektor:

```math
\mathbf{a}_t=
[a_{1,t},a_{2,t},\ldots,a_{N,t}]
```

Der Agent entscheidet damit, **ob und in welche Richtung** gehandelt wird, nicht über die Positionsgröße. Diese wird anschließend durch die separate Position-Sizing-Regel berechnet.

### Implementation Approach

XGBoost und der PPO-Agent werden getrennt trainiert. Zuerst wird XGBoost auf historischen Features und zukünftigen Renditen trainiert. Seine Point-in-Time-Prognosen werden anschließend zusammen mit Risiko-, Markt- und Portfolioinformationen als Input des RL-Agenten verwendet.

Die PPO-Policy erzeugt für jede Aktie drei Wahrscheinlichkeiten für Buy, Hold und Sell. Nach Auswahl und Ausführung der Aktionen berechnet das Environment die Positionsgrößen, wendet die Portfolio Constraints an und bestimmt die neuen Portfolio-Gewichte, Transaktionskosten, Portfoliorendite und den Reward.

Eine gemeinsame Klassifikation aller Aktionskombinationen wird vermieden, da bei `N` Aktien bereits $3^N$ Kombinationen entstehen. Stattdessen verwendet PPO eine separate diskrete Aktionsverteilung pro Aktie. Geeignete DQN-Varianten können später als Vergleich untersucht werden.

## 6. Risk-Adjusted Position Sizing

Die Position-Sizing-Regel übersetzt die diskrete RL-Aktion in eine konkrete Veränderung des Portfolio-Gewichts:

```math
\Delta w_{i,t}
=
a_{i,t}\cdot B_t\cdot q_{i,t}
```

Dabei bezeichnet:

- $\Delta w_{i,t}$ die Veränderung des Portfolio-Gewichts der Aktie `i` zum Zeitpunkt `t`
- $a_{i,t}\in\{-1,0,+1\}$ die RL-Aktion Sell, Hold oder Buy
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

Alle Varianten werden auf identischen Daten und Testzeiträumen verglichen:

1. **Buy & Hold**
2. **Equal Weight**
3. **Alpha Model + regelbasiertes Position Sizing**
4. **Alpha Model + Risk-Adjusted Position Sizing**
5. **Alpha Model + Risk-Adjusted Position Sizing + RL**

Die zentrale Forschungsfrage lautet:

> Liefert der RL-Agent zusätzlichen Nutzen gegenüber einer direkten regelbasierten Verwendung derselben Alpha-Signale?

## 10. Component Responsibilities

```text
Data / Features      Informationen über Unternehmen und Markt
        ↓
Alpha Model          Wie attraktiv ist die Aktie?
        ↓
Risk Adjustment      Wie stark ist das Signal relativ zum Risiko?
        ↓
RL Agent             Buy, Hold oder Sell?
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
- **RL Agent:** trifft kontextabhängige Buy/Hold/Sell-Entscheidungen.
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