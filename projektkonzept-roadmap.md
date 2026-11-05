# Diffusion-based Innovation Adoption Curve Simulator — Gesamtkonzept

Stand: 05.11.2026 · Arbeitsname Package: `adoptsim` (PyPI-Verfügbarkeit noch prüfen) · Repo: `innovation-adoption-simulator`

---

## 1. Problemstellung

Für neue (Deep-Tech-)Produkte müssen Strategie-, Investitions- und Kapazitätsentscheidungen getroffen werden, **bevor** es Absatzdaten gibt. Das Bass-Modell beschreibt Adoptionskurven gut, seine Parameter (p, q, m) lassen sich aber erst schätzen, wenn der Wendepunkt fast erreicht ist. In der Praxis werden Parameter daher per Analogie gewählt („verhält sich wie Produkt X“) — subjektiv, schlecht dokumentiert und mit hohen Fehlern (Goodwin et al. 2013). Zudem werden fast immer Punktprognosen kommuniziert, obwohl die Unsicherheit dominant ist.

**Ziel:** Ein offenes, praxistaugliches Werkzeug, das

1. aus einer Produktbeschreibung automatisch Analog-Produkte findet (Transformer-Embeddings),
2. daraus eine Prior-Verteilung über die Diffusionsparameter ableitet,
3. mit einem stochastischen Bass-Modell (SDE) kalibrierte Prognosebänder erzeugt,
4. sich mit den ersten realen Datenpunkten bayesianisch aktualisiert.

## 2. Forschungsfragen & Hypothesen

| ID | Frage | Hypothese | Test |
|---|---|---|---|
| H1 | Tragen Text-Embeddings Information über Diffusionsdynamik? | Embedding-Distanz korreliert positiv mit Distanz der Kurvenform (q, T₁₀→₉₀) | Mantel-Test (Permutation), Spike |
| H2 | Schlägt embedding-basierte Analogie-Suche einfache Baselines? | kNN-Prior auf Embeddings hat kleineren LOO-Fehler als Globalmittel, Zufallsanalogien und TF-IDF | Gepaarter Permutationstest auf LOO-Fehlern |
| H3 | Sind die Prognosebänder kalibriert? | Empirische Coverage der 80 %-Intervalle liegt in [0.75, 0.85] | Coverage, PIT, CRPS |
| H4 | Lohnt das bayesianische Update? | Fehler sinkt mit Anzahl beobachteter Perioden T schneller als beim NLS-Fit; Vorteil v. a. bei T ≤ 4 | Fehler-vs-T-Kurven |

H1 ist der **Go/No-Go-Punkt** (Phase 0). Fällt H1 durch, wird nicht in Phase 2 investiert, sondern auf strukturierte Attribute bzw. reine Unsicherheitsmodellierung umgeschwenkt.

## 3. Abgrenzung zum Stand der Forschung

- **Bass (1969); Bass, Krishnan & Jain (1994):** deterministische Basis, Generalized Bass Model mit Marketing-Effort x(t).
- **Skiadas & Giovanis (1997); Sebastiani (2019); Guarnera (2013):** stochastische Bass-Modelle mit SDE, Lösbarkeit, Zulässigkeit, Inferenz. → Wir übernehmen diese Theorie, Neuheit liegt nicht hier.
- **Lee, Kim, Park & Kang (2014, TFSC 86):** ML auf *strukturierten* Produktattributen → (p, q). → Engster Vorläufer; wir ersetzen handkodierte Attribute durch unstrukturierten Text.
- **Goodwin, Dyussekeneva & Meeran (2013, IMA JMM):** Analogie-Prognosen fehlerbehaftet; Mitteln über ~5–6 Analogien optimal. → Baseline-Design und Wahl von k.
- **Ramírez-Hassan & Montoya-Blandón (2020, IJF 36(2)):** Bayes-GBM mit Prior aus ähnlichen Märkten. → Methodische Vorlage für Prior → Posterior.
- **Li & Terui (Tohoku):** Social-Media-Text als Kovariaten zeitvariabler Bass-Parameter (Sentiment/Topics). → Wir nutzen Text *vor* Launch und moderne Embeddings.
- **Software:** PyPI-Pakete beschränken sich auf deterministische Fits (z. B. `innovationdiffusion`, `markbassmodel`, `bassmodelgens`).

**Beitrag:** (1) Text-Embeddings als Analogie-Raum, (2) kalibrierte Unsicherheit aus Parameter- *und* Prozessrauschen, (3) durchgängiges Pre-Launch→Early-Data-Modell, (4) offenes, getestetes Package.

## 4. Modell

### 4.1 Deterministischer Kern

Kumulierte Adoption N(t), Marktpotenzial m, Anteil F = N/m:

    dN/dt = (p + q·N/m)(m − N),   N(t₀) = N₀

Geschlossene Lösung mit beliebigem Anfangswert (f₀ = N₀/m, r = q/p):

    N(t) = m · [1 − C·e^{−(p+q)(t−t₀)}] / [1 + r·C·e^{−(p+q)(t−t₀)}],   C = (1 − f₀)/(1 + r·f₀)

Für f₀ = 0 ergibt sich die Standard-Bass-Lösung. Der Anfangswert ist wichtig, weil historische Reihen oft erst nach Markteinführung beginnen — so muss kein unbeobachteter Launch-Zeitpunkt geschätzt werden (der mit p schlecht identifizierbar ist).

**Formdeskriptoren** (parametrisierungsrobust, für Spike und Kommunikation): T₁₀→₉₀ = Dauer von 10 % auf 90 % von m; Peak-Zeitpunkt t* = ln(q/p)/(p+q) (Standardfall).

### 4.2 Stochastische Erweiterung (SDE)

    dF_t = (p + q·F_t)(1 − F_t)·x(t) dt + σ·√(F_t(1 − F_t)) dW_t

- Diffusionsterm verschwindet an 0 und 1 → Zustandsraum bleibt [0, 1] (Wright-Fisher-Struktur).
- x(t) ≡ 1 → Bass; x(t) ≠ 1 → GBM-Szenarien (Marketing-Push, Subventionen).
- Numerik: Euler–Maruyama mit Projektion auf [0, 1] (Baseline); Ablation: Simulation im Logit-Raum (Itô-korrigiert), Milstein.
- Ablationen des Diffusionsterms: additiv σ, multiplikativ σF, Sebastiani-Form.
- Validierung: σ → 0 ⇒ Pfade → ODE; Monte-Carlo-Mittel vs. ODE für kleine σ; Randverletzungen = 0.

### 4.3 Hierarchischer Prior aus Text

    θ_i = (log p_i, log q_i)               # Kurvenform
    e_i = Encoder(Beschreibung_i)          # Embedding
    θ_i | e_i ~ N(μ(e_i), Σ(e_i))

Stufen für μ, Σ (Komplexität nur bei Bedarf erhöhen):

1. **kNN-Prior:** gewichtete Mittel/Kovarianz der k nächsten Analogien (k ≈ 5, vgl. Goodwin et al.). Interpretierbar: Analogien werden angezeigt.
2. **Gauss-Prozess / Ridge** auf (dimensionsreduzierten) Embeddings, Σ aus prädiktiver Varianz.
3. **MLP-Head** nur, wenn Datensatz groß genug (> ~500 Kurven).

**Marktpotenzial m** ist aus Text kaum ableitbar und wird als User-Input mit Unsicherheit modelliert (z. B. TAM-Szenario lognormal). Der Text liefert primär die *Form* (p, q).

### 4.4 Inferenz

- **Pre-Launch:** Monte-Carlo über θ ~ Prior, m ~ User-Prior, SDE-Pfade → Fan-Chart (P10/P50/P90).
- **Early-Data:** Posterior p(θ, m, σ | y₁:T) via NumPyro (NUTS), Likelihood über Euler-Übergangsdichten (für jährliche Daten ausreichend, Ablation mit feinerem Gitter / Datenaugmentation). Für das Package zusätzlich schnelle Laplace-Approximation.
- **Klassische Referenz:** NLS-Fit (scipy `least_squares`, Multi-Start, Log-Parametrisierung).

## 5. Daten

### 5.1 Quellen

| Quelle | Inhalt | Status |
|---|---|---|
| Our World in Data – „Technology adoption by households in the US“ (u. a. Comin & Hobijn) | Haushaltsanteil (%) für ~30–40 Technologien, teils seit 1900 | **v0**, Loader vorhanden |
| CHAT (Comin & Hobijn 2009) | 104 Technologien, 161 Länder, 200 Jahre; viele *intensive* Maße (pro Kopf) | v1: braucht Normierung + Sättigungsannahme |
| Eigene CSV im Tidy-Schema | z. B. Branchen-/Unternehmensdaten | Loader vorhanden |
| Wikipedia (REST-Summary, später historische Revisionen) | Beschreibungstexte | **v0**, Loader vorhanden |
| Patentabstracts, frühe Presse | Leakage-arme Texte | v1 |

**Erwartung:** OWID allein liefert nur ~30–40 Kurven → reicht für den Spike (H1), nicht für ein robustes Modell. Für Phase 2 brauchen wir ≥ 100–200 Kurven (CHAT-Länderkurven, ggf. Branchendaten).

### 5.2 Tidy-Schema

| Spalte | Typ | Bedeutung |
|---|---|---|
| `technology` | str | eindeutiger Name |
| `region` | str | z. B. `USA` |
| `year` | int | Jahr |
| `adoption` | float | Anteil in [0, 1] |

Beschreibungen: `technology, description, source`.

### 5.3 Leakage-Kontrolle (kritisch)

Heutige Wikipedia-Texte beschreiben den Erfolg („wurde in 90 % der Haushalte…“). Gegenmaßnahmen, gestuft:

1. **v0 (Spike):** Maskierung von Jahreszahlen, Prozenten, Zahlen und Erfolgs-/Verbreitungsvokabular; Kürzung auf die ersten Sätze (Definition statt Geschichte).
2. **v1:** Wikipedia-Revision aus dem Launch-Jahrzehnt bzw. frühestmögliche Revision; Patentabstracts.
3. **Sensitivitätsanalyse:** Ergebnis mit maskierten vs. unmaskierten Texten vergleichen — schrumpft der Effekt stark, war Leakage im Spiel.

### 5.4 Survivorship Bias

Historische Datensätze enthalten fast nur erfolgreiche Technologien. Das Modell prognostiziert daher „Adoption *gegeben* Erfolg“. Kommunikation als explizite Limitation; v2: separate Erfolgswahrscheinlichkeit (z. B. aus gescheiterten Produkten/Startups).

## 6. Evaluation

**Protokoll:** Leave-one-out über Technologien; Variante „zeitlich geordnet“ (nur früher eingeführte Technologien dürfen Analogien sein) als realistischere Prüfung.

**Baselines:**
1. Globalmittel von (log p, log q)
2. Zufällige k Analogien
3. TF-IDF-Embeddings (ist ein Transformer überhaupt nötig?)
4. Strukturierte Attribute (Kategorie, Preisklasse) à la Lee et al. — sobald annotiert
5. Early-Data: NLS-Bass, Logistic, Gompertz

**Metriken:**
- Parameter: MAE auf log q, log p, T₁₀→₉₀
- Kurve: MAE/sMAPE der normierten kumulierten Adoption über Horizont H
- Verteilung: CRPS, Coverage 50/80/90 %, PIT-Histogramm
- Signifikanz: gepaarte Permutationstests bzw. Bootstrap-KIs; Holm-Korrektur bei mehreren Vergleichen

**Kontrollen:** Positivkontrolle (synthetische Daten mit eingebautem Text-Signal → Test muss anschlagen), Negativkontrolle (permutierte Texte → Test darf nicht anschlagen).

## 7. Go/No-Go-Kriterien Phase 0

Go, wenn auf den realen Daten (OWID v0, maskierte Texte):
- Mantel-Test für log q (oder T₁₀→₉₀) mit p < 0.05 **und**
- Embedding-kNN schlägt Globalmittel und Zufallsanalogien beim LOO-MAE auf log q signifikant (p < 0.05) **und**
- Positiv-/Negativkontrolle verhalten sich wie erwartet.

„Weich-Go“: Signal vorhanden, aber nicht besser als TF-IDF → Projekt läuft weiter, Transformer-Claim wird abgeschwächt (dann ist die Unsicherheitsquantifizierung der Hauptbeitrag).

No-Go: kein Signal → Pivot auf strukturierte Attribute + SDE-Unsicherheit als Hauptbeitrag.

## 8. Roadmap

| Phase | Dauer (grob) | Deliverable | Validierungskriterium |
|---|---|---|---|
| 0 – Spike | 1–2 Wo. | Daten v0, Bass-Fits, Embeddings, H1/H2-Test | Go/No-Go (Abschnitt 7) |
| 1 – Core-Modelle | 2–3 Wo. | ODE/SDE-Bass, NLS-Fit, Tests | Analytik = Numerik; Parameter-Recovery; σ→0-Konvergenz |
| 2 – Analogie-Prior | 3–4 Wo. | Encoder-Abstraktion, kNN/GP-Prior, LOO-Benchmark | Schlägt Baselines 1–3 bei CRPS; Coverage kalibriert |
| 3 – Bayes-Update | 2–3 Wo. | NumPyro-Posterior, Laplace-Approx. | H4 erfüllt |
| 4 – Package v0.1 | 2 Wo. | High-Level-API, CLI, Doku, PyPI-Release | Tests/CI grün, Beispiel reproduzierbar |
| 5 – Paper | 3–4 Wo. | arXiv-Preprint, Reproduktions-Skripte | Alle Zahlen aus `benchmarks/` reproduzierbar |
| 6 – optional | — | HF Space (Gradio), Deep-Tech-Fallstudien (Quantencomputing, eVTOL, Fusion, AR-Brillen) | Space importiert Package |

## 9. Architektur

```
adoptsim/
├── models/        bass.py (ODE + Fit), sde.py (Simulation), [generalized.py]
├── data/          schema.py, loaders.py (OWID, Tidy-CSV), descriptions.py (Wikipedia, Maskierung), synthetic.py
├── embeddings/    base.py (Encoder-ABC + Cache), tfidf.py, sentence.py (sentence-transformers)
├── evaluation/    analogs.py (kNN-LOO, Mantel, Permutationstests)
├── [priors/]      Phase 2: knn.py, gp.py
├── [inference/]   Phase 3: bayes.py, laplace.py
├── [forecast.py]  Phase 4: High-Level-API
└── [cli.py]
```

**Prinzipien:** austauschbare Komponenten über ABCs (`Encoder`, später `PriorModel`, `Inferencer`); reine Funktionen für die Numerik; Embeddings einmal berechnen und cachen (Key = Encoder + Version + Text-Hash), damit Modellierung ohne GPU läuft; schwere Abhängigkeiten (torch, sentence-transformers, numpyro) als optionale Extras.

**Geplante API (Phase 4):**

```python
from adoptsim import forecast
fc = forecast("Solid-state battery for passenger EVs ...", market_size=(2e6, 8e6), horizon=15)
fc.fan_chart(); fc.analogs
fc = fc.update(observed=[1200, 5400, 15800])
```

## 10. Infrastruktur

- **Env:** uv, Python 3.12, Lockfile committen.
- **Qualität:** ruff (Lint + Format), mypy, pytest + Coverage, hypothesis für numerische Kerne.
- **CI:** GitHub Actions (Lint, Typecheck, Tests auf 3.12/3.13), Release per Tag → PyPI via Trusted Publishing.
- **Compute:** lokale Embeddings (MacBook M5 via MPS, RTX 5070 via CUDA); für große Encoder Docker-Image (`docker/Dockerfile.gpu`) auf A100.
- **Reproduzierbarkeit:** feste Seeds, YAML-Configs in `configs/`, Ergebnisse mit Config-Hash, MLflow ab Phase 2.
- **Git:** dev → main per Merge-Commit/Rebase (kein Squash); `CLAUDE.md`, `ROADMAP.md`, `.claude/`, `*-out/` in `.gitignore`.

## 11. Risiken

| Risiko | Wirkung | Gegenmaßnahme |
|---|---|---|
| Zu wenige Kurven | Keine Signifikanz | CHAT-Länderkurven, Branchendaten; Spike bewusst als Signal-Test, nicht als Modell |
| Text-Leakage | Überschätzte Güte | Maskierung, historische Revisionen, Sensitivitätsanalyse |
| Survivorship Bias | Zu optimistische Prognosen | Als Limitation kommunizieren; v2 Erfolgswahrscheinlichkeit |
| p schlecht identifiziert | Instabile Priors | Anfangswert-Lösung, Formdeskriptoren, Fokus auf q/T₁₀→₉₀ |
| Historische ≠ Deep-Tech-Dynamik | Übertragbarkeit fraglich | Zeitlich geordnete Evaluation, Fallstudien, ehrliche Bänder |
| Haushalts- vs. B2B-Adoption | Falsche Analogien bei B2B-Deep-Tech | Segment als Metadatum, später getrennte Priors |

## 12. Literatur

- Bass, F. M. (1969). A new product growth for model consumer durables. *Management Science* 15(5).
- Bass, F. M., Krishnan, T. V., Jain, D. C. (1994). Why the Bass model fits without decision variables. *Marketing Science* 13(3).
- Comin, D., Hobijn, B. (2009). The CHAT Dataset. NBER WP 15319.
- Goodwin, P., Dyussekeneva, K., Meeran, S. (2013). The use of analogies in forecasting the annual sales of new electronics products. *IMA J. Management Mathematics*, Heft 4, 407–422.
- Guarnera, U. (2013). Stochastic Bass diffusion: model properties, exact simulation and statistical inference. Diss., Roma Tre.
- Lee, H., Kim, S. G., Park, H., Kang, P. (2014). Pre-launch new product demand forecasting using the Bass model. *TFSC* 86, 49–64.
- Ramírez-Hassan, A., Montoya-Blandón, S. (2020). Forecasting from others' experience: Bayesian estimation of the generalized Bass model. *IJF* 36(2), 442–465.
- Sebastiani, G. (2019). Bass diffusion: a stochastic differential equation approach. CNR.
- Skiadas, C., Giovanis, A. (1997). A stochastic Bass innovation diffusion model for studying the growth of electricity consumption in Greece. *Applied Stochastic Models and Data Analysis* 13(2).
