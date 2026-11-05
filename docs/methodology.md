# Methodology

## Hypotheses

| ID | Hypothesis | Test |
|---|---|---|
| H1 | Embedding distance correlates with distance in diffusion shape (log q, T₁₀→₉₀) | One-sided Mantel test (Spearman, permutation) |
| H2 | A kNN analog prior on embeddings beats global mean, random analogs and TF-IDF | LOO absolute error, paired sign-flip test, Holm correction |
| H3 | Forecast intervals are calibrated | Coverage of 50/80/90 % intervals, PIT, CRPS |
| H4 | Bayesian updating beats NLS refitting for few observed periods | Error as a function of observed periods T |

H1 is the go/no-go criterion of phase 0.

## Deterministic core

Cumulative adoption N(t) with market potential m:

$$\frac{dN}{dt} = \Big(p + q\frac{N}{m}\Big)(m - N), \qquad N(t_0) = N_0.$$

With $f_0 = N_0/m$, $r = q/p$ and $C = (1 - f_0)/(1 + r f_0)$:

$$N(t) = m\,\frac{1 - C e^{-(p+q)(t-t_0)}}{1 + rC e^{-(p+q)(t-t_0)}}.$$

Anchoring at the first observation avoids estimating an unobserved launch date, which is poorly
identified jointly with p when historical series start after launch.

Shape descriptors used as robust targets: the 10→90 % takeoff duration and, for $N_0 = 0$, the
peak time $t^* = \ln(q/p)/(p+q)$.

## Stochastic extension

On the adoption fraction $F = N/m$:

$$dF_t = (p + qF_t)(1 - F_t)\,x(t)\,dt + \sigma\sqrt{F_t(1 - F_t)}\,dW_t.$$

The Wright–Fisher diffusion term vanishes at 0 and 1, keeping the state in the unit interval;
$x(t)$ is the Generalized Bass effort multiplier (Bass, Krishnan & Jain 1994). Simulation uses
Euler–Maruyama with projection; the σ → 0 limit is checked against the closed form.

## Analog prior

$\theta_i = (\log p_i, \log q_i)$, $e_i = \text{Encoder}(\text{description}_i)$,
$\theta_i \mid e_i \sim \mathcal{N}(\mu(e_i), \Sigma(e_i))$, with μ, Σ from

1. the k nearest analogs (k ≈ 5, following Goodwin et al. 2013),
2. a Gaussian process on reduced embeddings,
3. an MLP head if the dataset grows beyond a few hundred curves.

Market potential m is supplied by the user as a scenario distribution; text informs the curve
shape, not the market size.

## Data and leakage control

- Curves: Our World in Data US household adoption panel (v0), CHAT country panels (v1),
  any tidy CSV (`technology, region, year, adoption`).
- Preparation: truncate at the maximum (decline reflects substitution), cut series at gaps
  longer than 10 years, require ≥ 8 points and peak ≥ 15 %, keep fits with R² ≥ 0.95.
- Descriptions: Wikipedia summaries truncated to the defining sentences; years, numbers,
  percentages and outcome vocabulary are masked. Masked vs. unmasked results are compared
  as a leakage diagnostic. Leakage-free manual descriptions take precedence when provided.

## Evaluation protocol

- Leave-one-out over technologies; temporal variant where only earlier technologies may serve
  as analogs.
- Baselines: global mean, random k analogs, TF-IDF, structured attributes (later),
  NLS/logistic/Gompertz fits for the early-data setting.
- Controls: synthetic data with planted text signal (must be detected), permuted descriptions
  (must not be detected).

## Limitations

Small n in phase 0; survivorship bias (all historical curves are successes, so forecasts are
conditional on success); household adoption may transfer poorly to B2B deep-tech markets.

## References

- Bass (1969), *Management Science* 15(5).
- Bass, Krishnan & Jain (1994), *Marketing Science* 13(3).
- Comin & Hobijn (2009), The CHAT Dataset, NBER WP 15319.
- Goodwin, Dyussekeneva & Meeran (2013), *IMA Journal of Management Mathematics*.
- Lee, Kim, Park & Kang (2014), *Technological Forecasting and Social Change* 86, 49–64.
- Ramírez-Hassan & Montoya-Blandón (2020), *International Journal of Forecasting* 36(2), 442–465.
- Sebastiani (2019), Bass diffusion: a stochastic differential equation approach.
- Skiadas & Giovanis (1997), *Applied Stochastic Models and Data Analysis* 13(2).
