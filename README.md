# innovation-adoption-simulator

Pre-launch adoption forecasting for new technologies: a stochastic Bass diffusion model whose
parameter prior comes from text-embedding analogs, updated with early sales data.

> Status: phase 0 (research spike). The API is unstable.

## Idea

The Bass model describes adoption curves well, but its parameters can only be estimated once
the inflection point is near. Before launch, practitioners pick parameters by analogy to
earlier products, which is subjective and error-prone. This project

1. embeds a product description and retrieves historically similar technologies,
2. turns their fitted Bass parameters into a prior over the curve shape,
3. propagates parameter and process uncertainty through a Bass SDE into fan charts,
4. updates the prior with the first observed periods.

See [`docs/methodology.md`](docs/methodology.md) for the model and evaluation protocol.

## Installation

```bash
uv sync --group dev                 # core + dev tools
uv sync --group dev --extra embeddings --extra notebooks   # transformer encoders, Jupyter
```

## Quick start

```python
import numpy as np
from adoptsim import bass_cumulative, fit_bass, simulate_bass_sde
from adoptsim.models import fan_quantiles

t = np.arange(1990, 2010)
y = bass_cumulative(t - 1990 + 1, p=0.02, q=0.4, m=0.8)
fit = fit_bass(t, y, m_bounds=(y.max(), 1.0))

_, paths = simulate_bass_sde(fit.params.p, fit.params.q, sigma=0.3, horizon=25, n_paths=2000)
bands = fan_quantiles(paths)  # P10 / P50 / P90 of the adoption fraction
```

## Phase-0 spike

`experiments/00_spike_embedding_signal.ipynb` tests whether embedding similarity predicts
similarity in diffusion dynamics (Mantel test, leave-one-out analog prior vs. baselines,
positive/negative controls, leakage and temporal sensitivity) and prints a go/no-go verdict.

```bash
uv sync --group dev --extra embeddings --extra notebooks
uv run jupyter lab experiments/00_spike_embedding_signal.ipynb
```

The notebook downloads the Our World in Data US household adoption panel and Wikipedia
summaries on first run. Without network access it falls back to synthetic data and labels the
result as a pipeline check only.

## Repository layout

```
src/adoptsim/      package: models (Bass ODE/SDE), data loaders, encoders, evaluation
tests/             unit, property and regression tests
experiments/       spike notebook and exploratory code (not public API)
benchmarks/        reproducible baseline comparisons (phase 2+)
docs/              methodology and documentation source
data/              raw/interim/processed (ignored), external/ (curated, versioned)
docker/            GPU image for batch embedding with large encoders
```

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

## Citation

See [`CITATION.cff`](CITATION.cff). Data sources and their citations are listed in
[`data/README.md`](data/README.md).

## License

Apache-2.0
