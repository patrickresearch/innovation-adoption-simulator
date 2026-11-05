"""Stochastic Bass diffusion with text-embedding analog priors."""

from adoptsim.models import BassParams, bass_cumulative, fit_bass, simulate_bass_sde

__version__ = "0.0.1"
__all__ = ["BassParams", "__version__", "bass_cumulative", "fit_bass", "simulate_bass_sde"]
