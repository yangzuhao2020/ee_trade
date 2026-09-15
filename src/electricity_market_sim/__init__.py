"""Lightweight electricity spot-market simulation package."""

from .models import SimulationResult
from .plotting import generate_plots
from .simulation import run_simulation, simulate

__all__ = ["SimulationResult", "generate_plots", "run_simulation", "simulate"]
