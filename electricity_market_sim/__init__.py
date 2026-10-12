"""Lightweight electricity spot-market simulation package."""

from .models import SimulationResult
from .plotting import generate_plots
from .simulation import (
    LearningEpisodeRunner,
    run_simulation,
    simulate,
    simulate_learning_episode,
)
