"""Version one and two EOM simulation entry point."""

from __future__ import annotations

from pathlib import Path

from ..models import MarketOpening, MarketSettings, SimulationResult
from .eom import simulate_eom_market


def simulate_v1_v2_market(
    input_path: Path,
    settings: MarketSettings,
    openings: tuple[MarketOpening, ...],
) -> SimulationResult:
    """Calculate one V1/V2 scenario from validated settings and openings."""

    return simulate_eom_market(input_path, settings, openings)
