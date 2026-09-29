"""Shared rendering utilities for PNG market plots."""

from __future__ import annotations

from math import isclose
from pathlib import Path

from ..errors import PlottingError
from ..market_models import MarketClearingResult
from ..models import StorageDispatchResult


_ENERGY_TOLERANCE_MWH = 1e-7
_OPENING_PLOT_FILENAMES = (
    "price_and_dispatch.png",
    "offer_acceptance.png",
)
_UNIT_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#D4A72C",
)


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.dates as dates
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        raise PlottingError(
            "PNG generation requires Matplotlib. Install matplotlib or run with --no-plots."
        ) from exc
    return pyplot, dates


def _validate_plot_data(results: tuple[MarketClearingResult, ...]) -> None:
    if not results:
        raise PlottingError("Cannot generate plots because the simulation has no delivery products.")

    for market in results:
        if market.duration_hours <= 0:
            raise PlottingError(
                f"Product at {market.delivery_start!s} has a non-positive duration."
            )
        if not isclose(
            market.requested_demand_mwh,
            market.cleared_energy_mwh + market.unserved_demand_mwh,
            rel_tol=0.0,
            abs_tol=_ENERGY_TOLERANCE_MWH,
        ):
            raise PlottingError(
                "Demand energy must equal cleared energy plus unmet demand for "
                f"{market.delivery_start:%Y-%m-%d %H:%M}."
            )


def _make_unit_colors(
    results: tuple[MarketClearingResult, ...],
    storage_results: tuple[StorageDispatchResult, ...] = (),
) -> dict[str, str]:
    unit_names = sorted(
        {
            cleared.offer.unit_name
            for market in results
            for cleared in market.offers
        }
        | {storage.unit_name for storage in storage_results}
    )
    return {
        name: _UNIT_COLORS[index % len(_UNIT_COLORS)]
        for index, name in enumerate(unit_names)
    }


def _format_time_axis(axis, dates) -> None:
    locator = dates.AutoDateLocator(minticks=4, maxticks=10)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
    axis.tick_params(axis="x", rotation=20)
    axis.margins(x=0)


def _save_figure(pyplot, figure, path: Path) -> Path:
    try:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        pyplot.close(figure)
    return path


def _annotate_no_supply(axis, message: str = "No supply offers submitted") -> None:
    """Mark a valid market state whose supply order book is empty."""

    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F3F4F6",
            "edgecolor": "#D1D5DB",
        },
        zorder=6,
    )
