"""Public entry point for simulation PNG visualisations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..errors import PlottingError
from ..models import SimulationResult
from .common import _load_matplotlib, _make_unit_colors, _validate_plot_data
from .dispatch import (
    _plot_dispatch_by_unit,
    _plot_operator_profit,
    _plot_storage_dispatch,
)
from .market import (
    _plot_first_merit_order,
    _plot_first_pay_as_bid_order_book,
    _plot_market_overview,
    _plot_market_summary,
)
from .opening import (
    _plot_offer_acceptance,
    _plot_opening_dispatch,
    _remove_stale_opening_plots,
    _select_opening_results,
)


# Keep the stable package-level API intentionally small; chart implementations
# live in focused modules and are imported by this orchestration entry point.
__all__ = ["generate_plots"]


def generate_plots(
    result: SimulationResult,
    output_dir: str | Path,
    opening_time: datetime | None = None,
) -> tuple[Path, ...]:
    """Render durable PNG summaries from a complete simulation result.

    This function deliberately owns no simulation or CSV-writing work. Call it
    only after ``write_results`` has completed so a rendering failure cannot
    discard valid market results.
    """

    results = tuple(
        sorted(result.market_results, key=lambda market: market.delivery_start)
    )
    storage_results = tuple(
        sorted(
            result.storage_results,
            key=lambda storage: (
                storage.delivery_start,
                storage.opening_time,
                storage.unit_name,
            ),
        )
    )
    _validate_plot_data(results)
    pyplot, dates = _load_matplotlib()

    plot_directory = Path(output_dir)
    plot_directory.mkdir(parents=True, exist_ok=True)
    unit_colors = _make_unit_colors(results, storage_results)

    try:
        paths = [
            _plot_market_overview(
                pyplot, dates, results, plot_directory / "market_overview.png"
            ),
            _plot_market_summary(
                pyplot, dates, results, plot_directory / "market_summary.png"
            ),
            _plot_dispatch_by_unit(
                pyplot,
                dates,
                results,
                unit_colors,
                plot_directory / "dispatch_by_unit.png",
            ),
            _plot_operator_profit(
                pyplot,
                results,
                storage_results,
                plot_directory / "operator_profit.png",
            ),
        ]
        storage_plot_path = plot_directory / "storage_dispatch.png"
        if storage_results:
            paths.append(
                _plot_storage_dispatch(
                    pyplot,
                    dates,
                    storage_results,
                    unit_colors,
                    storage_plot_path,
                )
            )
        else:
            storage_plot_path.unlink(missing_ok=True)
        if result.settings.market_mechanism == "complex_clearing":
            # A directory may have been rendered by an older version which
            # incorrectly treated complex clearing as a simple merit order.
            (plot_directory / "merit_order_first_product.png").unlink(
                missing_ok=True
            )
            (plot_directory / "pay_as_bid_first_product.png").unlink(
                missing_ok=True
            )
            opening_results = _select_opening_results(results, opening_time)
            selected_opening_time = (
                opening_results[0].opening_time or opening_results[0].delivery_start
            )
            opening_directory = (
                plot_directory
                / "openings"
                / selected_opening_time.strftime("%Y-%m-%d_%H-%M")
            )
            _remove_stale_opening_plots(
                plot_directory / "openings",
                keep=opening_directory,
            )
            opening_directory.mkdir(parents=True, exist_ok=True)
            paths.extend(
                (
                    _plot_opening_dispatch(
                        pyplot,
                        dates,
                        opening_results,
                        unit_colors,
                        opening_directory / "price_and_dispatch.png",
                    ),
                    _plot_offer_acceptance(
                        pyplot,
                        opening_results,
                        opening_directory / "offer_acceptance.png",
                    ),
                )
            )
        elif result.settings.market_mechanism == "pay_as_bid":
            _remove_stale_opening_plots(plot_directory / "openings")
            (plot_directory / "merit_order_first_product.png").unlink(
                missing_ok=True
            )
            paths.append(
                _plot_first_pay_as_bid_order_book(
                    pyplot,
                    results[0],
                    unit_colors,
                    plot_directory / "pay_as_bid_first_product.png",
                )
            )
        else:
            _remove_stale_opening_plots(plot_directory / "openings")
            (plot_directory / "pay_as_bid_first_product.png").unlink(
                missing_ok=True
            )
            paths.append(
                _plot_first_merit_order(
                    pyplot,
                    results[0],
                    unit_colors,
                    plot_directory / "merit_order_first_product.png",
                )
            )
    except PlottingError:
        raise
    except Exception as exc:
        raise PlottingError(f"Could not generate PNG plots: {exc}") from exc

    return tuple(paths)
