"""Public entry point for simulation PNG visualisations."""

from __future__ import annotations

from pathlib import Path

from ..errors import PlottingError
from ..models import SimulationResult
from .common import (
    _load_matplotlib,
    _make_unit_colors,
    _remove_stale_opening_plots,
    _validate_plot_data,
)
from .dispatch import (
    _plot_dispatch_by_unit,
    _plot_operator_profit,
    _plot_storage_dispatch,
)
from .industry import _plot_industry_dispatch
from .learning import (
    _plot_evaluation_comparison,
    _plot_learning_curves,
    _plot_learning_unit,
    _read_learning_metrics,
)
from .market import (
    _plot_first_pay_as_bid_order_book,
    _plot_market_overview,
    _plot_market_summary,
)


# Keep the stable package-level API intentionally small; chart implementations
# live in focused modules and are imported by this orchestration entry point.
__all__ = ["generate_learning_plots", "generate_plots"]

_LEARNING_UNIT_PLOT = "learning_unit.png"
_LEARNING_CURVES_PLOT = "learning_curves.png"
_LEARNING_COMPARISON_PLOT = "learning_evaluation_comparison.png"


def generate_plots(
    result: SimulationResult,
    output_dir: str | Path,
    learning_episode_label: str = "Actor evaluation",
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
                    storage_plot_path,
                )
            )
        else:
            storage_plot_path.unlink(missing_ok=True)
        industry_plot_path = plot_directory / "industry_dispatch.png"
        if result.industry_results:
            paths.append(
                _plot_industry_dispatch(
                    pyplot,
                    dates,
                    result.industry_results,
                    result.industry_flexibility_results,
                    industry_plot_path,
                )
            )
        else:
            industry_plot_path.unlink(missing_ok=True)
        if result.learning_steps:
            paths.append(
                _plot_learning_unit(
                    pyplot,
                    dates,
                    results,
                    result.learning_steps,
                    plot_directory / _LEARNING_UNIT_PLOT,
                    learning_episode_label,
                )
            )
        else:
            for name in (
                _LEARNING_UNIT_PLOT,
                _LEARNING_CURVES_PLOT,
                _LEARNING_COMPARISON_PLOT,
            ):
                (plot_directory / name).unlink(missing_ok=True)
        if result.settings.market_mechanism == "pay_as_bid":
            paths.append(
                _plot_first_pay_as_bid_order_book(
                    pyplot,
                    results[0],
                    unit_colors,
                    plot_directory / "pay_as_bid_first_product.png",
                )
            )
        else:
            # A directory may have been rendered by an older version which
            # incorrectly treated complex clearing as a simple merit order.
            (plot_directory / "pay_as_bid_first_product.png").unlink(missing_ok=True)
        # Older versions rendered per-opening detail plots under openings/.
        _remove_stale_opening_plots(plot_directory / "openings")
    except PlottingError:
        raise
    except Exception as exc:
        raise PlottingError(f"Could not generate PNG plots: {exc}") from exc

    return tuple(paths)


def generate_learning_plots(
    metrics_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Render Version 5 learning curves and the Actor/baseline comparison.

    The metrics CSV written by training or evaluation is the only input, so the
    learning API keeps returning a plain ``SimulationResult``.
    """

    rows = _read_learning_metrics(Path(metrics_path))
    curve_rows = [
        row
        for row in rows
        if row["phase"] in {"initial_experience", "train", "validation"}
    ]
    comparison_rows = [
        row for row in rows if row["phase"] in {"evaluation", "baseline"}
    ]
    pyplot, _ = _load_matplotlib()
    plot_directory = Path(output_dir)
    plot_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        if curve_rows:
            paths.append(
                _plot_learning_curves(
                    pyplot, curve_rows, plot_directory / _LEARNING_CURVES_PLOT
                )
            )
        else:
            (plot_directory / _LEARNING_CURVES_PLOT).unlink(missing_ok=True)
        if comparison_rows:
            paths.append(
                _plot_evaluation_comparison(
                    pyplot,
                    comparison_rows,
                    plot_directory / _LEARNING_COMPARISON_PLOT,
                )
            )
        else:
            (plot_directory / _LEARNING_COMPARISON_PLOT).unlink(missing_ok=True)
    except PlottingError:
        raise
    except Exception as exc:
        raise PlottingError(f"Could not generate learning PNG plots: {exc}") from exc
    return tuple(paths)
