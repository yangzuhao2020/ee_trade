""""""

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
from .household import _plot_household_dispatch, _plot_household_savings
from .learning import (
    _plot_evaluation_comparison,
    _plot_learning_curves,
    _plot_learning_unit,
    _read_learning_metrics,
    comparison_periods,
)
from .market import (
    _plot_market_overview,
    _plot_market_summary,
)


_LEARNING_UNIT_PLOT = "learning_unit.png"
_LEARNING_CURVES_PLOT = "learning_curves.png"
_LEARNING_COMPARISON_PATTERN = "learning_*_comparison.png"
_LEGACY_COMPARISON_PLOT = "learning_evaluation_comparison.png"


def _comparison_plot_name(period: str) -> str:
    return f"learning_{period}_comparison.png"


def generate_plots(
    result: SimulationResult,
    output_dir: str | Path,
    learning_episode_label: str = "策略评估",
) -> tuple[Path, ...]:
    """根据完整的仿真结果生成持久保存的 PNG 汇总图。
    此函数刻意不承担任何仿真计算或 CSV 写出工作。只能在 ``write_results`` 完成后
    调用，这样即使绘图失败，也不会丢失有效的市场结果。
    """

    results = tuple(
        sorted(result.market_results, key=lambda market: market.delivery_start))
    storage_results = tuple(
        sorted(
            result.storage_results,
            key=lambda storage: (
                storage.delivery_start,
                storage.opening_time,
                storage.unit.name,
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
                pyplot, dates, results, plot_directory / "market_overview.png",
                forecast_prices={
                    household.dispatch.plan.delivery_start: household.dispatch.plan.forecast_price_eur_per_mwh
                    for household in result.household_results
                },
            ),
            _plot_market_summary(
                pyplot, dates, results, plot_directory / "market_summary.png"),
            _plot_dispatch_by_unit(
                pyplot,
                dates,
                results,
                plot_directory / "dispatch_by_unit.png",),
        ]
        operator_plot_path = plot_directory / "operator_profit.png"
        if result.household_results:
            # 家庭场景不展示发电与储能运营商利润，并清理旧版输出。
            operator_plot_path.unlink(missing_ok=True)
        else:
            paths.append(
                _plot_operator_profit(
                    pyplot, results, storage_results, operator_plot_path,
                )
            )
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
                    industry_plot_path,
                )
            )
        else:
            industry_plot_path.unlink(missing_ok=True)
        household_plot_path = plot_directory / "household_dispatch.png"
        household_zoom_path = plot_directory / "household_dispatch_zoom.png"
        if result.household_results:
            for path, days in ((household_plot_path, None), (household_zoom_path, 3)):
                paths.append(
                    _plot_household_dispatch(
                        pyplot,
                        dates,
                        result.household_results,
                        path,
                        days=days,
                    )
                )
        else:
            household_plot_path.unlink(missing_ok=True)
            household_zoom_path.unlink(missing_ok=True)
        household_savings_path = plot_directory / "household_savings.png"
        if result.household_results:
            paths.append(
                _plot_household_savings(
                    pyplot,
                    dates,
                    result.household_results,
                    household_savings_path,
                )
            )
        else:
            household_savings_path.unlink(missing_ok=True)
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
            for name in (_LEARNING_UNIT_PLOT, _LEARNING_CURVES_PLOT):
                (plot_directory / name).unlink(missing_ok=True)
            for stale in plot_directory.glob(_LEARNING_COMPARISON_PATTERN):
                stale.unlink()
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
    """绘制第五版的学习曲线，以及各数据时段上 Actor 与基准策略的对比图。

    唯一的输入是训练、评估或测试写出的指标 CSV，因此学习接口仍只返回普通的
    ``SimulationResult``。只重绘本指标文件包含的图表，不删除输出目录中其他命令
    生成的学习曲线或其他时段的对比图。
    """

    rows = _read_learning_metrics(Path(metrics_path))
    curve_rows = [
        row
        for row in rows
        if row["phase"] in {"initial_experience", "train", "validation"}
    ]
    periods = comparison_periods(rows)
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
        # 旧版本未区分数据时段的对比图已不再生成。
        (plot_directory / _LEGACY_COMPARISON_PLOT).unlink(missing_ok=True)
        for period in periods:
            paths.append(
                _plot_evaluation_comparison(
                    pyplot,
                    rows,
                    plot_directory / _comparison_plot_name(period),
                    period,
                )
            )
    except PlottingError:
        raise
    except Exception as exc:
        raise PlottingError(f"Could not generate learning PNG plots: {exc}") from exc
    return tuple(paths)
