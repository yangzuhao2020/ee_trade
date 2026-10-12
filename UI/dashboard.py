"""使用 Streamlit 浏览 outputs/<数据集>/<场景> 下的仿真结果。
展示基于 CSV 的交互式图表，并在存在时显示学习场景的 PNG 图表。
在项目根目录的 ee_trade 环境中运行：
    streamlit run UI/dashboard.py"""

from pathlib import Path

import streamlit as st

import charts

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
MAX_CHARTS = 6
# Learning charts have no interactive version yet; their PNGs are shown when present.
LEARNING_PLOTS = ("learning_unit.png", "learning_curves.png", "learning_evaluation_comparison.png")


def subdirectories(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(child.name for child in path.iterdir() if child.is_dir())


def scenarios_with_results(dataset: str) -> list[str]:
    return [
        scenario
        for scenario in subdirectories(OUTPUTS_DIR / dataset)
        if charts.has_results(OUTPUTS_DIR / dataset / scenario)
    ]


def results_state(run_dir: Path) -> tuple:
    """Existence, modification time and size of every CSV the charts read."""

    state = []
    for filename in charts.TABLES.values():
        try:
            stat = (run_dir / filename).stat()
        except FileNotFoundError:
            state.append((filename, None))
        else:
            state.append((filename, stat.st_mtime_ns, stat.st_size))
    return tuple(state)


@st.cache_data(show_spinner="正在读取结果 CSV…", max_entries=8)
def load_charts(run_dir: str, files_state: tuple):
    """Build the run's figures; files_state invalidates the cache when any CSV changes."""

    return charts.build_charts(charts.load_run(Path(run_dir)))


st.set_page_config(page_title="电力市场仿真结果", layout="wide")
st.title("电力市场仿真结果")

runs = {dataset: scenarios_with_results(dataset) for dataset in subdirectories(OUTPUTS_DIR)}
runs = {dataset: scenarios for dataset, scenarios in runs.items() if scenarios}
if not runs:
    st.warning(f"未在 {OUTPUTS_DIR} 下找到任何仿真结果，请先运行仿真。")
    st.stop()

dataset_column, scenario_column = st.columns(2)
dataset = dataset_column.selectbox("数据集", list(runs))
scenario = scenario_column.selectbox("场景", runs[dataset])

run_dir = OUTPUTS_DIR / dataset / scenario
figures = load_charts(str(run_dir), results_state(run_dir))
learning_plots = [run_dir / "plots" / name for name in LEARNING_PLOTS if (run_dir / "plots" / name).is_file()]
items = [*figures, *learning_plots][:MAX_CHARTS]
st.caption(
    f"{run_dir.relative_to(OUTPUTS_DIR.parent).as_posix()} · {len(items)} 张图 · "
    "悬停查看数值，拖动框选缩放、双击还原，点击图例隐藏曲线"
)

for item in items:
    if isinstance(item, Path):
        st.image(str(item), caption=item.stem, width="stretch")
    else:
        st.plotly_chart(item, width="stretch")
