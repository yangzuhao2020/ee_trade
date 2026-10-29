"""使用 Streamlit 浏览 outputs/<数据集>/<场景> 下的仿真结果。
展示基于 CSV 的交互式图表，并兼容旧输出中的学习 PNG 图表。
在项目根目录的 ee_trade 环境中运行：
    streamlit run UI/dashboard.py"""

from pathlib import Path

import streamlit as st

import charts
import learning_charts

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
LEARNING_ERROR_LABELS = {
    "learning_unit": "学习机组图",
    "learning_curves": "学习曲线图",
    "learning_train_comparison": "训练期对比图",
    "learning_validation_comparison": "验证期对比图",
    "learning_test_comparison": "测试期对比图",
}


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
    """返回图表读取的每个 CSV 文件是否存在、修改时间和文件大小。"""

    state = []
    for filename in (*charts.TABLES.values(), *learning_charts.DATA_FILES):
        try:
            stat = (run_dir / filename).stat()
        except FileNotFoundError:
            state.append((filename, None))
        else:
            state.append((filename, stat.st_mtime_ns, stat.st_size))
    for path in sorted((run_dir / "plots").glob("learning_*.png")):
        stat = path.stat()
        state.append((str(path.relative_to(run_dir)), stat.st_mtime_ns, stat.st_size))
    return tuple(state)


@st.cache_data(show_spinner="正在读取结果 CSV…", max_entries=8)
def load_charts(run_dir: str, files_state: tuple):
    """构建中文图表；有居民需求时展示居民子图，其余场景保留市场供需概览。"""

    path = Path(run_dir)
    market_figures = charts.build_charts(charts.load_run(path))
    learning_figures, learning_errors = learning_charts.build_learning_charts(path)
    return market_figures, learning_figures, learning_errors


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
figures, learning_figures, learning_errors = load_charts(str(run_dir), results_state(run_dir))
learning_items = learning_charts.learning_items(run_dir, learning_figures, learning_errors)
chart_count = len(figures) + sum(kind != "error" for kind, _, _ in learning_items)
st.caption(
    f"{run_dir.relative_to(OUTPUTS_DIR.parent).as_posix()} · {chart_count} 张图 · "
    "交互图可悬停查看数值、框选缩放、双击还原或点击图例隐藏曲线"
)

for figure in figures:
    st.plotly_chart(figure, width="stretch")
for kind, name, item in learning_items:
    if kind == "figure":
        st.plotly_chart(item, width="stretch")
    elif kind == "image":
        st.image(str(item), caption=LEARNING_ERROR_LABELS.get(name, name), width="stretch")
    else:
        st.error(f"{LEARNING_ERROR_LABELS.get(name, name)}：{item}")
for name, error in learning_errors.items():
    if name not in learning_charts.CHART_NAMES:
        st.error(f"学习指标文件 {name}：{error}")
