# Project Rules

- **未经用户明确许可，不得修改代码。** 默认只做分析、解释和运行验证；任何代码修改必须先获得用户同意。
- **代码默认保持精简，禁止过度防御。** 只实现明确需求，不为假想异常增加分支、配置、抽象或兼容层。
- **必要校验只在输入边界做一次。** 优先复用参数解析器、类型声明及已有校验；内部逻辑不重复检查已成立的前提。
- **不擅自增加兜底、重试、恢复或静默降级。** 仅在明确需求或实际问题需要时引入；不用宽泛的 `try/except` 或默认值掩盖错误。

# Verification

- 运行测试需使用 `ee_trade` conda 环境（默认 base 环境缺少 scipy）：
  `C:\Users\yang\miniconda3\envs\ee_trade\python.exe -m unittest discover -s tests`
- 示例场景回归快照（改动前先记录，改动后对比；ARCHIVE 放在仓库外，如 `%TEMP%`）：
  `python -B tests/eom_regression.py baseline ARCHIVE` / `python -B tests/eom_regression.py compare ARCHIVE`（需设置 `PYTHONPATH=.`）
- `outputs/` 和大部分 `examples/input/*/*.csv` 被 .gitignore 忽略；`examples/input/example_01h_saving/` 已放行。
