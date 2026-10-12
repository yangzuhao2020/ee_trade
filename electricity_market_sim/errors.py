"""电力市场仿真的输入、运行状态及绘图异常类型。"""

# 基类：__main__.py 捕获它后只打印一行错误信息并以退出码 2 结束，不输出堆栈。
# KeyError、ZeroDivisionError 等不属于此体系的异常视为程序缺陷，会带完整堆栈抛出。
class SimulationError(Exception):
    """Base exception for invalid simulation input or state."""


# 输入错误：配置字段缺失或越界、CSV 格式错误、时段数据缺失、机组参数不合法等。
# 在 config/、inputs/、simulators/、clearing/ 等模块中广泛使用。
class InputValidationError(SimulationError):
    """Raised when an input file or configured scenario is invalid."""


# 绘图错误：仅在 plotting/ 中使用。抛出时结果 CSV 已经写出，只是 PNG 未能生成。
# plotting/__init__.py 会把绘图过程中的其他异常统一包装成该类型。
# 注意：__main__.py 中绘图位于 except SimulationError 之外，实际由 except Exception 捕获。
class PlottingError(SimulationError):
    """Raised when a completed simulation result cannot be rendered as plots."""
