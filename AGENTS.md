# Project Rules

- **未经用户明确许可，不得修改代码。** 默认只做分析、解释和运行验证；任何代码修改必须先获得用户同意。

# Verification

- 运行测试需使用 `ee_trade` conda 环境（默认 base 环境缺少 scipy）：
  `C:\Users\yang\miniconda3\envs\ee_trade\python.exe -m unittest discover -s tests`
