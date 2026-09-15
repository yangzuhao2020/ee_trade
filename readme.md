# 第一版
开发一个面向学习和研究的轻量级电力现货市场仿真平台。它不是真实交易系统，而是用来研究“机组如何报价、市场如何出清、参与者如何获得利润”。
## 实现功能
### 用户可以配置：
仿真开始和结束时间
时间步长，例如1小时
市场最高价和最低价
出清方式 第一版仅支持 pay_as_clear
发电机组参数
各时刻的负荷
燃料价格和碳价

**第一版可以继续使用 YAML 和 CSV 文件输入。**
### 市场出清
第一版采用单节点、统一出清机制：
	收集需求侧买单和所有机组卖单。
	按报价从低到高排序。
	依次接受报价，直到满足负荷。
	最后一笔被接受的机组卖单价格成为市场出清价格。
	最后一台机组允许部分成交。
	所有成交机组按照统一价格结算。
	如果全部机组的可用容量仍不能满足需求，则接受全部有效卖单，并记录未满足负荷。
### 输出结果
平台输出：
	各机组报价
	各机组成交功率和成交电量
	市场出清价格
	各机组收入、成本和利润
	市场总成交电量
	市场总交易金额
	未满足负荷
	各运营商总利润
### 文件映射
| 文件                     | 第一版用途            |
| ---------------------- | ---------------- |
| `config.yaml`          | 读取仿真时间、市场限价和出清机制 |
| `demand_df.csv`        | 提供各时段负荷          |
| `demand_units.csv`     | 定义负荷参与者及其运营商     |
| `fuel_prices_df.csv`   | 提供燃料价格与碳价        |
| `powerplant_units.csv` | 建立机组并计算边际成本      |
## 数据来源
输入：`examples/input/example_01a`
输出：`outputs/example_01a`

## 第一版范围

第一版只支持 `config.yaml` 中的 `base` 场景：单节点、每小时一个 1 小时交付产品、`pay_as_clear` 统一出清。`exchange_units`、日前多产品和 `complex_clearing` 明确不在第一版范围内。

机组的 `min_power` 会被读取和校验，但第一版不模拟启停、最小开机时间或爬坡约束；因此每个时段的可报价容量为 `max_power`。

同价卖单按照机组名称升序打破平局，以保证同一输入每次运行产生相同结果。

边际成本与 ASSUME 的 `example_01a` 保持一致：

$$
MC = \frac{fuel\_price + co2\_price \times emission\_factor}{efficiency} + additional\_cost
$$

## 运行

安装依赖后执行：

```bash
python -m pip install -e .
electricity-market-sim
```

默认读取 `examples/input/example_01a`，并写入以下文件：

- `outputs/example_01a/market_results.csv`：逐交付时段的出清价、成交电量、交易金额和未满足负荷；
- `outputs/example_01a/unit_results.csv`：逐机组的报价、成交功率/电量、收入、成本和利润；
- `outputs/example_01a/operator_results.csv`：各发电运营商的累计成交电量、收入、成本和利润。

CSV 写入成功后，默认还会在 `outputs/example_01a/plots/` 生成：

- `market_overview.png`：出清价格、需求电量、成交电量和未满足负荷；
- `dispatch_by_unit.png`：逐机组成交功率、需求功率和边际机组标记；
- `operator_profit.png`：各发电运营商累计利润；
- `merit_order_first_product.png`：首个产品的供给曲线、需求、出清价和边际机组。

图表从完整的仿真结果对象单独生成，且只在 CSV 成功写入之后执行；即使 Matplotlib 或绘图发生错误，已有 CSV 结果也会保留。若不需要图表，可加入 `--no-plots`：

```bash
electricity-market-sim --no-plots
```

也可以显式指定目录：

```bash
electricity-market-sim \
  --input-dir examples/input/example_01a \
  --output-dir outputs/example_01a \
  --scenario base
```

## 报价方式
### 需求侧报价
`demand_energy_naive` 用于表示完全无弹性负荷：
- 针对市场提供的交付产品生成买单。
- 买单价格设置为市场最高价，本例为 `3000 EUR/MWh`。
- 买单电量等于对应交付时段的需求电量。
- 在 ASSUME 中，买入电量使用负数表示。

### 发电侧报价
`powerplant_energy_naive` 表示机组按照边际成本报价：
- 针对市场提供的交付产品生成卖单。
- 报价价格等于机组边际成本。
- 报价电量等于机组当前可用容量。
- 每台机组为下一个交付时段提交一个卖单。

## 负荷数据需要重采样
> `demand_df.csv` 是 15 分钟数据，但仿真时间步长是 1 小时。

采用小时平均功率：
$$
D_h = \frac{D_1 + D_2 + D_3 + D_4}{4}
$$
例如 00:00—01:00：
$$
D_{00:00-01:00} = \frac{2196.6 + 2173.0 + 2147.6 + 2136.0}{4} = 2163.3\ \mathrm{MW}
$$
由于交付时间为 1 小时：
$$
E = 2163.3 \times 1 = 2163.3\ \mathrm{MWh}
$$
01:00—02:00 的平均负荷为：
$$
D_{01:00-02:00} = 2082.7\ \mathrm{MW}
$$
**因此，仿真在00:00开始后，第一次生成的订单对应01:00—02:00，而不是00:00—01:00**

00:00 市场开启 → 创建 01:00—02:00 的交易产品 → demand_energy_naive 为该产品提交需求买单 → 发电机也为该产品提交卖单 → 01:00 市场关闭并出清 → 01:00—02:00 交付电能

第一版默认使用 `config.yaml` 中的 `base` 场景。平台根据 `start_date` 和 `end_date` 读取对应负荷数据，只创建交付结束时间不晚于 `end_date` 的市场产品。
