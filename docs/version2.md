# 第二版
在第一版基础上，加入启发式报价策略 和 考虑机组可用率。
## 机组配置
继续使用 `powerplant_units.csv`。
主要字段：
```text
name
technology
bidding_EOM
fuel_type
emission_factor
max_power
min_power
efficiency
additional_cost
unit_operator
```
支持：
```text
powerplant_energy_naive
powerplant_energy_heuristic_flexable
```
### 市场出清
和第一版本相同。
但是，存在块报价时，将同一次市场开放包含的全部时段联合出清，并使用混合整数优化处理块报价的全接受或全拒绝约束。
## 报价方式
### 需求侧报价
`demand_energy_naive` 和第一版本相同。
	读取 `demand_df.csv`。
`demand_energy_heuristic_elastic` [新增]
	`demand_elastic` 不读取 `demand_df.csv`，而是直接根据以下参数生成需求。
同时报价不再是只有一份报价，而是多份报价。
### 发电侧报价
`powerplant_energy_naive` 和第一版本相同。
`powerplant_energy_heuristic_flexable` [新增]
`powerplant_energy_heuristic_block`[新增]
`powerplant_energy_heuristic_linked`[新增]
### 文件映射
| 文件                     | 第一版用途             |
| ---------------------- | ----------------- |
| `config.yaml`          | 读取仿真时间、市场限价和出清机制  |
| `demand_df.csv`        | 提供各时段负荷           |
| `demand_units.csv`     | 定义负荷参与者及其运营商      |
| `fuel_prices_df.csv`   | 提供燃料价格与碳价         |
| `powerplant_units.csv` | 建立机组并计算边际成本       |
| `exchange_units.csv`   | 外部电网              |
| `exchanges_df.csv`     | 进口量和出口量均以正数保存     |
| `availability_df.csv`  | 随时间变化的可用率         |
| `config.yaml`          | 交易中的配置文件          |
| `storage_units.csv`    | 储能参与电能量市场的启发式报价策略 |
`availability_df.csv` 中的可用率范围为 `[0,1]`。

`unit_results.csv` 保留全部发电机组在各交付时段的汇总结果。
`offer_results.csv` 只保留需要分段或约束信息的机组报价：同一交付时段有多段报价，或者存在块报价、关联报价、父报价关系或非零最小接受比例。
一旦某机组在一次开市中满足上述条件，该机组在该次开市的全部报价都会保留，包括未成交报价。各时段仅有一段无关联、最小接受比例为零的普通报价时，只保留机组汇总结果；筛选依据是实际报价结构，与发电技术类型及分段名称无关。
筛选后没有明细时不生成 `offer_results.csv`，并清理当前输出目录中上次运行遗留的同名文件。全市场数量和金额的统计应使用 `unit_results.csv`。

# powerplant_energy_heuristic_flexable
考虑最低出力、启停状态、启动成本、爬坡、未来价格和热电联产。
	将一台机组拆成两条报价。第一部分属于最低稳定出力，然后第二部分属于灵活出力。
## 灵活出力定价
- 灵活出力通常按边际成本报价。增加出力会增加总燃料与碳排放成本；
- 但在当前实现中，单位电量的边际成本不随出力变化。
## 最低出力怎么定价
最低出力的价格取决于机组当前是“运行”还是“停机”。
注意：如果 csv 表格中设定 min_power = 0 就意味着全部都是灵活定价。
### 停机
如果要让机组启动，除了燃料成本，还要支付启动成本。$$p_{\text{inflex}} = C_{\text{marginal}} + \frac{C_{\text{start}}}{T \times q_{\text{inflex}}}$$其中： $$T=\max(\text{avg\_op\_time},\text{min\_operating\_time})$$
$$\text{avg\_op\_time} = \frac{\text{累计出力大于 0 的时段数}}{\text{截至当前的总时段数}}$$
	注意：名称虽为 op_time，但不是平均连续运行小时数。
	q_inflex 当可用出力不低于 `min_power` 时，最低出力报价量等于 `min_power`。
	所以，边际成本 + 启动费用=最低出力报价。($C_{start}$启动成本可能为0)
### 运行
如果现在关闭机组，将来再启动，还会产生启动成本。所以机组可能愿意让最低出力暂时少赚一点，甚至亏一点，也不愿意关闭。$$p_{\text{inflex}} = C_{\text{flex}}^* - \text{避免重启带来的价值} - \text{供热相关的报价下调（成本分摊）}$$
当未来 12 小时累计预期收益**非负**、且当前预测电价低于边际成本时，代码会将 $C_{\text{flex}}^{*}$ 设为 0。
未来价格预测：
	该策略默认读取从当前时点开始、长度约 12 小时的 `EOM` 价格预测，并计算累计预期收益：
$$
R_t=\sum_{\tau=t}^{t+12h}\left(p_{\tau}^{\text{forecast}}-C_{\text{marginal}}\right)$$
  当 $R_t \ge 0$，且当前小时预测价格 $p_t^{\text{forecast}} < C_{\text{marginal}}$ 时，代码会将 $C_{\text{flex}}^{*}$ 设为 0。

```mermaid
flowchart LR
  A["demand_df.csv<br/>未来每个时段的负荷"] --> B["加载并按仿真步长重采样"]
  B --> C["DemandForecaster.demand<br/>demand = -abs(demand_df[unit])"]
  C --> D["calculate_sum_demand()<br/>取绝对值并汇总所有非弹性需求"]
  D --> E["calculate_naive_price()<br/>与同一时段的可用供给比较"]
  E --> F["price['EOM'][t]<br/>边际机组成本"]
```
### 成交
注意：
- 报价前：可用出力低于 `min_power`，不报价；
- 出清时：已经提交的最低出力段允许部分成交。
## 数据来源
### 测试1
输入：`examples/input/example_01b`
输出：`outputs/example_01b`
### 测试2
输入：`examples/input/example_01c`
输出：`outputs/example_01c`

| `demand_df.csv` 列 | `demand_units.csv` 中的负荷 |
| ----------------- | ----------------------- |
| `demand_EOM`      | `demand_EOM`            |
| `demand_CRM_pos`  | 正备用容量需求                 |
| `demand_CRM_neg`  | 负备用容量需求                 |
| 无对应列              | `demand_elastic`        |
### 其他的设置
启动成本 = 0
爬坡约束 = 不启用
供热出力 = 0

`exchanges_df.csv` 同样是15分钟功率数据，采用和第一版相同的方式计算。

# powerplant_energy_heuristic_block
该策略将机组的最低稳定出力打包成跨时段块报价，最低稳定出力以上的灵活出力仍然逐小时报价。

例如：
> 我愿意在这三个小时里每小时至少发200 MW，但这三小时的最低出力必须整体接受；不能只接受其中一个小时。
## 块报价范围
一个最低出力块覆盖同一次市场开放中的全部交付产品。
块报价测试使用 `config.yaml` 中的：dam_with_complex_opt_clearing
其市场配置为：
```yaml
products:
  - duration: 1h
    count: 24
    first_delivery: 24h
opening_frequency: 24h
market_mechanism: complex_clearing
additional_fields:
  - bid_type
  - min_acceptance_ratio
```
因此，市场每次联合出清24个1小时产品，最低出力形成一个24小时块。
**这里要考虑为未来交付时，处于运行还是停机状态。**
## 生成两类报价
$$P_{t}^{\text{available}} = P^{\max} a_{t},\quad a_{t}\in[0,1] $$当$P_t^{\text{available}} < P^{\min}$：$$
P_t^{\text{inflex}} = 0, \qquad P_t^{\text{flex}} = 0.
$$否则：
$$
P_t^{\text{inflex}} = P^{\min}, \qquad P_t^{\text{flex}} = P_t^{\text{available}} - P^{\min}.
$$此时：$$P_{t}^{available}=P_{t}^{\mathrm{inflex}}+P_{t}^{\mathrm{flex}}$$

| 组成                        | 含义          | 报价形式        |
| ------------------------- | ----------- | ----------- |
| $P_{t}^{\mathrm{inflex}}$ | 机组最低稳定出力    | 多个小时合成一个块报价 |
| $P_{t}^{\mathrm{flex}}$   | 最低出力以上的额外能力 | 每小时一个普通报价   |
## 块报价的价格怎么得到？
程序先分别计算每小时的最低出力报价，然后求电量加权平均价：
$$
C^{\text{block}} = \frac{\sum_{t} C_{t}^{\text{inflex}} P_{t}^{\text{inflex}} \Delta t}{\sum_{t} P_{t}^{\text{inflex}} \Delta t}.
$$这个公式是在计算多个小时最低出力报价的“电量加权平均价格”。其中的 $C^{\mathrm{block}}$ 为整个块只有一个报价价格，但是每个小时的电量可以不同。

最低出力块设置 `MAR=1`，意味着接受比例只能是 `0` 或 `1`，不能部分小时成交，也不能部分接受块内电量。

灵活出力仍然是独立的逐小时普通报价。因此可能出现：
> 最低出力块被拒绝，但某些小时的灵活出力成交。

第二版保留该行为，不建立最低出力块与灵活报价之间的父子约束。该行为可能导致机组出力低于最低稳定出力，属于本策略的已知限制。
# powerplant_energy_heuristic_linked
该策略在 `powerplant_energy_heuristic_block` 的基础上，为最低稳定出力和逐小时灵活出力建立父子关系。
- 最低稳定出力组成一个父块报价；
- 每小时的灵活出力分别作为子报价；
- 父块被拒绝时，所有子报价都必须被拒绝；
- 父块被接受后，各小时子报价才有资格成交。

情况1
	父块：拒绝
	08:00子报价：必须拒绝
	09:00子报价：必须拒绝
	10:00子报价：必须拒绝
情况2
	父块：接受
	三个子报价：全部拒绝
情况3
	父 和 子 都成交。

父块和子报价在同一次市场出清中联合决定。只有父块被接受，灵活出力子报价才允许成交。

## 配置文件
`powerplant_energy_heuristic_linked` 必须使用复杂出清，并在 `config.yaml` 中配置以下字段：
market_mechanism: complex_clearing
additional_fields:
- bid_type
- min_acceptance_ratio
- parent_bid_id

- `bid_type`：区分普通报价、块报价和关联报价；
    - `SB`：普通逐时报价；
    - `BB`：父块报价；
    - `LB`：关联子报价。
- `min_acceptance_ratio`：控制报价的最小接受比例。父块设置为 `1`，表示只能全部接受或全部拒绝。
- `parent_bid_id`：记录子报价对应的父块编号，用于建立父子依赖关系。
$$x_{\mathrm{child},t}\leq x_{\mathrm{parent}}.$$
- $x_parent​$：父块的接受比例；
- $x_{\mathrm{child},t}$：时段 tt 的子报价接受比例。

当父块被拒绝时，$x_{\mathrm{parent}}=0$，因此所有子报价的接受比例也只能为0。
# demand_energy_heuristic_elastic
把一个负荷拆成多档需求报价。电价低时多用电，电价高时少用电。

{
    "name": "demand_elastic",
    "technology": "elastic_demand",
    "bidding_EOM": "demand_energy_heuristic_elastic",
    "max_power": 20, # 最大可消费功率
    "min_power": 0,
    "unit_operator": "elastic_de",
    "elasticity": -0.05, # 需求价格弹性
    "elasticity_model": "isoelastic",  # 使用等弹性需求曲线
    "max_price": 3000, # 最高愿付价格
    "num_bids": 10 # 把需求曲线拆成10档报价
}

其中的弹性 `−0.05` 是什么意思呢？
$$ E = \frac{\text{需求量变化百分比}}{\text{价格变化百分比}} \quad E = -0.05$$
大致表示：
> 电价上涨1%，需求量下降0.05%。负号表示价格上涨时，用电量下降。
## 等弹性需求曲线
使用：
$$ Q(P) = Q_{\max} P^{E} $$
该负荷的最高愿付价格 `max_price` 为 3000 EUR/MWh。本例中它也等于市场价格上限。
$$
Q_{\text{first}} = Q_{\max} P_{\max}^{E}
$$
代入：
$$
Q_{\text{first}} = 20 \times 3000^{-0.05} \approx 13.402\ \text{MW}
$$
这表示：
> 即使电价达到3000 EUR/MWh，这个负荷仍然希望购买约13.402 MW。

因此第一档报价为：
```python
{
    "volume": 13.402,
    "price": 3000
}
```
剩余待拆分需求量为：20−13.402=6.598 MW
`num_bids=10`，除第一档外还需要生成9档，因此每档数量为：
$$
\frac{6.598}{9} \approx 0.733\ \text{MW}
$$
本平台沿用第一版规则，需求买单数量统一使用正数。

| 档位  |   本档数量    |   累计需求    | 最高愿付价格(EUR/MWh) |
| :-: | :-------: | :-------: | :-------------: |
|  1  | 13.402 MW | 13.402 MW |     3000.00     |
|  2  | 0.733 MW  | 14.135 MW |     1034.05     |
|  3  | 0.733 MW  | 14.868 MW |     376.15      |
|  4  | 0.733 MW  | 15.601 MW |     143.65      |
|  5  | 0.733 MW  | 16.335 MW |      57.34      |
|  6  | 0.733 MW  | 17.068 MW |      23.83      |
|  7  | 0.733 MW  | 17.801 MW |      10.28      |
|  8  | 0.733 MW  | 18.534 MW |      4.58       |
|  9  | 0.733 MW  | 19.267 MW |      2.11       |
| 10  | 0.733 MW  | 20.000 MW |      1.00       |
已知需求 如何求解价格呢？
举例：使用公式：
$$
Q_i = Q_{\text{first}} + i\Delta Q \quad i=1,...,9
$$
$$
P_i = \left( \frac{Q_i}{Q_{\max}} \right)^{1/E}
$$已知：
- $i = 1$：第二档；
- $i = 9$：第十档；
- 第一档直接使用 $Q_{\text{first}}$ 和 $P_{\max}$。

第二档：
$$
\begin{align*}
Q_1 &= 13.402133 + 0.733096 = 14.135229\ \text{MW} \\
P_1 &= \left(\frac{14.135229}{20}\right)^{-20} \approx 1034.05\ \text{EUR/MWh}
\end{align*}
$$
第一档的含义也可以进一步表述为：理论曲线上累计需求低于 $13.402\text{ MW}$ 的部分，愿付价格都高于市场限价 `3000`，因此统一合并为一档，以 `3000 EUR/MWh` 报价。
# storage_energy_heuristic_flexable
## 设备归属与经营目标
储能通过 `storage_units.csv` 中的 `unit_operator` 指定所属运营商。设备归属与交易方向分别定义：属于发电卖方运营商的储能，充电时仍可作为买方购电，放电时作为卖方售电。

在 `examples/input/example_01c` 中，`Storage 1` 与核电机组 `Unit 1` 均属于 `Operator 1`。本场景将储能定义为发电卖方运营商持有的经营资产，不将其解释为另一个独立第三方，也不将其绑定为某个用电买方的内部设备。

运营商的经营目标是自身资产组合的利润最大化，应综合考虑发电与储能售电收入、储能购电支出、发电成本及储能运行成本。该目标与市场出清的社会福利最大化目标不同。

第二版采用下述设备级启发式策略生成储能报价。发电机组和储能仍分别按各自策略报价；相同的 `unit_operator` 表示共同归属，不自动建立专线供电关系或联合调度、联合报价优化。启发式策略服务于盈利目标，但不代表已经求得运营商整体利润的最大值。

## 报价策略与市场范围
同一设备可针对不同市场配置不同策略：

| 配置字段 | `example_01c` 中的策略 | 适用市场 |
| --- | --- | --- |
| `bidding_EOM` | `storage_energy_heuristic_flexable` | 电能量市场 |
| `bidding_CRM_pos` | `storage_capacity_heuristic_balancing_pos` | 正向备用容量市场 |
| `bidding_CRM_neg` | `storage_capacity_heuristic_balancing_neg` | 负向备用容量市场 |

第二版的储能交易范围为 EOM，仅执行 `bidding_EOM` 策略。CSV 中的两个备用策略名称作为配置保留，本版不执行备用报价、备用出清或备用调用，不得将这些名称视为已经实现的算法能力。

## 充电来源与需求关系
储能通过 EOM 提交充电买单，从市场购电。充电没有指定由同一运营商旗下的某台发电机供电；市场供给可来自发电机组、外部电网进口及其他储能放电，且均受各自报价和物理约束限制。只有实际成交的充电电量才能计入储能补能；未成交的充电买单不增加 SOC。

`natural_inflow` 表示市场购电之外的自然补能来源。`example_01c` 中该值为 `0`，因此除初始储电量外，后续补能来自市场购电。

`demand_df.csv` 提供非弹性负荷，弹性需求由参数生成。非弹性负荷参与 EOM 预测电价计算，各类买单参与实际市场出清。储能通过预测价格间接感知供需，不直接按某个买方的用电量确定充放电计划，也不承担保障某个指定买方用电的义务。

## 设备参数
```json
{
    "name": "储能设备名称",
    "technology": "储能技术类型",
    "bidding_EOM": "电能量市场报价策略",
    "bidding_CRM_pos": "正向备用市场报价策略",
    "bidding_CRM_neg": "负向备用市场报价策略",
    "max_power_charge": "最大充电功率（MW）",
    "max_power_discharge": "最大放电功率（MW）",
    "efficiency_charge": "充电效率",
    "efficiency_discharge": "放电效率",
    "min_soc": "最低荷电状态（0～1）",
    "max_soc": "最高荷电状态（0～1）",
    "capacity": "储能容量（MWh）",
    "additional_cost_charge": "充电附加成本（EUR/MWh）",
    "additional_cost_discharge": "放电附加成本（EUR/MWh）",
    "natural_inflow": "自然流入带来的补能；普通电池通常填 0",
    "unit_operator": "所属运营商",
    "initial_soc": "初始荷电状态比例（0～1）"
}
```

储能报价与交付遵循以下规则：

- 充电时，储能是需求方，报价量为负数；
- 放电时，储能是供给方，报价量为正数；
- 根据预测电价决定充电还是放电；
- 受到 SOC、容量、充放电功率和效率限制。
## 基本规则：
```
当前预测电价 < 预测窗口平均电价
→ 现在便宜
→ 充电买电

当前预测电价 > 预测窗口平均电价
→ 现在较贵
→ 放电卖电

当前预测电价 = 预测窗口平均电价
→ 不提交充放电报价

储能净报价量：充电为负，放电为正。生成本平台内部 `DemandBid` 时，充电买单的 `volume_mwh` 存正的购买量；进入复杂出清模型时，需求量转换为负数。充电成交后，再用负的实际成交电量更新 SOC。转换电量符号时，报价价格不变。
```

|           参数           |    数值    |     含义      |
| :--------------------: | :------: | :---------: |
|   `max_power_charge`   | 1000 MW  | 本平台采用最大充电功率 |
| `max_power_discharge`  |  992 MW  | 本平台采用最大放电功率 |
|  `efficiency_charge`   |   0.86   |    充电效率     |
| `efficiency_discharge` |   0.90   |    放电效率     |
|       `min_soc`        |    0     |   最低荷电状态    |
|       `max_soc`        |    1     |   最高荷电状态    |
|       `capacity`       | 6076 MWh |   最大储能容量    |
## SOC 是什么，如何更新？
SOC 是 State of Charge，中文叫荷电状态，表示储能当前有多满。其中：${initial\_soc}\in[0,1]$ 初始荷电状态比例。

| SOC  |   含义    |
| :--: | :-----: |
|  0   |  完全没电   |
| 0.25 | 剩余25%容量 |
| 0.5  | 剩余50%容量 |
|  1   |  完全充满   |

假设充电功率为 $P_t$ < 0，则：$SOC_{t+1} = SOC_t + (-P_t * Δt * η_c) / C$
其中：
- $P_t$：实际成交电量除以时长，充电时为负；
- $Δt$：时间长度；
- $η_c$：充电效率；
- $C$：储能容量。

假设放电功率为 $P_t$ > 0，则：
$$
SOC_{t+1} = SOC_t - \frac{P_t \Delta t}{\eta_d C}
$$
## 储能什么时候充电、什么时候放电
- 策略根据预测价格决定提交哪一种报价。
- 市场出清后，才决定是否真的充电或放电。

在 `storage_energy_heuristic_flexable` 中，默认计算当前时刻前后约 12 小时(12小时是本平台选定的预测窗口)价格预测的平均值：$$\bar{p} = \operatorname{mean}(p_{t-12h},\ \ldots,\ p_t,\ \ldots,\ p_{t+12h})$$

仿真起止边界按照 ASSUME 的方式处理：当完整的前后各 12 小时窗口超出价格预测序列范围时，将窗口截断到已有预测范围，即
$$
t_{\mathrm{window,start}}=\max(t-12h,\ t_{\mathrm{forecast,start}}),\qquad
t_{\mathrm{window,end}}=\min(t+12h,\ t_{\mathrm{forecast,end}}).
$$
边界窗口可以少于 25 个小时点，不因缺少仿真开始前或结束后的预测数据而跳过开市；但截断后窗口内部需要使用的数据若缺失，仍跳过该次开市并报告数据缺口。

- 当前预测价格较高，SOC 允许放电，策略提交：
	- 正功率报价；
	- 作为供给方卖电；
	- 最大功率受到放电功率上限和 SOC 约束，本版不启用爬坡约束。
	
	放电报价：$$p_{\text{discharge}} = \frac{\bar{p}}{\eta_d}$$
- 当前预测价格较低，SOC 允许充电，策略提交：
	- 负功率报价；
	- 作为需求方购买电能充电；
	- 最大功率受到充电功率上限和 SOC 约束，本版不启用爬坡约束。
	
	充电报价：$$p_{\text{charge}} = \bar{p}\,\eta_c$$

## 充放电申报量与实际成交量
价格比较决定报价方向，设备物理条件决定申报量上限，市场出清决定实际成交量。选择某一方向后，本启发式策略按该方向的可用上限申报，不直接按市场需求量缩减申报，也不求解“当前出售多少、为未来保留多少”的跨时段利润优化问题。

设 $E_t$ 为交付时段开始时的报价参考内部储电量，单位 MWh；$C$ 为容量，$\Delta t$ 为产品时长，单位 h。选择充电时，向市场申报的购电量为：
$$
E_{\text{charge},t}^{\text{bid}}
=\min\left(P_{\text{charge}}^{\max}\Delta t,\ \frac{SOC_{\max}C-E_t}{\eta_c}\right).
$$
选择放电时，向市场申报的售电量为：
$$
E_{\text{discharge},t}^{\text{bid}}
=\min\left(P_{\text{discharge}}^{\max}\Delta t,\ (E_t-SOC_{\min}C)\eta_d\right).
$$
上述电量均为非负的电网侧电量；可申报量为零时不提交对应订单。同一交付时段只选择一个充放电方向。

首个交付时段的参考储电量应计入此前已成交但尚未交付的安排。一次开市有多个产品时，生成后续报价所用的参考电量轨迹按此前申报全部成交推算，仅用于生成报价，不作为真实 SOC。复杂出清必须按实际成交量对全部产品施加跨时段 SOC 约束；真实状态在交付后按实际执行量更新。

例如，`example_01c` 的储能容量为 `6076 MWh`、最大放电功率为 `992 MW`、放电效率为 `0.90`。当最低 SOC 为 `0`、报价参考 SOC 为 `0.5` 时，一小时最多申报 `min(992, 3038 × 0.90) = 992 MWh`。若只成交 `300 MWh`，则实际向市场交付 `300 MWh`，内部储电量减少 `300 / 0.90` MWh，不按申报的 `992 MWh` 更新 SOC。

## 成本与结算
`additional_cost_charge` 和 `additional_cost_discharge` 不进入储能报价价格。这里与 ASSUME 保持一致：报价阶段只考虑预测均价和充放电效率；市场出清后，再按照实际成交电量计算附加成本。

若充电成交电量和放电成交电量分别为 $E_{\text{charge}}$、$E_{\text{discharge}}$，则：
$$
C_{\text{additional,charge}}
=E_{\text{charge}}\,c_{\text{additional,charge}},
$$
$$
C_{\text{additional,discharge}}
=E_{\text{discharge}}\,c_{\text{additional,discharge}}.
$$

其中成交电量均按非负数计算。单个交付时段的储能净现金流为：
$$
\text{net cash flow}
=E_{\text{discharge}}p_{\text{clear}}
-E_{\text{charge}}p_{\text{clear}}
-C_{\text{additional,charge}}
-C_{\text{additional,discharge}}.
$$

上述净现金流是储能设备的交易收支组成部分，不等于所属运营商全部资产的利润。运营商整体利润应另行汇总其发电机组与储能的收入、支出和成本。

因此，附加成本会降低储能最终利润，但不会改变该储能订单的申报价，也不会仅因为这项成本而改变市场出清顺序。

例如：**预测均价为 50 EUR/MWh 的例子**：预测均价为 50，且已分别提交充电单或放电单，市场价格本身不能决定储能报哪种单。43 是充电买单的最高愿付价，55.56 是放电卖单的最低愿售价。预测价决定**报哪种单**，实际出清决定**是否成交**；

|   市场价格    | 经济动作 | 结果   |
| :-------: | :--: | ---- |
|  不高于 43   | 适合充电 | 可能成交 |
| 不低于 55.56 | 适合放电 | 可能成交 |

## 未成交与需求缺口的统计
选择充电不保证买单成交。例如，预测均价为 `50 EUR/MWh`、当前预测价为 `45 EUR/MWh` 时，策略会选择充电，但充电效率为 `0.86` 时最高愿付价仅为 `43 EUR/MWh`；若可成交供给的价格为 `45 EUR/MWh`，这笔买单不能成交。供给不足、其他买单竞争或跨时段 SOC 约束也可能限制成交量，不能将所有未成交都归因于价格。

储能未成交的充电电量表示未执行的充电计划，不属于未满足需求，也不等同于终端用电负荷缺供。输出应区分：

- `unserved_load_mwh`：仅统计刚性负荷的未满足电量，不计入储能充电未成交量。
- `unserved_total_demand_mwh`：除储能充电以外全部买单的未成交电量合计，不能直接解释为刚性负荷缺供。
- `unaccepted_storage_charge_mwh`：储能充电未成交量，即充电申报电量减去充电成交电量，单独统计；单台储能的数值可由 `storage_results.csv` 中的 `offered_charge_mwh` 与 `accepted_charge_mwh` 计算。

`market_results.csv` 满足：`requested_demand_energy_mwh = cleared_energy_mwh + unserved_total_demand_mwh + unaccepted_storage_charge_mwh`。

图表中，市场概览的“总申报需求”及成交需求堆叠均不含储能充电，也不展示储能未成交充电。

预测价格选择报价方向→SOC确定可报数量→市场出清→成交后才真正更新SOC。

- `SOC = min_soc`：达到最低保留电量，不能放电；
- `SOC = max_soc`：达到最高允许电量，不能继续充电；
- `min_soc < SOC < max_soc`：可能充电或放电，但数量还受功率及配置的 SOC 上下限限制。

|阶段|使用的量|作用|
|---|---|---|
|制定报价|可充/放功率、预测价格、当前 SOC|决定最多报多少|
|市场出清|`accepted_volume`|确定实际充电或放电量|
|状态更新|实际成交量|更新真实 SOC|
## 完整流程
- SOC 表示剩余电量比例；
- 预测电价决定想充电还是放电；
- 报价价格考虑充放电效率；
- 实际成交量决定 SOC 更新。

预测价格→生成储能报价→进入市场排序→确定出清价格→得到成交量→计算收益→更新SOC
边界表述 `SOC=min_soc` 不能放电、`SOC=max_soc` 不能充电。
# 确定最后的价格
## 复杂出清
当订单簿中出现 `BB` 块报价或 `LB` 关联报价时，ASSUME 将同一次市场开放包含的全部交付时段放入同一个优化模型中联合出清。

优化目标本质上是最大化社会福利：
$$
\max(\text{需求愿付价值} - \text{供给报价成本})
$$
ASSUME 实际写成：$$
\min \sum_{o} \sum_{t \in T_o} p_o q_{o,t} x_o
$$ASSUME 使用带符号的订单量：
- 供给报价量为正数；
- 需求报价量为负数。

因为需求量是负数，所以需求项会降低目标值，效果等价于最大化社会福利。
每个时段满足：$$
\sum_{o} q_{o,t} x_o = 0
$$$x_o$  表示订单 o 的接受比例。对于块报价，同一个 $x_o$ 应用于块内全部交付时段。
并加入：
- 普通报价接受比例；
- 当块报价的 `min_acceptance_ratio=1` 时，块报价只能整体接受或整体拒绝。
- 最小接受比例；
- 子报价的接受比例不超过父报价的接受比例。
例如：$$
x_{\text{child}} \leq x_{\text{parent}}
$$子报价的接受比例不能超过父报价的接受比例。
## 如何确定价格
ASSUME 使用两阶段求解：
1. 求解 MILP，确定块报价接受还是拒绝；
2. 固定 MILP 求得的整数接受决策。
3. 重新求解连续 LP；
4. 读取每个时段供需平衡约束的对偶值。
$$
p_t^{\text{clear}} = \lambda_t
$$
其中，$\lambda_t$ 是时段供需平衡约束的对偶值。
因此，一次联合出清包含24个1小时交付产品时，最终会得到24个时段价格，而不是一个24小时统一价格。

块报价虽然只有一个 `bid_price`，但成交后的 `accepted_price` 按小时保存。例如：
块报价价格：40 EUR/MWh  
	08:00 结算价格：35 EUR/MWh  
	09:00 结算价格：42 EUR/MWh  
	10:00 结算价格：51 EUR/MWh

块报价的 `bid_price` 是覆盖全部时段的统一报价价格；块报价被接受后，块内各时段分别按照对应时段的市场出清价格结算。因此，`accepted_price` 不一定等于 `bid_price`，各时段之间也可以不同。
单个时段的结算价格可以低于块报价价格，此时该时段产生负盈余，但不会单独拒绝该时段；ASSUME按照整个块的合计盈余进行检查。$$
S_{\text{block}} = \sum_{t \in T} Q_t \Delta t \left( P_t^{\text{clear}} - P^{\text{bid}} \right) 
$$其中：
- $Q_t$：块报价在时段 $t$ 的功率，单位 MW；
- $\Delta t$：时段长度，单位 h；
- $Q_t \Delta t$：该时段成交电量，单位 MWh；
- $S_{\text{block}}$：整个块的盈余，单位 EUR。

ASSUME 随后检查每个订单的盈余。对于普通块报价，按照 $S_{\text{block}}$ 检查。对于存在关联子报价的父块，按照父子报价的合计盈余检查：
$$
S_{\text{family}} = S_{\text{parent}} + \sum_{c} Q_c x_c \Delta t \left( P_c^{\text{clear}} - P_c^{\text{bid}} \right)
$$
其中，$x_c$ 是子报价的接受比例，$Q_c x_c \Delta t$ 是子报价的实际成交电量。关联子报价的正盈余可以补偿父块的负盈余。若 $S_{\text{family}} < 0$，则删除父块及其关联子报价，并重新出清。

`bid_price` 参与优化目标和块报价接受决策；`accepted_price` 用于计算各时段的实际结算金额。

# 注意的地方
demand_EOM → 进入EOM demand_elastic → 进入EOM demand_CRM_pos → 忽略 demand_CRM_neg → 忽略

当一次市场开放要求生成24个1小时交付产品，但仿真结束时间或输入数据不足以覆盖完整24小时时，本次市场开放整体跳过。不得截断产品数量、缩短块报价或使用零值补齐。程序应输出警告，说明被跳过的开市时间及缺失时段。

多个负盈余块同时出现时，删除顺序是什么？
	每轮出清后，计算各已接受块报价的盈余；对关联报价，计算父块与子报价的合计盈余。若有多个负盈余报价组，按盈余从低到高选取一个删除（同值时按 `bid_id` 排序），然后重新出清。重复此过程，直到没有负盈余报价组。

某个 BB 在 24 小时内全部因可用容量低于 `min_power` 而为零时，应当“不提交 BB，也不提交零量 SB/LB”。

只要市场配置为 `complex_clearing`，就使用优化模型出清，并从各时段供需平衡约束的对偶值取价。不要根据这次订单簿有没有 `BB/LB` 临时切换出清算法。

对使用启发式最低稳定出力定价的机组，完整性检查还必须覆盖从每个交付时段开始、长度为 12 小时的价格预测所依赖的数据。若该预测所需的需求、可用率或其他输入数据缺失，也跳过整次市场开放；警告应注明这是报价预测数据缺失。
