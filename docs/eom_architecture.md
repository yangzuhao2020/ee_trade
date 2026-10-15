# EOM 仿真分层

V1/V2、V4、V5 共用常规市场。V1/V2 的入口 `simulate_v1_v2_market` 位于
`eom.py`，V4 的入口在 `industry_market.py`，V5 保留在 `v5.py`。V1/V2、V5
共用事件引擎；V4 与 V3 一样在场景入口中直接组织运行流程。V4 调用基础市场
完成常规报价、储能、出清及交付状态更新。

```text
simulation.py
  ├─ eom.py：共用事件引擎、扩展协议；simulate_v1_v2_market 为基础市场入口
  ├─ industry_market.py：simulate_v4_market 直接组织工业计划、出清和交付事件
  ├─ v5.py：基础市场 + 学习扩展
  └─ household_market.py：原有 V3 家庭仿真

base_market/
  ├─ market.py：数据准备、预检、报价出清组织、状态初始化和回写
  ├─ demand.py：刚性/弹性需求与 Exchange 订单
  ├─ plants.py：常规机组报价、启动成本和运行状态
  ├─ storage.py：储能报价、承诺电量投影和 SOC 回写
  └─ forecasts.py：朴素价格预测、预测窗口及开放完整性检查

eom.py：V1/V2、V5 的开市、交割开始/结束事件，结果队列、扩展协议和 V1/V2 入口
```

具体报价公式保留在 `bidding.py`，出清算法保留在 `clearing_algorithm/`。

## 第四版直接调用流程

`simulate_v4_market()` 创建 `BaseMarket`，通过 `prepare(input_loader=...)`
加载工业预测和原料价格，再创建 `IndustryRollingCoordinator`。输入加载器仅用于
准备数据，交易阶段直接调用协调器：

```text
simulate_v4_market()
  ├─ 准备基础市场、工业输入及滚动协调器
  └─ 按时间处理事件
       ├─ 交割结束：market.end_delivery() → coordinator.record_delivery()
       ├─ 开市：coordinator.bids_for_products() → market.clear_opening()
       │         优化或复用生产计划、生成工业订单，再与常规及储能订单一起出清
       └─ 交割开始：market.start_delivery()，补计机组启动成本
```

工业订单通过 `clear_opening(..., additional_demand_bids=...)` 显式传入。
所有事件完成后检查钢需求，并汇总市场、储能、工业执行、灵活性和窗口成本结果。

`MarketSettings.opening_duration` 是必填配置，读取时检查为正值；目前未用于事件调度，
出清发生在开市事件中，没有单独的关市事件。`PowerPlant.min_down_time_hours` 会被读取
并检查为非负值，但目前未实现最小停机时间约束。`min_operating_time_hours` 用于启动
成本分摊，不代表已经实现最小连续运行时间约束。

学习会话负责动作维度、有限值校验和按需裁剪，再把已处理动作传给 `learning_offers()`。
经验记录保留动作原顺序，只有生成报价时对价格排序。

## 共用事件引擎的扩展生命周期

1. 基础市场读取机组、储能、需求曲线和可用率，调用扩展的 `prepare(market)`。
   扩展声明额外参与者、燃料/价格数据和预测时刻偏移；V5 在这里校验学习机组。
2. 基础市场检查开放窗口并生成预测、初始化运行状态，调用 `initialize(market)`。
   V5 以有效交割时段计算剩余负荷及归一化基准，然后建立学习会话。
3. 每次开市先收集 `bids_for_products` 的额外需求；逐台机组调用
   `offers_for_plant`。返回 `None` 表示使用常规报价，返回空元组表示不报价，
   返回订单表示完整替换该机组报价。预测价格始终按原有边际成本计算。
4. 交割开始附加启动成本后调用 `record_delivery_start`；交割结束回写常规状态，
   再调用 `record_delivery`；最终调用 `finalize` 附加各扩展的结果。

同一时刻的顺序保持为：

| 场景 | 顺序 |
| --- | --- |
| V1/V2、V4 | 交割结束 → 开市 → 交割开始 |
| V5 | 交割结束 → 交割开始及奖励记录 → 开市及下一动作 |

V1/V2、V5 的事件引擎通过 `settle_before_opening` 声明这个顺序；
V4 在自己的事件循环中明确按“交割结束 → 开市 → 交割开始”执行。

## 回归验证

`tests/eom_regression.py`、`tests/test_eom_engine.py` 和
`tests/test_industry_market.py` 纳入版本管理。
示例 YAML 由 Git 管理，CSV 数据目前不随仓库分发。运行回归前，需要在
`examples/input/` 下补齐 `example_01a`、`example_01b`、`example_01c`、
`example_01h`、`example_03` 和 `example_02a` 的配套 CSV。

基线应从待修改版本生成。当前代码已包含修改时，应将旧提交检出到独立工作树，
补齐相同 CSV 后在那里生成基线；随后在当前工作树中比较。两边使用同一 Python 环境、
配置和输入数据。已有基线继续用于比较，不覆盖或重新生成。

在项目根目录执行；`baseline` 命令仅在待修改版本且基线目录不存在时运行：

```powershell
conda activate ee_trade
python -B -m unittest discover -s tests -v
python -B tests/eom_regression.py baseline outputs/eom_baseline
# 修改实现后，以同一个基准目录进行比较。
python -B tests/eom_regression.py compare outputs/eom_baseline
```

基准脚本遍历六个示例数据集的 YAML 场景，保存完整内存结果、警告、回调序列
和 CSV。比较包括文件集合、CSV 字节、完整结果以及动作/经验回调顺序。
原有失败场景也记录异常类型和消息，并要求重构后保持一致。
动作边界模式、越界动作、非法维度及 NaN/Inf 由 `test_eom_engine.py` 补充验证。
第四版独立事件时序、待交付产量、部分成交补产及最终缺口由
`test_industry_market.py` 使用内存输入验证，不依赖示例 CSV。

当前上游 `example_02a` YAML 缺少部分必填训练字段；脚本保留原始配置的失败结果，
另以仅存在于内存的补全测试配置运行 `base` 和 `tiny`，使用依赖状态及前次奖励的
确定性策略。原始 YAML、数据集与常规输出目录不被修改，不执行网络训练。
