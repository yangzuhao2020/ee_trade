# EOM 仿真分层

V1/V2、V4、V5 共用常规市场。V1/V2 和 V4 的入口合并在 `eom_scenarios.py`，
V5 保留在 `v5.py`。各入口负责准备基础市场并选择参与者扩展，共用事件引擎。

```text
simulation.py
  ├─ eom_scenarios.py
  │    ├─ simulate_v1_v2_market：基础市场，无扩展
  │    └─ simulate_v4_market：基础市场 + 工业扩展
  ├─ v5.py：基础市场 + 学习扩展
  └─ household_market.py：原有 V3 家庭仿真

base_market/
  ├─ market.py：数据准备、预检、报价出清组织、状态初始化和回写
  ├─ demand.py：刚性/弹性需求与 Exchange 订单
  ├─ plants.py：常规机组报价、启动成本和运行状态
  ├─ storage.py：储能报价、承诺电量投影和 SOC 回写
  └─ forecasts.py：朴素价格预测、预测窗口及开放完整性检查

eom.py：开市、交割开始/结束事件，结果队列和扩展协议
```

具体报价公式保留在 `bidding.py`，出清算法保留在 `clearing/`。

`MarketSettings.opening_duration` 是必填配置，读取时检查为正值；目前未用于事件调度，
出清发生在开市事件中，没有单独的关市事件。`PowerPlant.min_down_time_hours` 会被读取
并检查为非负值，但目前未实现最小停机时间约束。`min_operating_time_hours` 用于启动
成本分摊，不代表已经实现最小连续运行时间约束。

学习会话负责动作维度、有限值校验和按需裁剪，再把已处理动作传给 `learning_offers()`。
经验记录保留动作原顺序，只有生成报价时对价格排序。

## 扩展生命周期

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

`settle_before_opening` 明确声明这个顺序，事件引擎不判断版本号或学习策略。

## 回归验证

`tests/eom_regression.py` 和 `tests/test_eom_engine.py` 纳入版本管理。
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

当前上游 `example_02a` YAML 缺少部分必填训练字段；脚本保留原始配置的失败结果，
另以仅存在于内存的补全测试配置运行 `base` 和 `tiny`，使用依赖状态及前次奖励的
确定性策略。原始 YAML、数据集与常规输出目录不被修改，不执行网络训练。
