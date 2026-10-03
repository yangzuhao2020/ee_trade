# EOM 仿真分层

V1/V2、V4、V5 共用常规市场。版本入口只负责选择参与者扩展，彼此不导入。

```text
simulation.py
  ├─ v1_v2.py：基础市场，无扩展
  ├─ v4.py：基础市场 + 工业扩展
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

在项目根目录执行：

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

当前上游 `example_02a` YAML 缺少部分必填训练字段；脚本保留原始配置的失败结果，
另以仅存在于内存的补全测试配置运行 `base` 和 `tiny`，使用依赖状态及前次奖励的
确定性策略。原始 YAML、数据集与常规输出目录不被修改，不执行网络训练。
