# CSHDA 通用决策引擎：完整技术方案

> 代号：StallMate Engine
> 语言：Python 3.11+
> 目标：一个通用决策引擎，覆盖六种决策类型，LLM仅参与语义提取，后续全部由Embedding量化+符号推理完成

---

## 一、系统总览

### 1.1 设计哲学

三条不可违反的原则：

1. **LLM只做翻译，不做决策**——LLM的唯一角色是把自然语言转换成结构化数据，一旦输出结构化数据就永久退场
2. **决策过程100%确定性**——同一个结构化输入永远产出同一个决策结果，不存在任何随机性
3. **零任务特调**——框架中不包含任何特定领域的知识、规则、权重预设，所有参数从数据结构本身推导

### 1.2 四层架构

```
自然语言输入
      │
      ▼
┌─────────────────────────────────────┐
│  Layer 1: 语义提取层                  │  技术: LLM (仅此一层)
│  输入: 自然语言                       │
│  输出: UniversalDecisionSpec (UDS)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Layer 2: 量化与建模层                │  技术: Sentence Embedding + 规则映射
│  输入: UDS                           │
│  输出: MathematicalFormulation (MF)   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Layer 3: 符号求解层                  │  技术: 六种确定性求解器
│  输入: MF                            │
│  输出: DecisionResult (DR)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Layer 4: 一致性保障层                │  技术: 公理验证 + 偏好图 + 审计日志
│  输入: DR + 历史决策                  │
│  输出: FinalOutput (验证后的决策+报告) │
└─────────────────────────────────────┘
```

### 1.3 六种决策类型

| 类型ID | 名称 | 典型问题 | 数学本质 |
|--------|------|---------|---------|
| T1 | 多属性单选 | 选坑位、选餐厅、选方案 | 评分矩阵 + 加权排序 |
| T2 | 约束组合选择 | 背包问题、预算内购物、课程选择 | 整数规划 / 动态规划 |
| T3 | 排序与排列 | 任务优先级、日程排序、偏好排名 | 偏序融合 + 拓扑排序 |
| T4 | 资源分配 | 预算分配、人员指派、负载均衡 | 线性规划 / 匹配问题 |
| T5 | 序贯决策 | 路径规划、多阶段投资、治疗计划 | 动态规划 / 决策树 |
| T6 | 博弈决策 | 谈判、定价竞争、合作策略 | 纳什均衡 / 极小极大 |

---

## 二、项目结构

```
stallmate/
├── README.md
├── pyproject.toml
├── config/
│   ├── llm_config.yaml              # LLM 接入配置（API key、模型选择）
│   ├── embedding_config.yaml         # Embedding 模型配置
│   └── solver_config.yaml            # 求解器参数配置
│
├── stallmate/
│   ├── __init__.py
│   ├── engine.py                     # 主引擎入口，串联四层
│   │
│   ├── layer1_extraction/
│   │   ├── __init__.py
│   │   ├── extractor.py              # LLM 调用封装
│   │   ├── prompts.py                # 提取用的 system prompt 模板
│   │   ├── schema.py                 # UDS 的 Pydantic 数据模型定义
│   │   ├── normalizer.py             # UDS 规范化处理
│   │   └── validator.py              # UDS 结构校验
│   │
│   ├── layer2_quantification/
│   │   ├── __init__.py
│   │   ├── embedder.py               # Embedding 模型封装
│   │   ├── polarity_scorer.py        # 语义极性轴投影评分
│   │   ├── relation_analyzer.py      # 属性/实体关系分析
│   │   ├── type_classifier.py        # 决策类型识别
│   │   ├── weight_calculator.py      # 通用客观权重计算（方差/熵/CRITIC）
│   │   └── formulator.py             # UDS → MathematicalFormulation 转换
│   │
│   ├── layer3_solver/
│   │   ├── __init__.py
│   │   ├── dispatcher.py             # 类型分发器
│   │   ├── base_solver.py            # 求解器基类（接口定义）
│   │   ├── t1_selection.py           # T1: 多属性单选求解器
│   │   ├── t2_knapsack.py            # T2: 约束组合选择求解器
│   │   ├── t3_ranking.py             # T3: 排序与排列求解器
│   │   ├── t4_allocation.py          # T4: 资源分配求解器
│   │   ├── t5_sequential.py          # T5: 序贯决策求解器
│   │   ├── t6_game.py                # T6: 博弈决策求解器
│   │   └── tiebreaker.py             # 通用平局处理
│   │
│   ├── layer4_consistency/
│   │   ├── __init__.py
│   │   ├── axiom_checker.py          # 公理验证器
│   │   ├── preference_graph.py       # 偏好图维护
│   │   ├── consistency_report.py     # 一致性报告生成
│   │   └── audit_logger.py           # 完整决策审计日志
│   │
│   └── utils/
│       ├── __init__.py
│       ├── math_utils.py             # 数学工具函数
│       └── visualization.py          # 决策过程可视化
│
├── tests/
│   ├── test_layer1/
│   ├── test_layer2/
│   ├── test_layer3/
│   ├── test_layer4/
│   └── test_integration/
│       ├── test_toilet.py            # 厕所问题端到端测试
│       ├── test_knapsack.py          # 背包问题端到端测试
│       ├── test_scheduling.py        # 排序问题端到端测试
│       ├── test_allocation.py        # 分配问题端到端测试
│       ├── test_sequential.py        # 序贯决策端到端测试
│       └── test_game.py              # 博弈决策端到端测试
│
├── experiments/
│   ├── phase1_comparison/            # 与 LLM Phase 1 数据的对比实验
│   ├── phase2_comparison/            # 与 LLM Phase 2 数据的对比实验
│   └── cross_type_consistency/       # 跨类型一致性实验
│
└── docs/
    ├── architecture.md               # 架构文档
    ├── data_contracts.md             # 层间数据契约
    └── solver_specs.md               # 各求解器的算法说明
```

---

## 三、层间数据契约

### 3.1 Layer 1 输出 → Layer 2 输入：UniversalDecisionSpec (UDS)

UDS 是整个系统的核心数据结构。所有类型的决策问题都映射到同一个 UDS 格式。

```
UniversalDecisionSpec:
  metadata:
    raw_input: str                           # 原始自然语言输入
    extraction_model: str                    # 使用的 LLM 名称
    extraction_timestamp: datetime           # 提取时间

  entities:                                  # 涉及的所有实体
    - id: str                                # 唯一标识 (e.g., "item_1", "stall_3")
      label: str                             # 自然语言标签 (e.g., "坑位3", "笔记本电脑")
      entity_type: str                       # 实体类型 (option/resource/task/agent)
      properties:                            # 该实体的属性描述
        - key: str                           # 属性名 (e.g., "weight", "privacy")
          value_description: str             # 属性值的自然语言描述
          numeric_value: float | null        # 如果原文包含数值则直接填入
          unit: str | null                   # 单位 (e.g., "kg", "元")

  objectives:                                # 优化目标
    - id: str
      description: str                       # 目标的自然语言描述
      direction: "maximize" | "minimize" | "target"
      target_value: float | null             # 仅 direction=target 时有值

  constraints:                               # 约束条件
    - id: str
      description: str                       # 约束的自然语言描述
      constraint_type: str                   # 类型标签 (capacity/budget/time/availability/dependency/mutex)
      involves: list[str]                    # 涉及的实体 id
      numeric_limit: float | null            # 数值上限/下限
      limit_direction: "le" | "ge" | "eq" | null

  relations:                                 # 实体之间的关系
    - source: str                            # 源实体 id
      target: str                            # 目标实体 id
      relation_type: str                     # 关系类型
      description: str                       # 关系的自然语言描述
      strength: "strong" | "moderate" | "weak" | null

  decision_context:                          # 决策情境
    - factor: str                            # 情境因素名
      description: str                       # 描述
      influence_on: list[str]                # 影响的目标/属性 id

  decision_type_hint: str | null             # LLM 对决策类型的初步判断（仅供参考，Layer 2 会独立判断）
```

#### 关系类型枚举（relation_type 的合法值）

| 关系类型 | 含义 | 适用场景 |
|---------|------|---------|
| depends_on | A 依赖 B（B 必须在 A 之前）| 任务排序、序贯决策 |
| conflicts_with | A 和 B 互斥，不能同时选择 | 组合选择、资源分配 |
| synergizes_with | A 和 B 同时选择时有额外收益 | 组合选择 |
| competes_with | A 和 B 是竞争关系 | 博弈 |
| cooperates_with | A 和 B 是合作关系 | 博弈 |
| precedes | A 在时间上先于 B | 序贯决策、排序 |
| substitutes | A 可以替代 B | 单选、分配 |
| part_of | A 是 B 的一部分 | 层级结构 |
| influences | A 的状态影响 B 的属性值 | 因果关系 |

#### 各类型问题的 UDS 填充模式

**T1 多属性单选（厕所问题）**：

- entities: 每个坑位一个实体，entity_type="option"
- objectives: "最大化综合舒适度"
- constraints: 被占用的坑位标记为 availability 约束
- relations: 通常为空或包含 influences 关系

**T2 约束组合选择（背包问题）**：

- entities: 每个物品一个实体，properties 包含 weight 和 value（带数值）
- objectives: "最大化总价值"
- constraints: 包含一条 capacity 约束，numeric_limit=背包容量
- relations: 可能包含 conflicts_with（互斥物品）或 synergizes_with（组合加成）

**T3 排序（任务优先级）**：

- entities: 每个任务一个实体，entity_type="task"
- objectives: "最优排列顺序" 或 "最小化总完成时间"
- constraints: 可能包含 deadline 约束
- relations: 包含 depends_on 和 precedes 关系

**T4 资源分配（预算分配）**：

- entities: 包含两类——resource 类型（预算、人力）和 option 类型（部门、项目）
- objectives: "最大化总收益" 或 "最均衡分配"
- constraints: 资源总量约束
- relations: 可能包含最低保障要求

**T5 序贯决策（多阶段投资）**：

- entities: 每个阶段的可选行动，entity_type="option"
- objectives: "最大化最终收益"
- constraints: 各阶段的状态转移约束
- relations: precedes 关系定义阶段顺序，influences 关系定义状态转移

**T6 博弈决策（定价竞争）**：

- entities: 包含两类——agent 类型（玩家）和 option 类型（各玩家的可选策略）
- objectives: 每个 agent 有自己的 objective
- constraints: 博弈规则
- relations: competes_with 或 cooperates_with

### 3.2 Layer 2 输出 → Layer 3 输入：MathematicalFormulation (MF)

MF 是针对具体决策类型的数学化表示。不同类型的 MF 结构不同，但共享一个统一的外壳。

```
MathematicalFormulation:
  decision_type: T1 | T2 | T3 | T4 | T5 | T6   # 确定的类型（Layer 2 判定）
  type_confidence: float                          # 类型判定置信度
  
  formulation: T1_Formulation | T2_Formulation | ... | T6_Formulation  # 类型特定的数学结构
  
  embedding_artifacts:                            # 保留的 embedding 产物（供 Layer 4 审计）
    entity_embeddings: dict[str, vector]
    attribute_embeddings: dict[str, vector]
    polarity_axes: dict[str, (vector, vector)]    # 每个属性的正负锚点向量
    similarity_matrix: matrix                      # 属性间相似度矩阵
```

#### T1_Formulation（多属性单选）

```
T1_Formulation:
  score_matrix: matrix[n_options, n_attributes]   # 评分矩阵，值域 [0, 1]
  weights: vector[n_attributes]                   # 属性权重，和为 1
  weight_method: str                              # 使用的权重方法名
  option_ids: list[str]                           # 选项 id（与矩阵行对应）
  attribute_ids: list[str]                        # 属性 id（与矩阵列对应）
  attribute_directions: list[str]                 # 每个属性的优化方向
```

#### T2_Formulation（约束组合选择）

```
T2_Formulation:
  n_items: int
  item_ids: list[str]
  
  value_vector: vector[n_items]                   # 每个物品的价值
  
  constraint_dimensions: int                      # 约束维度数（背包=1，多维背包>1）
  cost_matrix: matrix[n_items, constraint_dimensions]  # 每个物品在每个约束维度上的消耗
  capacity_vector: vector[constraint_dimensions]  # 每个约束维度的容量上限
  
  mutex_pairs: list[(str, str)]                   # 互斥对
  synergy_pairs: list[(str, str, float)]          # 协同对及加成系数
  
  item_bounds: list[(int, int)]                   # 每个物品的选择数量上下界（0-1背包则全为(0,1)）
```

#### T3_Formulation（排序与排列）

```
T3_Formulation:
  n_items: int
  item_ids: list[str]
  
  priority_matrix: matrix[n_items, n_criteria]    # 每个元素在每个排序标准下的优先级分
  criteria_weights: vector[n_criteria]            # 排序标准权重
  
  precedence_pairs: list[(str, str)]              # 硬性先后约束 (A 必须在 B 前)
  
  # 如果是调度问题（带时间）：
  processing_times: vector[n_items] | null        # 每个任务的处理时间
  deadlines: vector[n_items] | null               # 每个任务的截止时间
  release_times: vector[n_items] | null           # 每个任务的最早开始时间
```

#### T4_Formulation（资源分配）

```
T4_Formulation:
  n_receivers: int                                # 接收者数量（部门/项目）
  n_resources: int                                # 资源类型数量
  receiver_ids: list[str]
  resource_ids: list[str]
  
  utility_matrix: matrix[n_receivers, n_resources] # 每个接收者获得每单位各类资源的边际效用
  
  resource_totals: vector[n_resources]            # 各类资源总量
  
  min_allocations: matrix[n_receivers, n_resources] | null  # 最低保障
  max_allocations: matrix[n_receivers, n_resources] | null  # 分配上限
  
  fairness_constraint: str | null                 # 公平性约束类型 (none/proportional/equal_floor)
  
  # 如果是指派问题（一对一匹配）：
  is_assignment: bool                             # 是否为指派问题
  assignment_cost_matrix: matrix | null            # 指派成本矩阵（仅 is_assignment=True 时）
```

#### T5_Formulation（序贯决策）

```
T5_Formulation:
  n_stages: int                                   # 阶段数
  
  stages:
    - stage_id: str
      state_space: list[str]                      # 该阶段的可能状态
      action_space: list[str]                     # 该阶段的可选行动
      
      transition_table: dict[(state, action), list[(next_state, probability)]]
        # 状态转移表：(当前状态, 行动) → [(下一状态, 概率)]
        # 确定性问题中概率全为 1.0
      
      reward_table: dict[(state, action), float]
        # 即时奖励表：(当前状态, 行动) → 奖励值
  
  initial_state: str                              # 初始状态
  discount_factor: float                          # 折扣因子（默认 1.0）
  terminal_rewards: dict[str, float] | null       # 终止状态的额外奖励
```

#### T6_Formulation（博弈决策）

```
T6_Formulation:
  n_players: int
  player_ids: list[str]
  
  # 每个玩家的策略集
  strategy_sets: dict[str, list[str]]             # player_id → 可选策略列表
  
  # 支付矩阵（收益张量）
  # 对于两人博弈：payoff_tensor[i][j] = (player1_payoff, player2_payoff)
  # 对于 n 人博弈：payoff_tensor[s1][s2]...[sn] = (p1, p2, ..., pn)
  payoff_tensor: nested_dict | matrix
  
  game_type: str                                  # simultaneous（同时博弈）| sequential（序贯博弈）
  
  # 序贯博弈的额外信息
  move_order: list[str] | null                    # 行动顺序
  information_sets: dict | null                   # 信息集（不完全信息博弈）
  
  # 重复博弈的额外信息
  is_repeated: bool
  n_rounds: int | null
```

### 3.3 Layer 3 输出 → Layer 4 输入：DecisionResult (DR)

```
DecisionResult:
  decision_type: str
  
  # 统一的结果字段
  chosen: list[str]                               # 被选中的实体 id
  ranking: list[str] | null                       # 完整排序（如适用）
  allocation: dict[str, float] | null             # 分配方案（如适用）
  strategy: dict[str, str] | null                 # 博弈策略（如适用）
  action_sequence: list[str] | null               # 行动序列（如适用）
  
  # 决策质量指标
  objective_value: float                          # 目标函数值
  optimality_gap: float | null                    # 与最优解的差距（如可计算）
  
  # 完整的求解过程记录
  solver_name: str                                # 使用的求解器名称
  solver_trace: list[dict]                        # 求解过程的逐步记录
  intermediate_values: dict                       # 中间计算值（评分、权重等）
  
  # 敏感性信息
  margin: float                                   # 最优与次优的差距
  critical_parameters: list[str]                  # 影响结果的关键参数
```

### 3.4 Layer 4 输出：FinalOutput

```
FinalOutput:
  decision: DecisionResult                        # 最终决策（与 Layer 3 相同或经修正）
  
  consistency_report:
    transitivity: "PASS" | "FAIL" | "N/A"
    iia: "PASS" | "FAIL" | "N/A"
    frame_invariance: "PASS" | "FAIL" | "N/A"
    constraint_satisfaction: "PASS" | "FAIL"      # 所有约束是否满足
    type_specific_checks: dict[str, str]          # 类型特定的验证结果
  
  confidence_score: float                         # 综合置信度 (0-1)
  confidence_breakdown:
    extraction_stability: float                   # Layer 1 多次提取的一致率
    quantification_robustness: float              # Layer 2 权重敏感性
    solution_margin: float                        # Layer 3 解的余量
  
  audit_trail:                                    # 完整审计链
    raw_input: str
    extracted_uds: UDS
    mathematical_formulation: MF
    solver_result: DR
    consistency_checks: dict
    timestamp: datetime
```

---

## 四、Layer 1 详细设计：语义提取层

### 4.1 职责边界

Layer 1 做的事情：

- 从自然语言中识别实体、属性、目标、约束、关系
- 将信息组织为 UDS 格式
- 对决策类型做初步判断（hint，不是最终判定）

Layer 1 绝不做的事情：

- 不输出任何选择或建议
- 不评估任何属性的好坏
- 不设定任何权重或优先级
- 不做任何推理或推断（只提取文本中明确或直接可推断的信息）

### 4.2 LLM System Prompt 设计

系统提示分为三部分：角色定义、提取规则、输出格式。

**角色定义**：声明 LLM 的角色是"结构化信息提取器"，不是"决策顾问"。明确禁止任何形式的建议、偏好表达、价值判断。

**提取规则**：

- 穷举所有可选实体，不遗漏
- 对每个实体，提取文本中提到的所有属性描述
- 如果文本包含明确的数值（"重量5kg"），直接填入 numeric_value
- 如果文本只有定性描述（"特别干净"），填入 value_description
- 对于文本未提及的属性，不要臆测
- 约束条件必须从文本中明确找到，不能添加隐含约束
- 关系必须从文本中的语义逻辑推导，不能凭空创造

**输出格式**：严格遵循 UDS 的 JSON Schema，不允许额外字段。

### 4.3 多次提取与规范化

**多次提取策略**：

对同一输入运行 K 次提取（默认 K=3）：

- K 次使用完全相同的 system prompt 和 user prompt
- 采样温度设为 0（贪心解码），但仍可能因 API 的内部变化产生差异
- 对 K 次结果做字段级别的对齐和投票

**字段级投票规则**：

- entities: 取所有 K 次结果的并集，然后对每个实体的 properties 取众数
- objectives: 取众数（如果 K 次结果中 ≥ 2 次出现同一目标）
- constraints: 取并集（约束宁多勿少，遗漏约束比多出约束更危险）
- relations: 取众数（≥ 2 次出现的关系才保留）
- decision_type_hint: 取众数

**规范化步骤**：

1. 实体 id 标准化：按类型+序号重新编号（option_1, option_2, ...）
2. 属性名去重：使用 embedding 计算属性名之间的余弦相似度，>0.85 的合并
3. 约束格式统一：所有约束转化为标准的 (involves, type, limit, direction) 格式
4. 关系去重和方向校验：去除重复关系，检查依赖关系是否成环

### 4.4 极性锚点生成

在提取主流程之外，额外运行一次 LLM 调用，专门生成极性锚点：

对 UDS 中的每个 objective 和 每个 entity property key：

- 生成该维度的正面极端描述（positive_anchor）
- 生成该维度的负面极端描述（negative_anchor）

这些锚点在 Layer 2 中用于构建语义极性轴。

锚点生成也遵循多次提取+众数的策略（但由于锚点通常简短，LLM 的输出稳定性较高，K=2 即可）。

---

## 五、Layer 2 详细设计：量化与建模层

### 5.1 Embedding 模型选择

使用预训练的 sentence embedding 模型，要求：

- 开源、可本地部署（避免对外部 API 的依赖）
- 支持中英文
- 模型参数量 < 500M（保持轻量）
- 在语义相似度任务上表现良好

推荐候选：BGE-M3（BAAI）、multilingual-e5-large（Microsoft）、GTE-large（Alibaba）

所有 embedding 计算在本地完成，确保确定性——同一模型、同一输入永远产出同一向量。

### 5.2 语义极性轴投影

**轴构建**：

对属性 attr_j：

- pos_vec = embed(attr_j.positive_anchor)
- neg_vec = embed(attr_j.negative_anchor)
- axis_vec = pos_vec - neg_vec（极性方向向量）
- axis_unit = axis_vec / ||axis_vec||（单位方向向量）

**投影评分**：

对实体 entity_i 在属性 attr_j 上的描述 desc_ij：

- 如果 desc_ij 有 numeric_value：直接使用数值（经 min-max 归一化到 [0, 1]）
- 如果 desc_ij 只有 value_description：
  - desc_vec = embed(value_description)
  - centered = desc_vec - neg_vec
  - raw_score = dot(centered, axis_unit) / ||axis_vec||
  - score = clip(raw_score, 0, 1)

**评分含义**：

- 0.0 = 该描述在语义上接近负面极端
- 1.0 = 该描述在语义上接近正面极端
- 0.5 = 中性或无信息

### 5.3 属性关系分析

**属性间相似度矩阵**：

对所有属性对 (attr_i, attr_j)：

- sim_ij = cosine_similarity(embed(attr_i.label), embed(attr_j.label))
- 构建相似度矩阵 S，S[i][j] = sim_ij

**冗余检测**：

如果 S[i][j] > threshold（默认 0.80），标记 attr_i 和 attr_j 为冗余对。后续权重计算中会折减冗余属性的权重。

**情境影响分析**：

对每个 context_factor cf 和每个 attribute attr_j：

- influence_score = cosine_similarity(embed(cf.description), embed(attr_j.label))
- 构建情境影响矩阵 CI，CI[cf][j] = influence_score

### 5.4 决策类型识别

**基于结构特征的规则分类**（不使用 ML，纯符号规则）：

```
判断逻辑（按优先级）：

1. 如果 entities 中存在 entity_type="agent" 且 >1 个 agent
   → T6 博弈决策

2. 如果 relations 中存在 precedes/depends_on 关系
   且这些关系构成多阶段结构（可以划分为 ≥2 个阶段）
   → T5 序贯决策

3. 如果 entities 中存在 entity_type="resource"
   且 objectives 中包含"分配"/"allocate"语义
   → T4 资源分配

4. 如果 objectives 中包含"排序"/"优先级"/"顺序"语义
   或 relations 中存在大量 precedes 关系但无多阶段结构
   → T3 排序与排列

5. 如果 constraints 中存在 capacity/budget 约束
   且 entities 数量 > 1 且每个 entity 有 value 和 cost 属性
   → T2 约束组合选择

6. 其他情况（从多个选项中选一个最优）
   → T1 多属性单选
```

**语义辅助验证**：

类型判断后，用 embedding 验证：将问题描述和六种类型的标准描述计算相似度，检查规则判断是否与语义判断一致。如果不一致，记录 type_confidence 较低值。

### 5.5 UDS → MathematicalFormulation 转换

**转换器为每种类型实现一个专用转换函数**，负责将通用的 UDS 映射为该类型的 MF：

**T1 转换**：

- 从 entities 的 properties 和 assessments 构建评分矩阵
- 用极性轴投影计算每个格子的评分
- 用方差法/熵权法/CRITIC 计算权重
- 输出 T1_Formulation

**T2 转换**：

- 从 entities 中提取 value 和 cost 属性（应为数值）
- 从 constraints 中提取容量限制
- 从 relations 中提取互斥和协同关系
- 输出 T2_Formulation

**T3 转换**：

- 从 entities 的 properties 构建优先级矩阵
- 从 relations 中提取先后约束
- 从 constraints 中提取截止时间等
- 输出 T3_Formulation

**T4 转换**：

- 从 entities 中分离 resource 和 receiver
- 构建效用矩阵（每个 receiver 对每种 resource 的边际效用）
- 从 constraints 中提取资源总量和最低保障
- 输出 T4_Formulation

**T5 转换**：

- 从 relations 的时序结构中划分阶段
- 对每个阶段识别状态空间和行动空间
- 从 entities 的属性和 constraints 推导状态转移表
- 从 objectives 推导奖励函数
- 输出 T5_Formulation

**T6 转换**：

- 从 entities 中识别 agents 和各自的策略
- 从 objectives 和 entities 的属性构建支付矩阵
- 从 relations 判断博弈类型（同时/序贯）
- 输出 T6_Formulation

### 5.6 权重计算（适用于 T1、T3、T4）

**方法一：方差权重法**

对评分矩阵中每个属性列计算方差，归一化为权重。

**方法二：熵权法**

对评分矩阵中每个属性列计算信息熵，用 1-熵 作为区分度，归一化为权重。

**方法三：CRITIC 法**

综合考虑标准差（波动性）和属性间相关性（独立性）。对高度冗余的属性自动折减权重。

**最终权重**：三种方法的加权平均（等权），并在论文/博客中报告三种方法各自的结果。

**冗余折减**：对相似度矩阵中 >0.80 的属性对，按 CRITIC 法的思路折减权重。

---

## 六、Layer 3 详细设计：符号求解层

### 6.1 求解器分发

dispatcher 模块根据 MF.decision_type 字段，将 MF 分发到对应的求解器。每个求解器实现统一的接口。

**求解器基类接口**：

- solve(formulation) → DecisionResult
- validate(formulation) → bool（检查输入格式是否合法）
- explain(result) → str（生成可读的求解过程解释）

### 6.2 T1 求解器：多属性单选

**算法：加权求和法（SAW）+ TOPSIS 交叉验证**

主算法使用 SAW（Simple Additive Weighting）：

- 对每个选项 i：U(i) = Σ_j w_j × s_ij
- 选择 U 最大的选项

交叉验证使用 TOPSIS（Technique for Order Preference by Similarity to Ideal Solution）：

- 计算正理想解 A+ 和负理想解 A-
- 对每个选项计算到 A+ 和 A- 的加权欧氏距离
- 选择相对接近度最高的选项

如果 SAW 和 TOPSIS 结果一致，置信度高。如果不一致，记录差异并报告。

**平局处理**：

- 第一规则：在非区分属性上寻找微小差异
- 第二规则：选属性分向量的字典序最大者
- 第三规则：选 id 字典序最小者

### 6.3 T2 求解器：约束组合选择

**算法：动态规划（精确解）+ 贪心法（快速验证）**

**0-1 背包（单约束维度）**：

- 标准 DP：O(n × W)，n=物品数，W=容量
- 同时记录 DP 表的回溯路径，支持解的可解释性

**多维背包（多约束维度）**：

- 使用分支定界法（Branch and Bound）
- 上界估计：LP 松弛
- 搜索策略：最佳优先搜索

**互斥约束处理**：

- 在 DP/分支定界中加入互斥检查：如果当前选集包含互斥对，剪枝

**协同效应处理**：

- 对 synergy_pairs 中的每对 (i, j)，如果同时选择 i 和 j，总价值额外增加 synergy_bonus
- 在 DP 状态中加入协同标记

**贪心交叉验证**：

- 按价值密度（value / cost）排序，依次放入
- 如果贪心解与 DP 解相同，置信度高
- 如果不同，记录差异（贪心解是下界，DP 解是最优解）

### 6.4 T3 求解器：排序与排列

**算法：偏序融合 + 拓扑排序**

**Step 1：构建偏序关系**

- 从 precedence_pairs 中获取硬性先后约束
- 从 priority_matrix 和 criteria_weights 中计算加权优先级总分
- 优先级总分定义了"软"排序偏好

**Step 2：拓扑排序**

- 构建有向图（硬性约束为边）
- 检测环路（如果有环 → 报告约束冲突）
- 在拓扑排序的自由度内（同层节点的排列），按优先级总分排序

**Step 3：如果是调度问题（有时间属性）**

- 使用 SPT（Shortest Processing Time）规则最小化平均完成时间
- 或 EDD（Earliest Due Date）规则最小化最大延迟
- 选择哪种规则由 objectives 的语义决定

**平局处理**：优先级总分相同的元素按 id 字典序排列

### 6.5 T4 求解器：资源分配

**子类型判断**：根据 is_assignment 字段分流

**连续分配（is_assignment=False）**：

- 构建线性规划问题：
  - 变量：x_ij = receiver_i 分配到的 resource_j 的数量
  - 目标：max Σ utility_ij × x_ij
  - 约束：Σ_i x_ij ≤ resource_total_j（资源总量）
  - 约束：x_ij ≥ min_allocation_ij（最低保障）
  - 约束：x_ij ≤ max_allocation_ij（分配上限）
- 使用单纯形法或内点法求解
- Python 实现：调用 scipy.optimize.linprog

**离散指派（is_assignment=True）**：

- 使用匈牙利算法（Kuhn-Munkres）求最优匹配
- Python 实现：调用 scipy.optimize.linear_sum_assignment

**公平性约束处理**：

- proportional：按某个基准（如人数、贡献度）成比例分配
- equal_floor：先均分到最低保障，剩余按效用最大化分配

### 6.6 T5 求解器：序贯决策

**算法：动态规划（值迭代 / 后向归纳）**

**有限阶段、确定性**：

- 从最后一个阶段向前递推
- V(s, t) = max_a [R(s, a, t) + V(s', t+1)]，其中 s' 是确定性转移的下一状态
- 记录每个阶段每个状态的最优行动

**有限阶段、随机性**：

- V(s, t) = max_a [R(s, a, t) + Σ_{s'} P(s'|s,a) × V(s', t+1)]
- 标准值迭代

**决策树回溯**：

- 构建完整的决策树（阶段数有限时可行）
- 叶节点评分为终止奖励
- 非叶节点选择子节点中最优的行动
- 完整记录树结构，支持可视化

### 6.7 T6 求解器：博弈决策

**子类型判断**：根据 game_type 分流

**同时博弈（Normal Form）**：

- 求纳什均衡
- 纯策略纳什均衡：枚举所有策略组合，检查单方偏离条件
- 混合策略纳什均衡：使用 Lemke-Howson 算法（两人博弈）或 support enumeration
- Python 实现：调用 nashpy 库

**序贯博弈（Extensive Form）**：

- 使用后向归纳法（Backward Induction）
- 从博弈树的叶节点回溯，每个决策节点选择使自身收益最大的行动
- 输出子博弈完美均衡（SPE）

**极小极大（零和博弈）**：

- 如果检测到零和结构（一方收益 = 另一方损失），使用极小极大算法
- 可选 alpha-beta 剪枝优化

**重复博弈**：

- 对有限重复：后向归纳（可能退化为单次均衡）
- 对无限重复：计算触发策略（如针锋相对 tit-for-tat）

**输出**：

- 如果存在纯策略均衡：输出均衡策略组合
- 如果只有混合策略均衡：输出各策略的概率分布
- 如果存在多个均衡：列出所有均衡，按帕累托效率排序，推荐帕累托最优的

### 6.8 通用平局处理器

所有求解器共享一个平局处理模块。当出现效用完全相同的选项/方案时：

```
TieBreaker 规则（按优先级）：
1. 在非区分维度上寻找微小差异（访问原始 UDS 中被标记为 non-discriminative 的属性）
2. Minimax Regret：选择"最坏情况最好"的选项（鲁棒性原则）
3. 字典序：选 id 字典序最小者（纯确定性兜底）
```

---

## 七、Layer 4 详细设计：一致性保障层

### 7.1 公理验证器

**通用公理（所有类型都验证）**：

- **确定性**：同一 MF 输入多次运行求解器，结果是否完全相同
- **约束满足**：解是否满足 MF 中的所有约束

**T1 专项公理**：

- **传递性**：从多组 T1 结果中提取成对偏好，构建偏好图，检测环路
- **IIA**：对一个 T1 问题增加或移除选项，检查剩余选项的相对排序是否变化
- **框架不变性**：同一场景不同表述（经 Layer 1 提取后），决策是否相同

**T2 专项公理**：

- **最优性证明**：对 DP 解，验证没有可行的单步改进（交换一个物品进出）
- **单调性**：如果某物品的价值增加（其他不变），该物品在最优解中的出现概率不应下降

**T3 专项公理**：

- **偏序一致性**：输出的全序与输入的偏序约束不矛盾
- **排序稳定性**：交换排序标准的权重微小量（±5%），排序前 K 名是否变化

**T4 专项公理**：

- **效率性**：分配方案是否帕累托最优（不存在让某人更好而不让任何人更差的替代方案）
- **资源用尽**：可分配资源是否被完全使用

**T5 专项公理**：

- **时间一致性**：在阶段 t 做出的最优决策，是否在阶段 t+1 仍然是最优子策略（贝尔曼最优性原理）
- **子博弈一致性**：从任意中间状态开始的子问题，其最优策略是否是全局最优策略的子序列

**T6 专项公理**：

- **均衡验证**：任何单个玩家单方面偏离建议策略是否会导致其收益降低
- **帕累托效率**：建议的均衡是否是帕累托最优的（如果存在多个均衡）

### 7.2 偏好图

维护一个全局有向图 G = (V, E)：

- 节点 V：历史决策中出现过的所有选项/方案
- 有向边 (A, B) ∈ E：在某次决策中 A 被选择而 B 未被选择（A ≻ B）
- 边属性：决策时间戳、决策类型、效用差距

**操作**：

- add_preference(A, B, metadata)：添加偏好边
- check_transitivity()：DFS 检测环路，O(V+E)
- get_inconsistencies()：返回所有环路（传递性违反）
- resolve_conflict(cycle)：基于时间衰减/效用差距选择保留哪些边

### 7.3 审计日志

对每次决策记录完整的审计日志，包含：

- 原始输入
- Layer 1 的提取结果（包括多次提取的原始结果和投票后的结果）
- Layer 2 的 embedding 向量、评分矩阵、权重计算过程
- Layer 3 的求解过程（DP 表、LP 单纯形迭代、博弈论计算）
- Layer 4 的一致性检查结果
- 最终输出

日志格式：JSON Lines，每行一个完整的决策记录。支持后续的批量分析。

---

## 八、依赖库

### 8.1 核心依赖

| 库 | 用途 | Layer |
|----|------|-------|
| pydantic | 数据模型定义和验证 | 全局 |
| openai / anthropic / httpx | LLM API 调用 | Layer 1 |
| sentence-transformers | Sentence Embedding 模型 | Layer 2 |
| numpy | 矩阵运算 | Layer 2, 3 |
| scipy | 线性规划、匈牙利算法、优化 | Layer 3 (T4) |
| nashpy | 纳什均衡求解 | Layer 3 (T6) |
| networkx | 图操作（偏好图、拓扑排序、环路检测） | Layer 3 (T3), Layer 4 |
| pyyaml | 配置文件读取 | 全局 |

### 8.2 可选依赖

| 库 | 用途 | Layer |
|----|------|-------|
| ortools | 更强大的组合优化求解器 | Layer 3 (T2, T4) |
| pulp | LP/MIP 建模（scipy 的替代） | Layer 3 (T4) |
| matplotlib / plotly | 决策过程可视化 | 可视化 |
| rich | 终端美化输出 | CLI |
| pytest | 测试框架 | 测试 |

### 8.3 不使用的库

| 类别 | 说明 |
|------|------|
| 深度学习框架 (PyTorch/TF) | 不需要。Embedding 推理用 sentence-transformers 即可 |
| 大模型推理框架 (vLLM/TGI) | 不需要。LLM 通过 API 调用，不本地部署 |
| AutoML / scikit-learn | 不需要。不做任何机器学习训练 |

---

## 九、测试策略

### 9.1 单元测试

每个模块独立测试，使用固定的输入数据（不依赖 LLM）。

**Layer 1 测试**：

- 测试规范化逻辑（输入两个语义等价的 UDS，验证规范化后相同）
- 测试校验逻辑（输入不合法的 UDS，验证能检测到错误）

**Layer 2 测试**：

- 测试极性轴投影（构造已知正负极端的句子，验证投影分数合理）
- 测试类型分类（构造每种类型的典型 UDS，验证分类正确）
- 测试权重计算（构造已知方差的评分矩阵，验证权重计算正确）

**Layer 3 测试**：

- T1：构造已知最优解的评分矩阵，验证求解器输出正确
- T2：使用经典背包测试用例（已知最优解），验证 DP 输出正确
- T3：构造已知拓扑序的 DAG，验证排序正确
- T4：构造简单 LP 问题（已知最优解），验证分配正确
- T5：构造简单 MDP（已知最优策略），验证值迭代正确
- T6：使用经典博弈（囚徒困境、性别战争），验证均衡求解正确

**Layer 4 测试**：

- 构造包含传递性违反的偏好图，验证能检测到
- 构造满足所有公理的决策序列，验证全部 PASS

### 9.2 集成测试

端到端测试，从自然语言输入到最终输出。

**每种类型至少 3 个端到端测试用例**：

T1 测试用例：

- 厕所问题（5 个坑位，2 个被占）
- 选餐厅（4 家餐厅，各有价格、距离、评分）
- 选笔记本电脑（3 款，各有性能、价格、重量）

T2 测试用例：

- 经典 0-1 背包（10 个物品，已知最优解）
- 预算内购物（多个商品，有预算限制）
- 课程选择（多门课，有学分上限和先修约束）

T3 测试用例：

- 任务优先级排序（5 个任务，有依赖关系）
- 面试排序（4 个候选人，多个评价维度）
- 旅行景点排序（按偏好和地理位置）

T4 测试用例：

- 预算分配（100 万分给 4 个部门）
- 人员指派（5 个人分配到 5 个岗位）
- 时间分配（一天的时间分给多个活动）

T5 测试用例：

- 多阶段投资（3 个阶段，每阶段选保守/激进）
- 路径规划（从 A 到 D，经过中间节点，各段有不同成本/收益）
- 治疗方案（先做检查 → 根据结果选药物 → 评估效果）

T6 测试用例：

- 囚徒困境
- 定价竞争（两家公司选高价/低价）
- 拍卖策略（两个竞标者，各有估值）

### 9.3 一致性测试（专项）

**表述不变性测试**：

- 对每个集成测试用例，生成 3 种不同的自然语言表述
- 验证三种表述经过完整 pipeline 后产出相同的决策

**参数敏感性测试**：

- 对 Layer 2 的权重方法（方差/熵/CRITIC），验证三种方法是否产出相同决策
- 对 Layer 2 的 embedding 模型，替换为不同模型，验证结果稳定性

**规模压力测试**：

- T1：选项数从 3 扩展到 100，验证性能和一致性
- T2：物品数从 10 扩展到 1000，验证 DP 的时间和内存
- T3：任务数从 5 扩展到 200，验证拓扑排序性能
- T6：策略数从 2×2 扩展到 10×10，验证均衡求解性能

---

## 十、与 LLM 的对比实验

### 10.1 实验设计

对每种决策类型，构造一组测试问题，同时发给 LLM（直接决策）和 CSHDA（框架决策），对比两者的表现。

**对比维度**：

| 维度 | 测量方法 |
|------|---------|
| 一致性 | 同一问题多次运行的 MCR |
| 表述稳定性 | 不同表述下选择分布的 JSD |
| 最优性 | 在有已知最优解的问题上，解的质量 |
| 可解释性 | 是否能输出完整的决策推理过程 |
| 计算效率 | FLOPs / 延迟 / token 消耗 |
| 公理符合性 | 传递性、IIA 等的违反率 |

### 10.2 已有数据的利用

Phase 1 和 Phase 2 的厕所实验数据可以直接用于 T1 的对比：

- 将实验中使用的 prompt 输入 CSHDA
- 对比 CSHDA 的输出与 glm-5.1 的输出
- 重点对比 MCR、JSD、条件响应合理性

### 10.3 新增实验

为 T2-T6 各设计一组对比实验：

- 选择有公认最优解的经典问题（如背包、旅行商、囚徒困境）
- 用自然语言描述这些问题，发给 LLM 和 CSHDA
- 对比两者的解质量和一致性

---

## 十一、实现优先级

### Phase A：核心骨架（先跑通一条线）

| 优先级 | 任务 | 产出 |
|--------|------|------|
| A1 | 定义所有数据模型（UDS, MF, DR, FinalOutput）| schema.py |
| A2 | 实现 Layer 1 提取器 + system prompt | extractor.py, prompts.py |
| A3 | 实现 Layer 2 极性轴投影 + 方差权重 | polarity_scorer.py, weight_calculator.py |
| A4 | 实现 T1 求解器 | t1_selection.py |
| A5 | 实现 Layer 4 基础验证 | axiom_checker.py |
| A6 | 串联 engine.py，厕所问题端到端跑通 | engine.py |

### Phase B：扩展类型

| 优先级 | 任务 | 产出 |
|--------|------|------|
| B1 | 实现类型分类器 | type_classifier.py |
| B2 | 实现 T2 求解器（背包 DP）| t2_knapsack.py |
| B3 | 实现 T3 求解器（拓扑排序）| t3_ranking.py |
| B4 | 实现 T4 求解器（LP + 匈牙利）| t4_allocation.py |
| B5 | 实现 T5 求解器（值迭代）| t5_sequential.py |
| B6 | 实现 T6 求解器（纳什均衡）| t6_game.py |

### Phase C：完善与测试

| 优先级 | 任务 | 产出 |
|--------|------|------|
| C1 | 完善 Layer 4 的类型专项验证 | axiom_checker.py 扩展 |
| C2 | 实现 CRITIC 权重和熵权法 | weight_calculator.py 扩展 |
| C3 | 全部单元测试和集成测试 | tests/ |
| C4 | 与 LLM 的对比实验 | experiments/ |
| C5 | 可视化模块 | visualization.py |
| C6 | CLI 入口 | cli.py |

### Phase D：文档与发布

| 优先级 | 任务 | 产出 |
|--------|------|------|
| D1 | README.md | 项目说明 |
| D2 | 架构文档 | docs/architecture.md |
| D3 | 博客文章 | blog/ |
| D4 | GitHub 发布 | release |

---

## 十二、六种类型的端到端示例

以下为每种决策类型提供一个完整的数据流示例，展示从自然语言输入到最终输出的全过程。所有示例均不包含任何特调逻辑——同一个 engine 处理所有类型。

### 12.1 T1 示例：选坑位

**自然语言输入**：
"你走进一间公共厕所，面前有一排5个独立的坑位，编号1到5。其中第2个和第4个有人正在使用。你会选择哪一个？"

**Layer 1 输出（UDS 摘要）**：

```
entities:
  - id: stall_1, type: option, properties: [{key: position, value: "最左端, 编号1"}]
  - id: stall_3, type: option, properties: [{key: position, value: "正中间, 编号3"}]
  - id: stall_5, type: option, properties: [{key: position, value: "最右端, 编号5"}]

objectives:
  - description: "选择最舒适的坑位"
    direction: maximize

constraints:
  - type: availability, involves: [stall_2], description: "2号有人使用"
  - type: availability, involves: [stall_4], description: "4号有人使用"

relations:
  - source: stall_2, target: stall_1, type: influences, description: "2号有人影响1号的隐私"
  - source: stall_2, target: stall_3, type: influences, description: "2号有人影响3号的隐私"
  - source: stall_4, target: stall_3, type: influences, description: "4号有人影响3号的隐私"
  - source: stall_4, target: stall_5, type: influences, description: "4号有人影响5号的隐私"
```

**Layer 2 输出（MF 摘要）**：

类型判定：T1（多属性单选），置信度 0.95

极性锚点示例：

- 属性"隐私性"：positive="完全独立无人打扰"，negative="两侧紧邻都有人"
- 属性"与占用者距离"：positive="距离所有占用者非常远"，negative="紧挨着占用者"
- 属性"便利性"：positive="离入口非常近"，negative="离入口非常远"

评分矩阵（极性轴投影后）：

```
            隐私性    距离    便利性
stall_1     0.82     0.24    0.85
stall_3     0.18     0.21    0.52
stall_5     0.83     0.79    0.16
```

权重（方差法）：

```
隐私性: 0.377, 距离: 0.368, 便利性: 0.255
```

**Layer 3 求解（T1 SAW）**：

```
U(stall_1) = 0.377×0.82 + 0.368×0.24 + 0.255×0.85 = 0.614
U(stall_3) = 0.377×0.18 + 0.368×0.21 + 0.255×0.52 = 0.278
U(stall_5) = 0.377×0.83 + 0.368×0.79 + 0.255×0.16 = 0.644

排序: stall_5 (0.644) > stall_1 (0.614) > stall_3 (0.278)
选择: stall_5
```

**Layer 4 验证**：

- 传递性：PASS（5>1>3 → 5>3 ✓）
- 约束满足：PASS（未选择被占坑位）
- 置信度：0.95（解余量 0.030，stall_5 vs stall_1 差距不大但方向稳定）

### 12.2 T2 示例：背包问题

**自然语言输入**：
"你有一个能装15公斤的背包。有5件物品可选：帐篷重5kg值60元，睡袋重3kg值40元，食物重4kg值50元，水壶重2kg值20元，急救包重1kg值10元。选哪些物品装进去，让总价值最大？"

**Layer 1 输出（UDS 摘要）**：

```
entities:
  - id: tent, type: option, properties: [{key: weight, numeric: 5, unit: kg}, {key: value, numeric: 60, unit: 元}]
  - id: sleeping_bag, type: option, properties: [{key: weight, numeric: 3}, {key: value, numeric: 40}]
  - id: food, type: option, properties: [{key: weight, numeric: 4}, {key: value, numeric: 50}]
  - id: bottle, type: option, properties: [{key: weight, numeric: 2}, {key: value, numeric: 20}]
  - id: first_aid, type: option, properties: [{key: weight, numeric: 1}, {key: value, numeric: 10}]

objectives:
  - description: "最大化总价值", direction: maximize

constraints:
  - type: capacity, description: "背包容量15kg", numeric_limit: 15, limit_direction: le
```

**Layer 2 输出（MF 摘要）**：

类型判定：T2（约束组合选择），置信度 0.98

```
T2_Formulation:
  n_items: 5
  value_vector: [60, 40, 50, 20, 10]
  cost_matrix: [[5], [3], [4], [2], [1]]
  capacity_vector: [15]
  item_bounds: [(0,1), (0,1), (0,1), (0,1), (0,1)]
```

**Layer 3 求解（T2 DP）**：

DP 表构建过程（容量 0-15，物品逐个考虑），回溯得到最优子集。

```
最优解: {tent, sleeping_bag, food, bottle, first_aid}
总重量: 5+3+4+2+1 = 15kg（恰好装满）
总价值: 60+40+50+20+10 = 180元
```

贪心验证：按价值密度排序（帐篷12, 睡袋13.3, 食物12.5, 急救包10, 水壶10），贪心也选全部。DP 和贪心一致。

**Layer 4 验证**：

- 约束满足：PASS（15 ≤ 15）
- 最优性：PASS（无法通过单步交换改进）

### 12.3 T3 示例：任务排序

**自然语言输入**：
"今天有4个任务要做：写报告（需要3小时）、开会（需要1小时，必须在写报告之前完成）、回邮件（需要30分钟）、代码审查（需要2小时，必须在回邮件之后做）。按什么顺序做最好？"

**Layer 1 输出（UDS 摘要）**：

```
entities:
  - id: report, type: task, properties: [{key: duration, numeric: 180, unit: min}]
  - id: meeting, type: task, properties: [{key: duration, numeric: 60}]
  - id: email, type: task, properties: [{key: duration, numeric: 30}]
  - id: code_review, type: task, properties: [{key: duration, numeric: 120}]

objectives:
  - description: "最优排列顺序", direction: minimize (总完成时间/等待时间)

relations:
  - source: meeting, target: report, type: precedes, description: "开会必须在写报告之前"
  - source: email, target: code_review, type: precedes, description: "回邮件必须在代码审查之前"
```

**Layer 2 输出（MF 摘要）**：

类型判定：T3（排序与排列），置信度 0.96

```
T3_Formulation:
  precedence_pairs: [(meeting, report), (email, code_review)]
  processing_times: [180, 60, 30, 120]
```

**Layer 3 求解（T3 拓扑排序 + SPT）**：

```
依赖图：meeting → report, email → code_review
两条独立链：{meeting → report} 和 {email → code_review}

链内顺序由依赖关系决定：
  链1: meeting(60min) → report(180min)
  链2: email(30min) → code_review(120min)

链间交织按 SPT（短任务优先）：
  可选起始: meeting(60min) 或 email(30min) → 选 email

最终排序: email → meeting → code_review → report
总加权完成时间: 30 + 90 + 210 + 390 = 720 分钟
```

**Layer 4 验证**：

- 偏序一致性：PASS（meeting 在 report 前 ✓，email 在 code_review 前 ✓）

### 12.4 T4 示例：预算分配

**自然语言输入**：
"公司有100万预算，分给三个部门：研发、市场、运营。研发每投入1万产出1.5万收益，市场每投入1万产出1.2万收益，运营每投入1万产出0.8万收益。每个部门最少分配10万。怎么分配让总收益最大？"

**Layer 1 输出（UDS 摘要）**：

```
entities:
  - id: rd, type: option, properties: [{key: marginal_return, numeric: 1.5}]
  - id: marketing, type: option, properties: [{key: marginal_return, numeric: 1.2}]
  - id: operations, type: option, properties: [{key: marginal_return, numeric: 0.8}]
  - id: budget, type: resource, properties: [{key: total, numeric: 100, unit: 万}]

objectives:
  - description: "最大化总收益", direction: maximize

constraints:
  - type: capacity, numeric_limit: 100, description: "总预算100万"
  - type: budget, involves: [rd], numeric_limit: 10, limit_direction: ge, description: "研发至少10万"
  - type: budget, involves: [marketing], numeric_limit: 10, limit_direction: ge, description: "市场至少10万"
  - type: budget, involves: [operations], numeric_limit: 10, limit_direction: ge, description: "运营至少10万"
```

**Layer 2 输出（MF 摘要）**：

类型判定：T4（资源分配），置信度 0.97

```
T4_Formulation:
  utility_matrix: [[1.5], [1.2], [0.8]]
  resource_totals: [100]
  min_allocations: [[10], [10], [10]]
  is_assignment: false
```

**Layer 3 求解（T4 线性规划）**：

```
LP 建模:
  max 1.5*x_rd + 1.2*x_mkt + 0.8*x_ops
  s.t. x_rd + x_mkt + x_ops = 100
       x_rd >= 10, x_mkt >= 10, x_ops >= 10

由于边际收益线性且研发最高：
  先满足最低保障: rd=10, mkt=10, ops=10（用掉30万）
  剩余70万全部给边际收益最高的研发: rd += 70

最优分配: rd=80万, mkt=10万, ops=10万
总收益: 80×1.5 + 10×1.2 + 10×0.8 = 120 + 12 + 8 = 140万
```

**Layer 4 验证**：

- 约束满足：PASS（80+10+10=100, 各≥10）
- 效率性：PASS（帕累托最优，无法在不减少某部门的情况下增加另一部门的收益）

### 12.5 T5 示例：序贯投资

**自然语言输入**：
"你有10万元。第一年可以选择投股票（50%概率变成15万，50%概率变成7万）或存银行（确定变成10.5万）。第二年不管第一年结果如何，同样可以选股票或银行，规则相同。怎么操作两年后期望收益最大？"

**Layer 1 输出（UDS 摘要）**：

```
entities:
  - id: stock, type: option, properties: [{key: return_up, numeric: 1.5}, {key: return_down, numeric: 0.7}, {key: prob_up, numeric: 0.5}]
  - id: bank, type: option, properties: [{key: return, numeric: 1.05}]

objectives:
  - description: "最大化两年后的期望收益", direction: maximize

relations:
  - source: year1, target: year2, type: precedes
```

**Layer 2 输出（MF 摘要）**：

类型判定：T5（序贯决策），置信度 0.94

```
T5_Formulation:
  n_stages: 2
  stages:
    - stage_id: year1
      state_space: [10]  (初始资金10万)
      action_space: [stock, bank]
      transition_table:
        (10, stock) → [(15, 0.5), (7, 0.5)]
        (10, bank)  → [(10.5, 1.0)]
      reward_table: 即时奖励为0，最终收益在终止状态计算

    - stage_id: year2
      state_space: [15, 10.5, 7]  (第一年可能的结果)
      action_space: [stock, bank]
      transition_table:
        (15, stock) → [(22.5, 0.5), (10.5, 0.5)]
        (15, bank)  → [(15.75, 1.0)]
        (10.5, stock) → [(15.75, 0.5), (7.35, 0.5)]
        (10.5, bank) → [(11.025, 1.0)]
        (7, stock)  → [(10.5, 0.5), (4.9, 0.5)]
        (7, bank)   → [(7.35, 1.0)]

  initial_state: 10
  terminal_rewards: 终态的值就是资金本身
```

**Layer 3 求解（T5 后向归纳）**：

```
第二年（后向）：
  状态=15: E[stock]=0.5×22.5+0.5×10.5=16.5, E[bank]=15.75 → 选 stock (16.5)
  状态=10.5: E[stock]=0.5×15.75+0.5×7.35=11.55, E[bank]=11.025 → 选 stock (11.55)
  状态=7: E[stock]=0.5×10.5+0.5×4.9=7.7, E[bank]=7.35 → 选 stock (7.7)

第一年（后向）：
  状态=10: 
    E[stock] = 0.5×V(15) + 0.5×V(7) = 0.5×16.5 + 0.5×7.7 = 12.1
    E[bank]  = V(10.5) = 11.55
    → 选 stock (12.1)

最优策略: 两年都选股票
期望最终资金: 12.1万
```

**Layer 4 验证**：

- 时间一致性：PASS（第二年的最优策略不依赖第一年的选择，满足贝尔曼最优性原理）

### 12.6 T6 示例：定价博弈

**自然语言输入**：
"两家咖啡店相邻。每家可以定高价（30元）或低价（20元）。如果都定高价，各赚50万；如果都定低价，各赚30万；如果一家高价一家低价，高价的赚10万，低价的赚60万。各自应该怎么定价？"

**Layer 1 输出（UDS 摘要）**：

```
entities:
  - id: shop_A, type: agent
  - id: shop_B, type: agent
  - id: high_price, type: option, properties: [{key: price, numeric: 30}]
  - id: low_price, type: option, properties: [{key: price, numeric: 20}]

objectives:
  - id: obj_A, description: "店A利润最大化", direction: maximize
  - id: obj_B, description: "店B利润最大化", direction: maximize
```

**Layer 2 输出（MF 摘要）**：

类型判定：T6（博弈决策），置信度 0.99

```
T6_Formulation:
  n_players: 2
  strategy_sets: {shop_A: [high, low], shop_B: [high, low]}
  payoff_tensor:
              shop_B:high   shop_B:low
  shop_A:high  (50, 50)     (10, 60)
  shop_A:low   (60, 10)     (30, 30)
  
  game_type: simultaneous
```

**Layer 3 求解（T6 纳什均衡）**：

```
检查纯策略纳什均衡：
  (high, high): A偏离到low → 60>50 ✓ → 不是均衡
  (high, low):  A偏离到low → 30<10? 不，30>10 ✓ → A想偏离; B偏离到high → 50>60? 不 → 不是均衡
  (low, high):  对称分析 → 不是均衡
  (low, low):   A偏离到high → 10<30 → A不想偏离 ✓; B偏离到high → 10<30 → B不想偏离 ✓
  → (low, low) 是唯一纯策略纳什均衡

纳什均衡: 两家都定低价
均衡收益: 各赚30万
```

说明：这是经典的囚徒困境结构——合作（都高价）对双方都更好（各50万），但理性的个体均衡是都低价（各30万）。

**Layer 4 验证**：

- 均衡验证：PASS（任何单方偏离都会导致收益降低）
- 帕累托效率：WARN（(low,low)不是帕累托最优，(high,high)帕累托优于它，但不是均衡）

---

## 十三、错误处理与边界情况

### 13.1 Layer 1 错误处理

| 错误类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| LLM 返回非法 JSON | JSON 解析失败 | 重试最多 3 次；若仍失败，返回 ExtractionError |
| LLM 返回空 entities | 未识别到任何决策实体 | 返回 EmptyEntitiesError，附带原始输入供人工检查 |
| LLM 返回与 schema 不符的字段 | Pydantic 校验失败 | 记录具体校验错误，重试 1 次（附带纠错提示） |
| 多次提取结果严重不一致 | K 次提取中 entities 数量差异 > 50% | 标记 extraction_stability = low，发出警告但不中断 |
| LLM 在输出中夹带了决策建议 | 检测到 "建议"/"应该选" 等关键词 | 过滤掉建议内容，仅保留结构化提取部分 |
| LLM API 超时或限速 | 网络异常 | 指数退避重试，最多 5 次 |

### 13.2 Layer 2 错误处理

| 错误类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| 所有属性评分完全相同 | 评分矩阵每列方差 = 0 | 所有属性设为等权重，发出警告"无区分力" |
| 极性锚点语义过近 | positive 和 negative anchor 的 embedding 余弦相似度 > 0.9 | 该属性标记为"极性不可靠"，使用默认中性评分 0.5 |
| 类型分类无法确定 | 规则分类和语义分类结果不一致 | 取规则分类结果（更可靠），标记 type_confidence = low |
| 数值属性缺失单位 | numeric_value 有值但 unit 为 null | 按无量纲处理，归一化到 [0,1] |
| 属性数量为 0 | UDS 中无属性信息 | 仅基于约束做可行性筛选，从可行集中随机选择（标记 confidence = very_low） |

### 13.3 Layer 3 错误处理

| 错误类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| 无可行解 | 约束互相矛盾，没有任何解满足所有约束 | 返回 InfeasibleError，列出互相矛盾的约束对 |
| DP 内存溢出 | T2 背包容量过大（W > 10^7）| 切换到贪心近似算法，标记 optimality_gap = estimated |
| LP 无界 | T4 分配问题目标函数无上界 | 返回 UnboundedError，提示检查约束是否完整 |
| 博弈无纯策略均衡 | T6 枚举所有纯策略组合均非均衡 | 求混合策略均衡，输出概率分布 |
| 决策树过大 | T5 状态空间 × 阶段数 > 10^6 | 使用近似方法（蒙特卡洛树搜索或启发式剪枝），标记 approximate = true |
| 完全平局 | 所有选项效用值完全相同 | 执行 TieBreaker 规则链，记录使用了哪条规则 |

### 13.4 Layer 4 错误处理

| 错误类型 | 触发条件 | 处理方式 |
|---------|---------|---------|
| 传递性违反检测到环路 | 偏好图中存在环 | 报告环路涉及的选项和决策，按冲突解决策略处理 |
| IIA 违反 | 增加选项后排序翻转 | 记录违反详情，报告使用了哪种权重方法，建议切换到固定权重法 |
| 约束不满足 | 求解器输出不满足 MF 中的约束 | 严重错误——标记为 solver_bug，返回错误而非不满足约束的解 |

### 13.5 通用错误原则

1. **永远不输出不满足硬约束的解**——宁可返回错误也不返回违规的解
2. **区分"最优解"和"近似解"**——如果使用了近似算法，必须在输出中标记
3. **所有错误都记入审计日志**——即使错误被优雅处理，日志中也要有完整记录
4. **错误信息可读**——面向用户的错误信息用自然语言，面向开发者的信息附带技术详情

---

## 十四、性能基准与计算量分析

### 14.1 各层的计算复杂度

| 层 | 核心操作 | 时间复杂度 | 典型耗时 |
|---|---------|----------|---------|
| Layer 1 | LLM API 调用 × K 次 | O(K × LLM_latency) | 3-10秒（K=3） |
| Layer 2 | Embedding 推理 × (实体数+属性数) | O((N+M) × embed_dim) | 0.1-1秒 |
| Layer 2 | 权重计算 | O(N × M) | <1ms |
| Layer 3 (T1) | SAW 加权求和 | O(N × M) | <1ms |
| Layer 3 (T2) | DP 背包 | O(N × W) | <1秒（N≤1000, W≤10^6） |
| Layer 3 (T3) | 拓扑排序 | O(N + E) | <1ms |
| Layer 3 (T4) | 线性规划（单纯形） | O(N^2 × M) 平均情况 | <1秒 |
| Layer 3 (T5) | 值迭代 | O(S × A × T) | <1秒（小规模） |
| Layer 3 (T6) | 纳什均衡枚举 | O(S1 × S2 × ... × Sn) | <1秒（策略数≤20） |
| Layer 4 | 环路检测 | O(V + E) | <1ms |

**瓶颈在 Layer 1**：LLM API 调用占总耗时的 90%+。Layer 2-4 的总计算量约为 10^3-10^6 FLOPs，比 LLM 直接决策（10^12 FLOPs）低 6-9 个数量级。

### 14.2 可扩展性边界

| 参数 | 推荐上限 | 原因 | 突破方案 |
|------|---------|------|---------|
| T1 选项数 N | 10,000 | O(N×M) 线性增长，无瓶颈 | 无需突破 |
| T2 物品数 N | 1,000 | DP 表大小 N×W | 改用贪心近似或分支定界 |
| T2 容量 W | 10^6 | DP 内存 O(W) | 改用分支定界 |
| T3 任务数 N | 10,000 | 拓扑排序 O(N+E)，很快 | 无需突破 |
| T4 接收者数 | 1,000 | LP 变量数 N×M | 用 scipy sparse 或 ortools |
| T5 状态空间 S | 10,000 | 值迭代 O(S×A×T) | 近似 DP 或 MCTS |
| T6 策略数 | 20 per player | 纯策略枚举 exponential | 支持枚举（support enumeration） |

### 14.3 内存消耗估算

| 组件 | 内存占用 | 说明 |
|------|---------|------|
| Embedding 模型 | ~500MB-1GB | 模型加载后常驻内存 |
| UDS 数据结构 | <1MB | 单次决策的提取结果 |
| 评分矩阵 | <10KB（N=100, M=20） | numpy array |
| DP 表 | N × W × sizeof(float) | T2 背包，1000×10^6 ≈ 8GB → 需优化 |
| 偏好图 | O(V + E) | 历史决策积累 |

### 14.4 基准测试方案

对每种类型运行标准测试集，记录：

- 端到端延迟（包含 LLM 调用）
- Layer 2-4 延迟（不含 LLM 调用）
- 峰值内存占用
- 解的最优性（与已知最优解对比）

---

## 十五、配置系统设计

### 15.1 配置文件结构

**llm_config.yaml**：

```
# LLM 接入配置
provider: openai | anthropic | zhipu | local
model: gpt-4o | claude-3.5-sonnet | glm-4 | ...
api_key_env: OPENAI_API_KEY          # 环境变量名
base_url: null                        # 自定义 API 地址（本地模型时使用）
temperature: 0                        # 固定为 0，确保确定性
max_tokens: 4096
timeout_seconds: 30
retry_max: 3
retry_backoff: exponential

# 多次提取配置
extraction_rounds: 3                  # K 值
extraction_vote_threshold: 2          # 至少 K 中几次一致才保留
```

**embedding_config.yaml**：

```
# Embedding 模型配置
model_name: BAAI/bge-m3              # HuggingFace 模型名
device: cpu | cuda                    # 推理设备
batch_size: 32                        # 批量 embedding 的 batch size
cache_enabled: true                   # 是否缓存 embedding 结果
cache_dir: .cache/embeddings

# 极性轴配置
polarity_similarity_threshold: 0.85   # 属性去重的余弦相似度阈值
anchor_similarity_min: 0.3            # 正负锚点最低差异度（低于此值则锚点无效）
```

**solver_config.yaml**：

```
# 求解器全局配置
weight_method: variance | entropy | critic | ensemble  # 权重方法
weight_ensemble_weights: [0.33, 0.33, 0.34]            # ensemble 时三种方法的权重

# T1 配置
t1:
  cross_validate_topsis: true         # 是否用 TOPSIS 交叉验证 SAW

# T2 配置
t2:
  max_dp_capacity: 1000000            # DP 容量上限，超过则切换分支定界
  use_greedy_validation: true         # 是否用贪心法交叉验证

# T3 配置
t3:
  scheduling_rule: spt | edd | auto   # 调度规则（auto 由 objectives 语义决定）

# T4 配置
t4:
  lp_solver: scipy | ortools          # LP 求解器选择
  allow_fractional: true              # 是否允许非整数分配

# T5 配置
t5:
  max_state_space: 100000             # 状态空间上限，超过则用近似方法
  discount_factor_default: 1.0

# T6 配置
t6:
  max_pure_strategy_enumeration: 1000 # 纯策略枚举上限
  compute_mixed_equilibrium: true     # 是否计算混合策略均衡

# 通用平局处理
tiebreaker:
  rules: [secondary_attributes, minimax_regret, lexicographic]
```

### 15.2 配置优先级

命令行参数 > 环境变量 > 配置文件 > 默认值

所有配置在系统启动时加载一次，运行过程中不可变——保证确定性。

---

## 十六、CLI 接口设计

### 16.1 基本用法

```
stallmate decide "你走进厕所，5个坑位，第2和第4个有人，选哪个？"

stallmate decide --file problem.txt

cat problem.txt | stallmate decide --stdin
```

### 16.2 输出模式

```
# 简洁模式（默认）：只输出决策结果
stallmate decide "..." --output brief

# 详细模式：输出完整决策过程
stallmate decide "..." --output detailed

# JSON 模式：输出完整 FinalOutput JSON
stallmate decide "..." --output json

# 审计模式：输出完整审计日志
stallmate decide "..." --output audit
```

### 16.3 高级选项

```
# 指定权重方法
stallmate decide "..." --weight-method critic

# 指定 LLM 提供商
stallmate decide "..." --llm-provider anthropic --llm-model claude-3.5-sonnet

# 指定 embedding 模型
stallmate decide "..." --embedding-model BAAI/bge-m3

# 强制指定决策类型（跳过自动分类）
stallmate decide "..." --force-type T2

# 敏感性分析模式
stallmate decide "..." --sensitivity-analysis

# 对比模式：同时用 LLM 直接决策和 CSHDA 决策，输出对比
stallmate decide "..." --compare-with-llm

# 批量模式
stallmate batch --input problems.jsonl --output results.jsonl
```

### 16.4 交互模式

```
stallmate interactive

> 请输入决策问题：
  你有一个能装15公斤的背包...

> 类型识别: T2 (约束组合选择), 置信度 0.98
> 求解中...

> 决策结果:
  选择: 帐篷, 睡袋, 食物, 水壶, 急救包
  总价值: 180元, 总重量: 15kg
  置信度: 0.99

> 一致性验证: 全部 PASS
> 输入 'explain' 查看详细过程, 'next' 输入下一个问题, 'quit' 退出
```

---

## 十七、可视化方案

### 17.1 决策过程可视化

**T1（单选）**：

- 评分矩阵热力图：行=选项，列=属性，颜色深浅=评分
- 权重饼图：各属性权重占比
- 效用柱状图：各选项的最终效用值
- 雷达图：选中选项 vs 次优选项在各维度的对比

**T2（背包）**：

- DP 表热力图：行=物品，列=容量，颜色=最优价值
- 物品选择甘特图：每个物品是否被选中
- 价值密度排序图：贪心路径可视化

**T3（排序）**：

- 依赖关系 DAG 图：节点=任务，有向边=依赖关系
- 甘特图：最终排序的时间线展示
- 优先级矩阵热力图

**T4（分配）**：

- 分配桑基图：资源→接收者的流向
- 分配比例堆叠柱状图

**T5（序贯）**：

- 决策树可视化：节点=状态，边=行动，叶节点=收益
- 最优路径高亮

**T6（博弈）**：

- 支付矩阵表格（带纳什均衡标记）
- 最优反应图：每个玩家对对方策略的最优反应

### 17.2 对比可视化（CSHDA vs LLM）

- MCR 对比柱状图：相同场景下 CSHDA（恒为1.0）vs LLM 的 MCR
- JSD 对比热力图：CSHDA（恒为0.0）vs LLM 的跨模板 JSD
- 公理违反率雷达图：CSHDA（全0）vs LLM 的各公理违反率
- 计算量对比柱状图（对数刻度）

### 17.3 可视化实现

使用 matplotlib 生成静态图（论文/博客用），plotly 生成交互图（Web 演示用）。

所有可视化函数接受 DecisionResult 和 FinalOutput 作为输入，自动根据 decision_type 选择对应的可视化方案。

---

## 十八、扩展性设计

### 18.1 新增决策类型

如果未来需要支持第七种决策类型（如多目标优化）：

1. 在 schema.py 中定义 T7_Formulation 数据模型
2. 在 type_classifier.py 中添加 T7 的分类规则
3. 在 formulator.py 中添加 UDS → T7_Formulation 的转换逻辑
4. 创建 t7_solver.py 实现求解器
5. 在 axiom_checker.py 中添加 T7 的专项公理检查
6. 在 dispatcher.py 的分发表中注册 T7

其余代码（Layer 1 提取、Layer 2 embedding 量化、Layer 4 偏好图）无需修改。

### 18.2 替换组件

| 可替换组件 | 接口要求 | 替换场景 |
|-----------|---------|---------|
| LLM 提供商 | 接受 prompt，返回 JSON 字符串 | 切换到本地部署的模型 |
| Embedding 模型 | 接受字符串，返回固定维度向量 | 切换到更好的 embedding 模型 |
| LP 求解器 | scipy.linprog 兼容接口 | 切换到 Gurobi/CPLEX 等商业求解器 |
| 博弈论求解器 | nashpy 兼容接口 | 切换到 Gambit 等更强大的求解器 |

### 18.3 多语言支持

当前设计已天然支持多语言：

- Layer 1 的 LLM 支持任何语言的输入
- Layer 2 的 embedding 模型选用多语言模型（BGE-M3）
- Layer 3-4 完全在数值/符号层面工作，与语言无关

唯一需要注意的是：极性锚点的生成质量可能因语言而异。可通过增加 extraction_rounds 来提高稳定性。

---

## 十九、安全与隐私

### 19.1 数据流向

- 用户输入 → LLM API（外部传输）→ UDS（本地）→ 后续全部本地计算
- 敏感数据仅在 Layer 1 与 LLM API 交互时离开本地环境
- 如果使用本地 LLM（通过 base_url 配置），则全部数据留在本地

### 19.2 审计与合规

- 所有决策过程的完整审计日志存储在本地
- 日志中不存储 LLM API key
- 支持配置日志的保留期限和自动清理

### 19.3 对抗性输入处理

- Layer 1 的 LLM 提取 prompt 中明确禁止执行任何指令注入
- UDS 的 Pydantic 校验拒绝不符合 schema 的输入
- Layer 3 的求解器对输入做 bounds checking，防止数值溢出

---

## 二十、博客写作大纲

### 20.1 建议标题

"StallMate：当大模型走进公共厕所——我们为什么需要一个全新的决策引擎"

### 20.2 章节结构

**Part 1：问题发现**

- 用厕所问题引出LLM决策不一致的发现
- Phase 1 和 Phase 2 的核心数据和结论
- 核心论点：LLM的问题不是理解力，而是决策结构的缺失

**Part 2：解决方案**

- CSHDA 的设计哲学：解耦理解与决策
- 四层架构概述
- 六种决策类型的覆盖
- 关键创新：语义极性轴投影、自动权重、因果图自动构建

**Part 3：实战演示**

- 六种类型各一个端到端示例
- 重点展示：相同场景下 CSHDA vs LLM 的对比
- 一致性、最优性、计算量的硬数据

**Part 4：技术深潜**

- Layer 2 的 embedding 极性轴投影原理
- Layer 3 各求解器的算法选择理由
- Layer 4 的公理验证机制

**Part 5：讨论与展望**

- 系统的局限性（诚实讨论）
- 与现有 neuro-symbolic 工作的关系
- 未来方向：更多决策类型、在线学习、人类偏好适配

### 20.3 预计篇幅

约 15,000-20,000 字，配 15-20 张图表。
