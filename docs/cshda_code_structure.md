# CSHDA 通用决策引擎：代码结构设计 | Code Structure

> 本文档描述 CSHDA（Consistent Symbolic-Heuristic Decision Architecture）决策引擎的完整代码结构。
> 基于 `docs/our_method.md` 的技术方案，适配当前 `stall-mate` 项目的工程规范。

---

## 一、整体架构概览

### 1.1 双阶段工程结构

项目包含两个紧密关联但又独立可运行的阶段：

```
stall-mate/
│
├── 【已有】Phase 1/2 实验框架               ← 已完成，89 测试通过
│   用于验证 LLM 决策不一致性，为 CSHDA 提供对比基线
│   src/stall_mate/{types,schema,config,prompt,client,runner,recorder,analysis}
│
└── 【新增】CSHDA 决策引擎                    ← 待实现
    通用确定性决策系统，覆盖 6 种决策类型
    src/stall_mate/cshda/
```

**设计原则**：CSHDA 作为 `stall_mate` 的一个子包（`cshda/`），与已有实验框架共享 `types/`、`config/`、`client/` 等基础设施，但不修改任何已有模块的行为。

### 1.2 四层流水线

```
自然语言
   │
   ▼
┌──────────────────────────────────────┐
│ Layer 1 · 语义提取                     │  LLM（仅此一层使用）
│ 自然语言 → UniversalDecisionSpec (UDS) │  复用: client/, instructor
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Layer 2 · 量化建模                     │  Embedding + 规则
│ UDS → MathematicalFormulation (MF)    │  新增: sentence-transformers
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Layer 3 · 符号求解                     │  6 种确定性求解器
│ MF → DecisionResult (DR)             │  新增: scipy, nashpy, networkx
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Layer 4 · 一致性保障                   │  公理验证 + 审计
│ DR + 历史 → FinalOutput              │
└──────────────────────────────────────┘
```

---

## 二、目录结构

### 2.1 CSHDA 子包结构

```
src/stall_mate/cshda/
├── __init__.py                     # 公开 API: CSHDAEngine, decide()
│
├── engine.py                       # 主引擎：串联四层流水线
│
├── schema/                         # 所有 Pydantic 数据模型
│   ├── __init__.py
│   ├── uds.py                      # UniversalDecisionSpec
│   ├── formulation.py              # MathematicalFormulation + T1~T6 子模型
│   ├── result.py                   # DecisionResult
│   └── output.py                   # FinalOutput + ConsistencyReport
│
├── layer1_extraction/              # Layer 1: 语义提取
│   ├── __init__.py
│   ├── extractor.py                # LLM 调用封装 + 多次提取 + 投票
│   ├── prompts.py                  # System prompt 模板（提取指令 + 输出格式）
│   ├── anchor_generator.py         # 极性锚点生成（额外 LLM 调用）
│   └── normalizer.py               # UDS 规范化（ID 标准化、属性去重、约束格式统一）
│
├── layer2_quantification/          # Layer 2: 量化建模
│   ├── __init__.py
│   ├── embedder.py                 # Embedding 模型封装（sentence-transformers）
│   ├── polarity_scorer.py          # 语义极性轴投影评分
│   ├── type_classifier.py          # 决策类型识别（规则 + 语义辅助）
│   ├── weight_calculator.py        # 客观权重：方差法 / 熵权法 / CRITIC
│   ├── relation_analyzer.py        # 属性冗余检测 + 情境影响矩阵
│   └── formulator.py               # UDS → MF 转换（按类型分派）
│
├── layer3_solver/                  # Layer 3: 符号求解
│   ├── __init__.py
│   ├── base.py                     # 求解器抽象基类 BaseSolver
│   ├── dispatcher.py               # 类型分发器（MF.decision_type → 对应 Solver）
│   ├── t1_selection.py             # T1: SAW + TOPSIS 交叉验证
│   ├── t2_knapsack.py              # T2: 动态规划 + 贪心验证
│   ├── t3_ranking.py               # T3: 偏序融合 + 拓扑排序
│   ├── t4_allocation.py            # T4: 线性规划 + 匈牙利算法
│   ├── t5_sequential.py            # T5: 后向归纳 / 值迭代
│   ├── t6_game.py                  # T6: 纳什均衡 + 后向归纳
│   └── tiebreaker.py               # 通用平局处理链
│
└── layer4_consistency/             # Layer 4: 一致性保障
    ├── __init__.py
    ├── axiom_checker.py            # 公理验证器（通用 + 类型专项）
    ├── preference_graph.py         # 偏好图维护（networkx 有向图 + 环检测）
    ├── confidence_scorer.py        # 综合置信度计算
    └── audit_logger.py             # 审计日志（JSONL 格式）
```

### 2.2 新增配置文件

```
configs/
├── cshda/                          # CSHDA 专属配置
│   ├── llm_config.yaml             # LLM 提取配置（provider, model, extraction_rounds）
│   ├── embedding_config.yaml       # Embedding 模型配置（model_name, device, 缓存）
│   └── solver_config.yaml          # 求解器参数（权重方法、DP 上限、平局规则）
```

### 2.3 测试结构

```
tests/
├── ...                             # 已有 89 个测试（Phase 1/2）
├── cshda/                          # CSHDA 测试
│   ├── test_schema.py              # 数据模型校验测试
│   ├── test_layer1/                # Layer 1 单元测试
│   │   ├── test_extractor.py       #   提取 + 投票逻辑（mock LLM）
│   │   ├── test_normalizer.py      #   规范化逻辑
│   │   └── test_anchor.py          #   锚点生成
│   ├── test_layer2/                # Layer 2 单元测试
│   │   ├── test_polarity_scorer.py #   极性轴投影（已知正负锚点）
│   │   ├── test_type_classifier.py #   类型分类（构造典型 UDS）
│   │   ├── test_weight_calculator.py # 权重计算（已知方差矩阵）
│   │   └── test_formulator.py      #   UDS → MF 转换
│   ├── test_layer3/                # Layer 3 单元测试
│   │   ├── test_t1.py              #   SAW/TOPSIS（已知最优解）
│   │   ├── test_t2.py              #   经典背包用例
│   │   ├── test_t3.py              #   已知拓扑序 DAG
│   │   ├── test_t4.py              #   简单 LP / 匈牙利
│   │   ├── test_t5.py              #   简单 MDP
│   │   ├── test_t6.py              #   囚徒困境、性别战争
│   │   └── test_tiebreaker.py      #   平局处理
│   ├── test_layer4/                # Layer 4 单元测试
│   │   ├── test_axiom_checker.py   #   公理验证
│   │   └── test_preference_graph.py #  偏好图 + 环检测
│   └── test_integration/           # 端到端集成测试
│       ├── test_e2e_t1_toilet.py   #   厕所问题
│       ├── test_e2e_t2_knapsack.py #   背包问题
│       ├── test_e2e_t3_scheduling.py # 排序问题
│       ├── test_e2e_t4_budget.py   #   预算分配
│       ├── test_e2e_t5_invest.py   #   多阶段投资
│       └── test_e2e_t6_pricing.py  #   定价博弈
```

### 2.4 实验脚本

```
scripts/
├── ...                             # 已有脚本
├── run_cshda.py                    # CSHDA 单次决策入口
├── run_cshda_batch.py              # CSHDA 批量实验
└── run_comparison.py               # CSHDA vs LLM 对比实验
```

---

## 三、核心数据模型（schema/）

### 3.1 模型继承关系

```
                     ┌──────────────────┐
                     │  UDS (layer1)     │  自然语言 → 结构化提取
                     │  UniversalDecisionSpec
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  MF (layer2)      │  结构化 → 数学化
                     │  MathematicalFormulation
                     │  ┌─ T1_Formulation│  评分矩阵 + 权重
                     │  ├─ T2_Formulation│  价值向量 + 容量约束
                     │  ├─ T3_Formulation│  优先级矩阵 + 先后约束
                     │  ├─ T4_Formulation│  效用矩阵 + 资源总量
                     │  ├─ T5_Formulation│  阶段 + 状态转移表
                     │  └─ T6_Formulation│  支付矩阵 + 策略集
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  DR (layer3)      │  数学化 → 决策结果
                     │  DecisionResult   │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  FinalOutput (layer4)│  验证后的最终输出
                     │  + ConsistencyReport │
                     │  + AuditTrail        │
                     └──────────────────┘
```

### 3.2 关键字段

**UDS（UniversalDecisionSpec）**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `metadata` | `ExtractionMeta` | 原始输入、LLM 名称、提取时间 |
| `entities` | `list[Entity]` | 所有决策实体（id, label, type, properties） |
| `objectives` | `list[Objective]` | 优化目标（maximize/minimize/target） |
| `constraints` | `list[Constraint]` | 约束条件（capacity/budget/time...） |
| `relations` | `list[Relation]` | 实体间关系（depends_on/conflicts_with...） |
| `decision_context` | `list[ContextFactor]` | 情境因素 |
| `decision_type_hint` | `str \| None` | LLM 的类型初判（仅供参考） |

**MF（MathematicalFormulation）**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `decision_type` | `Literal["T1","T2","T3","T4","T5","T6"]` | Layer 2 确定的类型 |
| `type_confidence` | `float` | 类型判定置信度 |
| `formulation` | `T1_Formulation \| ... \| T6_Formulation` | 类型特定的数学结构 |
| `embedding_artifacts` | `EmbeddingArtifacts` | 保留的 embedding 产物（供审计） |

**DR（DecisionResult）**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `chosen` | `list[str]` | 被选中的实体 id |
| `ranking` | `list[str] \| None` | 完整排序 |
| `allocation` | `dict[str, float] \| None` | 分配方案 |
| `strategy` | `dict[str, str] \| None` | 博弈策略 |
| `objective_value` | `float` | 目标函数值 |
| `margin` | `float` | 最优与次优的差距 |
| `solver_trace` | `list[dict]` | 求解过程记录 |

### 3.3 Pydantic 鉴别器模式

使用 Pydantic v2 的 `Literal` 鉴别器区分类型特定模型：

```python
# formulation.py
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, Discriminator

class T1_Formulation(BaseModel):
    decision_type: Literal["T1"] = "T1"
    score_matrix: list[list[float]]
    weights: list[float]
    # ...

class T2_Formulation(BaseModel):
    decision_type: Literal["T2"] = "T2"
    value_vector: list[float]
    # ...

Formulation = Annotated[
    Union[T1_Formulation, T2_Formulation, ..., T6_Formulation],
    Discriminator("decision_type"),
]

class MathematicalFormulation(BaseModel):
    decision_type: str
    type_confidence: float
    formulation: Formulation
    embedding_artifacts: EmbeddingArtifacts
```

---

## 四、四层实现细节

### 4.1 Layer 1：语义提取层

**模块职责**：将自然语言转换为 UDS，LLM 仅做翻译不做决策。

| 模块 | 核心功能 | 关键依赖 |
|------|---------|---------|
| `extractor.py` | 多次提取 + 字段级投票 | `client/`（已有 LLMClient）、`instructor` |
| `prompts.py` | System prompt 模板管理 | YAML 配置 |
| `anchor_generator.py` | 极性锚点生成（额外 LLM 调用） | `client/` |
| `normalizer.py` | ID 标准化、属性去重、约束格式统一 | `embedder.py`（余弦相似度去重） |

**与已有代码的复用关系**：

```
已有 client/LLMClient.query_structured()  ← 复用
已有 schema/StallChoice                   ← 不复用（CSHDA 用 UDS schema）
已有 config/loader.py                     ← 复用配置加载模式
已有 recorder/jsonl.py                    ← 复用审计日志记录模式
```

**多次提取投票策略**：

```python
# extractor.py 伪代码
async def extract(self, natural_input: str) -> UDS:
    # K 次提取（默认 K=3, T=0）
    raw_results = [self._single_extract(natural_input) for _ in range(K)]

    # 字段级投票
    entities = field_level_union_and_vote(raw_results, "entities")
    objectives = field_level_majority(raw_results, "objectives", threshold=2)
    constraints = field_level_union(raw_results, "constraints")  # 宁多勿少
    relations = field_level_majority(raw_results, "relations", threshold=2)

    uds = UDS(entities=entities, objectives=objectives, ...)
    return self.normalizer.normalize(uds)
```

### 4.2 Layer 2：量化建模层

**模块职责**：将 UDS 量化为数学公式，包含评分、权重、类型判定。

| 模块 | 核心功能 | 算法 |
|------|---------|------|
| `embedder.py` | 文本 → 向量 | `sentence-transformers` (BGE-M3) |
| `polarity_scorer.py` | 属性评分 [0,1] | SemAxis：`score = cos(desc - neg, pos - neg)` |
| `type_classifier.py` | 决策类型识别 | 规则优先（entity_type → relation → constraint → objective） |
| `weight_calculator.py` | 属性权重计算 | 方差法 / 熵权法 / CRITIC，等权集成 |
| `relation_analyzer.py` | 属性冗余检测 | 余弦相似度矩阵，>0.80 折减 |
| `formulator.py` | UDS → MF | 按 `decision_type` 分派到 6 个转换函数 |

**Embedding 封装**：

```python
# embedder.py
from sentence_transformers import SentenceTransformer

class Embedder:
    """封装 sentence-transformers 模型，确保确定性输出。"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> np.ndarray:
        """单文本 → 向量 (1, dim)。"""
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """批量文本 → 矩阵 (n, dim)。"""
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32)
```

**极性轴投影评分（核心创新）**：

```
对属性 attr_j：
  axis = embed(positive_anchor) - embed(negative_anchor)
  axis_unit = axis / ||axis||

对实体 entity_i 在属性 attr_j 上的描述 desc_ij：
  如果有 numeric_value → min-max 归一化到 [0, 1]
  如果只有 value_description：
    score = cos(embed(desc) - embed(neg_anchor), axis_unit)
    score = clip(score, 0, 1)
```

### 4.3 Layer 3：符号求解层

**模块职责**：确定性数学求解，每种类型一个独立求解器。

| 求解器 | 算法 | 交叉验证 | 核心依赖 |
|--------|------|---------|---------|
| `t1_selection.py` | SAW 加权求和 | TOPSIS | `numpy` |
| `t2_knapsack.py` | 动态规划 | 贪心法 | `numpy` |
| `t3_ranking.py` | 拓扑排序 + SPT/EDD | — | `networkx` |
| `t4_allocation.py` | 线性规划 / 匈牙利算法 | — | `scipy.optimize` |
| `t5_sequential.py` | 后向归纳 / 值迭代 | — | `numpy` |
| `t6_game.py` | 纳什均衡枚举 / Lemke-Howson | — | `nashpy` |

**求解器基类接口**：

```python
# base.py
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    """所有求解器的抽象基类。"""

    @abstractmethod
    def solve(self, formulation) -> DecisionResult:
        """求解并返回结果。"""
        ...

    @abstractmethod
    def validate(self, formulation) -> bool:
        """校验输入格式是否合法。"""
        ...

    def explain(self, result: DecisionResult) -> str:
        """生成可读的求解过程解释。"""
        return str(result.solver_trace)
```

**分发器**：

```python
# dispatcher.py
_SOLVERS: dict[str, type[BaseSolver]] = {
    "T1": T1SelectionSolver,
    "T2": T2KnapsackSolver,
    "T3": T3RankingSolver,
    "T4": T4AllocationSolver,
    "T5": T5SequentialSolver,
    "T6": T6GameSolver,
}

def dispatch(mf: MathematicalFormulation) -> DecisionResult:
    solver_cls = _SOLVERS[mf.decision_type]
    solver = solver_cls()
    if not solver.validate(mf.formulation):
        raise ValueError(f"Invalid {mf.decision_type} formulation")
    return solver.solve(mf.formulation)
```

### 4.4 Layer 4：一致性保障层

**模块职责**：验证决策结果的公理合规性，维护全局偏好图。

| 模块 | 核心功能 | 验证内容 |
|------|---------|---------|
| `axiom_checker.py` | 通用 + 类型专项公理 | 确定性、约束满足、传递性、IIA、时间一致性... |
| `preference_graph.py` | 偏好有向图 + 环检测 | `networkx.DiGraph`，DFS 环检测 O(V+E) |
| `confidence_scorer.py` | 综合置信度 | extraction_stability × quantification_robustness × solution_margin |
| `audit_logger.py` | 完整审计日志 | JSONL 格式，复用 `recorder/jsonl.py` 的模式 |

**公理检查矩阵**：

| 公理 | T1 | T2 | T3 | T4 | T5 | T6 |
|------|----|----|----|----|----|----|
| 确定性（重复运行结果一致） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 约束满足 | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| 传递性 | ✓ | — | — | — | — | — |
| IIA（无关选项独立性） | ✓ | — | — | — | — | — |
| 框架不变性 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 最优性证明 | — | ✓ | — | — | — | — |
| 偏序一致性 | — | — | ✓ | — | — | — |
| 排序稳定性 | — | — | ✓ | — | — | — |
| 帕累托效率 | — | — | — | ✓ | — | ✓ |
| 贝尔曼最优性 | — | — | — | — | ✓ | — |
| 均衡验证 | — | — | — | — | — | ✓ |

---

## 五、主引擎串联（engine.py）

```python
# engine.py 核心逻辑
class CSHDAEngine:
    """CSHDA 决策引擎：串联四层流水线。"""

    def __init__(self, config_path: Path):
        self.config = load_cshda_config(config_path)
        self.extractor = Extractor(self.config.llm)
        self.embedder = Embedder(self.config.embedding)
        self.scorer = PolarityScorer(self.embedder)
        self.classifier = TypeClassifier()
        self.weight_calc = WeightCalculator()
        self.formulator = Formulator(self.scorer, self.weight_calc)
        self.dispatcher = SolverDispatcher()
        self.axiom_checker = AxiomChecker()
        self.preference_graph = PreferenceGraph()
        self.audit_logger = AuditLogger(self.config.audit)

    def decide(self, natural_input: str) -> FinalOutput:
        """完整的四层决策流水线。"""

        # Layer 1: 语义提取
        uds = self.extractor.extract(natural_input)

        # Layer 2: 量化建模
        mf = self.formulator.formulate(uds)

        # Layer 3: 符号求解
        dr = self.dispatcher.dispatch(mf)

        # Layer 4: 一致性保障
        consistency = self.axiom_checker.check(dr, mf, self.preference_graph)
        self.preference_graph.add_decision(dr, mf)

        # 构建最终输出
        output = FinalOutput(
            decision=dr,
            consistency_report=consistency,
            confidence_score=self._compute_confidence(uds, mf, dr, consistency),
            audit_trail=self.audit_logger.build_trail(natural_input, uds, mf, dr, consistency),
        )
        self.audit_logger.log(output)
        return output
```

---

## 六、新增依赖

### 6.1 pyproject.toml 新增项

```toml
[project]
dependencies = [
    # 已有依赖保持不变
    "pydantic>=2.0",
    "instructor>=1.0",
    "pyyaml>=6.0",
    "httpx>=0.27",
    "rich>=13.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    # CSHDA 新增
    "sentence-transformers>=2.2",    # Layer 2: Embedding 模型
    "nashpy>=0.0.39",                # Layer 3 (T6): 纳什均衡
    "networkx>=3.0",                 # Layer 3 (T3) + Layer 4: 图操作
]

[project.optional-dependencies]
cshda-heavy = [
    "ortools>=9.6",                  # Layer 3: 更强的组合优化（可选）
]
```

### 6.2 依赖分层

```
Layer 1: openai + instructor (已有)
Layer 2: sentence-transformers (新增) + numpy (已有)
Layer 3: numpy (已有) + scipy (已有) + nashpy (新增) + networkx (新增)
Layer 4: networkx (新增)
全局:   pydantic (已有) + pyyaml (已有) + rich (已有)
```

---

## 七、与已有代码的复用关系

### 7.1 直接复用（不改代码）

| 已有模块 | CSHDA 用途 |
|---------|-----------|
| `client/LLMClient` | Layer 1 的 LLM API 调用 |
| `config/loader.py` 的 YAML 加载模式 | CSHDA 配置加载 |
| `recorder/jsonl.py` 的 JSONL 记录模式 | Layer 4 审计日志 |
| `types/record.py` 的 Pydantic 模式 | schema 设计参考 |

### 7.2 模式复用（参考风格，新写代码）

| 已有模式 | CSHDA 对应 |
|---------|-----------|
| `config/loader.py` 的 `ExperimentConfig` | CSHDA 的 `CSHDAConfig`、各层配置 |
| `runner/experiment.py` 的 `RunStats` | CSHDA 的 `ConfidenceBreakdown` |
| `analysis/metrics.py` 的统计函数 | Layer 2 的权重计算 |
| `analysis/visualize.py` 的图表模式 | CSHDA 的决策可视化 |

### 7.3 数据复用

| 已有数据 | CSHDA 用途 |
|---------|-----------|
| `data/phase1.jsonl` (1980 条) | T1 厕所问题对比实验基线 |
| `data/phase2.jsonl` (2340 条) | T1 条件场景对比实验基线 |
| `output/final_report/` | 论文/博客的 Phase 1/2 结论引用 |

---

## 八、实现路线图

### Phase A：核心骨架（T1 端到端跑通）

```
A1 ─── schema/ 全部数据模型定义
 │
A2 ─── layer1_extraction/ 提取器 + system prompt
 │
A3 ─── layer2_quantification/ 极性轴投影 + 方差权重
 │
A4 ─── layer3_solver/t1_selection.py SAW + TOPSIS
 │
A5 ─── layer4_consistency/ 基础公理检查
 │
A6 ─── engine.py 串联，厕所问题端到端验证
```

**验收标准**：同一个厕所场景 10 次调用，MCR = 1.0，与 Phase 1 LLM 的 MCR 0.53~0.90 形成对比。

### Phase B：扩展 6 种类型

```
B1 ─── type_classifier.py 决策类型识别
 │
 ├── B2 ─── t2_knapsack.py
 ├── B3 ─── t3_ranking.py
 ├── B4 ─── t4_allocation.py
 ├── B5 ─── t5_sequential.py
 └── B6 ─── t6_game.py
```

**验收标准**：每种类型至少 3 个端到端测试通过。

### Phase C：完善 + 对比实验

```
C1 ─── axiom_checker.py 类型专项验证扩展
 │
C2 ─── weight_calculator.py CRITIC + 熵权法
 │
C3 ─── 全部单元测试 + 集成测试
 │
C4 ─── run_comparison.py CSHDA vs LLM 对比实验
 │
C5 ─── 决策过程可视化
```

**验收标准**：对比实验报告完成，CSHDA 在一致性指标上 100% 满分。

---

## 九、配置示例

### 9.1 configs/cshda/llm_config.yaml

```yaml
# LLM 提取配置
provider: openai
model: glm-5.1
base_url: "http://localhost:3000/v1"
temperature: 0
max_tokens: 4096
timeout_seconds: 60
retry_max: 3
retry_backoff: exponential

# 多次提取
extraction_rounds: 3
extraction_vote_threshold: 2
anchor_generation_rounds: 2
```

### 9.2 configs/cshda/embedding_config.yaml

```yaml
# Embedding 模型
model_name: "BAAI/bge-m3"
device: cpu
batch_size: 32
cache_enabled: true
cache_dir: ".cache/embeddings"

# 极性轴
polarity_similarity_threshold: 0.85
anchor_similarity_min: 0.3
```

### 9.3 configs/cshda/solver_config.yaml

```yaml
# 求解器全局配置
weight_method: ensemble
weight_ensemble_weights: [0.33, 0.33, 0.34]

t1:
  cross_validate_topsis: true

t2:
  max_dp_capacity: 1000000
  use_greedy_validation: true

t3:
  scheduling_rule: auto

t4:
  lp_solver: scipy
  allow_fractional: true

t5:
  max_state_space: 100000
  discount_factor_default: 1.0

t6:
  max_pure_strategy_enumeration: 1000
  compute_mixed_equilibrium: true

tiebreaker:
  rules: [secondary_attributes, minimax_regret, lexicographic]
```

---

## 十、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| CSHDA 作为子包而非新项目 | `src/stall_mate/cshda/` | 共享基础设施，对比实验可直接引用已有数据 |
| Pydantic v2 Discriminator | `Literal` 联合类型 | 类型安全的 formulation 分派，编译期检查完备性 |
| Embedding 模型 | BGE-M3 | 1024 维，原生中英双语，<500M 参数 |
| 权重方法 | 三法等权集成 | 避免单一方法偏差，CRITIC 自动折减冗余属性 |
| 求解器分发 | 静态字典映射 | 简单、可扩展、运行时零开销 |
| 审计日志 | JSONL | 与已有 `recorder/` 一致，支持流式追加 |
| 配置加载 | YAML | 与已有 `config/` 一致 |
| 测试策略 | 先单元后集成 | Layer 2/3 全部 mock-free（纯数值），Layer 1 mock LLM |
