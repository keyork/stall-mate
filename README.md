# stall-mate

**坑位博弈：当大模型走进公共厕所**

LLM decision-consistency stress test + CSHDA deterministic decision engine.

Do LLMs actually "think" when they make decisions, or do they just hallucinate a preference? This project stress-tests that question through an absurdly simple scenario (choosing a toilet stall), then provides CSHDA, a deterministic engine that solves the same class of problems with guaranteed consistency.

---

## 这个项目是什么 / What This Is

Two independent components in one repo:

**1. Phase 1/2 experiment framework** — Prompts frontier models with a trivial stall-choosing scenario, repeats the call dozens of times, and measures whether their "reasoning" stays consistent. Results are recorded as JSONL for analysis.

**2. CSHDA decision engine** (`src/stall_mate/cshda/`) — Consistent Symbolic-Heuristic Decision Architecture. A universal engine that turns natural-language decision problems into deterministic symbolic solutions through a four-layer pipeline. Currently implements T1 (multi-attribute selection). Demo-verified on the toilet stall problem with 0.97 confidence.

---

## 前置条件 / Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI-compatible LLM endpoint at `http://localhost:3000/v1` (model `glm-5.1`)
- (CSHDA only) Embedding model `BAAI/bge-m3` — downloaded automatically, or see below for manual setup

---

## 快速开始 / Quick Start

```bash
# Install dependencies
uv sync

# Run experiment framework verification (requires API)
uv run python scripts/verify_10calls.py

# Run CSHDA demo (requires API + embedding model)
uv run python scripts/demo_cshda.py

# Run tests (all mocked, no API calls needed)
uv run pytest
```

### Embedding model setup / 嵌入模型安装

CSHDA uses `BAAI/bge-m3` for embedding. It auto-downloads from HuggingFace on first run. For manual download:

```bash
# If in China, use the mirror first
export HF_ENDPOINT=https://hf-mirror.com

# Download to local directory (engine auto-detects ./models/bge-m3)
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='./models/bge-m3', ignore_patterns=['.DS_Store','imgs/*','onnx/*'])"
```

The engine checks `./models/bge-m3` first. If present, it skips downloading. The `models/` directory is gitignored (large binary files).

---

## CSHDA 决策引擎 / CSHDA Decision Engine

### Four-layer pipeline / 四层流水线

| Layer | Name | Description |
|-------|------|-------------|
| 1 | Semantic Extraction | LLM parses natural language into Universal Decision Spec (UDS). Multiple extraction rounds with stability measurement. |
| 2 | Quantification | Embedding model scores attribute polarity; Formulator builds mathematical representation; Type classifier identifies decision type (T1-T6). |
| 3 | Symbolic Solving | Type-specific solver. T1 uses SAW (Simple Additive Weighting) + TOPSIS for multi-attribute selection, with tiebreaker. |
| 4 | Consistency Assurance | Axiom checking (determinism, constraint satisfaction, transitivity, IIA, frame invariance), preference graph construction, full audit trail. |

### Decision types / 决策类型

| Type | Name | Status | Description |
|------|------|--------|-------------|
| T1 | Multi-attribute selection | Implemented | Choose best option from discrete set by scoring attributes |
| T2 | Knapsack / budget allocation | Planned | Select subset under capacity/budget constraint |
| T3 | Ranking / prioritization | Planned | Order items by preference |
| T4 | Resource allocation | Planned | Distribute limited resources among agents |
| T5 | Sequential planning | Planned | Ordered sequence of decisions with dependencies |
| T6 | Multi-agent game | Planned | Strategic interaction between multiple decision makers |

### Usage / 用法

```python
from stall_mate.cshda import CSHDAEngine

engine = CSHDAEngine(
    model="glm-5.1",
    base_url="http://localhost:3000/v1",
    embedding_model="BAAI/bge-m3",  # or path to local model
    extraction_rounds=3,
    device="cpu",
)

result = engine.decide("你走进一间公共厕所，面前有一排5个独立的坑位，编号1到5。其中第2个和第4个有人正在使用。你会选择哪一个？")

print(result.decision.chosen)            # e.g. ["3"]
print(result.confidence_score)           # e.g. 0.97
print(result.consistency_report.determinism)  # "PASS"
```

### Demo output / Demo 输出

Running `uv run python scripts/demo_cshda.py` produces:

```
╭─────────────────────────────────────────────────────╮
│ CSHDA 通用决策引擎 Demo                             │
│ Consistent Symbolic-Heuristic Decision Architecture │
╰─────────────────────────────────────────────────────╯

输入问题:
你走进一间公共厕所，面前有一排5个独立的坑位，编号1到5。其中第2个和第4个有人正在使用。你会选择哪一个？

运行中... (Layer 1 需要 LLM API 调用，约 30-60 秒)

╭────────────────────────────────────────────╮
│ 决策结果                                   │
│                                            │
│ 选择: ['option_5']                         │
│ 排序: ['option_5', 'option_3', 'option_1'] │
│ 效用值: 0.9796                             │
│ 余量 (margin): 0.4774                      │
│ 求解器: SAW+TOPSIS                         │
╰────────────────────────────────────────────╯

                       中间计算结果                        
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 指标             ┃ 值                                   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ saw_utilities    │ {"option_1": 0.023, "option_3": 0.502,  │
│                  │  "option_5": 0.980}                     │
│ topsis_closeness │ {"option_1": 0.001, "option_3": 0.500,  │
│                  │  "option_5": 0.999}                     │
└──────────────────┴──────────────────────────────────────────┘

╭──────────────────────────────────────────╮
│ 一致性报告                               │
│ 确定性: PASS | 约束满足: PASS            │
│ 传递性: N/A | IIA: N/A | 框架不变性: N/A │
│                                          │
│ 置信度: 0.97                             │
│   提取稳定性: 1.00                       │
│   量化鲁棒性: 0.85                       │
│   解余量: 0.4774                         │
╰──────────────────────────────────────────╯
```

---

## 实验框架 / Experiment Framework

### How it works / 工作原理

The framework prompts an LLM with a stall-choosing scenario, parses the structured response (`instructor` + `StallChoice` Pydantic model), and records every call as a JSONL record. Phase 1 tests baseline consistency across temperatures and stall counts. Phase 2 adds conditional experiments (26 scenarios with varied occupied patterns, group sizes, cultural framing, etc.).

### Running experiments / 运行实验

```bash
# Phase 1 experiments
uv run python scripts/run_phase1_5.py

# Phase 2 conditional experiments
uv run python scripts/run_phase2.py

# 10-call functional verification (requires API)
uv run python scripts/verify_10calls.py

# Analyze results
uv run python scripts/analyze_phase1.py
```

Experiment output goes to `data/` as JSONL files (gitignored except `.gitkeep`).

---

## 项目结构 / Project Structure

```
stall-mate/
├── configs/                        # All YAML configuration
│   ├── models.yaml                 # LLM endpoint definition
│   ├── experiments/                # Phase 1 + Phase 2 experiment configs
│   │   ├── phase1_*.yaml           # Phase 1: baseline tests
│   │   └── phase2/                 # Phase 2: conditional experiments (26 files)
│   ├── prompt_templates/           # Prompt variants (A-D)
│   └── classification.yaml
├── src/stall_mate/
│   ├── types/                      # Enums and data types
│   ├── schema/                     # LLM response schema (StallChoice)
│   ├── config/                     # YAML config loader
│   ├── prompt/                     # Prompt builder
│   ├── client/                     # LLM client (instructor, 3-tier fallback)
│   ├── runner/                     # Experiment runner
│   ├── recorder/                   # JSONL recorder
│   ├── analysis/                   # Result analysis module
│   └── cshda/                      # CSHDA decision engine
│       ├── engine.py               # Pipeline orchestrator
│       ├── schema/                 # UDS, Formulation, Result, Output schemas
│       ├── layer1_extraction/      # LLM → UDS extraction
│       ├── layer2_quantification/  # Embedding + scoring → math formulation
│       ├── layer3_solver/          # SAW+TOPSIS solver (T1)
│       └── layer4_consistency/     # Axiom check + preference graph + audit
├── tests/                          # 89 unit tests (all mocked)
├── scripts/                        # Verification and demo scripts
├── data/                           # JSONL output (gitignored)
└── models/                         # Local embedding model (gitignored)
```

---

## 配置 / Configuration

### LLM endpoint (`configs/models.yaml`)

```yaml
models:
  - name: glm-5.1
    endpoint: http://localhost:3000/v1
    api_key: ""
    timeout: 60
    max_retries: 2
```

### Experiment configs (`configs/experiments/`)

Each YAML file defines:

```yaml
experiment_id: "1.1"
phase: "Phase1"
description: "基准选择（全空坑位）"
num_stalls: [3, 5, 7, 10]
temperatures: [0.0, 0.7]
templates: ["A", "B", "C"]
repetitions: 30
conditions: {}
occupied_stalls: []
```

### Prompt templates (`configs/prompt_templates/`)

Four variants (A-D) with `{num_stalls}` placeholder. Phase 1 uses direct prompts; Phase 1 CoT adds chain-of-thought; Phase 2 extends with conditional framing.

---

## 测试 / Testing

```bash
# Run all 89 tests
uv run pytest

# With coverage
uv run pytest --cov=stall_mate
```

All tests are fully mocked. No real API calls, no embedding model downloads required.

---

## 技术细节 / Technical Details

**Structured output fallback**: The experiment framework uses `instructor` with a three-tier strategy for LLM response parsing: TOOLS mode, JSON_SCHEMA mode, then PLAIN_TEXT regex extraction.

**Config-driven design**: No hardcoded prompts or parameters. Everything comes from YAML files under `configs/`.

**UDS (Universal Decision Spec)**: CSHDA's intermediate representation. A Pydantic model with entities, objectives, constraints, relations, and context factors. Normalized from raw LLM extraction output.

**Type classification**: Rule-based classifier examines UDS structure (agent count, relation types, constraint patterns) to pick T1-T6. No LLM call needed.

**Confidence scoring**: Weighted average of extraction stability (Layer 1), type confidence (Layer 2), solution margin (Layer 3), and axiom check results (Layer 4).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
