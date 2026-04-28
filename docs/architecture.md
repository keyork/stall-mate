# Stall Mate 架构文档

> 坑位博弈 — 大模型决策一致性压力测试框架

---

## 1. 概述

Stall Mate 测试前沿大语言模型在面对极其简单的场景时，是否能做出一致的决策：在一排条件完全相同的厕所隔间中，仅凭位置差异做出选择。通过在不同参数组合下重复相同（或等价）的提示词数百次，可以衡量模型的"推理"究竟是稳定的偏好，还是随口编造的事后合理化。

---

## 2. 架构总览

### 数据流

```
                    ┌─────────────┐
                    │  YAML       │
                    │  configs/   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  配置加载器  │  读取 YAML，验证为
                    │  Config     │  Pydantic 模型
                    │  Loader     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐    │     ┌──────▼──────┐
       │  提示词     │    │     │  模型配置   │
       │  构建器     │    │     │  Model      │
       │  Builder    │    │     │  Config     │
       └──────┬──────┘    │     └──────┬──────┘
              │            │            │
              │    ┌───────▼───────┐    │
              │    │  实验运行器   │    │
              │    │  Experiment   │    │
              │    │  Runner       │◄───┘
              │    │  (编排器)     │
              │    └───┬───────┬───┘
              │        │       │
       ┌──────▼────────▼─┐   │
       │  LLM 客户端     │   │
       │  (三级回退)     │   │
       └──────┬──────────┘   │
              │              │
       ┌──────▼──────────────▼──┐
       │  JSONL 记录器          │
       │  (追加写入日志)        │
       └───────────────────────┘
```

**流水线步骤：**

1. **配置加载器** 读取 YAML 文件，验证为 Pydantic 模型
2. **实验运行器** 生成所有参数组合（num_stalls x temperatures x templates x repetitions），随机打乱以避免顺序效应
3. 对每组参数，**提示词构建器** 填充模板中的占位符
4. **LLM 客户端** 发送提示词，带结构化输出约束（三级回退）
5. 响应被解析、分类（VALID/REFUSED/AMBIGUOUS）并记录
6. **JSONL 记录器** 每行追加一条 `ExperimentRecord`

---

## 3. 模块详解

### 3.1 `types/` — 数据类型

定义所有枚举和核心数据记录模型。

| 符号 | 类型 | 说明 |
|------|------|------|
| `ExperimentPhase` | `str, Enum` | `PHASE1`、`PHASE2` — 实验阶段标识 |
| `PromptTemplate` | `str, Enum` | `A`、`B`、`C`、`D` — 提示词模板标识 |
| `ChoiceStatus` | `str, Enum` | `VALID`、`REFUSED`、`AMBIGUOUS` — 响应分类 |
| `ExperimentRecord` | `BaseModel` | 单次模型调用的完整记录（18 个字段） |

**`ExperimentRecord` 字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `record_id` | `str` | 唯一标识（UUID hex，12 位） |
| `experiment_phase` | `ExperimentPhase` | Phase1 或 Phase2 |
| `experiment_group` | `str` | 实验分组标识（如 "1.1"） |
| `model_name` | `str` | 模型名称（来自配置） |
| `model_version` | `str` | 模型版本（默认 "unknown"） |
| `temperature` | `float` | 采样温度 |
| `prompt_template` | `PromptTemplate` | 使用的模板（A/B/C/D） |
| `prompt_text` | `str` | 完整渲染后的提示词文本 |
| `num_stalls` | `int` | 坑位总数 |
| `occupied_stalls` | `list[int]` | 已占用的坑位编号（Phase 2 使用） |
| `conditions` | `dict` | 附加实验条件 |
| `raw_response` | `str` | 模型原始响应文本 |
| `extracted_choice` | `int | None` | 解析出的坑位编号（提取失败为 None） |
| `choice_status` | `ChoiceStatus` | VALID / REFUSED / AMBIGUOUS |
| `reasoning_present` | `bool` | 是否捕获到结构化推理 |
| `extracted_reasoning` | `str` | 结构化输出中的推理文本 |
| `response_tokens` | `int` | API 响应的 token 数 |
| `latency_ms` | `int` | 请求延迟（毫秒） |
| `timestamp` | `datetime` | 调用的 UTC 时间戳 |

### 3.2 `schema/` — LLM 响应模式

定义 LLM 被要求产生的结构化输出格式。

- **`StallChoice(BaseModel)`** — 结构化响应模型：
  - `chosen_stall: int` — 选择的坑位编号（通过 `[1, num_stalls]` 范围验证）
  - `chain_of_thought: str` — 模型推理过程（最小长度 10）
  - `confidence: float` — 信心程度 [0.0, 1.0]
  - 包含 `validate_stall_range` 字段验证器，检查 `context["num_stalls"]` 范围

- **`get_stallchoice_json_schema() -> dict`** — 返回 `StallChoice` 的 JSON Schema，用于 PLAIN_TEXT 回退时的系统提示

### 3.3 `config/` — 配置加载

从 YAML 文件加载并验证为类型化的 Pydantic 模型。

**Pydantic 配置模型：**

| 模型 | 来源 | 关键字段 |
|------|------|----------|
| `ModelConfig` | `configs/models.yaml` | `name`、`endpoint`、`api_key`、`version`、`timeout`、`max_retries`、`probe_message` |
| `ExperimentConfig` | `configs/experiments/*.yaml` | `experiment_id`、`num_stalls`、`temperatures`、`templates`、`repetitions`、`conditions`、`occupied_stalls` |
| `PromptTemplateConfig` | `configs/prompt_templates/*.yaml` | `templates: dict[str, str]`、`system_message_template` |
| `ClassificationConfig` | `configs/classification.yaml` | `refusal_keywords`、`chinese_patterns`、`english_patterns`、`trailing_digit_pattern`、`general_digit_pattern`、`direction_reversal` |
| `DirectionReversalPair` | （嵌套在 ClassificationConfig 中） | `source`、`target` |

**加载函数：**

| 函数 | 输入 | 输出 |
|------|------|------|
| `load_yaml(path)` | YAML 文件路径 | `dict` |
| `load_model_config(path)` | models.yaml 路径 | `ModelConfig` |
| `load_experiment_config(path)` | 实验配置 YAML 路径 | `ExperimentConfig` |
| `load_prompt_templates(path)` | 模板 YAML 路径 | `PromptTemplateConfig` |
| `load_classification_config(path)` | 分类配置 YAML 路径 | `ClassificationConfig` |
| `discover_experiments(config_dir)` | 实验配置目录 | `list[ExperimentConfig]` |

`ClassificationConfig` 提供两个辅助方法：
- `to_extraction_patterns() -> dict` — 转换为 `ExperimentRunner` 所需的格式
- `to_reversal_pairs() -> list[dict]` — 转换为 `build_reverse_prompt()` 所需的格式

### 3.4 `prompt/` — 提示词构建

通过替换占位符渲染提示词模板，构建结构化输出的系统消息。

- **`build_prompt(template_text, num_stalls, **kwargs) -> str`** — 替换模板中的 `{num_stalls}` 及其他占位符

- **`build_system_message(num_stalls, template=None) -> str`** — 返回结构化 JSON 输出的系统消息。使用传入的模板（通常从 `PromptTemplateConfig.system_message_template` 加载），或回退到内置默认模板。模板必须包含 `{num_stalls}` 占位符。

- **`build_reverse_prompt(template_text, num_stalls, reversal_pairs=None, **kwargs) -> str`** — 通过应用 `{"source": "...", "target": "..."}` 替换列表，构建方向反转的提示词。默认使用内置的中英文方向对。用于对称性实验 1.2。

### 3.5 `client/` — LLM 客户端

与 OpenAI 兼容 API 通信，支持自动能力探测和优雅降级。

```
LLMClient(endpoint, model, api_key="", timeout=60, max_retries=2, probe_message="Say OK")
```

**初始化：** 延迟初始化 — 底层 `OpenAI` 客户端在首次使用时创建，而非构造时。所有参数有合理默认值，通常从 `ModelConfig` 加载。

**API 探测 — `probe_api() -> str`：**

按顺序尝试三种模式，确定 API 支持哪种：

| 优先级 | 模式 | 机制 | 适用场景 |
|--------|------|------|----------|
| 1 | `TOOLS` | `instructor.Mode.TOOLS`（函数调用） | OpenAI，能力最强的 API |
| 2 | `JSON_SCHEMA` | `instructor.Mode.JSON_SCHEMA` | 支持 JSON 模式但不支持函数调用的 API |
| 3 | `PLAIN_TEXT` | 原始 `OpenAI` 客户端，不用 instructor | 最小 API 支持 |

**查询方法：**

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `query_structured(prompt, system_message, temperature, num_stalls)` | `(StallChoice | None, str, int, int)` | TOOLS/JSON_SCHEMA 模式使用 instructor；回退到 `query_plain()` |
| `query_plain(prompt, system_message, temperature)` | `(str, int, int)` | 直接 OpenAI 调用，返回 (原始响应, token 数, 延迟毫秒) |

### 3.6 `recorder/` — JSONL 记录器

以追加写入的 JSONL 格式持久化实验结果（每行一个 JSON 对象）。

```
JSONLRecorder(output_path: Path)
```

| 方法 | 说明 |
|------|------|
| `record(record: ExperimentRecord)` | 追加一条 JSON 记录 |
| `record_batch(records: list[ExperimentRecord])` | 追加多条记录 |
| `read_all() -> list[ExperimentRecord]` | 读取并解析所有记录 |
| `count() -> int` | 统计行数（不解析内容） |
| `clear()` | 删除输出文件 |

输出目录在初始化时自动创建。

### 3.7 `runner/` — 实验运行器

编排完整流水线 — 参数组合生成、提示词构建、API 调用、响应分类和记录。

```
ExperimentRunner(client, recorder, model_config, refusal_keywords=None, extraction_patterns=None)
```

**参数：**
- `refusal_keywords` — 拒绝关键词字符串列表（从 `ClassificationConfig.refusal_keywords` 加载）。默认使用内置的中英文关键词。
- `extraction_patterns` — 文本提取的正则模式字典（通过 `ClassificationConfig.to_extraction_patterns()` 加载）。默认使用内置模式。

**公开方法：**

- **`run_single(prompt, system_message, temperature, num_stalls, metadata) -> ExperimentRecord`** — 执行一次 API 调用，分类响应，记录结果

- **`run_experiment(experiment_config, templates) -> list[ExperimentRecord]`** — 运行完整实验：
  1. 生成所有参数组合：`num_stalls x temperatures x templates x repetitions`
  2. **随机打乱** 以避免顺序效应
  3. 依次通过 `run_single()` 执行每组参数
  4. 返回所有记录

**响应分类 — `_classify_response()`：**

| 状态 | 条件 |
|------|------|
| `VALID` | `extracted_choice` 不为 None 且在 `[1, num_stalls]` 范围内 |
| `REFUSED` | 原始响应包含拒绝关键词（如 "无法"、"refuse"、"cannot"） |
| `AMBIGUOUS` | 无法提取有效选择且未检测到拒绝 |

**文本提取 — `_extract_choice_from_text()`：**

当结构化输出失败（PLAIN_TEXT 模式）时，通过正则模式级联提取坑位编号：

1. 中文模式：`第(\d+)个`、`(\d+)号`、`选择.*?(\d+)`
2. 英文模式：`stall (\d+)`、`number (\d+)`
3. 响应末尾的裸数字
4. 响应中最后一个在有效范围内的数字

---

## 4. 配置指南

### 4.1 模型配置

**文件：** `configs/models.yaml`

```yaml
models:
  - name: glm-5.1
    endpoint: http://localhost:3000/v1    # 基址 — SDK 自动拼接 /chat/completions
    api_key: ""                            # 可选，默认为 "unused"
    version: "latest"                      # 可选，默认为 "unknown"
    timeout: 60                            # API 调用超时（秒）
    max_retries: 2                         # 结构化输出最大重试次数
    probe_message: "Say OK"                # 探测 API 能力时使用的消息
```

注意：`endpoint` 是基址 URL。OpenAI SDK 会自动拼接 `/chat/completions`。

### 4.1.1 分类配置

**文件：** `configs/classification.yaml`

控制响应分类和文本提取的行为：

```yaml
refusal_keywords: ["无法", "不能", "拒绝", "refuse", "cannot", "won't", "I can't", "inappropriate"]
chinese_patterns: ["第\\s*(\\d+)\\s*个", "(\\d+)\\s*号", "选择.*?(\\d+)"]
english_patterns: ["stall\\s*(\\d+)", "number\\s*(\\d+)"]
trailing_digit_pattern: "(\\d+)\\s*[。.!?]?\\s*$"
general_digit_pattern: "\\b(\\d+)\\b"
direction_reversal:
  - source: "从左到右"
    target: "从右到左"
  - source: "from left to right"
    target: "from right to left"
```

所有字段都有合理默认值 — 文件可以为空或不存在。

### 4.2 实验配置

**目录：** `configs/experiments/`

每个 YAML 文件定义一个实验：

```yaml
experiment_id: "1.1"                    # 唯一标识
experiment_group: "1.1"                 # 分组键
phase: "Phase1"                         # ExperimentPhase 枚举值
description: "基准选择"                  # 可读描述
num_stalls: [3, 5, 7, 10]              # 要测试的坑位数量列表
temperatures: [0.0, 0.7]               # 采样温度
templates: ["A", "B", "C"]             # 使用的提示词模板键
repetitions: 30                         # 每组参数的重复次数
conditions: {}                          # 附加实验条件
occupied_stalls: []                     # 已占用的坑位（Phase 2 使用）
```

**每个实验的总调用数：** `len(num_stalls) * len(temperatures) * len(templates) * repetitions`

**当前实验：**

| ID | 说明 | num_stalls | 总调用数 |
|----|------|------------|----------|
| 1.1 | 基准选择 — 全部空坑位 | [3, 5, 7, 10] | 4 x 2 x 3 x 30 = 720 次 |
| 1.2 | 对称性 — 方向反转 | [5] | 1 x 2 x 3 x 30 = 180 次 |
| 1.3 | 选项数量效应 | [3, 5, 7, 10, 15, 20] | 6 x 2 x 3 x 30 = 1,080 次 |

### 4.3 提示词模板

**文件：** `configs/prompt_templates/phase1.yaml`

```yaml
templates:
  A: "一排{num_stalls}个坑位，全部空闲，干净程度完全相同，除位置外没有任何差异。你选第几个？只回答数字。"
  B: "你走进一间公共厕所，面前有一排{num_stalls}个独立的坑位，从左到右依次编号1到{num_stalls}。所有坑位都没有人使用..."
  C: "一个人走进公共厕所，看到一排{num_stalls}个空坑位，编号1到{num_stalls}。所有坑位的卫生状况和硬件条件完全一样..."
  D: "面前有一排{num_stalls}个坑位，全部空闲，干净程度相同，设施完全一样，除位置外没有其他差别。你可以选最左边的、最右边的..."
```

- `{num_stalls}` 是占位符，运行时由 `build_prompt()` 替换
- 所有模板都强调"除位置外条件完全相同"
- 模板 A 最精简；B 和 C 增加场景描述；D 显式列举选项

---

## 5. 数据格式

### JSONL 记录示例

输出 JSONL 文件中每一行是一条完整的 `ExperimentRecord`：

```json
{
  "record_id": "a3f8b2c1e4d5",
  "experiment_phase": "Phase1",
  "experiment_group": "1.1",
  "model_name": "glm-5.1",
  "model_version": "latest",
  "temperature": 0.0,
  "prompt_template": "A",
  "prompt_text": "一排5个坑位，全部空闲，干净程度完全相同，除位置外没有任何差异。你选第几个？只回答数字。",
  "num_stalls": 5,
  "occupied_stalls": [],
  "conditions": {},
  "raw_response": "{\"chosen_stall\": 3, \"chain_of_thought\": \"...\", \"confidence\": 0.8}",
  "extracted_choice": 3,
  "choice_status": "VALID",
  "reasoning_present": true,
  "extracted_reasoning": "中间位置最安全",
  "response_tokens": 0,
  "latency_ms": 15234,
  "timestamp": "2026-04-28T10:30:00.123456+00:00"
}
```

---

## 6. 关键设计决策

### 6.1 三级结构化输出回退

**决策：** `LLMClient.probe_api()` 依次尝试 TOOLS、JSON_SCHEMA，最终回退到 PLAIN_TEXT。拒绝关键词、文本提取正则模式、方向反转字符串全部可通过 `configs/classification.yaml` 配置。

**原因：** 自托管或本地模型 API（如 vLLM、Ollama、自定义端点）通常不支持完整的 OpenAI 函数调用。客户端优雅降级而非直接失败。在 PLAIN_TEXT 模式下，运行器通过基于正则的文本提取（`_extract_choice_from_text()`）从自由格式响应中解析坑位编号。将这些模式外化到配置文件，研究人员可以针对新语言或模型行为调整分类规则，无需修改代码。

### 6.2 不使用 litellm

**决策：** 直接使用 `instructor` + `openai` SDK，不引入 `litellm`。

**原因：** Stall Mate 每次实验运行只针对一个模型端点。`litellm` 的多提供商路由能力在此场景下是多余的。`instructor` 以最小依赖提供我们所需的结构化输出提取。

### 6.3 不使用异步

**决策：** 所有 API 调用都是顺序执行的（同步）。

**原因：** 完整 Phase 1 实验需要约 1,980 次调用，每次约 15-30 秒（总计约 8-16 小时）。这是批处理任务，不是延迟敏感的服务。顺序执行避免并发 bug、简化调试、保证记录顺序确定性。需要时可后续添加异步支持。

### 6.4 配置驱动

**决策：** 所有实验参数、模型设置、提示词模板和分类模式都存放在 YAML 文件中。Python 代码中无硬编码值。

**原因：** 将实验设计与实现分离。研究人员可以通过编辑 YAML 文件定义新实验、调整分类阈值或修改系统提示词，无需触碰代码。降低意外代码修改影响实验条件的风险。

### 6.5 JSONL 输出

**决策：** 每行一条 `ExperimentRecord`，通过标准库 `json` 写入。

**原因：** JSONL 支持追加写入（中断运行也安全）、按行分隔（易于用 `jq`、`wc -l`、`grep` 等标准工具处理）、不需要额外库。每行自包含，可独立解析。

### 6.6 随机打乱

**决策：** `run_experiment()` 生成所有参数组合后随机打乱再执行。

**原因：** 顺序执行相同参数可能与 API 端的缓存或限流产生交互。打乱将不同参数条件分散到时间轴上，减少时间混杂因素。

### 6.7 API 端点使用基址

**决策：** `configs/models.yaml` 存储基址 URL（如 `http://localhost:3000/v1`），而非完整的 chat completions 路径。

**原因：** OpenAI Python SDK 自动在基址后拼接 `/chat/completions`。只存储基址避免双重路径问题（`/v1/chat/completions/chat/completions`）。

---

## 7. 扩展指南

### 7.1 添加新提示词模板

1. 在 `configs/prompt_templates/phase1.yaml` 中添加模板：
   ```yaml
   E: "你的新模板，包含{num_stalls}个坑位..."
   ```
2. 在 `src/stall_mate/types/record.py` 的 `PromptTemplate` 枚举中添加 `"E"`
3. 在实验 YAML 的 `templates` 字段中引用模板 `"E"`

### 7.2 添加新实验

在 `configs/experiments/` 目录下创建新 YAML 文件：
```yaml
experiment_id: "1.4"
experiment_group: "1.4"
phase: "Phase1"
description: "你的实验描述"
num_stalls: [5]
temperatures: [0.0]
templates: ["A", "B"]
repetitions: 50
conditions: {}
occupied_stalls: []
```

使用 `discover_experiments()` 自动加载目录下所有实验配置。

### 7.3 添加新模型

在 `configs/models.yaml` 中添加：
```yaml
models:
  - name: glm-5.1
    endpoint: http://localhost:3000/v1
    api_key: ""
    version: "latest"
    timeout: 60
    max_retries: 2
    probe_message: "Say OK"
  - name: gpt-4o
    endpoint: https://api.openai.com/v1
    api_key: "sk-..."
    version: "2024-08-06"
    timeout: 120
    max_retries: 3
```

目前 `load_model_config()` 加载第一条记录。如需多模型支持，修改加载器以接受模型名称参数。

### 7.4 自定义分类

编辑 `configs/classification.yaml` 来修改：
- **拒绝关键词** — 为新语言或模型行为增删关键词
- **文本提取模式** — 为新的响应格式添加正则模式
- **方向反转对** — 为对称性测试添加新语言对

所有字段都有默认值；文件可以部分填写。

### 7.5 扩展到 Phase 2

Phase 2 引入已占用坑位（部分坑位有人，模型必须在剩余坑位中选择）：

1. `ExperimentPhase` 枚举中已有 `PHASE2 = "Phase2"`（无需修改）
2. 创建带有非空 `occupied_stalls` 的实验配置：
   ```yaml
   occupied_stalls: [1, 3, 5]  # 这些坑位已被占用
   ```
3. 更新 `build_prompt()` 在提示词文本中融入已占用坑位信息
4. `ExperimentRunner` 和 `ExperimentRecord` 已支持 `occupied_stalls` — 无需修改

### 7.6 添加异步支持

大规模实验的未来优化方向：

1. 将 `LLMClient.query_structured()` 和 `query_plain()` 改为 `async`
2. 使用 `asyncio.Semaphore` 限制并发请求数
3. 用 `asyncio.Queue` 替代 `random.shuffle()` 实现并发消费
4. `JSONLRecorder.record()` 需要加锁以支持并发写入
5. 保持同步 API 为默认，异步作为可选

---

## 8. 验证

### 功能验证脚本

**文件：** `scripts/verify_10calls.py`

运行 10 次 API 调用，覆盖不同参数组合，验证完整流水线：

| 组别 | 参数 | 目的 |
|------|------|------|
| 1（3 次） | N=5, T=0.0, 模板 A/B/C | 确定性温度下的模板变化 |
| 2（3 次） | N=3/7/10, T=0.7, 模板 A | 坑位数量变化 |
| 3（2 次） | N=5, T=0.0, 模板 D | 一致性检查 — 相同参数调用两次 |
| 4（2 次） | N=15/20, T=0.7, 模板 B/C | 边界情况 — 大坑位数 |

**验证步骤：**
1. 从 `configs/` 加载配置
2. 初始化 `LLMClient`、`JSONLRecorder`、`ExperimentRunner`
3. 执行 10 次调用，打印进度和结果
4. 回读 JSONL 文件并验证：记录数匹配、所有必填字段存在
5. 退出码 0 = 通过，1 = 失败

**运行：**
```bash
uv run python scripts/verify_10calls.py
```

注意：每次调用约 15-30 秒。总运行时间约 3-5 分钟。

### 单元测试

**运行：**
```bash
uv run pytest
```

85 个测试覆盖所有模块。所有测试使用 mock — 测试期间不发起真实 API 调用。测试文件位于 `tests/` 目录，结构与源代码对应。
