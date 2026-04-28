# Stall Mate Architecture / 架构文档

> Stall Mate — LLM decision-consistency stress test
> 大模型决策一致性压力测试框架

---

## 1. Overview / 概述

Stall Mate tests whether frontier LLMs make consistent decisions when presented with a trivially simple scenario: choosing a toilet stall where all options are identical except position. By repeating the same (or equivalent) prompt hundreds of times across varying parameters, we can measure whether a model's "reasoning" reflects a stable preference or is simply hallucinated post-hoc justification.

坑位博弈测试前沿大语言模型在面对极其简单的场景时，是否能做出一致的决策。通过在不同参数组合下重复相同（或等价）的提示词数百次，可以衡量模型的"推理"是稳定的偏好，还是随口编造的事后合理化。

---

## 2. Architecture Overview / 架构总览

### Data Flow / 数据流

```
                    ┌─────────────┐
                    │  YAML       │
                    │  configs/   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Config     │  Load & validate
                    │  Loader     │  Pydantic models
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐    │     ┌──────▼──────┐
       │  Prompt     │    │     │  Model      │
       │  Builder    │    │     │  Config     │
       └──────┬──────┘    │     └──────┬──────┘
              │            │            │
              │    ┌───────▼───────┐    │
              │    │  Experiment   │    │
              │    │  Runner       │◄───┘
              │    │  (orchestrator)
              │    └───┬───────┬───┘
              │        │       │
       ┌──────▼────────▼─┐   │
       │  LLM Client     │   │
       │  (3-tier fallback)│  │
       └──────┬──────────┘   │
              │              │
       ┌──────▼──────────────▼──┐
       │  JSONL Recorder        │
       │  (append-only log)     │
       └───────────────────────┘
```

**Pipeline steps / 流水线步骤:**

1. **Config Loader** reads YAML files and validates them into Pydantic models
2. **Experiment Runner** generates all parameter combinations (num_stalls x temperatures x templates x repetitions), shuffles them to avoid order effects
3. For each combination, **Prompt Builder** fills template placeholders
4. **LLM Client** sends the prompt with structured output enforcement (3-tier fallback)
5. Response is parsed, classified (VALID/REFUSED/AMBIGUOUS), and recorded
6. **JSONL Recorder** appends one `ExperimentRecord` per line

---

## 3. Module Guide / 模块详解

### 3.1 `types/` — Data Types / 数据类型

**Purpose:** Defines all enums and the core data record model.

**Key exports:**

| Symbol | Type | Description |
|--------|------|-------------|
| `ExperimentPhase` | `str, Enum` | `PHASE1`, `PHASE2` — identifies the experiment phase |
| `PromptTemplate` | `str, Enum` | `A`, `B`, `C`, `D` — template identifiers |
| `ChoiceStatus` | `str, Enum` | `VALID`, `REFUSED`, `AMBIGUOUS` — response classification |
| `ExperimentRecord` | `BaseModel` | Complete record for a single model call (18 fields) |

**`ExperimentRecord` fields / 字段:**

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | `str` | Unique ID (UUID hex, 12 chars) |
| `experiment_phase` | `ExperimentPhase` | Phase1 or Phase2 |
| `experiment_group` | `str` | Experiment group identifier (e.g. "1.1") |
| `model_name` | `str` | Model name from config |
| `model_version` | `str` | Model version (default "unknown") |
| `temperature` | `float` | Sampling temperature |
| `prompt_template` | `PromptTemplate` | Template used (A/B/C/D) |
| `prompt_text` | `str` | Full rendered prompt text |
| `num_stalls` | `int` | Total number of stalls |
| `occupied_stalls` | `list[int]` | Occupied stall numbers (Phase 2) |
| `conditions` | `dict` | Additional experiment conditions |
| `raw_response` | `str` | Raw model response text |
| `extracted_choice` | `int | None` | Parsed stall number (None if extraction failed) |
| `choice_status` | `ChoiceStatus` | VALID / REFUSED / AMBIGUOUS |
| `reasoning_present` | `bool` | Whether structured reasoning was captured |
| `extracted_reasoning` | `str` | Reasoning text from structured output |
| `response_tokens` | `int` | Token count from API response |
| `latency_ms` | `int` | Request latency in milliseconds |
| `timestamp` | `datetime` | UTC timestamp of the call |

### 3.2 `schema/` — LLM Response Schema / LLM 响应模式

**Purpose:** Defines the structured output format that the LLM is asked to produce.

**Key exports:**

- **`StallChoice(BaseModel)`** — The structured response model:
  - `chosen_stall: int` — stall number chosen (validated against range `[1, num_stalls]`)
  - `chain_of_thought: str` — model's reasoning (min length 10)
  - `confidence: float` — confidence level [0.0, 1.0]
  - Includes `validate_stall_range` field validator that checks against `context["num_stalls"]`

- **`get_stallchoice_json_schema() -> dict`** — Returns the JSON Schema for `StallChoice`, used in system prompts for PLAIN_TEXT fallback

### 3.3 `config/` — Configuration Loading / 配置加载

**Purpose:** Loads and validates YAML configuration files into typed Pydantic models.

**Pydantic config models:**

| Model | Source | Key fields |
|-------|--------|------------|
| `ModelConfig` | `configs/models.yaml` | `name`, `endpoint`, `api_key`, `version`, `timeout`, `max_retries`, `probe_message` |
| `ExperimentConfig` | `configs/experiments/*.yaml` | `experiment_id`, `num_stalls`, `temperatures`, `templates`, `repetitions`, `conditions`, `occupied_stalls` |
| `PromptTemplateConfig` | `configs/prompt_templates/*.yaml` | `templates: dict[str, str]`, `system_message_template` |
| `ClassificationConfig` | `configs/classification.yaml` | `refusal_keywords`, `chinese_patterns`, `english_patterns`, `trailing_digit_pattern`, `general_digit_pattern`, `direction_reversal` |
| `DirectionReversalPair` | (embedded in ClassificationConfig) | `source`, `target` |

**Loading functions:**

| Function | Input | Output |
|----------|-------|--------|
| `load_yaml(path)` | YAML file path | `dict` |
| `load_model_config(path)` | models.yaml path | `ModelConfig` |
| `load_experiment_config(path)` | experiment YAML path | `ExperimentConfig` |
| `load_prompt_templates(path)` | template YAML path | `PromptTemplateConfig` |
| `load_classification_config(path)` | classification YAML path | `ClassificationConfig` |
| `discover_experiments(config_dir)` | experiments directory | `list[ExperimentConfig]` |

`ClassificationConfig` provides two helper methods:
- `to_extraction_patterns() -> dict` — converts to the format expected by `ExperimentRunner`
- `to_reversal_pairs() -> list[dict]` — converts to the format expected by `build_reverse_prompt()`

### 3.4 `prompt/` — Prompt Builder / 提示词构建

**Purpose:** Renders prompt templates by substituting placeholders, and constructs system messages for structured output.

**Key functions:**

- **`build_prompt(template_text, num_stalls, **kwargs) -> str`** — Substitutes `{num_stalls}` and any additional placeholders in the template string

- **`build_system_message(num_stalls, template=None) -> str`** — Returns a system message for structured JSON output. Uses the provided template (typically loaded from `PromptTemplateConfig.system_message_template`) or falls back to a built-in default. The template must contain a `{num_stalls}` placeholder.

- **`build_reverse_prompt(template_text, num_stalls, reversal_pairs=None, **kwargs) -> str`** — Creates a direction-reversed prompt by applying a list of `{"source": "...", "target": "..."}` replacements. Defaults to the built-in Chinese/English direction pairs. Used in symmetry experiment 1.2.

### 3.5 `client/` — LLM Client / LLM 客户端

**Purpose:** Communicates with the OpenAI-compatible API, with automatic capability detection and graceful fallback.

**`LLMClient` class:**

```
LLMClient(endpoint, model, api_key="", timeout=60, max_retries=2, probe_message="Say OK")
```

**Initialization:** Lazy — the underlying `OpenAI` client is created on first use, not at construction time. All parameters default to sensible values but are typically loaded from `ModelConfig`.

**API probing — `probe_api() -> str`:**

Tries three modes in order to determine what the API supports:

| Priority | Mode | Mechanism | When to use |
|----------|------|-----------|-------------|
| 1 | `TOOLS` | `instructor.Mode.TOOLS` (function calling) | OpenAI, most capable APIs |
| 2 | `JSON_SCHEMA` | `instructor.Mode.JSON_SCHEMA` | APIs with JSON mode but no tools |
| 3 | `PLAIN_TEXT` | Raw `OpenAI` client, no instructor | Minimal API support |

**Query methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `query_structured(prompt, system_message, temperature, num_stalls)` | `(StallChoice | None, str, int, int)` | Uses instructor if TOOLS/JSON_SCHEMA mode; falls back to `query_plain()` |
| `query_plain(prompt, system_message, temperature)` | `(str, int, int)` | Direct OpenAI call, returns (raw_response, tokens, latency_ms) |

### 3.6 `recorder/` — JSONL Recorder / JSONL 记录器

**Purpose:** Persist experiment results as append-only JSONL (one JSON object per line).

**`JSONLRecorder` class:**

```
JSONLRecorder(output_path: Path)
```

| Method | Description |
|--------|-------------|
| `record(record: ExperimentRecord)` | Append one record as a JSON line |
| `record_batch(records: list[ExperimentRecord])` | Append multiple records |
| `read_all() -> list[ExperimentRecord]` | Read and parse all records |
| `count() -> int` | Count lines without parsing |
| `clear()` | Delete the output file |

The output directory is created automatically on initialization.

### 3.7 `runner/` — Experiment Runner / 实验运行器

**Purpose:** Orchestrates the full pipeline — parameter combination generation, prompt building, API calls, response classification, and recording.

**`ExperimentRunner` class:**

```
ExperimentRunner(client, recorder, model_config, refusal_keywords=None, extraction_patterns=None)
```

**Parameters:**
- `refusal_keywords` — list of refusal keyword strings (loaded from `ClassificationConfig.refusal_keywords`). Defaults to built-in Chinese/English keywords.
- `extraction_patterns` — dict of regex pattern lists for text extraction (loaded via `ClassificationConfig.to_extraction_patterns()`). Defaults to built-in patterns.

**Public methods:**

- **`run_single(prompt, system_message, temperature, num_stalls, metadata) -> ExperimentRecord`** — Executes one API call, classifies the response, records the result

- **`run_experiment(experiment_config, templates) -> list[ExperimentRecord]`** — Runs a full experiment:
  1. Generates all combinations: `num_stalls x temperatures x templates x repetitions`
  2. **Shuffles** the list randomly to avoid order effects
  3. Executes each combination via `run_single()`
  4. Returns all records

**Response classification — `_classify_response()`:**

| Status | Condition |
|--------|-----------|
| `VALID` | `extracted_choice` is not None and in range `[1, num_stalls]` |
| `REFUSED` | Raw response contains refusal patterns (e.g. "无法", "refuse", "cannot") |
| `AMBIGUOUS` | No valid choice extracted and no refusal detected |

**Text extraction — `_extract_choice_from_text()`:**

When structured output fails (PLAIN_TEXT mode), extracts the stall number using a cascade of regex patterns:

1. Chinese patterns: `第(\d+)个`, `(\d+)号`, `选择.*?(\d+)`
2. English patterns: `stall (\d+)`, `number (\d+)`
3. Bare digit at end of response
4. Last digit found in the response that falls within valid range

---

## 4. Configuration Guide / 配置指南

### 4.1 Model Configuration / 模型配置

**File:** `configs/models.yaml`

```yaml
models:
  - name: glm-5.1
    endpoint: http://localhost:3000/v1    # Base URL — SDK appends /chat/completions
    api_key: ""                            # Optional, defaults to "unused"
    version: "latest"                      # Optional, defaults to "unknown"
    timeout: 60                            # API call timeout in seconds
    max_retries: 2                         # Max retries for structured output
    probe_message: "Say OK"                # Message used to probe API capabilities
```

Note: `endpoint` is the base URL. The OpenAI SDK automatically appends `/chat/completions`.

### 4.1.1 Classification Configuration / 分类配置

**File:** `configs/classification.yaml`

Controls response classification and text extraction behavior:

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

All fields have sensible defaults — the file can be empty or missing.

### 4.2 Experiment Configuration / 实验配置

**Directory:** `configs/experiments/`

Each YAML file defines one experiment:

```yaml
experiment_id: "1.1"                    # Unique identifier
experiment_group: "1.1"                 # Grouping key
phase: "Phase1"                         # ExperimentPhase enum value
description: "Baseline choice"          # Human-readable description
num_stalls: [3, 5, 7, 10]              # List of stall counts to test
temperatures: [0.0, 0.7]               # Sampling temperatures
templates: ["A", "B", "C"]             # Prompt template keys to use
repetitions: 30                         # Repetitions per parameter combo
conditions: {}                          # Additional conditions metadata
occupied_stalls: []                     # Pre-occupied stalls (Phase 2)
```

**Total calls per experiment:** `len(num_stalls) * len(temperatures) * len(templates) * repetitions`

**Current experiments / 当前实验:**

| ID | Description | num_stalls | Total combos |
|----|-------------|------------|-------------|
| 1.1 | Baseline — all stalls empty | [3, 5, 7, 10] | 4 x 2 x 3 x 30 = 720 calls |
| 1.2 | Symmetry — direction reversal | [5] | 1 x 2 x 3 x 30 = 180 calls |
| 1.3 | Option count effect | [3, 5, 7, 10, 15, 20] | 6 x 2 x 3 x 30 = 1,080 calls |

### 4.3 Prompt Templates / 提示词模板

**File:** `configs/prompt_templates/phase1.yaml`

```yaml
templates:
  A: "一排{num_stalls}个坑位，全部空闲，干净程度完全相同，除位置外没有任何差异。你选第几个？只回答数字。"
  B: "你走进一间公共厕所，面前有一排{num_stalls}个独立的坑位，从左到右依次编号1到{num_stalls}。所有坑位都没有人使用..."
  C: "一个人走进公共厕所，看到一排{num_stalls}个空坑位，编号1到{num_stalls}。所有坑位的卫生状况和硬件条件完全一样..."
  D: "面前有一排{num_stalls}个坑位，全部空闲，干净程度相同，设施完全一样，除位置外没有其他差别。你可以选最左边的、最右边的..."
```

- `{num_stalls}` is the placeholder, replaced by `build_prompt()` at runtime
- All templates emphasize that conditions are identical except position
- Template A is the most minimal; B and C add framing context; D explicitly names options

---

## 5. Data Format / 数据格式

### JSONL Record Example / JSONL 记录示例

Each line in the output JSONL file is a complete `ExperimentRecord`:

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

## 6. Key Design Decisions / 关键设计决策

### 6.1 Three-Tier Structured Output Fallback / 三级结构化输出回退

**Decision:** `LLMClient.probe_api()` tries TOOLS, then JSON_SCHEMA, then falls back to PLAIN_TEXT. Refusal keywords, text extraction regex patterns, and direction-reversal strings are all configurable via `configs/classification.yaml`.

**Why:** Self-hosted or local model APIs (e.g. vLLM, Ollama, custom endpoints) often lack full OpenAI function calling support. Rather than failing, the client gracefully degrades. In PLAIN_TEXT mode, the runner uses regex-based text extraction (`_extract_choice_from_text()`) to parse the stall number from free-form responses. Externalizing these patterns to config allows researchers to adapt classification for new languages or model behaviors without code changes.

### 6.2 No litellm / 不使用 litellm

**Decision:** Use `instructor` directly with the `openai` SDK, not `litellm`.

**Why:** Stall Mate targets a single model endpoint per experiment run. `litellm` adds complexity for multi-provider routing that is unnecessary here. `instructor` provides exactly the structured output extraction we need with minimal dependencies.

### 6.3 No Async / 不使用异步

**Decision:** All API calls are sequential (synchronous).

**Why:** The full Phase 1 experiment requires ~1,980 calls at ~15-30s each (~8-16 hours). This is a batch job, not a latency-sensitive service. Sequential execution avoids concurrency bugs, simplifies debugging, and ensures deterministic recording order. Async can be added later if needed.

### 6.4 Config-Driven / 配置驱动

**Decision:** All experiment parameters, model settings, prompt templates, and classification patterns live in YAML files. No hardcoded values in Python.

**Why:** Separates experiment design from implementation. Researchers can define new experiments, tune classification thresholds, or change system prompts by editing YAML files without touching code. Reduces risk of accidental code changes affecting experiment conditions.

### 6.5 JSONL Output / JSONL 输出

**Decision:** One `ExperimentRecord` per line, written via stdlib `json`.

**Why:** JSONL is append-only (safe for interrupted runs), line-delimited (easy to process with standard tools: `jq`, `wc -l`, `grep`), and requires no additional libraries. Each line is self-contained and independently parseable.

### 6.6 Random Shuffle / 随机打乱

**Decision:** `run_experiment()` generates all parameter combinations and shuffles them before execution.

**Why:** Sequential execution of identical parameters could interact with API-side caching or rate limiting. Shuffling distributes parameter conditions across the time axis, reducing temporal confounds.

### 6.7 API Endpoint as Base URL / API 端点为基址

**Decision:** `configs/models.yaml` stores the base URL (e.g. `http://localhost:3000/v1`), not the full chat completions path.

**Why:** The OpenAI Python SDK automatically appends `/chat/completions` to the base URL. Storing only the base URL avoids double-path bugs (`/v1/chat/completions/chat/completions`).

---

## 7. Extension Guide / 扩展指南

### 7.1 Add a New Prompt Template / 添加新提示词模板

1. Add the template to `configs/prompt_templates/phase1.yaml`:
   ```yaml
   E: "你的新模板，包含{num_stalls}个坑位..."
   ```
2. Add `"E"` to the `PromptTemplate` enum in `src/stall_mate/types/record.py`
3. Reference template `"E"` in experiment YAML `templates` field

### 7.2 Add a New Experiment / 添加新实验

Create a new YAML file in `configs/experiments/`:
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

Use `discover_experiments()` to auto-load all experiment configs from the directory.

### 7.3 Add a New Model / 添加新模型

Add to `configs/models.yaml`:
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

Currently `load_model_config()` loads the first entry. For multi-model support, modify the loader to accept a model name parameter.

### 7.4 Customize Classification / 自定义分类

Edit `configs/classification.yaml` to change:
- **Refusal keywords** — add/remove keywords for new languages or model behaviors
- **Text extraction patterns** — add regex patterns for new response formats
- **Direction reversal pairs** — add new language pairs for symmetry tests

All fields have defaults; the file can be partially filled.

### 7.5 Extend to Phase 2 / 扩展到第二阶段

Phase 2 introduces occupied stalls (some stalls are taken, the model must choose among remaining ones):

1. Add `PHASE2 = "Phase2"` to the `ExperimentPhase` enum (already exists)
2. Create experiment configs with non-empty `occupied_stalls`:
   ```yaml
   occupied_stalls: [1, 3, 5]  # These stalls are already occupied
   ```
3. Update `build_prompt()` to incorporate occupied stall information into the prompt text
4. The `ExperimentRunner` and `ExperimentRecord` already support `occupied_stalls` — no changes needed

### 7.6 Add Async Support / 添加异步支持

Future optimization for large-scale experiments:

1. Convert `LLMClient.query_structured()` and `query_plain()` to `async`
2. Use `asyncio.Semaphore` to limit concurrent requests
3. Replace `random.shuffle()` with `asyncio.Queue` for concurrent consumption
4. `JSONLRecorder.record()` needs a lock for concurrent writes
5. Keep the sequential API as default; async as opt-in

---

## 8. Verification / 验证

### Functional Verification Script / 功能验证脚本

**File:** `scripts/verify_10calls.py`

Runs 10 API calls across diverse parameter combinations to verify the full pipeline:

| Group | Parameters | Purpose |
|-------|-----------|---------|
| 1 (3 calls) | N=5, T=0.0, templates A/B/C | Template variation at deterministic temperature |
| 2 (3 calls) | N=3/7/10, T=0.7, template A | Stall count variation |
| 3 (2 calls) | N=5, T=0.0, template D | Consistency check — same params, two calls |
| 4 (2 calls) | N=15/20, T=0.7, templates B/C | Edge cases — large stall counts |

**Verification steps:**
1. Loads config from `configs/`
2. Initializes `LLMClient`, `JSONLRecorder`, `ExperimentRunner`
3. Executes 10 calls, prints progress and results
4. Reads back the JSONL file and verifies: record count matches, all required fields present
5. Exit code 0 = pass, 1 = fail

**Run:**
```bash
uv run python scripts/verify_10calls.py
```

Note: Each call takes ~15-30 seconds. Total runtime ~3-5 minutes.

### Unit Tests / 单元测试

**Run:**
```bash
uv run pytest
```

74 tests covering all modules. All tests use mocks — no real API calls are made during testing. Test files are in `tests/` and mirror the source structure.
