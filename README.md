# stall-mate

坑位博弈：当大模型走进公共厕所 | Do LLMs actually "think" when they make decisions, or just hallucinate a preference? We stress-test decision consistency of frontier LLMs through an absurdly simple scenario — choosing a toilet stall.

## Quick Start / 快速开始

```bash
uv sync
uv run python scripts/verify_10calls.py
```

## Project Structure / 项目结构

```
stall-mate/
├── configs/              # 实验配置 / Experiment configs
│   ├── models.yaml       # 模型定义 / Model definitions
│   ├── experiments/      # 实验参数 / Experiment parameters
│   └── prompt_templates/ # 提示词模板 / Prompt templates
├── src/stall_mate/       # 核心代码 / Core library
│   ├── types/            # 数据类型 / Data types & enums
│   ├── schema/           # LLM 响应模式 / LLM response schema
│   ├── config/           # 配置加载 / Config loading
│   ├── prompt/           # 提示词构建 / Prompt builder
│   ├── client/           # LLM 客户端 / LLM client
│   ├── runner/           # 实验运行器 / Experiment runner
│   └── recorder/         # JSONL 记录器 / JSONL recorder
├── tests/                # 单元测试 / Unit tests
├── scripts/              # 脚本 / Scripts
│   └── verify_10calls.py # 功能验证 / Functional verification
└── data/                 # 实验数据 / Experiment data
```

## Configuration / 配置

Model endpoint is defined in `configs/models.yaml`:

```yaml
models:
  - name: glm-5.1
    endpoint: http://localhost:3000/v1/chat/completions
```

Experiment definitions are in `configs/experiments/` — each YAML file defines stall counts, temperatures, templates, and repetition counts.

Prompt templates are in `configs/prompt_templates/` — four variants (A–D) with `{num_stalls}` placeholder.

## Testing / 测试

```bash
uv run pytest
```

## Data Format / 数据格式

Results are saved as JSONL (one JSON object per line). See `docs/idea.md` Section 5.2 for the full schema.

## License

Apache 2.0 — see [LICENSE](LICENSE).
