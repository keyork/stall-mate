# AGENTS.md — Stall Mate

## Project

LLM decision-consistency stress test. Prompts frontier models with a trivial scenario (choosing a toilet stall) and measures whether their "reasoning" is stable or hallucinated. Research/artifact repo, not a library or service.

## Status

Phase 1 & 2 framework implemented. Core pipeline + conditional experiments + analysis module all working.

## Commands

```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests (89 tests)
uv run python scripts/verify_10calls.py  # 10-call functional verification (requires API)
```

## Architecture

- `src/stall_mate/` — core library (types, schema, config, prompt, client, runner, recorder, analysis)
- `configs/` — YAML configs: `models.yaml` (API endpoint), `experiments/` (Phase 1, 2), `prompt_templates/`, `classification.yaml`
- `tests/` — pytest unit tests (all mocked, no real API calls)
- `data/` — JSONL output (gitignored except `.gitkeep`)
- `scripts/` — verification scripts

## Key patterns

- **Structured output**: `instructor` + `StallChoice` Pydantic model, 3-tier fallback (TOOLS → JSON_SCHEMA → PLAIN_TEXT)
- **Config-driven**: YAML defines experiments; no hardcoded prompts or params in Python
- **JSONL recording**: One `ExperimentRecord` per line, all fields from `docs/idea.md` §5.2
- **API endpoint**: OpenAI-compatible at `http://localhost:3000/v1` (configured in `configs/models.yaml`)

## License

Apache 2.0 — `# SPDX-License-Identifier: Apache-2.0` header on all `.py` files.

## Language

Bilingual: Chinese + English. Keep both in user-facing text (README, docstrings, prompts).

## Constraints

- No `litellm` dependency (single model, `instructor` sufficient)
- No async runtime (parallel via ThreadPoolExecutor; can add async later for scale)
- Phase 2 conditional experiments implemented (26 condition scenarios)
- API calls are slow (~15-30s each); set timeouts accordingly
