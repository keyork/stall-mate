# AGENTS.md — Stall Mate

## Project

LLM decision-consistency stress test. Prompts frontier models with a trivial scenario (choosing a toilet stall) and measures whether their "reasoning" is stable or hallucinated. Research/artifact repo, not a library or service.

## Status

Phase 1 framework implemented. Core pipeline works: config → prompt → API call → structured output → JSONL record.

## Commands

```bash
pip install -e .          # Install package
pytest                    # Run tests (74 tests)
python scripts/verify_10calls.py  # 10-call functional verification (requires API)
```

## Architecture

- `src/stall_mate/` — core library (types, schema, config, prompt, client, runner, recorder)
- `configs/` — YAML configs: `models.yaml` (API endpoint), `experiments/` (Phase 1), `prompt_templates/`
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

- No `litellm` dependency (single model, `instructor` sufficient for Phase 1)
- No async runtime (sequential calls; can add later for scale)
- No Phase 2 code yet (only Phase 1 experiment configs exist)
- API calls are slow (~15-30s each); set timeouts accordingly
