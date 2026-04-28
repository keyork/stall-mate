#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
10 次调用功能验证 / 10-call functional verification.

验证完整实验管线：配置加载 → 提示词构建 → API 调用 → 响应解析 → JSONL 记录。
Verifies the full pipeline: config → prompt → API call → parse → JSONL record.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stall_mate.client import LLMClient
from stall_mate.config import (
    load_model_config,
    load_prompt_templates,
    load_classification_config,
)
from stall_mate.prompt import build_prompt, build_system_message
from stall_mate.recorder import JSONLRecorder
from stall_mate.runner import ExperimentRunner


def main():
    # 1. Load configs
    model_cfg = load_model_config(ROOT / "configs" / "models.yaml")
    templates = load_prompt_templates(ROOT / "configs" / "prompt_templates" / "phase1.yaml")
    classification_cfg = load_classification_config(ROOT / "configs" / "classification.yaml")

    # 2. Initialize components
    client = LLMClient(
        endpoint=model_cfg.endpoint,
        model=model_cfg.name,
        api_key=model_cfg.api_key,
        timeout=model_cfg.timeout,
        max_retries=model_cfg.max_retries,
        probe_message=model_cfg.probe_message,
    )
    output_path = ROOT / "data" / "verify_10calls.jsonl"
    recorder = JSONLRecorder(output_path)
    # Clear previous results
    recorder.clear()
    runner = ExperimentRunner(
        client=client,
        recorder=recorder,
        model_config=model_cfg,
        refusal_keywords=classification_cfg.refusal_keywords,
        extraction_patterns=classification_cfg.to_extraction_patterns(),
    )

    # 3. Define 10 verification calls
    calls = [
        # Group 1: Temperature=0, deterministic (3 calls, template A,B,C)
        {"num_stalls": 5, "temp": 0.0, "template": "A", "label": "T0-tmplA-N5"},
        {"num_stalls": 5, "temp": 0.0, "template": "B", "label": "T0-tmplB-N5"},
        {"num_stalls": 5, "temp": 0.0, "template": "C", "label": "T0-tmplC-N5"},
        # Group 2: Variable stall counts at temp=0.7 (3 calls)
        {"num_stalls": 3, "temp": 0.7, "template": "A", "label": "T0.7-tmplA-N3"},
        {"num_stalls": 7, "temp": 0.7, "template": "A", "label": "T0.7-tmplA-N7"},
        {"num_stalls": 10, "temp": 0.7, "template": "A", "label": "T0.7-tmplA-N10"},
        # Group 3: Template D + consistency check (2 calls, same params)
        {"num_stalls": 5, "temp": 0.0, "template": "D", "label": "consistency-1"},
        {"num_stalls": 5, "temp": 0.0, "template": "D", "label": "consistency-2"},
        # Group 4: Edge cases - large stall count (2 calls)
        {"num_stalls": 15, "temp": 0.7, "template": "B", "label": "T0.7-tmplB-N15"},
        {"num_stalls": 20, "temp": 0.7, "template": "C", "label": "T0.7-tmplC-N20"},
    ]

    # 4. Execute each call
    print("=" * 70)
    print("Stall Mate — 10-call Functional Verification / 10 次调用功能验证")
    print("=" * 70)

    results = []
    for i, call in enumerate(calls):
        label = call["label"]
        print(
            f"\n[{i+1}/10] {label} "
            f"(N={call['num_stalls']}, T={call['temp']}, tmpl={call['template']})..."
        )

        prompt = build_prompt(templates.templates[call["template"]], call["num_stalls"])
        system_msg = build_system_message(
            call["num_stalls"], template=templates.system_message_template,
        )

        metadata = {
            "experiment_phase": "Phase1",
            "experiment_group": "verify",
            "prompt_template": call["template"],
            "prompt_text": prompt,
        }

        record = runner.run_single(
            prompt=prompt,
            system_message=system_msg,
            temperature=call["temp"],
            num_stalls=call["num_stalls"],
            metadata=metadata,
        )
        results.append(record)
        print(
            f"  → choice={record.extracted_choice}, "
            f"status={record.choice_status.value}, "
            f"latency={record.latency_ms}ms, "
            f"reasoning={'yes' if record.reasoning_present else 'no'}"
        )

    # 5. Print summary report
    print_summary(results, calls)

    # 6. Verify output
    passed = verify_jsonl(output_path, expected_count=10)

    if passed:
        print("\n" + "=" * 70)
        print("✅ PASS — All 10 calls verified successfully / 全部 10 次调用验证成功")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ FAIL — Verification failed / 验证失败")
        print("=" * 70)
        sys.exit(1)


def print_summary(results, calls):
    """Print a formatted summary table of all 10 results."""
    print("\n" + "-" * 70)
    print("Summary / 汇总")
    print("-" * 70)
    print(
        f"{'#':>2} | {'Label':<20} | {'Choice':>6} | "
        f"{'Status':<10} | {'Tokens':>6} | {'Latency':>8}"
    )
    print("-" * 70)
    for i, (r, c) in enumerate(zip(results, calls)):
        choice = str(r.extracted_choice) if r.extracted_choice is not None else "N/A"
        print(
            f"{i+1:>2} | {c['label']:<20} | {choice:>6} | "
            f"{r.choice_status.value:<10} | {r.response_tokens:>6} | {r.latency_ms:>6}ms"
        )

    valid_count = sum(1 for r in results if r.choice_status.value == "VALID")
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

    print("-" * 70)
    print(f"Valid: {valid_count}/10 | Avg latency: {avg_latency:.0f}ms")


def verify_jsonl(path: Path, expected_count: int) -> bool:
    """Verify JSONL file has expected records with all required fields."""
    if not path.exists():
        print(f"\n❌ JSONL file not found: {path}")
        return False

    recorder = JSONLRecorder(path)
    records = recorder.read_all()

    if len(records) != expected_count:
        print(f"\n❌ Expected {expected_count} records, found {len(records)}")
        return False

    required_fields = [
        "record_id",
        "experiment_phase",
        "experiment_group",
        "model_name",
        "temperature",
        "prompt_template",
        "prompt_text",
        "num_stalls",
        "raw_response",
        "choice_status",
        "timestamp",
    ]

    for i, r in enumerate(records):
        for field in required_fields:
            val = getattr(r, field, None)
            if val is None or val == "":
                print(f"\n❌ Record {i} missing or empty field: {field}")
                return False

    print(f"\n✅ JSONL verification: {len(records)} records, all fields present")
    return True


if __name__ == "__main__":
    main()
