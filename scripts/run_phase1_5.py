#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
Phase 1.5 CoT 实验执行脚本 | Phase 1.5 CoT experiment runner.

使用 Chain-of-Thought 提示模板运行关键条件的决策一致性实验。
Runs decision consistency experiments on key conditions using CoT prompt templates.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stall_mate.client import LLMClient
from stall_mate.config import (
    load_model_config,
    load_prompt_templates,
    load_classification_config,
)
from stall_mate.recorder import JSONLRecorder
from stall_mate.runner import ExperimentDisplay, ExperimentRunner, RunStats


def main() -> None:
    model_cfg = load_model_config(ROOT / "configs" / "models.yaml")
    cot_templates = load_prompt_templates(ROOT / "configs" / "prompt_templates" / "phase1_cot.yaml")
    classification_cfg = load_classification_config(ROOT / "configs" / "classification.yaml")

    from stall_mate.config.loader import ExperimentConfig

    cot_config = ExperimentConfig(
        experiment_id="1.5",
        experiment_group="1.5",
        phase="Phase1",
        description="Chain-of-Thought 决策一致性 / CoT decision consistency",
        num_stalls=[5, 10],
        temperatures=[0.0],
        templates=["A", "B", "C"],
        repetitions=30,
    )

    client = LLMClient(
        endpoint=model_cfg.endpoint,
        model=model_cfg.name,
        api_key=model_cfg.api_key,
        timeout=model_cfg.timeout,
        max_retries=model_cfg.max_retries,
        probe_message=model_cfg.probe_message,
    )

    output_path = ROOT / "data" / "phase1_1.5.jsonl"
    recorder = JSONLRecorder(output_path)

    display = ExperimentDisplay()

    runner = ExperimentRunner(
        client=client,
        recorder=recorder,
        model_config=model_cfg,
        refusal_keywords=classification_cfg.refusal_keywords,
        extraction_patterns=classification_cfg.to_extraction_patterns(),
        display=display,
        parallel_num=4,
    )

    print(f"\n{'='*80}")
    print("Phase 1.5: CoT 决策一致性实验 | CoT Decision Consistency Experiment")
    print(f"{'='*80}")
    print(f"模板: CoT (phase1_cot.yaml)")
    print(f"条件: N={cot_config.num_stalls}, T={cot_config.temperatures}, 模板={cot_config.templates}")
    print(f"重复: {cot_config.repetitions} 次")
    calls = len(cot_config.num_stalls) * len(cot_config.temperatures) * len(cot_config.templates) * cot_config.repetitions
    print(f"总调用: {calls} 次")
    print(f"输出: {output_path}")
    print(f"{'='*80}\n")

    stats = runner.run_experiment(cot_config, cot_templates)
    stats.end_time = time.time()

    print(f"\n{'='*80}")
    print("Phase 1.5 完成 | Phase 1.5 Complete")
    print(f"{'='*80}")
    print(stats.summary())


if __name__ == "__main__":
    main()
