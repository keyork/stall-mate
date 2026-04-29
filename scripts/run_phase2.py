#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 条件叠加实验执行脚本 | Phase 2 conditional experiment runner.

使用 build_phase2_prompt 逐实验配置构建条件提示词，
通过 ExperimentRunner.run_single 执行每次调用。
Builds per-condition prompts via build_phase2_prompt and runs each call
through ExperimentRunner.run_single.
"""

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stall_mate.client import LLMClient
from stall_mate.config import (
    ExperimentConfig,
    discover_experiments,
    load_classification_config,
    load_model_config,
    load_prompt_templates,
)
from stall_mate.prompt import build_phase2_prompt, build_system_message
from stall_mate.prompt.phase2_builder import build_conditions_text
from stall_mate.recorder import JSONLRecorder
from stall_mate.runner import ExperimentDisplay, ExperimentRunner, RunStats
from stall_mate.runner.experiment import _Task
from stall_mate.types import ChoiceStatus


def _build_phase2_tasks(
    exp_config: ExperimentConfig,
    templates_config: Any,
) -> list[_Task]:
    combos: list[tuple[int, float, str, int]] = []
    for ns in exp_config.num_stalls:
        for temp in exp_config.temperatures:
            for tpl_key in exp_config.templates:
                for rep_idx in range(exp_config.repetitions):
                    combos.append((ns, temp, tpl_key, rep_idx))

    random.shuffle(combos)

    tasks: list[_Task] = []
    for num_stalls, temperature, template_key, _rep_idx in combos:
        prompt = build_phase2_prompt(
            templates_config.templates[template_key],
            num_stalls,
            exp_config.conditions,
        )
        system_message = build_system_message(
            num_stalls, template=templates_config.system_message_template,
        )
        metadata: dict[str, Any] = {
            "experiment_phase": exp_config.phase,
            "experiment_group": exp_config.experiment_group,
            "model_name": "",
            "temperature": temperature,
            "prompt_template": template_key,
            "prompt_text": prompt,
            "num_stalls": num_stalls,
            "conditions": exp_config.conditions,
            "occupied_stalls": exp_config.occupied_stalls,
        }
        tasks.append(_Task(prompt, system_message, temperature, num_stalls, metadata))

    return tasks


def _run_experiment(
    runner: ExperimentRunner,
    exp_config: ExperimentConfig,
    templates_config: Any,
    display: ExperimentDisplay,
    max_retries: int = 3,
) -> RunStats:
    tasks = _build_phase2_tasks(exp_config, templates_config)
    stats = RunStats(start_time=time.time())

    display.print_experiment_header(
        exp_id=exp_config.experiment_id,
        description=exp_config.description,
        total_calls=len(tasks),
        output_path=runner.recorder.output_path,
    )

    failed = runner._run_task_batch(
        tasks, stats, total=len(tasks), label="🧠 模型正在选择坑位...",
    )

    for retry_round in range(1, max_retries + 1):
        if not failed:
            break
        stats.retries_used += len(failed)
        display.print_retry_round(retry_round, max_retries, len(failed))
        failed = runner._run_task_batch(
            failed, stats, total=len(failed),
            label=f"🔄 重试 {retry_round}/{max_retries}",
        )

    stats.end_time = time.time()

    if failed:
        display.print_retry_exhausted(len(failed), max_retries)

    display.print_experiment_summary(stats)
    return stats


def main() -> None:
    model_cfg = load_model_config(ROOT / "configs" / "models.yaml")
    phase2_templates = load_prompt_templates(
        ROOT / "configs" / "prompt_templates" / "phase2.yaml"
    )
    classification_cfg = load_classification_config(
        ROOT / "configs" / "classification.yaml"
    )

    experiment_configs = discover_experiments(
        ROOT / "configs" / "experiments" / "phase2"
    )

    client = LLMClient(
        endpoint=model_cfg.endpoint,
        model=model_cfg.name,
        api_key=model_cfg.api_key,
        timeout=model_cfg.timeout,
        max_retries=model_cfg.max_retries,
        probe_message=model_cfg.probe_message,
    )

    output_path = ROOT / "data" / "phase2.jsonl"
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

    total_calls = sum(
        len(ec.num_stalls)
        * len(ec.temperatures)
        * len(ec.templates)
        * ec.repetitions
        for ec in experiment_configs
    )

    print(f"\n{'='*80}")
    print("Phase 2: 条件叠加实验 | Conditional Experiment")
    print(f"{'='*80}")
    print(f"实验数量: {len(experiment_configs)}")
    print(f"总调用: {total_calls} 次")
    print(f"输出: {output_path}")
    print(f"{'='*80}\n")

    total_stats = RunStats(start_time=time.time())

    for i, exp_config in enumerate(experiment_configs, 1):
        print(f"\n--- [{i}/{len(experiment_configs)}] {exp_config.experiment_id}: {exp_config.description} ---")
        exp_stats = _run_experiment(runner, exp_config, phase2_templates, display)
        total_stats.merge(exp_stats)

    total_stats.end_time = time.time()

    print(f"\n{'='*80}")
    print("Phase 2 完成 | Phase 2 Complete")
    print(f"{'='*80}")
    print(total_stats.summary())


if __name__ == "__main__":
    main()
