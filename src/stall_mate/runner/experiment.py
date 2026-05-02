# SPDX-License-Identifier: Apache-2.0
"""实验运行器 | Experiment runner — orchestrates the full pipeline."""

from __future__ import annotations

import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stall_mate.client import LLMClient
from stall_mate.config import ClassificationConfig, ExperimentConfig, ModelConfig, PromptTemplateConfig
from stall_mate.prompt import build_prompt, build_system_message
from stall_mate.recorder import JSONLRecorder
from stall_mate.runner.display import ExperimentDisplay
from stall_mate.types import (
    ChoiceStatus,
    ExperimentPhase,
    ExperimentRecord,
    PromptTemplate,
)


@dataclass(frozen=True)
class _Task:
    """单次调用任务包 | Single call task bundle."""
    prompt: str
    system_message: str
    temperature: float
    num_stalls: int
    metadata: dict[str, Any]


@dataclass
class RunStats:
    """实验运行统计 | Experiment run statistics.

    Note: When using parallel execution, mutations (increment, append) must be
    performed under ``ExperimentRunner._lock`` to ensure thread-safety.
    """

    total_calls: int = 0
    valid: int = 0
    refused: int = 0
    ambiguous: int = 0
    error: int = 0
    retries_used: int = 0
    total_latency_ms: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    def merge(self, other: RunStats) -> None:
        self.total_calls += other.total_calls
        self.valid += other.valid
        self.refused += other.refused
        self.ambiguous += other.ambiguous
        self.error += other.error
        self.retries_used += other.retries_used
        self.total_latency_ms += other.total_latency_ms

    def summary(self) -> str:
        h = int(self.elapsed_seconds // 3600)
        m = int((self.elapsed_seconds % 3600) // 60)
        s = int(self.elapsed_seconds % 60)
        avg = (
            f"{self.total_latency_ms / self.total_calls:.0f}ms"
            if self.total_calls > 0
            else "N/A"
        )
        return (
            f"  总调用: {self.total_calls} | "
            f"VALID: {self.valid}, REFUSED: {self.refused}, "
            f"AMBIGUOUS: {self.ambiguous}, ERROR: {self.error}\n"
            f"  重试次数: {self.retries_used} | 平均延迟: {avg} | 耗时: {h}h {m}m {s}s"
        )


class ExperimentRunner:
    """实验运行器 | Experiment runner — orchestrates the full pipeline."""

    def __init__(
        self,
        client: LLMClient,
        recorder: JSONLRecorder,
        model_config: ModelConfig,
        refusal_keywords: list[str] | None = None,
        extraction_patterns: dict[str, list[str] | str] | None = None,
        display: ExperimentDisplay | None = None,
        parallel_num: int = 4,
    ):
        """初始化实验运行器 | Initialize the experiment runner."""
        self.client = client
        self.recorder = recorder
        self.model_config = model_config
        self._display = display or ExperimentDisplay()
        self._parallel_num = parallel_num
        self._lock = threading.Lock()

        classification = ClassificationConfig()
        keywords = refusal_keywords or classification.refusal_keywords
        self._refusal_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in keywords
        ]

        patterns = extraction_patterns or classification.to_extraction_patterns()
        _cp = patterns.get("chinese_patterns", [])
        self._chinese_patterns: list[str] = _cp if isinstance(_cp, list) else [_cp]
        _ep = patterns.get("english_patterns", [])
        self._english_patterns: list[str] = _ep if isinstance(_ep, list) else [_ep]
        _tp = patterns.get("trailing_digit_pattern", r"(\d+)\s*[。.!?]?\s*$")
        self._trailing_digit_pattern: str = _tp if isinstance(_tp, str) else str(_tp)
        _gp = patterns.get("general_digit_pattern", r"\b(\d+)\b")
        self._general_digit_pattern: str = _gp if isinstance(_gp, str) else str(_gp)

    def _build_record_base(
        self,
        prompt: str,
        temperature: float,
        num_stalls: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "record_id": uuid.uuid4().hex[:12],
            "experiment_phase": ExperimentPhase(metadata.get("experiment_phase", "Phase1")),
            "experiment_group": metadata.get("experiment_group", ""),
            "model_name": self.model_config.name,
            "model_version": self.model_config.version,
            "temperature": temperature,
            "prompt_template": PromptTemplate(metadata.get("prompt_template", "A")),
            "prompt_text": metadata.get("prompt_text", prompt),
            "num_stalls": num_stalls,
            "occupied_stalls": metadata.get("occupied_stalls", []),
            "conditions": metadata.get("conditions", {}),
            "timestamp": datetime.now(timezone.utc),
        }

    def run_single(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        num_stalls: int,
        metadata: dict[str, Any],
    ) -> ExperimentRecord:
        """执行单次模型调用并记录结果 | Execute a single model call and record the result."""
        try:
            return self._run_single_inner(
                prompt, system_message, temperature, num_stalls, metadata
            )
        except Exception as e:
            return self._make_error_record(
                prompt, temperature, num_stalls, metadata, f"{type(e).__name__}: {e}"
            )

    def _run_single_inner(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        num_stalls: int,
        metadata: dict[str, Any],
    ) -> ExperimentRecord:
        parsed, raw_response, response_tokens, latency_ms = (
            self.client.query_structured(prompt, system_message, temperature, num_stalls)
        )

        is_api_error = parsed is None and self._is_error_response(raw_response)

        if is_api_error:
            return self._make_error_record(
                prompt, temperature, num_stalls, metadata, raw_response, latency_ms
            )

        if parsed is not None:
            extracted_choice = parsed.chosen_stall
            extracted_reasoning = parsed.chain_of_thought
            reasoning_present = bool(parsed.chain_of_thought)
            choice_status = self._classify_response(
                raw_response, extracted_choice, num_stalls
            )
        else:
            extracted_choice = self._extract_choice_from_text(raw_response, num_stalls)
            extracted_reasoning = ""
            reasoning_present = False
            if extracted_choice is not None:
                choice_status = ChoiceStatus.VALID
            else:
                choice_status = self._classify_response(
                    raw_response, None, num_stalls
                )

        record = ExperimentRecord(
            **self._build_record_base(prompt, temperature, num_stalls, metadata),
            raw_response=raw_response,
            extracted_choice=extracted_choice,
            choice_status=choice_status,
            reasoning_present=reasoning_present,
            extracted_reasoning=extracted_reasoning,
            response_tokens=response_tokens,
            latency_ms=latency_ms,
        )

        self.recorder.record(record)
        return record

    def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        templates: PromptTemplateConfig,
        max_retries: int = 3,
    ) -> RunStats:
        """运行完整实验（含重试） | Run full experiment with retry logic."""
        combos = self._build_combos(experiment_config)
        random.shuffle(combos)

        tasks: list[_Task] = []
        for num_stalls, temperature, template_key, _rep_idx in combos:
            prompt = build_prompt(templates.templates[template_key], num_stalls)
            system_message = build_system_message(
                num_stalls, template=templates.system_message_template,
            )
            metadata: dict[str, Any] = {
                "experiment_phase": experiment_config.phase,
                "experiment_group": experiment_config.experiment_group,
                "model_name": self.model_config.name,
                "temperature": temperature,
                "prompt_template": template_key,
                "prompt_text": prompt,
                "num_stalls": num_stalls,
                "conditions": experiment_config.conditions,
                "occupied_stalls": experiment_config.occupied_stalls,
            }
            tasks.append(_Task(prompt, system_message, temperature, num_stalls, metadata))

        stats = RunStats(start_time=time.time())

        self._display.print_experiment_header(
            exp_id=experiment_config.experiment_id,
            description=experiment_config.description,
            total_calls=len(tasks),
            output_path=self.recorder.output_path,
        )

        failed = self._run_task_batch(tasks, stats, total=len(tasks), label="🧠 模型正在选择坑位...")

        for retry_round in range(1, max_retries + 1):
            if not failed:
                break
            stats.retries_used += len(failed)
            self._display.print_retry_round(retry_round, max_retries, len(failed))
            failed = self._run_task_batch(
                failed, stats, total=len(failed),
                label=f"🔄 重试 {retry_round}/{max_retries}",
            )

        stats.end_time = time.time()

        if failed:
            self._display.print_retry_exhausted(len(failed), max_retries)

        self._display.print_experiment_summary(stats)
        return stats

    def _run_task_batch(
        self,
        tasks: list[_Task],
        stats: RunStats,
        total: int = 0,
        label: str = "",
    ) -> list[_Task]:
        failed: list[_Task] = []
        if total == 0:
            total = len(tasks)

        progress = self._display.create_progress(total, label)
        with progress:
            task_id = progress.add_task(label, total=total)
            with ThreadPoolExecutor(max_workers=self._parallel_num) as executor:
                future_to_task = {}
                for task in tasks:
                    future = executor.submit(
                        self.run_single,
                        task.prompt, task.system_message,
                        task.temperature, task.num_stalls, task.metadata,
                    )
                    future_to_task[future] = task

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        record = future.result()
                    except Exception as e:
                        record = self._make_error_record(
                            task.prompt, task.temperature, task.num_stalls,
                            task.metadata, f"ThreadError: {e}",
                        )

                    with self._lock:
                        stats.total_calls += 1
                        stats.total_latency_ms += record.latency_ms
                        if record.choice_status == ChoiceStatus.VALID:
                            stats.valid += 1
                        elif record.choice_status == ChoiceStatus.REFUSED:
                            stats.refused += 1
                        elif record.choice_status == ChoiceStatus.AMBIGUOUS:
                            stats.ambiguous += 1
                        elif record.choice_status == ChoiceStatus.ERROR:
                            stats.error += 1
                            failed.append(task)

                    status_text = self._display.format_record_status(record)
                    choice_str = f"#{record.extracted_choice}" if record.extracted_choice is not None else "-"
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"[{record.choice_status.value}] {choice_str}",
                    )

        return failed

    @staticmethod
    def _is_error_response(raw_response: str) -> bool:
        error_prefixes = (
            "ConnectionError:",
            "Timeout:",
            "APIConnectionError:",
            "APIStatusError:",
            "RateLimitError:",
            "AuthenticationError:",
            "NotFoundError:",
            "InternalServerError:",
            "ServiceUnavailableError:",
        )
        return raw_response.startswith(error_prefixes)

    def _make_error_record(
        self,
        prompt: str,
        temperature: float,
        num_stalls: int,
        metadata: dict[str, Any],
        error_msg: str,
        latency_ms: int = 0,
    ) -> ExperimentRecord:
        record = ExperimentRecord(
            **self._build_record_base(prompt, temperature, num_stalls, metadata),
            raw_response=error_msg,
            extracted_choice=None,
            choice_status=ChoiceStatus.ERROR,
            reasoning_present=False,
            extracted_reasoning="",
            response_tokens=0,
            latency_ms=latency_ms,
        )
        self.recorder.record(record)
        return record

    @staticmethod
    def _build_combos(
        experiment_config: ExperimentConfig,
    ) -> list[tuple[int, float, str, int]]:
        combos: list[tuple[int, float, str, int]] = []
        for ns in experiment_config.num_stalls:
            for temp in experiment_config.temperatures:
                for tpl_key in experiment_config.templates:
                    for rep_idx in range(experiment_config.repetitions):
                        combos.append((ns, temp, tpl_key, rep_idx))
        return combos

    def _classify_response(
        self,
        raw_response: str,
        extracted_choice: int | None,
        num_stalls: int,
    ) -> ChoiceStatus:
        if extracted_choice is not None and 1 <= extracted_choice <= num_stalls:
            return ChoiceStatus.VALID

        for pat in self._refusal_patterns:
            if pat.search(raw_response):
                return ChoiceStatus.REFUSED

        return ChoiceStatus.AMBIGUOUS

    def _extract_choice_from_text(self, raw_response: str, num_stalls: int) -> int | None:
        for pat in self._chinese_patterns:
            m = re.search(pat, raw_response)
            if m:
                val = int(m.group(1))
                if 1 <= val <= num_stalls:
                    return val

        for pat in self._english_patterns:
            m = re.search(pat, raw_response, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 1 <= val <= num_stalls:
                    return val

        m = re.search(self._trailing_digit_pattern, raw_response)
        if m:
            val = int(m.group(1))
            if 1 <= val <= num_stalls:
                return val

        all_digits = re.findall(self._general_digit_pattern, raw_response)
        for d in reversed(all_digits):
            val = int(d)
            if 1 <= val <= num_stalls:
                return val

        return None
