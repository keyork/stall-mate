# SPDX-License-Identifier: Apache-2.0
"""
实验运行器 / Experiment runner.

编排提示词构建、LLM 调用、响应分类和数据记录的完整流程。
Orchestrates prompt building, LLM calls, response classification, and data recording.
"""

from __future__ import annotations

import random
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from stall_mate.client import LLMClient
from stall_mate.config import ExperimentConfig, ModelConfig, PromptTemplateConfig
from stall_mate.prompt import build_prompt, build_system_message
from stall_mate.recorder import JSONLRecorder
from stall_mate.schema import StallChoice
from stall_mate.types import (
    ChoiceStatus,
    ExperimentPhase,
    ExperimentRecord,
    PromptTemplate,
)


class _Task:
    """内部任务包 / Internal task bundle for retry tracking."""

    __slots__ = ("prompt", "system_message", "temperature", "num_stalls", "metadata")

    def __init__(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        num_stalls: int,
        metadata: dict[str, Any],
    ):
        self.prompt = prompt
        self.system_message = system_message
        self.temperature = temperature
        self.num_stalls = num_stalls
        self.metadata = metadata

# 默认拒绝关键词 / Default refusal keywords
_DEFAULT_REFUSAL_KEYWORDS: list[str] = [
    "无法",
    "不能",
    "拒绝",
    "refuse",
    "cannot",
    "won't",
    "I can't",
    "inappropriate",
]

# 默认文本提取模式 / Default text extraction patterns
_DEFAULT_EXTRACTION_PATTERS: dict[str, list[str] | str] = {
    "chinese_patterns": [
        r"第\s*(\d+)\s*个",
        r"(\d+)\s*号",
        r"选择.*?(\d+)",
    ],
    "english_patterns": [
        r"stall\s*(\d+)",
        r"number\s*(\d+)",
    ],
    "trailing_digit_pattern": r"(\d+)\s*[。.!?]?\s*$",
    "general_digit_pattern": r"\b(\d+)\b",
}


class ExperimentRunner:
    """实验运行器 / Experiment runner — orchestrates the full pipeline."""

    def __init__(
        self,
        client: LLMClient,
        recorder: JSONLRecorder,
        model_config: ModelConfig,
        refusal_keywords: list[str] | None = None,
        extraction_patterns: dict[str, list[str] | str] | None = None,
    ):
        self.client = client
        self.recorder = recorder
        self.model_config = model_config

        keywords = refusal_keywords if refusal_keywords is not None else _DEFAULT_REFUSAL_KEYWORDS
        self._refusal_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in keywords
        ]

        patterns = (
            extraction_patterns
            if extraction_patterns is not None
            else _DEFAULT_EXTRACTION_PATTERS
        )
        _cp = patterns.get("chinese_patterns", [])
        self._chinese_patterns: list[str] = _cp if isinstance(_cp, list) else [_cp]
        _ep = patterns.get("english_patterns", [])
        self._english_patterns: list[str] = _ep if isinstance(_ep, list) else [_ep]
        _tp = patterns.get("trailing_digit_pattern", r"(\d+)\s*[。.!?]?\s*$")
        self._trailing_digit_pattern: str = _tp if isinstance(_tp, str) else str(_tp)
        _gp = patterns.get("general_digit_pattern", r"\b(\d+)\b")
        self._general_digit_pattern: str = _gp if isinstance(_gp, str) else str(_gp)

    # ------------------------------------------------------------------
    # Public API / 公开接口
    # ------------------------------------------------------------------

    def run_single(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        num_stalls: int,
        metadata: dict[str, Any],
    ) -> ExperimentRecord:
        """运行单次 API 调用并记录结果 / Run a single API call and record the result.

        任何异常（网络错误、API 错误等）均被捕获，标记为 ERROR 状态，
        不会向外抛出。调用方可通过 choice_status == ERROR 判断是否需要重试。

        1. 调用 client.query_structured()
        2. 若结构化输出失败，回退到纯文本 + 手动解析
        3. 分类 choice_status: VALID / REFUSED / AMBIGUOUS / ERROR
        4. 构建 ExperimentRecord
        5. recorder.record()
        6. 返回记录
        """
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
        parsed: StallChoice | None
        raw_response: str
        response_tokens: int
        latency_ms: int

        parsed, raw_response, response_tokens, latency_ms = (
            self.client.query_structured(prompt, system_message, temperature, num_stalls)
        )

        # 检测 API 调用层面的错误（client 会将异常格式化为 "ErrorType: msg"）
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
            record_id=uuid.uuid4().hex[:12],
            experiment_phase=ExperimentPhase(metadata.get("experiment_phase", "Phase1")),
            experiment_group=metadata.get("experiment_group", ""),
            model_name=self.model_config.name,
            model_version=self.model_config.version,
            temperature=temperature,
            prompt_template=PromptTemplate(metadata.get("prompt_template", "A")),
            prompt_text=metadata.get("prompt_text", prompt),
            num_stalls=num_stalls,
            occupied_stalls=metadata.get("occupied_stalls", []),
            conditions=metadata.get("conditions", {}),
            raw_response=raw_response,
            extracted_choice=extracted_choice,
            choice_status=choice_status,
            reasoning_present=reasoning_present,
            extracted_reasoning=extracted_reasoning,
            response_tokens=response_tokens,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc),
        )

        self.recorder.record(record)
        return record

    def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        templates: PromptTemplateConfig,
        max_retries: int = 3,
    ) -> list[ExperimentRecord]:
        """运行完整实验 / Run a full experiment across all parameter combinations.

        按照 (num_stalls x temperatures x templates x repetitions) 生成所有
        参数组合，随机打乱后依次执行。失败的调用（choice_status == ERROR）
        会在每轮结束后收集并重试，最多重试 max_retries 轮。
        """
        combos = self._build_combos(experiment_config)
        random.shuffle(combos)

        records: list[ExperimentRecord] = []

        # 构建参数包 / Build parameter bundles
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

        # 主流程 / Main pass
        print(f"  主流程: {len(tasks)} 次调用...")
        records, failed = self._run_task_batch(tasks)
        total_failed = failed

        # 重试轮次 / Retry rounds
        for retry_round in range(1, max_retries + 1):
            if not total_failed:
                break
            print(f"  重试第 {retry_round}/{max_retries} 轮: {len(total_failed)} 次失败调用...")
            retry_records, total_failed = self._run_task_batch(total_failed)
            records.extend(retry_records)

        if total_failed:
            print(f"  ⚠ {len(total_failed)} 次调用在 {max_retries} 轮重试后仍然失败")

        return records

    # ------------------------------------------------------------------
    # Private helpers / 私有辅助方法
    # ------------------------------------------------------------------

    def _run_task_batch(
        self, tasks: list[_Task]
    ) -> tuple[list[ExperimentRecord], list[_Task]]:
        """执行一批任务，返回 (成功+可解析记录, 失败任务列表)。"""
        records: list[ExperimentRecord] = []
        failed: list[_Task] = []

        for task in tasks:
            record = self.run_single(
                task.prompt, task.system_message,
                task.temperature, task.num_stalls, task.metadata,
            )
            records.append(record)
            if record.choice_status == ChoiceStatus.ERROR:
                failed.append(task)

        return records, failed

    @staticmethod
    def _is_error_response(raw_response: str) -> bool:
        """判断 raw_response 是否是 API 调用错误（而非模型正常输出）。"""
        # client 格式化错误为 "ErrorType: msg"
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
        """生成一条 ERROR 状态的记录。"""
        record = ExperimentRecord(
            record_id=uuid.uuid4().hex[:12],
            experiment_phase=ExperimentPhase(metadata.get("experiment_phase", "Phase1")),
            experiment_group=metadata.get("experiment_group", ""),
            model_name=self.model_config.name,
            model_version=self.model_config.version,
            temperature=temperature,
            prompt_template=PromptTemplate(metadata.get("prompt_template", "A")),
            prompt_text=metadata.get("prompt_text", prompt),
            num_stalls=num_stalls,
            occupied_stalls=metadata.get("occupied_stalls", []),
            conditions=metadata.get("conditions", {}),
            raw_response=error_msg,
            extracted_choice=None,
            choice_status=ChoiceStatus.ERROR,
            reasoning_present=False,
            extracted_reasoning="",
            response_tokens=0,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc),
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
        """分类响应状态 / Classify the response status.

        Args:
            raw_response: 原始响应文本。
            extracted_choice: 已提取的选择（可能为 None）。
            num_stalls: 坑位总数。

        Returns:
            VALID / REFUSED / AMBIGUOUS
        """
        if extracted_choice is not None and 1 <= extracted_choice <= num_stalls:
            return ChoiceStatus.VALID

        for pat in self._refusal_patterns:
            if pat.search(raw_response):
                return ChoiceStatus.REFUSED

        return ChoiceStatus.AMBIGUOUS

    def _extract_choice_from_text(self, raw_response: str, num_stalls: int) -> int | None:
        """从纯文本中提取坑位编号 / Extract stall number from plain text.

        依次尝试中文模式、英文模式、末尾数字和最后一个在范围内的数字。
        """
        # 中文模式 / Chinese patterns
        for pat in self._chinese_patterns:
            m = re.search(pat, raw_response)
            if m:
                val = int(m.group(1))
                if 1 <= val <= num_stalls:
                    return val

        # 英文模式 / English patterns
        for pat in self._english_patterns:
            m = re.search(pat, raw_response, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 1 <= val <= num_stalls:
                    return val

        # 末尾裸数字 / Bare digit at end of response
        m = re.search(self._trailing_digit_pattern, raw_response)
        if m:
            val = int(m.group(1))
            if 1 <= val <= num_stalls:
                return val

        # 最后一个在范围内的数字 / Last digit in range
        all_digits = re.findall(self._general_digit_pattern, raw_response)
        for d in reversed(all_digits):
            val = int(d)
            if 1 <= val <= num_stalls:
                return val

        return None
