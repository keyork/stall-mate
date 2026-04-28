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

# 拒绝关键词 / Refusal keywords (case-insensitive check)
_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        "无法",
        "不能",
        "拒绝",
        "refuse",
        "cannot",
        "won't",
        "I can't",
        "inappropriate",
    )
]


class ExperimentRunner:
    """实验运行器 / Experiment runner — orchestrates the full pipeline."""

    def __init__(self, client: LLMClient, recorder: JSONLRecorder, model_config: ModelConfig):
        self.client = client
        self.recorder = recorder
        self.model_config = model_config

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

        1. 调用 client.query_structured()
        2. 若结构化输出失败，回退到纯文本 + 手动解析
        3. 分类 choice_status: VALID / REFUSED / AMBIGUOUS
        4. 构建 ExperimentRecord
        5. recorder.record()
        6. 返回记录
        """
        parsed: StallChoice | None
        raw_response: str
        response_tokens: int
        latency_ms: int

        parsed, raw_response, response_tokens, latency_ms = (
            self.client.query_structured(prompt, system_message, temperature, num_stalls)
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
    ) -> list[ExperimentRecord]:
        """运行完整实验 / Run a full experiment across all parameter combinations.

        按照 (num_stalls × temperatures × templates × repetitions) 生成所有
        参数组合，随机打乱后依次执行，避免顺序效应。
        """
        combos: list[tuple[int, float, str, int]] = []
        for ns in experiment_config.num_stalls:
            for temp in experiment_config.temperatures:
                for tpl_key in experiment_config.templates:
                    for rep_idx in range(experiment_config.repetitions):
                        combos.append((ns, temp, tpl_key, rep_idx))

        # 随机打乱避免顺序效应 / Shuffle to avoid order effects
        random.shuffle(combos)

        records: list[ExperimentRecord] = []
        for num_stalls, temperature, template_key, _rep_idx in combos:
            prompt = build_prompt(templates.templates[template_key], num_stalls)
            system_message = build_system_message(num_stalls)
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
            rec = self.run_single(
                prompt, system_message, temperature, num_stalls, metadata
            )
            records.append(rec)

        return records

    # ------------------------------------------------------------------
    # Private helpers / 私有辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_response(
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

        for pat in _REFUSAL_PATTERNS:
            if pat.search(raw_response):
                return ChoiceStatus.REFUSED

        return ChoiceStatus.AMBIGUOUS

    @staticmethod
    def _extract_choice_from_text(raw_response: str, num_stalls: int) -> int | None:
        """从纯文本中提取坑位编号 / Extract stall number from plain text.

        依次尝试中文模式、英文模式、末尾数字和最后一个在范围内的数字。
        """
        # 中文模式 / Chinese patterns
        cn_patterns = [
            r"第\s*(\d+)\s*个",
            r"(\d+)\s*号",
            r"选择.*?(\d+)",
        ]
        for pat in cn_patterns:
            m = re.search(pat, raw_response)
            if m:
                val = int(m.group(1))
                if 1 <= val <= num_stalls:
                    return val

        # 英文模式 / English patterns
        en_patterns = [
            r"stall\s*(\d+)",
            r"number\s*(\d+)",
        ]
        for pat in en_patterns:
            m = re.search(pat, raw_response, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 1 <= val <= num_stalls:
                    return val

        # 末尾裸数字 / Bare digit at end of response
        m = re.search(r"(\d+)\s*[。.!?]?\s*$", raw_response)
        if m:
            val = int(m.group(1))
            if 1 <= val <= num_stalls:
                return val

        # 最后一个在范围内的数字 / Last digit in range
        all_digits = re.findall(r"\b(\d+)\b", raw_response)
        for d in reversed(all_digits):
            val = int(d)
            if 1 <= val <= num_stalls:
                return val

        return None
