# SPDX-License-Identifier: Apache-2.0
"""
LLM 客户端 / LLM API client with structured output support.

通过 instructor 库实现结构化输出，包含三级回退链：
TOOLS → JSON_SCHEMA → PLAIN_TEXT。

Provides structured output via instructor with a 3-tier fallback chain:
TOOLS → JSON_SCHEMA → PLAIN_TEXT.
"""

from __future__ import annotations

import time
from typing import Any

import instructor
from openai import OpenAI
from pydantic import ValidationError

from stall_mate.schema import StallChoice


class LLMClient:
    """LLM 客户端 / LLM API client with structured output support.

    支持通过 instructor 进行结构化输出，自动探测 API 能力并回退。
    Supports structured output via instructor, auto-probes API capabilities
    and falls back gracefully.
    """

    def __init__(self, endpoint: str, model: str, api_key: str = "", timeout: int = 60):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._openai_client: OpenAI | None = None
        self._mode: str | None = None

    def _get_openai_client(self) -> OpenAI:
        """延迟初始化底层 OpenAI 客户端 / Lazy-init the underlying OpenAI client."""
        if self._openai_client is None:
            self._openai_client = OpenAI(
                base_url=self.endpoint,
                api_key=self.api_key or "unused",
                timeout=self.timeout,
            )
        return self._openai_client

    def probe_api(self) -> str:
        """探测 API 支持的最佳 instructor 模式 / Probe API to determine best instructor mode.

        Returns:
            "TOOLS" | "JSON_SCHEMA" | "PLAIN_TEXT"

        依次尝试 TOOLS、JSON_SCHEMA 模式，均失败则回退到 PLAIN_TEXT。
        Tries Mode.TOOLS, then Mode.JSON_SCHEMA; falls back to PLAIN_TEXT.
        """
        client = self._get_openai_client()

        # 尝试 TOOLS 模式 / Try TOOLS mode
        try:
            inst = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
            inst.chat.completions.create(
                model=self.model,
                response_model=str,
                messages=[{"role": "user", "content": "Say OK"}],
                max_retries=0,
            )
            self._mode = "TOOLS"
            return self._mode
        except Exception:
            pass

        # 尝试 JSON_SCHEMA 模式 / Try JSON_SCHEMA mode
        try:
            inst = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)
            inst.chat.completions.create(
                model=self.model,
                response_model=str,
                messages=[{"role": "user", "content": "Say OK"}],
                max_retries=0,
            )
            self._mode = "JSON_SCHEMA"
            return self._mode
        except Exception:
            pass

        # 全部失败，回退纯文本 / All failed, fallback to plain text
        self._mode = "PLAIN_TEXT"
        return self._mode

    def query_structured(
        self,
        prompt: str,
        system_message: str,
        temperature: float = 0.0,
        num_stalls: int = 5,
    ) -> tuple[StallChoice | None, str, int, int]:
        """结构化输出查询 / Query with structured output enforcement.

        Args:
            prompt: 用户提示词 / User prompt.
            system_message: 系统消息 / System message.
            temperature: 采样温度 / Sampling temperature.
            num_stalls: 坑位数量（用于验证）/ Number of stalls (for validation).

        Returns:
            (解析结果或 None, 原始响应, token 数, 延迟毫秒)
            (parsed_response or None, raw_response, response_tokens, latency_ms)
        """
        if self._mode is None:
            self.probe_api()

        start = time.time()

        try:
            client = self._get_openai_client()

            if self._mode in ("TOOLS", "JSON_SCHEMA"):
                mode = (
                    instructor.Mode.TOOLS
                    if self._mode == "TOOLS"
                    else instructor.Mode.JSON_SCHEMA
                )
                inst = instructor.from_openai(client, mode=mode)
                result: StallChoice = inst.chat.completions.create(
                    model=self.model,
                    response_model=StallChoice,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_retries=2,
                    context={"num_stalls": num_stalls},
                )
                latency = int((time.time() - start) * 1000)
                return result, result.model_dump_json(), 0, latency
            else:
                # PLAIN_TEXT 回退 — 获取纯文本 / PLAIN_TEXT fallback
                raw, tokens, latency = self.query_plain(
                    prompt, system_message, temperature
                )
                return None, raw, tokens, latency
        except (ValidationError, Exception) as e:
            latency = int((time.time() - start) * 1000)
            return None, str(e), 0, latency

    def query_plain(
        self,
        prompt: str,
        system_message: str,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        """纯文本回退查询 / Fallback plain text query without structured output.

        Args:
            prompt: 用户提示词 / User prompt.
            system_message: 系统消息 / System message.
            temperature: 采样温度 / Sampling temperature.

        Returns:
            (原始响应, token 数, 延迟毫秒)
            (raw_response, response_tokens, latency_ms)
        """
        start = time.time()
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        latency = int((time.time() - start) * 1000)
        raw = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return raw, tokens, latency
