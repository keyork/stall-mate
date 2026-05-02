# SPDX-License-Identifier: Apache-2.0
"""LLM 客户端 | LLM API client with structured output support.

通过 instructor 库实现结构化输出，包含三级回退链：
TOOLS → JSON_SCHEMA → PLAIN_TEXT。

Provides structured output via instructor with a 3-tier fallback chain:
TOOLS → JSON_SCHEMA → PLAIN_TEXT.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import instructor
from openai import OpenAI
from pydantic import ValidationError

from stall_mate.schema import StallChoice

_log = logging.getLogger(__name__)


class LLMClient:
    """LLM 客户端 / LLM API client with structured output support.

    支持通过 instructor 进行结构化输出，自动探测 API 能力并回退。
    Supports structured output via instructor, auto-probes API capabilities
    and falls back gracefully.
    """

    def __init__(self, endpoint: str, model: str, api_key: str = "", timeout: int = 60,
                 max_retries: int = 2, probe_message: str = "Say OK"):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.probe_message = probe_message
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

        # Try TOOLS mode
        try:
            inst = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
            inst.chat.completions.create(
                model=self.model,
                response_model=str,
                messages=[{"role": "user", "content": self.probe_message}],
                max_retries=0,
            )
            self._mode = "TOOLS"
            return self._mode
        except Exception as exc:
            _log.debug("TOOLS mode probe failed: %s", exc)

        # Try JSON_SCHEMA mode
        try:
            inst = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)
            inst.chat.completions.create(
                model=self.model,
                response_model=str,
                messages=[{"role": "user", "content": self.probe_message}],
                max_retries=0,
            )
            self._mode = "JSON_SCHEMA"
            return self._mode
        except Exception as exc:
            _log.debug("JSON_SCHEMA mode probe failed: %s", exc)

        # All failed, fallback to plain text
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

        任何异常（网络错误、API 错误、解析错误）均被捕获，不会向外抛出。
        All exceptions are caught internally; this method never raises.

        Args:
            prompt: 用户提示词 / User prompt.
            system_message: 系统消息 / System message.
            temperature: 采样温度 / Sampling temperature.
            num_stalls: 坑位数量（用于验证）/ Number of stalls (for validation).

        Returns:
            (解析结果或 None, 原始响应或错误信息, token 数, 延迟毫秒)
            (parsed_response or None, raw_response_or_error, response_tokens, latency_ms)
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
                    max_retries=self.max_retries,
                    context={"num_stalls": num_stalls},
                )
                latency = int((time.time() - start) * 1000)
                return result, result.model_dump_json(), 0, latency
            else:
                # PLAIN_TEXT fallback
                raw, tokens, latency = self.query_plain(
                    prompt, system_message, temperature
                )
                return None, raw, tokens, latency
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            return None, f"{type(e).__name__}: {e}", 0, latency

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
