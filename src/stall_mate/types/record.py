# SPDX-License-Identifier: Apache-2.0
"""
类型定义 — 实验数据结构
Type definitions — experiment data structures.

定义实验中使用的枚举类型和数据记录模型。
Defines enums and data record models used in the experiment.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ExperimentPhase(str, Enum):
    """实验阶段 / Experiment phase."""

    PHASE1 = "Phase1"
    PHASE2 = "Phase2"


class PromptTemplate(str, Enum):
    """Prompt 模板标识 / Prompt template identifier."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class ChoiceStatus(str, Enum):
    """选择状态 / Choice extraction status."""

    VALID = "VALID"
    REFUSED = "REFUSED"
    AMBIGUOUS = "AMBIGUOUS"


class ExperimentRecord(BaseModel):
    """单次实验记录 / Single experiment trial record.

    包含一次模型调用的完整上下文、响应和解析结果。
    Contains the full context, response, and parsed result of a single model call.
    """

    record_id: str
    experiment_phase: ExperimentPhase
    experiment_group: str
    model_name: str
    model_version: str = "unknown"
    temperature: float
    prompt_template: PromptTemplate
    prompt_text: str
    num_stalls: int
    occupied_stalls: list[int] = Field(default_factory=list)
    conditions: dict = Field(default_factory=dict)
    raw_response: str
    extracted_choice: int | None = None
    choice_status: ChoiceStatus
    reasoning_present: bool = False
    extracted_reasoning: str = ""
    response_tokens: int = 0
    latency_ms: int = 0
    timestamp: datetime
