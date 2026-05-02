# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pydantic import BaseModel, Field


class DecisionResult(BaseModel):
    decision_type: str
    chosen: list[str] = Field(default_factory=list)
    ranking: list[str] | None = None
    allocation: dict[str, float] | None = None
    strategy: dict[str, str] | None = None
    action_sequence: list[str] | None = None
    objective_value: float = 0.0
    optimality_gap: float | None = None
    solver_name: str = ""
    solver_trace: list[dict] = Field(default_factory=list)
    intermediate_values: dict = Field(default_factory=dict)
    margin: float = 0.0
    critical_parameters: list[str] = Field(default_factory=list)
