# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from stall_mate.cshda.schema.result import DecisionResult


class ConsistencyReport(BaseModel):
    determinism: str = "PASS"
    constraint_satisfaction: str = "PASS"
    transitivity: str = "N/A"
    iia: str = "N/A"
    frame_invariance: str = "N/A"
    type_specific_checks: dict[str, str] = Field(default_factory=dict)


class ConfidenceBreakdown(BaseModel):
    extraction_stability: float = 1.0
    quantification_robustness: float = 1.0
    solution_margin: float = 1.0


class AuditTrail(BaseModel):
    raw_input: str = ""
    extracted_uds: dict = Field(default_factory=dict)
    mathematical_formulation: dict = Field(default_factory=dict)
    solver_result: dict = Field(default_factory=dict)
    consistency_checks: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class FinalOutput(BaseModel):
    decision: DecisionResult
    consistency_report: ConsistencyReport = Field(default_factory=ConsistencyReport)
    confidence_score: float = 1.0
    confidence_breakdown: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    audit_trail: AuditTrail = Field(default_factory=AuditTrail)
