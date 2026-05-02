# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path

from stall_mate.cshda.schema.formulation import MathematicalFormulation
from stall_mate.cshda.schema.output import AuditTrail, FinalOutput
from stall_mate.cshda.schema.result import DecisionResult
from stall_mate.cshda.schema.uds import UniversalDecisionSpec

logger = logging.getLogger(__name__)


class AuditLogger:
    def __init__(self, output_path: Path | None = None) -> None:
        self._output_path = output_path
        if self._output_path is not None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, output: FinalOutput) -> None:
        if self._output_path is None:
            return
        with self._output_path.open("a", encoding="utf-8") as f:
            f.write(output.model_dump_json() + "\n")

    def build_trail(
        self,
        raw_input: str,
        uds: UniversalDecisionSpec,
        mf: MathematicalFormulation,
        dr: DecisionResult,
        consistency: ConsistencyReport,
    ) -> AuditTrail:
        return AuditTrail(
            raw_input=raw_input,
            extracted_uds=uds.model_dump(mode="json"),
            mathematical_formulation=mf.model_dump(mode="json"),
            solver_result=dr.model_dump(mode="json"),
            consistency_checks=consistency.model_dump(mode="json"),
        )
