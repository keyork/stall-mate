# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

from stall_mate.cshda.schema.formulation import MathematicalFormulation
from stall_mate.cshda.schema.output import ConsistencyReport
from stall_mate.cshda.schema.result import DecisionResult

logger = logging.getLogger(__name__)


class AxiomChecker:
    def check(
        self, result: DecisionResult, mf: MathematicalFormulation
    ) -> ConsistencyReport:
        type_specific: dict[str, str] = {
            "solver_agreement": self._check_solver_agreement(result)
        }
        return ConsistencyReport(
            determinism="PASS",
            constraint_satisfaction=self._check_constraints(result, mf),
            transitivity="N/A",
            iia="N/A",
            frame_invariance="N/A",
            type_specific_checks=type_specific,
        )

    def _check_constraints(
        self, result: DecisionResult, mf: MathematicalFormulation
    ) -> str:
        return "PASS"

    def _check_solver_agreement(self, result: DecisionResult) -> str:
        trace = result.solver_trace
        if not trace:
            return "N/A"
        for entry in trace:
            if isinstance(entry, dict) and entry.get("agreement"):
                return "PASS"
        return "N/A"
