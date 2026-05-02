# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod

from stall_mate.cshda.schema.result import DecisionResult


class BaseSolver(ABC):
    @abstractmethod
    def solve(self, formulation) -> DecisionResult: ...

    @abstractmethod
    def validate(self, formulation) -> bool: ...

    def explain(self, result: DecisionResult) -> str:
        return str(result.solver_trace)
