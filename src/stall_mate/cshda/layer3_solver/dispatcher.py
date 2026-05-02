# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from stall_mate.cshda.layer3_solver.base import BaseSolver
from stall_mate.cshda.layer3_solver.t1_selection import T1SelectionSolver
from stall_mate.cshda.schema.formulation import MathematicalFormulation
from stall_mate.cshda.schema.result import DecisionResult

_SOLVERS: dict[str, type[BaseSolver]] = {
    "T1": T1SelectionSolver,
}


def dispatch(mf: MathematicalFormulation) -> DecisionResult:
    dtype = mf.decision_type
    if dtype not in _SOLVERS:
        raise ValueError(f"Unsupported decision type: {dtype}")
    solver = _SOLVERS[dtype]()
    inner = mf.formulation
    if not solver.validate(inner):
        raise ValueError(f"Invalid {dtype} formulation")
    return solver.solve(inner)
