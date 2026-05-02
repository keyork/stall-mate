# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from stall_mate.cshda.layer3_solver.base import BaseSolver
from stall_mate.cshda.schema.formulation import T1Formulation
from stall_mate.cshda.schema.result import DecisionResult


class T1SelectionSolver(BaseSolver):

    def validate(self, formulation: T1Formulation) -> bool:
        n_opts = len(formulation.option_ids)
        n_attrs = len(formulation.attribute_ids)
        sm = formulation.score_matrix
        w = formulation.weights
        if len(sm) != n_opts:
            return False
        for row in sm:
            if len(row) != n_attrs:
                return False
        if len(w) != n_attrs:
            return False
        if abs(sum(w) - 1.0) >= 1e-6:
            return False
        for row in sm:
            for v in row:
                if v < -1e-9 or v > 1.0 + 1e-9:
                    return False
        return True

    def solve(self, formulation: T1Formulation) -> DecisionResult:
        ids = formulation.option_ids
        attr_ids = formulation.attribute_ids
        sm = formulation.score_matrix_np()
        w = formulation.weights_np()

        utilities = sm @ w
        saw_order = np.argsort(-utilities)
        saw_ranking = [ids[i] for i in saw_order]

        eps = 1e-10
        col_norms = np.linalg.norm(sm, axis=0) + eps
        normalized = sm / col_norms
        w_matrix = normalized * w
        pos_ideal = w_matrix.max(axis=0)
        neg_ideal = w_matrix.min(axis=0)
        d_pos = np.linalg.norm(w_matrix - pos_ideal, axis=1)
        d_neg = np.linalg.norm(w_matrix - neg_ideal, axis=1)
        closeness = d_neg / (d_pos + d_neg + eps)
        topsis_order = np.argsort(-closeness)
        topsis_ranking = [ids[i] for i in topsis_order]

        agreement = saw_ranking[0] == topsis_ranking[0]

        sorted_utils = sorted(utilities, reverse=True)
        margin = float(sorted_utils[0] - sorted_utils[1]) if len(sorted_utils) >= 2 else 1.0

        saw_util_map = {ids[i]: float(utilities[i]) for i in range(len(ids))}
        topsis_close_map = {ids[i]: float(closeness[i]) for i in range(len(ids))}

        trace = [
            {"step": "SAW", "utilities": saw_util_map},
            {"step": "TOPSIS", "closeness": topsis_close_map},
            {"step": "agreement", "saw_top": saw_ranking[0],
             "topsis_top": topsis_ranking[0], "agree": agreement},
        ]

        weight_pairs = list(zip(attr_ids, w))
        weight_pairs.sort(key=lambda p: float(p[1]), reverse=True)
        critical = [p[0] for p in weight_pairs[:2]]

        return DecisionResult(
            decision_type="T1",
            chosen=[saw_ranking[0]],
            ranking=saw_ranking,
            objective_value=float(utilities.max()),
            margin=margin,
            solver_name="SAW+TOPSIS",
            solver_trace=trace,
            intermediate_values={
                "saw_utilities": saw_util_map,
                "topsis_closeness": topsis_close_map,
            },
            critical_parameters=critical,
        )
