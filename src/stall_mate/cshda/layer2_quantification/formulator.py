# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

import numpy as np

from stall_mate.cshda.layer2_quantification.embedder import Embedder
from stall_mate.cshda.layer2_quantification.polarity_scorer import PolarityScorer
from stall_mate.cshda.layer2_quantification.type_classifier import (
    classify_decision_type,
)
from stall_mate.cshda.layer2_quantification.weight_calculator import WeightCalculator
from stall_mate.cshda.schema.formulation import (
    MathematicalFormulation,
    T1Formulation,
)
from stall_mate.cshda.schema.uds import UniversalDecisionSpec

_log = logging.getLogger(__name__)


class Formulator:
    def __init__(self, embedder: Embedder, scorer: PolarityScorer):
        self._embedder = embedder
        self._scorer = scorer
        self._weight_calc = WeightCalculator()

    def formulate(self, uds: UniversalDecisionSpec) -> MathematicalFormulation:
        decision_type, confidence = classify_decision_type(uds)
        if decision_type == "T1":
            formulation = self._formulate_t1(uds)
        else:
            raise NotImplementedError(
                f"Decision type {decision_type} formulation not yet implemented"
            )
        return MathematicalFormulation(
            decision_type=decision_type,
            type_confidence=confidence,
            formulation=formulation,
        )

    def _formulate_t1(self, uds: UniversalDecisionSpec) -> T1Formulation:
        unavailable_ids: set[str] = set()
        for constraint in uds.constraints:
            if "avail" in constraint.constraint_type.lower():
                unavailable_ids.update(constraint.involves)

        available_entities = [
            e for e in uds.entities if e.id not in unavailable_ids
        ]
        option_ids = [e.id for e in available_entities]

        all_keys_ordered: list[str] = []
        seen_keys: set[str] = set()
        for entity in available_entities:
            for prop in entity.properties:
                if prop.key not in seen_keys:
                    seen_keys.add(prop.key)
                    all_keys_ordered.append(prop.key)
        attribute_ids = all_keys_ordered

        numeric_by_key: dict[str, list[float]] = {}
        for key in attribute_ids:
            values = []
            for entity in available_entities:
                for prop in entity.properties:
                    if prop.key == key and prop.numeric_value is not None:
                        values.append(prop.numeric_value)
            numeric_by_key[key] = values

        score_matrix: list[list[float]] = []
        for entity in available_entities:
            row: list[float] = []
            for key in attribute_ids:
                prop = next(
                    (p for p in entity.properties if p.key == key), None
                )
                if prop is None:
                    row.append(0.5)
                    continue

                if (
                    prop.numeric_value is not None
                    and len(numeric_by_key[key]) > 1
                ):
                    score = self._scorer.score_numeric(
                        prop.numeric_value, numeric_by_key[key]
                    )
                elif (
                    prop.value_description
                    and prop.positive_anchor
                    and prop.negative_anchor
                ):
                    score = self._scorer.score_attribute(
                        prop.value_description,
                        prop.positive_anchor,
                        prop.negative_anchor,
                    )
                else:
                    score = 0.5
                row.append(score)
            score_matrix.append(row)

        matrix_np = np.array(score_matrix, dtype=np.float64)
        weights = self._weight_calc.ensemble_weights(matrix_np)

        return T1Formulation(
            score_matrix=score_matrix,
            weights=weights.tolist(),
            weight_method="ensemble",
            option_ids=option_ids,
            attribute_ids=attribute_ids,
        )
