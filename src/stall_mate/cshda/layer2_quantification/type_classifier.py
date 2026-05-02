# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

from stall_mate.cshda.schema.uds import UniversalDecisionSpec

_log = logging.getLogger(__name__)


def classify_decision_type(uds: UniversalDecisionSpec) -> tuple[str, float]:
    agent_entities = [
        e for e in uds.entities if e.entity_type == "agent"
    ]
    if len(agent_entities) > 1:
        return ("T6", 0.95)

    precedes_relations = [
        r for r in uds.relations
        if r.relation_type in ("precedes", "depends_on")
    ]
    if len(precedes_relations) >= 2:
        return ("T5", 0.90)

    resource_entities = [
        e for e in uds.entities if e.entity_type == "resource"
    ]
    if resource_entities:
        return ("T4", 0.90)

    objective_text = " ".join(o.description for o in uds.objectives).lower()
    if any(
        kw in objective_text
        for kw in ("排序", "优先级", "顺序")
    ) or len(precedes_relations) >= 1:
        return ("T3", 0.85)

    capacity_constraints = [
        c for c in uds.constraints
        if "capacity" in c.constraint_type.lower()
        or "budget" in c.constraint_type.lower()
    ]
    if capacity_constraints:
        has_value_cost = any(
            any(p.key in ("value", "cost") for p in e.properties)
            for e in uds.entities
        )
        if has_value_cost:
            return ("T2", 0.90)

    return ("T1", 0.85)
