# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import difflib
import logging
import uuid
from collections import defaultdict

from stall_mate.cshda.schema.uds import (
    Constraint,
    Entity,
    Relation,
    UniversalDecisionSpec,
)

_log = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.85
_CYCLE_RELATION_TYPES = {"depends_on", "precedes"}


class UDSNormalizer:
    def normalize(self, uds: UniversalDecisionSpec) -> UniversalDecisionSpec:
        id_map = self._standardize_entity_ids(uds.entities)
        self._deduplicate_property_keys(uds.entities)
        self._ensure_constraint_ids(uds.constraints)
        self._deduplicate_relations(uds.relations)
        self._check_circular_dependencies(uds.relations)
        self._remap_references(uds, id_map)
        return uds

    def _standardize_entity_ids(self, entities: list[Entity]) -> dict[str, str]:
        counters: dict[str, int] = defaultdict(int)
        id_map: dict[str, str] = {}
        for entity in entities:
            counters[entity.entity_type] += 1
            new_id = f"{entity.entity_type}_{counters[entity.entity_type]}"
            id_map[entity.id] = new_id
            entity.id = new_id
        return id_map

    def _deduplicate_property_keys(self, entities: list[Entity]) -> None:
        for entity in entities:
            merged: list = []
            kept_keys: list[str] = []
            for prop in entity.properties:
                similar = None
                for kept in kept_keys:
                    if difflib.SequenceMatcher(None, prop.key, kept).ratio() > _SIMILARITY_THRESHOLD:
                        similar = kept
                        break
                if similar is None:
                    kept_keys.append(prop.key)
                    merged.append(prop)
                else:
                    for existing in merged:
                        if existing.key == similar and not existing.value_description and prop.value_description:
                            existing.value_description = prop.value_description
            entity.properties = merged

    def _ensure_constraint_ids(self, constraints: list[Constraint]) -> None:
        for i, c in enumerate(constraints):
            if not c.id:
                c.id = f"constraint_{i + 1}_{uuid.uuid4().hex[:6]}"

    def _deduplicate_relations(self, relations: list[Relation]) -> None:
        seen: set[str] = set()
        unique: list[Relation] = []
        for r in relations:
            key = f"{r.source}->{r.target}:{r.relation_type}"
            if key not in seen:
                seen.add(key)
                unique.append(r)
        relations.clear()
        relations.extend(unique)

    def _check_circular_dependencies(self, relations: list[Relation]) -> None:
        edges: dict[str, list[str]] = defaultdict(list)
        for r in relations:
            if r.relation_type in _CYCLE_RELATION_TYPES:
                edges[r.source].append(r.target)

        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for neighbor in edges.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in in_stack:
                    return True
            in_stack.discard(node)
            return False

        all_nodes = set(edges.keys())
        for r in relations:
            if r.relation_type in _CYCLE_RELATION_TYPES:
                all_nodes.add(r.source)
                all_nodes.add(r.target)

        for node in all_nodes:
            if node not in visited:
                if dfs(node):
                    raise ValueError(
                        f"Circular dependency detected in relations involving node '{node}'"
                    )

    def _remap_references(self, uds: UniversalDecisionSpec, id_map: dict[str, str]) -> None:
        for c in uds.constraints:
            c.involves = [id_map.get(eid, eid) for eid in c.involves]
        for r in uds.relations:
            r.source = id_map.get(r.source, r.source)
            r.target = id_map.get(r.target, r.target)
        for cf in uds.decision_context:
            cf.influence_on = [id_map.get(eid, eid) for eid in cf.influence_on]
