# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime

import instructor
from openai import OpenAI

from stall_mate.cshda.layer1_extraction.prompts import ANCHOR_SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT
from stall_mate.cshda.schema.uds import (
    Constraint, ContextFactor, Entity, EntityProperty,
    ExtractionMeta, Objective, Relation, UniversalDecisionSpec,
)

_log = logging.getLogger(__name__)


class UDSExtractor:
    def __init__(
        self,
        model: str = "glm-5.1",
        base_url: str = "http://localhost:3000/v1",
        api_key: str = "",
        extraction_rounds: int = 3,
        anchor_rounds: int = 2,
        temperature: float = 0.0,
        timeout: int = 120,
    ):
        self.model = model
        self.extraction_rounds = extraction_rounds
        self.anchor_rounds = anchor_rounds
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key or "unused", timeout=timeout)

    def extract(self, natural_input: str) -> UniversalDecisionSpec:
        results = [self._single_extract(natural_input) for _ in range(self.extraction_rounds)]
        merged = self._vote_and_merge(results, natural_input)
        return merged

    def _single_extract(self, natural_input: str) -> UniversalDecisionSpec:
        inst = instructor.from_openai(self._client, mode=instructor.Mode.TOOLS)
        try:
            result = inst.chat.completions.create(
                model=self.model,
                response_model=UniversalDecisionSpec,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": natural_input},
                ],
                temperature=self.temperature,
                max_retries=1,
            )
            result.metadata.raw_input = natural_input
            result.metadata.extraction_model = self.model
            return result
        except Exception as exc:
            _log.warning("Extraction call failed: %s", exc)
            return UniversalDecisionSpec(
                metadata=ExtractionMeta(
                    raw_input=natural_input,
                    extraction_model=self.model,
                    extraction_timestamp=datetime.now(),
                    extraction_rounds=0,
                    extraction_stability=0.0,
                ),
            )

    def _vote_and_merge(self, results: list[UniversalDecisionSpec], raw_input: str) -> UniversalDecisionSpec:
        if not results:
            return UniversalDecisionSpec(metadata=ExtractionMeta(raw_input=raw_input))
        if len(results) == 1:
            return results[0]

        entity_map: dict[str, list[Entity]] = {}
        for r in results:
            for e in r.entities:
                entity_map.setdefault(e.id, []).append(e)
        merged_entities = []
        for eid, versions in entity_map.items():
            base = versions[0].model_copy()
            all_props: dict[str, list[EntityProperty]] = {}
            for v in versions:
                for p in v.properties:
                    all_props.setdefault(p.key, []).append(p)
            merged_props = []
            for pkey, pversions in all_props.items():
                desc_counter = Counter(p.value_description for p in pversions)
                most_common_desc = desc_counter.most_common(1)[0][0]
                numeric_vals = [p.numeric_value for p in pversions if p.numeric_value is not None]
                merged_props.append(EntityProperty(
                    key=pkey,
                    value_description=most_common_desc,
                    numeric_value=numeric_vals[0] if numeric_vals else None,
                    unit=pversions[0].unit,
                ))
            base.properties = merged_props
            merged_entities.append(base)

        obj_descs = Counter(o.description for r in results for o in r.objectives)
        threshold = max(2, len(results) // 2 + 1)
        merged_objectives = []
        seen_descs: set[str] = set()
        for r in results:
            for o in r.objectives:
                if o.description not in seen_descs and obj_descs[o.description] >= threshold:
                    merged_objectives.append(o)
                    seen_descs.add(o.description)

        constraint_set: dict[str, Constraint] = {}
        for r in results:
            for c in r.constraints:
                key = f"{c.constraint_type}:{c.description}"
                if key not in constraint_set:
                    constraint_set[key] = c

        rel_keys = Counter(f"{r.source}->{r.target}:{r.relation_type}" for res in results for r in res.relations)
        merged_relations = []
        seen_rels: set[str] = set()
        for r in results:
            for rel in r.relations:
                rk = f"{rel.source}->{rel.target}:{rel.relation_type}"
                if rk not in seen_rels and rel_keys[rk] >= threshold:
                    merged_relations.append(rel)
                    seen_rels.add(rk)

        type_hints = [r.decision_type_hint for r in results if r.decision_type_hint]
        hint_counter = Counter(type_hints)
        best_hint = hint_counter.most_common(1)[0][0] if hint_counter else None

        stability = len(merged_entities) / max(len(set(e.id for r in results for e in r.entities)), 1) if results else 0.0

        return UniversalDecisionSpec(
            metadata=ExtractionMeta(
                raw_input=raw_input,
                extraction_model=self.model,
                extraction_timestamp=datetime.now(),
                extraction_rounds=len(results),
                extraction_stability=min(stability, 1.0),
            ),
            entities=merged_entities,
            objectives=merged_objectives,
            constraints=list(constraint_set.values()),
            relations=merged_relations,
            decision_context=results[0].decision_context if results else [],
            decision_type_hint=best_hint,
        )

    def generate_anchors(self, uds: UniversalDecisionSpec) -> UniversalDecisionSpec:
        inst = instructor.from_openai(self._client, mode=instructor.Mode.TOOLS)
        for entity in uds.entities:
            for prop in entity.properties:
                if prop.positive_anchor and prop.negative_anchor:
                    continue
                try:
                    user_msg = json.dumps({
                        "entity_label": entity.label,
                        "entity_type": entity.entity_type,
                        "property_key": prop.key,
                        "value_description": prop.value_description,
                    }, ensure_ascii=False)
                    anchor_result = inst.chat.completions.create(
                        model=self.model,
                        response_model=EntityProperty,
                        messages=[
                            {"role": "system", "content": ANCHOR_SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.0,
                        max_retries=1,
                    )
                    if anchor_result.positive_anchor:
                        prop.positive_anchor = anchor_result.positive_anchor
                    if anchor_result.negative_anchor:
                        prop.negative_anchor = anchor_result.negative_anchor
                except Exception as exc:
                    _log.warning("Anchor generation failed for %s.%s: %s", entity.id, prop.key, exc)
                    if not prop.positive_anchor:
                        prop.positive_anchor = f"理想的{prop.key}状态"
                    if not prop.negative_anchor:
                        prop.negative_anchor = f"最差的{prop.key}状态"
        return uds
