# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExtractionMeta(BaseModel):
    raw_input: str
    extraction_model: str = ""
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    extraction_rounds: int = 1
    extraction_stability: float = 1.0

    @field_validator("extraction_timestamp", mode="before")
    @classmethod
    def _coerce_timestamp(cls, v: Any) -> Any:
        if isinstance(v, str):
            from dateutil.parser import parse as parse_dt
            return parse_dt(v)
        return v


class EntityProperty(BaseModel):
    key: str
    value_description: str = ""
    numeric_value: float | None = None
    unit: str | None = None
    positive_anchor: str = ""
    negative_anchor: str = ""


class Entity(BaseModel):
    id: str
    label: str
    entity_type: str = "option"
    properties: list[EntityProperty] = Field(default_factory=list)


class Objective(BaseModel):
    id: str = ""
    description: str
    direction: str = "maximize"
    target_value: float | None = None


class Constraint(BaseModel):
    id: str = ""
    description: str
    constraint_type: str = ""
    involves: list[str] = Field(default_factory=list)
    numeric_limit: float | None = None
    limit_direction: str | None = None


class Relation(BaseModel):
    source: str
    target: str
    relation_type: str
    description: str = ""
    strength: str | None = None


class ContextFactor(BaseModel):
    factor: str
    description: str = ""
    influence_on: list[str] = Field(default_factory=list)


class UniversalDecisionSpec(BaseModel):
    metadata: ExtractionMeta
    entities: list[Entity] = Field(default_factory=list)
    objectives: list[Objective] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    decision_context: list[ContextFactor] = Field(default_factory=list)
    decision_type_hint: str | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, v: Any) -> Any:
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v
