# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, Discriminator, Field


class T1Formulation(BaseModel):
    decision_type: Literal["T1"] = "T1"
    score_matrix: list[list[float]]
    weights: list[float]
    weight_method: str = "variance"
    option_ids: list[str]
    attribute_ids: list[str]
    attribute_directions: list[str] = Field(default_factory=list)

    def score_matrix_np(self) -> np.ndarray:
        return np.array(self.score_matrix)

    def weights_np(self) -> np.ndarray:
        return np.array(self.weights)


class T2Formulation(BaseModel):
    decision_type: Literal["T2"] = "T2"
    n_items: int
    item_ids: list[str]
    value_vector: list[float]
    cost_matrix: list[list[float]]
    capacity_vector: list[float]
    mutex_pairs: list[tuple[str, str]] = Field(default_factory=list)
    synergy_pairs: list[tuple[str, str, float]] = Field(default_factory=list)
    item_bounds: list[tuple[int, int]] | None = None


class T3Formulation(BaseModel):
    decision_type: Literal["T3"] = "T3"
    n_items: int
    item_ids: list[str]
    priority_matrix: list[list[float]] = Field(default_factory=list)
    criteria_weights: list[float] = Field(default_factory=list)
    precedence_pairs: list[tuple[str, str]] = Field(default_factory=list)
    processing_times: list[float] | None = None
    deadlines: list[float] | None = None
    release_times: list[float] | None = None


class T4Formulation(BaseModel):
    decision_type: Literal["T4"] = "T4"
    n_receivers: int
    n_resources: int
    receiver_ids: list[str]
    resource_ids: list[str]
    utility_matrix: list[list[float]]
    resource_totals: list[float]
    min_allocations: list[list[float]] | None = None
    max_allocations: list[list[float]] | None = None
    fairness_constraint: str | None = None
    is_assignment: bool = False
    assignment_cost_matrix: list[list[float]] | None = None


class T5Formulation(BaseModel):
    decision_type: Literal["T5"] = "T5"
    n_stages: int
    stages: list[dict]
    initial_state: str
    discount_factor: float = 1.0
    terminal_rewards: dict[str, float] | None = None


class T6Formulation(BaseModel):
    decision_type: Literal["T6"] = "T6"
    n_players: int
    player_ids: list[str]
    strategy_sets: dict[str, list[str]]
    payoff_tensor: list
    game_type: str = "simultaneous"
    move_order: list[str] | None = None
    information_sets: dict | None = None
    is_repeated: bool = False
    n_rounds: int | None = None


Formulation = Annotated[
    Union[T1Formulation, T2Formulation, T3Formulation,
          T4Formulation, T5Formulation, T6Formulation],
    Discriminator("decision_type"),
]


class EmbeddingArtifacts(BaseModel):
    entity_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    attribute_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    polarity_axes: dict[str, tuple[list[float], list[float]]] = Field(default_factory=dict)
    similarity_matrix: list[list[float]] | None = None


class MathematicalFormulation(BaseModel):
    decision_type: str
    type_confidence: float = 1.0
    formulation: Formulation
    embedding_artifacts: EmbeddingArtifacts = Field(default_factory=EmbeddingArtifacts)
