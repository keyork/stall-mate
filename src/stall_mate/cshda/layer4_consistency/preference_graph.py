# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)


class PreferenceGraph:
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_preference(
        self, preferred: str, not_preferred: str, metadata: dict | None = None
    ) -> None:
        self._graph.add_edge(preferred, not_preferred, **(metadata or {}))

    def check_transitivity(self) -> list[list[str]]:
        return list(nx.simple_cycles(self._graph))

    def get_inconsistencies(self) -> list[list[str]]:
        return self.check_transitivity()

    def clear(self) -> None:
        self._graph.clear()

    def __len__(self) -> int:
        return self._graph.number_of_edges()
