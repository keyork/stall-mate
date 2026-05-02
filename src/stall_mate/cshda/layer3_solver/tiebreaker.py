# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


class TieBreaker:
    def __init__(self, rules: list[str] | None = None) -> None:
        self.rules = rules or ["lexicographic"]

    def break_tie(self, tied_ids: list[str]) -> str:
        if len(tied_ids) == 1:
            return tied_ids[0]
        return min(tied_ids)
