# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from stall_mate.schema import StallChoice, get_stallchoice_json_schema


def test_valid_creation():
    sc = StallChoice(
        chosen_stall=3,
        chain_of_thought="I prefer the middle stall for privacy reasons.",
        confidence=0.75,
    )
    assert sc.chosen_stall == 3
    assert sc.confidence == 0.75


def test_confidence_bounds_valid():
    StallChoice(chosen_stall=1, chain_of_thought="minimum confidence", confidence=0.0)
    StallChoice(chosen_stall=1, chain_of_thought="maximum confidence", confidence=1.0)


@pytest.mark.parametrize("bad_conf", [-0.1, 1.1])
def test_confidence_bounds_invalid(bad_conf):
    with pytest.raises(ValidationError):
        StallChoice(
            chosen_stall=1, chain_of_thought="out of bounds test", confidence=bad_conf
        )


def test_json_schema():
    schema = get_stallchoice_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "chosen_stall" in schema["properties"]


def test_stall_range_valid_with_context():
    StallChoice.model_validate(
        {"chosen_stall": 3, "chain_of_thought": "a valid choice here", "confidence": 0.5},
        context={"num_stalls": 5},
    )


def test_stall_range_invalid_with_context():
    with pytest.raises(ValidationError):
        StallChoice.model_validate(
            {
                "chosen_stall": 10,
                "chain_of_thought": "out of range choice here",
                "confidence": 0.5,
            },
            context={"num_stalls": 5},
        )


def test_stall_range_no_context_allows_any():
    sc = StallChoice(
        chosen_stall=999,
        chain_of_thought="no context so anything goes ok",
        confidence=0.5,
    )
    assert sc.chosen_stall == 999


def test_chain_of_thought_min_length():
    with pytest.raises(ValidationError):
        StallChoice(chosen_stall=1, chain_of_thought="short", confidence=0.5)
