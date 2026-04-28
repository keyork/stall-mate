# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from stall_mate.types import (
    ChoiceStatus,
    ExperimentPhase,
    ExperimentRecord,
    PromptTemplate,
)


def _make_record(**overrides):
    base = dict(
        record_id="rec-001",
        experiment_phase=ExperimentPhase.PHASE1,
        experiment_group="1.1",
        model_name="gpt-4o",
        temperature=0.0,
        prompt_template=PromptTemplate.A,
        prompt_text="请选择一个隔间。",
        num_stalls=5,
        raw_response="我选择3号隔间。",
        choice_status=ChoiceStatus.VALID,
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return ExperimentRecord(**base)


class TestEnums:
    def test_experiment_phase_values(self):
        assert ExperimentPhase.PHASE1.value == "Phase1"
        assert ExperimentPhase.PHASE2.value == "Phase2"

    def test_prompt_template_values(self):
        assert PromptTemplate.A.value == "A"
        assert PromptTemplate.B.value == "B"
        assert PromptTemplate.C.value == "C"
        assert PromptTemplate.D.value == "D"

    def test_choice_status_values(self):
        assert ChoiceStatus.VALID.value == "VALID"
        assert ChoiceStatus.REFUSED.value == "REFUSED"
        assert ChoiceStatus.AMBIGUOUS.value == "AMBIGUOUS"

    def test_enums_are_strings(self):
        for phase in ExperimentPhase:
            assert isinstance(phase, str)
        for pt in PromptTemplate:
            assert isinstance(pt, str)
        for cs in ChoiceStatus:
            assert isinstance(cs, str)


class TestExperimentRecord:
    def test_create_with_required_fields_only(self):
        rec = _make_record()
        assert rec.record_id == "rec-001"
        assert rec.experiment_phase == ExperimentPhase.PHASE1
        assert rec.choice_status == ChoiceStatus.VALID

    def test_defaults(self):
        rec = _make_record()
        assert rec.model_version == "unknown"
        assert rec.occupied_stalls == []
        assert rec.conditions == {}
        assert rec.extracted_choice is None
        assert rec.reasoning_present is False
        assert rec.extracted_reasoning == ""
        assert rec.response_tokens == 0
        assert rec.latency_ms == 0

    def test_with_all_fields(self):
        rec = _make_record(
            model_version="2024-08-06",
            occupied_stalls=[1, 3],
            conditions={"cleanliness": "dirty"},
            extracted_choice=3,
            reasoning_present=True,
            extracted_reasoning="3号最远",
            response_tokens=42,
            latency_ms=1200,
        )
        assert rec.occupied_stalls == [1, 3]
        assert rec.conditions == {"cleanliness": "dirty"}
        assert rec.extracted_choice == 3
        assert rec.response_tokens == 42

    def test_json_roundtrip(self):
        rec = _make_record(occupied_stalls=[2, 4])
        json_str = rec.model_dump_json()
        rec2 = ExperimentRecord.model_validate_json(json_str)
        assert rec2 == rec

    def test_invalid_choice_status_raises(self):
        with pytest.raises(ValidationError):
            _make_record(choice_status="INVALID_STATUS")
