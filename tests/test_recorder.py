# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path

import pytest

from stall_mate.recorder import JSONLRecorder
from stall_mate.types import (
    ChoiceStatus,
    ExperimentPhase,
    ExperimentRecord,
    PromptTemplate,
)


def make_record(**overrides):
    defaults = dict(
        record_id="test-001",
        experiment_phase=ExperimentPhase.PHASE1,
        experiment_group="1.1",
        model_name="test-model",
        temperature=0.0,
        prompt_template=PromptTemplate.A,
        prompt_text="test prompt",
        num_stalls=5,
        raw_response="I choose stall 3",
        extracted_choice=3,
        choice_status=ChoiceStatus.VALID,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


class TestRecord:
    def test_record_writes_one_line(self, tmp_path):
        path = tmp_path / "out.jsonl"
        rec = JSONLRecorder(path)
        rec.record(make_record())
        assert path.exists()
        assert len(path.read_text().strip().splitlines()) == 1

    def test_record_batch_writes_five(self, tmp_path):
        path = tmp_path / "out.jsonl"
        rec = JSONLRecorder(path)
        records = [make_record(record_id=f"r-{i}") for i in range(5)]
        rec.record_batch(records)
        assert rec.count() == 5

    def test_read_all_after_write(self, tmp_path):
        path = tmp_path / "out.jsonl"
        rec = JSONLRecorder(path)
        written = [make_record(record_id=f"r-{i}") for i in range(3)]
        rec.record_batch(written)
        read_back = rec.read_all()
        assert len(read_back) == 3
        for w, r in zip(written, read_back):
            assert w.record_id == r.record_id

    def test_round_trip_all_fields(self, tmp_path):
        path = tmp_path / "out.jsonl"
        rec = JSONLRecorder(path)
        original = make_record(
            record_id="full-001",
            model_version="v2",
            occupied_stalls=[1, 3],
            conditions={"key": "value"},
            reasoning_present=True,
            extracted_reasoning="because",
            response_tokens=42,
            latency_ms=150,
        )
        rec.record(original)
        read_back = rec.read_all()
        assert len(read_back) == 1
        result = read_back[0]
        assert result.record_id == "full-001"
        assert result.experiment_phase == ExperimentPhase.PHASE1
        assert result.experiment_group == "1.1"
        assert result.model_name == "test-model"
        assert result.model_version == "v2"
        assert result.temperature == 0.0
        assert result.prompt_template == PromptTemplate.A
        assert result.prompt_text == "test prompt"
        assert result.num_stalls == 5
        assert result.occupied_stalls == [1, 3]
        assert result.conditions == {"key": "value"}
        assert result.raw_response == "I choose stall 3"
        assert result.extracted_choice == 3
        assert result.choice_status == ChoiceStatus.VALID
        assert result.reasoning_present is True
        assert result.extracted_reasoning == "because"
        assert result.response_tokens == 42
        assert result.latency_ms == 150
        assert result.timestamp == datetime(2025, 1, 1, 12, 0, 0)

    def test_count_returns_correct_number(self, tmp_path):
        path = tmp_path / "out.jsonl"
        rec = JSONLRecorder(path)
        assert rec.count() == 0
        rec.record(make_record())
        assert rec.count() == 1
        rec.record(make_record(record_id="r-2"))
        assert rec.count() == 2

    def test_read_all_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        rec = JSONLRecorder(path)
        assert rec.read_all() == []

    def test_count_nonexistent_returns_zero(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        rec = JSONLRecorder(path)
        assert rec.count() == 0

    def test_clear_deletes_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        rec = JSONLRecorder(path)
        rec.record(make_record())
        assert path.exists()
        rec.clear()
        assert not path.exists()

    def test_init_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.jsonl"
        rec = JSONLRecorder(path)
        assert (tmp_path / "nested" / "deep").is_dir()
