# SPDX-License-Identifier: Apache-2.0
"""ExperimentRunner 测试 | ExperimentRunner tests with mocked dependencies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stall_mate.config import ExperimentConfig, ModelConfig, PromptTemplateConfig
from stall_mate.runner import ExperimentRunner, RunStats
from stall_mate.schema import StallChoice
from stall_mate.types import ChoiceStatus


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def recorder() -> MagicMock:
    return MagicMock()


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(name="test-model", endpoint="http://localhost:1234", version="v1")


@pytest.fixture
def runner(client: MagicMock, recorder: MagicMock, model_config: ModelConfig) -> ExperimentRunner:
    return ExperimentRunner(client=client, recorder=recorder, model_config=model_config)


# ------------------------------------------------------------------
# _classify_response
# ------------------------------------------------------------------


class TestClassifyResponse:
    def test_classify_valid(self, runner: ExperimentRunner):
        assert runner._classify_response("I choose stall 3", 3, 5) == ChoiceStatus.VALID

    def test_classify_refused(self, runner: ExperimentRunner):
        assert runner._classify_response("I cannot help with that", None, 5) == ChoiceStatus.REFUSED

    def test_classify_refused_chinese(self, runner: ExperimentRunner):
        assert runner._classify_response("我无法回答这个问题", None, 5) == ChoiceStatus.REFUSED

    def test_classify_ambiguous(self, runner: ExperimentRunner):
        assert runner._classify_response("It depends on your preference", None, 5) == ChoiceStatus.AMBIGUOUS


# ------------------------------------------------------------------
# _extract_choice_from_text
# ------------------------------------------------------------------


class TestExtractChoiceFromText:
    def test_extract_chinese_patterns(self, runner: ExperimentRunner):
        assert runner._extract_choice_from_text("我选择第3个坑位", 5) == 3
        assert runner._extract_choice_from_text("3号坑位比较好", 5) == 3
        assert runner._extract_choice_from_text("我会选择5号", 5) == 5

    def test_extract_no_match(self, runner: ExperimentRunner):
        assert runner._extract_choice_from_text("随便选一个吧", 5) is None

    def test_extract_out_of_range(self, runner: ExperimentRunner):
        assert runner._extract_choice_from_text("我选第6个", 5) is None


# ------------------------------------------------------------------
# run_single
# ------------------------------------------------------------------


class TestRunSingle:
    def test_run_single_structured_ok(self, runner: ExperimentRunner, client: MagicMock, recorder: MagicMock):
        parsed = StallChoice(
            chosen_stall=2,
            chain_of_thought="中间位置比较安全，避免两端",
            confidence=0.85,
        )
        client.query_structured.return_value = (parsed, parsed.model_dump_json(), 50, 120)

        record = runner.run_single(
            prompt="Choose a stall with {num_stalls} stalls",
            system_message="system",
            temperature=0.0,
            num_stalls=5,
            metadata={
                "experiment_phase": "Phase1",
                "experiment_group": "G1",
                "prompt_template": "A",
                "prompt_text": "Choose a stall with 5 stalls",
                "num_stalls": 5,
                "conditions": {},
                "occupied_stalls": [],
            },
        )

        assert record.choice_status == ChoiceStatus.VALID
        assert record.extracted_choice == 2
        assert record.reasoning_present is True
        assert record.extracted_reasoning == parsed.chain_of_thought
        assert record.model_name == "test-model"
        assert record.response_tokens == 50
        assert record.latency_ms == 120
        recorder.record.assert_called_once_with(record)

    def test_run_single_plain_fallback(self, runner: ExperimentRunner, client: MagicMock, recorder: MagicMock):
        client.query_structured.return_value = (None, "我觉得第3个比较好", 10, 200)

        record = runner.run_single(
            prompt="Choose a stall",
            system_message="system",
            temperature=0.7,
            num_stalls=5,
            metadata={
                "experiment_phase": "Phase1",
                "experiment_group": "G2",
                "prompt_template": "B",
                "prompt_text": "Choose a stall",
                "num_stalls": 5,
                "conditions": {},
                "occupied_stalls": [],
            },
        )

        assert record.extracted_choice == 3
        assert record.choice_status == ChoiceStatus.VALID
        assert record.reasoning_present is False
        assert record.raw_response == "我觉得第3个比较好"
        recorder.record.assert_called_once()

    def test_run_single_api_error(self, runner: ExperimentRunner, client: MagicMock, recorder: MagicMock):
        client.query_structured.return_value = (None, "ConnectionError: Connection refused", 0, 0)

        record = runner.run_single(
            prompt="Choose a stall",
            system_message="system",
            temperature=0.0,
            num_stalls=5,
            metadata={
                "experiment_phase": "Phase1",
                "experiment_group": "G3",
                "prompt_template": "A",
                "prompt_text": "Choose a stall",
            },
        )

        assert record.choice_status == ChoiceStatus.ERROR
        assert record.extracted_choice is None
        assert "ConnectionError" in record.raw_response
        recorder.record.assert_called_once()

    def test_run_single_exception_caught(self, runner: ExperimentRunner, client: MagicMock, recorder: MagicMock):
        client.query_structured.side_effect = RuntimeError("unexpected boom")

        record = runner.run_single(
            prompt="Choose a stall",
            system_message="system",
            temperature=0.0,
            num_stalls=5,
            metadata={
                "experiment_phase": "Phase1",
                "experiment_group": "G4",
                "prompt_template": "A",
                "prompt_text": "Choose a stall",
            },
        )

        assert record.choice_status == ChoiceStatus.ERROR
        assert "RuntimeError" in record.raw_response


# ------------------------------------------------------------------
# run_experiment
# ------------------------------------------------------------------


class TestRunExperiment:
    def test_run_experiment_count(self, runner: ExperimentRunner, client: MagicMock):
        experiment_config = ExperimentConfig(
            experiment_id="exp1",
            experiment_group="G1",
            phase="Phase1",
            description="test",
            num_stalls=[3, 5],
            temperatures=[0.0, 0.7],
            templates=["A", "B"],
            repetitions=2,
        )
        templates = PromptTemplateConfig(
            templates={
                "A": "Pick from {num_stalls} stalls",
                "B": "Select one of {num_stalls} stalls",
            }
        )

        parsed = StallChoice(
            chosen_stall=1,
            chain_of_thought="reasoning text here that is long enough",
            confidence=0.5,
        )
        client.query_structured.return_value = (parsed, "{}", 0, 0)

        stats = runner.run_experiment(experiment_config, templates)

        expected_count = 2 * 2 * 2 * 2  # num_stalls × temps × templates × reps
        assert isinstance(stats, RunStats)
        assert stats.total_calls == expected_count
        assert stats.valid == expected_count
        assert stats.error == 0
        assert client.query_structured.call_count == expected_count

    def test_run_experiment_retries_on_error(self, runner: ExperimentRunner, client: MagicMock):
        experiment_config = ExperimentConfig(
            experiment_id="exp2",
            experiment_group="G2",
            phase="Phase1",
            description="retry test",
            num_stalls=[3],
            temperatures=[0.0],
            templates=["A"],
            repetitions=1,
        )
        templates = PromptTemplateConfig(
            templates={"A": "Pick from {num_stalls} stalls"},
        )

        parsed = StallChoice(
            chosen_stall=2,
            chain_of_thought="reasoning text here that is long enough",
            confidence=0.5,
        )
        # First call fails, second succeeds
        client.query_structured.side_effect = [
            (None, "ConnectionError: timeout", 0, 0),
            (parsed, "{}", 0, 0),
        ]

        stats = runner.run_experiment(experiment_config, templates, max_retries=3)

        assert isinstance(stats, RunStats)
        assert stats.total_calls == 2  # 1 error + 1 success
        assert stats.error == 1  # first attempt error
        assert stats.valid == 1  # retry success
        assert stats.retries_used == 1  # 1 task retried

    def test_run_experiment_exhausts_retries(self, runner: ExperimentRunner, client: MagicMock):
        experiment_config = ExperimentConfig(
            experiment_id="exp3",
            experiment_group="G3",
            phase="Phase1",
            description="exhaust retry test",
            num_stalls=[3],
            temperatures=[0.0],
            templates=["A"],
            repetitions=1,
        )
        templates = PromptTemplateConfig(
            templates={"A": "Pick from {num_stalls} stalls"},
        )

        # All calls fail
        client.query_structured.return_value = (None, "ConnectionError: timeout", 0, 0)

        stats = runner.run_experiment(experiment_config, templates, max_retries=2)

        # 1 original + 2 retries = 3 error records
        assert stats.total_calls == 3
        assert stats.error == 3
        assert stats.valid == 0
        assert stats.retries_used == 2  # 1 task retried twice (rounds 1 and 2)
        assert client.query_structured.call_count == 3
