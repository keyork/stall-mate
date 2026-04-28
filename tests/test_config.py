# SPDX-License-Identifier: Apache-2.0
"""配置加载测试 | Tests for stall_mate.config."""

from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from stall_mate.config import (
    ClassificationConfig,
    ExperimentConfig,
    ModelConfig,
    PromptTemplateConfig,
    discover_experiments,
    load_classification_config,
    load_experiment_config,
    load_model_config,
    load_prompt_templates,
    load_yaml,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

VALID_MODELS_YAML = {
    "models": [
        {
            "name": "gpt-4o",
            "endpoint": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "version": "2024-08-06",
        }
    ]
}

VALID_EXPERIMENT_YAML = {
    "experiment_id": "exp-001",
    "experiment_group": "baseline",
    "phase": "phase-1",
    "description": "Basic stall choice test",
    "num_stalls": [3, 5],
    "temperatures": [0.0, 0.7, 1.0],
    "templates": ["A", "B"],
    "repetitions": 30,
    "conditions": {"scenario": "default"},
    "occupied_stalls": [2],
}

VALID_TEMPLATES_YAML = {
    "templates": {
        "A": "You walk into a bathroom with {n} stalls...",
        "B": "There are {n} stalls in front of you...",
    }
}


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return path


# ------------------------------------------------------------------
# load_yaml
# ------------------------------------------------------------------


class TestLoadYaml:
    def test_reads_valid_yaml(self, tmp_path: Path):
        p = _write_yaml(tmp_path / "a.yaml", {"key": "value"})
        assert load_yaml(p) == {"key": "value"}

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert load_yaml(p) == {}

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_yaml(tmp_path / "nope.yaml")


# ------------------------------------------------------------------
# load_model_config
# ------------------------------------------------------------------


class TestLoadModelConfig:
    def test_valid_model(self, tmp_path: Path):
        p = _write_yaml(tmp_path / "models.yaml", VALID_MODELS_YAML)
        cfg = load_model_config(p)
        assert isinstance(cfg, ModelConfig)
        assert cfg.name == "gpt-4o"
        assert cfg.endpoint == "https://api.openai.com/v1"
        assert cfg.api_key == "sk-test"
        assert cfg.version == "2024-08-06"

    def test_defaults(self, tmp_path: Path):
        data = {"models": [{"name": "test", "endpoint": "http://localhost"}]}
        p = _write_yaml(tmp_path / "models.yaml", data)
        cfg = load_model_config(p)
        assert cfg.api_key == ""
        assert cfg.version == "unknown"

    def test_missing_models_key(self, tmp_path: Path):
        p = _write_yaml(tmp_path / "models.yaml", {"other": []})
        with pytest.raises(ValueError, match="No models list found"):
            load_model_config(p)

    def test_empty_models_list(self, tmp_path: Path):
        p = _write_yaml(tmp_path / "models.yaml", {"models": []})
        with pytest.raises(ValueError, match="No models list found"):
            load_model_config(p)


# ------------------------------------------------------------------
# load_experiment_config
# ------------------------------------------------------------------


class TestLoadExperimentConfig:
    def test_valid_experiment(self, tmp_path: Path):
        p = _write_yaml(tmp_path / "exp.yaml", VALID_EXPERIMENT_YAML)
        cfg = load_experiment_config(p)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.experiment_id == "exp-001"
        assert cfg.num_stalls == [3, 5]
        assert cfg.temperatures == [0.0, 0.7, 1.0]
        assert cfg.repetitions == 30
        assert cfg.conditions == {"scenario": "default"}
        assert cfg.occupied_stalls == [2]

    def test_defaults(self, tmp_path: Path):
        minimal = {
            "experiment_id": "exp-002",
            "experiment_group": "g",
            "phase": "p1",
            "num_stalls": [3],
            "temperatures": [0.5],
            "templates": ["A"],
        }
        p = _write_yaml(tmp_path / "exp.yaml", minimal)
        cfg = load_experiment_config(p)
        assert cfg.description == ""
        assert cfg.repetitions == 30
        assert cfg.conditions == {}
        assert cfg.occupied_stalls == []

    def test_missing_required_field_raises(self, tmp_path: Path):
        bad = {"experiment_id": "x"}  # missing required fields
        p = _write_yaml(tmp_path / "bad.yaml", bad)
        with pytest.raises(ValidationError):
            load_experiment_config(p)


# ------------------------------------------------------------------
# load_prompt_templates
# ------------------------------------------------------------------


class TestLoadPromptTemplates:
    def test_valid_templates(self, tmp_path: Path):
        p = _write_yaml(tmp_path / "tpl.yaml", VALID_TEMPLATES_YAML)
        cfg = load_prompt_templates(p)
        assert isinstance(cfg, PromptTemplateConfig)
        assert "A" in cfg.templates
        assert "{n}" in cfg.templates["A"]

    def test_missing_templates_key_raises(self, tmp_path: Path):
        p = _write_yaml(tmp_path / "tpl.yaml", {"other": {}})
        with pytest.raises(ValidationError):
            load_prompt_templates(p)


# ------------------------------------------------------------------
# discover_experiments
# ------------------------------------------------------------------


class TestDiscoverExperiments:
    def test_discovers_multiple_experiments(self, tmp_path: Path):
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()
        for i in range(3):
            data = {
                "experiment_id": f"exp-{i:03d}",
                "experiment_group": "g",
                "phase": "p1",
                "num_stalls": [3],
                "temperatures": [0.5],
                "templates": ["A"],
            }
            _write_yaml(exp_dir / f"exp_{i}.yaml", data)

        configs = discover_experiments(exp_dir)
        assert len(configs) == 3
        assert [c.experiment_id for c in configs] == [
            "exp-000",
            "exp-001",
            "exp-002",
        ]

    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        exp_dir = tmp_path / "empty"
        exp_dir.mkdir()
        assert discover_experiments(exp_dir) == []

    def test_nonexistent_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Config directory not found"):
            discover_experiments(tmp_path / "nope")

    def test_ignores_non_yaml_files(self, tmp_path: Path):
        exp_dir = tmp_path / "mixed"
        exp_dir.mkdir()
        data = {
            "experiment_id": "exp-100",
            "experiment_group": "g",
            "phase": "p1",
            "num_stalls": [3],
            "temperatures": [0.5],
            "templates": ["A"],
        }
        _write_yaml(exp_dir / "valid.yaml", data)
        (exp_dir / "notes.txt").write_text("not yaml")
        configs = discover_experiments(exp_dir)
        assert len(configs) == 1
        assert configs[0].experiment_id == "exp-100"


# ------------------------------------------------------------------
# ModelConfig new fields
# ------------------------------------------------------------------


class TestModelConfigNewFields:
    def test_new_defaults(self, tmp_path: Path):
        data = {"models": [{"name": "test", "endpoint": "http://localhost"}]}
        p = _write_yaml(tmp_path / "models.yaml", data)
        cfg = load_model_config(p)
        assert cfg.timeout == 60
        assert cfg.max_retries == 2
        assert cfg.probe_message == "Say OK"

    def test_new_fields_from_yaml(self, tmp_path: Path):
        data = {
            "models": [{
                "name": "test",
                "endpoint": "http://localhost",
                "timeout": 120,
                "max_retries": 5,
                "probe_message": "Ping",
            }]
        }
        p = _write_yaml(tmp_path / "models.yaml", data)
        cfg = load_model_config(p)
        assert cfg.timeout == 120
        assert cfg.max_retries == 5
        assert cfg.probe_message == "Ping"


# ------------------------------------------------------------------
# load_classification_config
# ------------------------------------------------------------------


class TestLoadClassificationConfig:
    def test_valid_classification(self, tmp_path: Path):
        data = {
            "refusal_keywords": ["拒绝", "refuse"],
            "chinese_patterns": [r"第\s*(\d+)\s*个"],
            "english_patterns": [r"stall\s*(\d+)"],
            "trailing_digit_pattern": r"(\d+)$",
            "general_digit_pattern": r"\b(\d+)\b",
            "direction_reversal": [
                {"source": "左", "target": "右"},
            ],
        }
        p = _write_yaml(tmp_path / "classification.yaml", data)
        cfg = load_classification_config(p)
        assert isinstance(cfg, ClassificationConfig)
        assert cfg.refusal_keywords == ["拒绝", "refuse"]
        assert cfg.chinese_patterns == [r"第\s*(\d+)\s*个"]
        assert cfg.direction_reversal[0].source == "左"

    def test_defaults(self, tmp_path: Path):
        p = _write_yaml(tmp_path / "classification.yaml", {})
        cfg = load_classification_config(p)
        assert isinstance(cfg, ClassificationConfig)
        assert "无法" in cfg.refusal_keywords
        assert len(cfg.chinese_patterns) >= 1

    def test_empty_file_uses_defaults(self, tmp_path: Path):
        p = tmp_path / "classification.yaml"
        p.write_text("")
        cfg = load_classification_config(p)
        assert isinstance(cfg, ClassificationConfig)
        assert len(cfg.refusal_keywords) > 0

    def test_to_extraction_patterns(self, tmp_path: Path):
        data = {
            "chinese_patterns": [r"第(\d+)个"],
            "english_patterns": [r"stall (\d+)"],
            "trailing_digit_pattern": r"(\d+)$",
            "general_digit_pattern": r"\b(\d+)\b",
        }
        p = _write_yaml(tmp_path / "classification.yaml", data)
        cfg = load_classification_config(p)
        patterns = cfg.to_extraction_patterns()
        assert patterns["chinese_patterns"] == [r"第(\d+)个"]
        assert patterns["trailing_digit_pattern"] == r"(\d+)$"

    def test_to_reversal_pairs(self, tmp_path: Path):
        data = {
            "direction_reversal": [
                {"source": "A", "target": "B"},
                {"source": "C", "target": "D"},
            ],
        }
        p = _write_yaml(tmp_path / "classification.yaml", data)
        cfg = load_classification_config(p)
        pairs = cfg.to_reversal_pairs()
        assert pairs == [{"source": "A", "target": "B"}, {"source": "C", "target": "D"}]


# ------------------------------------------------------------------
# PromptTemplateConfig system_message_template
# ------------------------------------------------------------------


class TestSystemMessageTemplate:
    def test_default_system_message(self, tmp_path: Path):
        data = {"templates": {"A": "Pick from {num_stalls}"}}
        p = _write_yaml(tmp_path / "tpl.yaml", data)
        cfg = load_prompt_templates(p)
        assert "{num_stalls}" in cfg.system_message_template

    def test_custom_system_message(self, tmp_path: Path):
        data = {
            "templates": {"A": "Pick"},
            "system_message_template": "Custom {num_stalls} msg",
        }
        p = _write_yaml(tmp_path / "tpl.yaml", data)
        cfg = load_prompt_templates(p)
        assert cfg.system_message_template == "Custom {num_stalls} msg"
