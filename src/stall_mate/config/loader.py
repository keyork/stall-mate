# SPDX-License-Identifier: Apache-2.0
"""
配置加载工具 | Configuration loading utilities.

通过 Pydantic 模型定义和 YAML 文件加载实验配置。
Defines experiment configuration via Pydantic models and loads from YAML files.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic configuration models / Pydantic 配置模型
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """大语言模型配置 | LLM model configuration."""

    name: str
    endpoint: str
    api_key: str = ""
    version: str = "unknown"
    timeout: int = 60
    max_retries: int = 2
    probe_message: str = "Say OK"


class ExperimentConfig(BaseModel):
    """实验配置 | Experiment configuration."""

    experiment_id: str
    experiment_group: str
    phase: str
    description: str = ""
    num_stalls: list[int]
    temperatures: list[float]
    templates: list[str]
    repetitions: int = 30
    conditions: dict = {}
    occupied_stalls: list[int] = []


class PromptTemplateConfig(BaseModel):
    """提示词模板配置 | Prompt template configuration."""

    templates: dict[str, str]
    system_message_template: str = (
        "你是一个正在选择公共厕所隔间的人。请用 JSON 格式回复。\n"
        "You are choosing a toilet stall. Respond in JSON format.\n\n"
        "有效坑位范围 / Valid stall range: 1 to {num_stalls}\n"
        '回复格式 / Response format:\n'
        '{{"chosen_stall": <int>, "chain_of_thought": "<str>", "confidence": <float>}}'
    )


class DirectionReversalPair(BaseModel):
    """方向反转替换对 / Direction reversal string pair."""

    source: str
    target: str


class ClassificationConfig(BaseModel):
    """响应分类配置 / Response classification configuration."""

    refusal_keywords: list[str] = [
        "无法",
        "不能",
        "拒绝",
        "refuse",
        "cannot",
        "won't",
        "I can't",
        "inappropriate",
    ]
    chinese_patterns: list[str] = [
        r"第\s*(\d+)\s*个",
        r"(\d+)\s*号",
        r"选择.*?(\d+)",
    ]
    english_patterns: list[str] = [
        r"stall\s*(\d+)",
        r"number\s*(\d+)",
    ]
    trailing_digit_pattern: str = r"(\d+)\s*[。.!?]?\s*$"
    general_digit_pattern: str = r"\b(\d+)\b"
    direction_reversal: list[DirectionReversalPair] = [
        DirectionReversalPair(source="从左到右", target="从右到左"),
        DirectionReversalPair(source="from left to right", target="from right to left"),
    ]

    def to_extraction_patterns(self) -> dict[str, list[str] | str]:
        """转换为 ExperimentRunner 所需的 extraction_patterns 格式。
        Convert to extraction_patterns dict format expected by ExperimentRunner.
        """
        return {
            "chinese_patterns": self.chinese_patterns,
            "english_patterns": self.english_patterns,
            "trailing_digit_pattern": self.trailing_digit_pattern,
            "general_digit_pattern": self.general_digit_pattern,
        }

    def to_reversal_pairs(self) -> list[dict[str, str]]:
        """转换为 builder 所需的 reversal_pairs 格式。
        Convert to reversal_pairs list format expected by build_reverse_prompt.
        """
        return [{"source": p.source, "target": p.target} for p in self.direction_reversal]


# ---------------------------------------------------------------------------
# YAML loading helpers / YAML 加载辅助函数
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    """读取 YAML 文件并返回字典 | Read a YAML file and return a dict.

    Args:
        path: YAML 文件路径 | Path to the YAML file.

    Returns:
        解析后的字典 | Parsed dictionary.

    Raises:
        FileNotFoundError: 文件不存在 | File does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到 | Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_model_config(path: Path) -> ModelConfig:
    """从 YAML 加载模型配置 | Load model config from YAML.

    读取 ``models`` 键下的第一个元素并验证为 ModelConfig。
    Reads the first item under the ``models`` key and validates as ModelConfig.
    """
    data = load_yaml(path)
    models = data.get("models", [])
    if not models:
        raise ValueError(
            "models.yaml 中未找到 models 列表 | No models list found in models.yaml"
        )
    return ModelConfig.model_validate(models[0])


def load_experiment_config(path: Path) -> ExperimentConfig:
    """从 YAML 加载实验配置 | Load experiment config from YAML."""
    data = load_yaml(path)
    return ExperimentConfig.model_validate(data)


def load_prompt_templates(path: Path) -> PromptTemplateConfig:
    """从 YAML 加载提示词模板 | Load prompt templates from YAML."""
    data = load_yaml(path)
    return PromptTemplateConfig.model_validate(data)


def load_classification_config(path: Path) -> ClassificationConfig:
    """从 YAML 加载分类配置 | Load classification config from YAML."""
    data = load_yaml(path)
    if not data:
        return ClassificationConfig()
    return ClassificationConfig.model_validate(data)


def discover_experiments(config_dir: Path) -> list[ExperimentConfig]:
    """扫描目录中的 YAML 文件并加载所有实验配置。

    Glob for ``*.yaml`` in *config_dir* and load each as ExperimentConfig.
    """
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"配置目录未找到 | Config directory not found: {config_dir}"
        )
    configs: list[ExperimentConfig] = []
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        configs.append(load_experiment_config(yaml_file))
    return configs
