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
