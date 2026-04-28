# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for Stall Mate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_jsonl_path(tmp_path: Path) -> Path:
    """Return a path to a temporary .jsonl file (not yet created)."""
    return tmp_path / "output.jsonl"


@pytest.fixture()
def sample_model_config_dict() -> dict:
    """Return a minimal model configuration dict for testing."""
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1024,
    }


@pytest.fixture()
def sample_experiment_config_dict() -> dict:
    """Return a minimal experiment configuration dict for testing."""
    return {
        "name": "smoke-test",
        "prompt_template": "stall_choice_v1",
        "models": ["gpt-4o-mini"],
        "num_repeats": 3,
        "output_dir": "data",
    }
