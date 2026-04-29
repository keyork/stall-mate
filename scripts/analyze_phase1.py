#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stall_mate.analysis.report import generate_phase1_report

output_dir = ROOT / "output" / "phase1_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

generate_phase1_report(
    experiment_data_dir=ROOT / "data",
    output_dir=output_dir,
)
