#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Phase 1 全量实验执行脚本
# 运行 configs/experiments/ 下的所有 Phase1 实验配置
#
# 用法:
#   ./scripts/run_phase1.sh              # 运行全部实验
#   ./scripts/run_phase1.sh --dry-run    # 仅显示实验计划，不实际执行

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "========================================"
echo "  Stall Mate — Phase 1 全量实验"
echo "========================================"
echo ""

# 检查依赖
if ! command -v uv &>/dev/null; then
    echo "错误: 未找到 uv，请先安装 (https://docs.astral.sh/uv/)"
    exit 1
fi

# 检查 API 是否可达
ENDPOINT=$(uv run python -c "
import yaml
with open('$ROOT/configs/models.yaml') as f:
    print(yaml.safe_load(f)['models'][0]['endpoint'])
" 2>/dev/null)

if [[ -z "$ENDPOINT" ]]; then
    echo "错误: 无法读取 models.yaml 中的 endpoint"
    exit 1
fi

echo "API 端点: $ENDPOINT"
echo ""

# 计算各实验的调用量
echo "----------------------------------------"
echo "  实验计划"
echo "----------------------------------------"

TOTAL=0
while IFS= read -r line; do
    echo "  $line"
    COUNT=$(echo "$line" | grep -oP '\d+(?= 次$)' || true)
    if [[ -n "$COUNT" ]]; then
        TOTAL=$((TOTAL + COUNT))
    fi
done < <(uv run python -c "
from pathlib import Path
from stall_mate.config import discover_experiments

ROOT = Path('$ROOT')
configs = discover_experiments(ROOT / 'configs' / 'experiments')
total = 0
for cfg in configs:
    calls = len(cfg.num_stalls) * len(cfg.temperatures) * len(cfg.templates) * cfg.repetitions
    print(f'{cfg.experiment_id}: {cfg.description}')
    print(f'  N={cfg.num_stalls}, T={cfg.temperatures}, 模板={cfg.templates}, 重复={cfg.repetitions}')
    print(f'  调用次数: {calls} 次')
    total += calls
print(f'')
print(f'总计: {total} 次调用')
print(f'预估耗时: {int(total * 20 / 3600)}~{int(total * 30 / 3600)} 小时（按每次 20~30 秒估算）')
" 2>/dev/null)

echo "----------------------------------------"
echo ""

if $DRY_RUN; then
    echo "--dry-run 模式，仅显示计划，不执行。"
    exit 0
fi

# 确认执行
read -p "确认开始全量实验? (y/N) " -r
echo ""
if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "已取消。"
    exit 0
fi

# 执行
echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') 开始执行..."
echo ""

uv run python -c "
import sys
import time
from pathlib import Path

ROOT = Path('$ROOT')
sys.path.insert(0, str(ROOT / 'src'))

from stall_mate.client import LLMClient
from stall_mate.config import (
    discover_experiments,
    load_model_config,
    load_prompt_templates,
    load_classification_config,
)
from stall_mate.recorder import JSONLRecorder
from stall_mate.runner import ExperimentRunner

model_cfg = load_model_config(ROOT / 'configs' / 'models.yaml')
templates = load_prompt_templates(ROOT / 'configs' / 'prompt_templates' / 'phase1.yaml')
classification_cfg = load_classification_config(ROOT / 'configs' / 'classification.yaml')
experiment_configs = discover_experiments(ROOT / 'configs' / 'experiments')

client = LLMClient(
    endpoint=model_cfg.endpoint,
    model=model_cfg.name,
    api_key=model_cfg.api_key,
    timeout=model_cfg.timeout,
    max_retries=model_cfg.max_retries,
    probe_message=model_cfg.probe_message,
)

total_calls = 0
total_valid = 0
total_refused = 0
total_ambiguous = 0
start_time = time.time()

for cfg in experiment_configs:
    output_path = ROOT / 'data' / f'phase1_{cfg.experiment_id}.jsonl'
    recorder = JSONLRecorder(output_path)
    runner = ExperimentRunner(
        client=client,
        recorder=recorder,
        model_config=model_cfg,
        refusal_keywords=classification_cfg.refusal_keywords,
        extraction_patterns=classification_cfg.to_extraction_patterns(),
    )

    calls_in_exp = len(cfg.num_stalls) * len(cfg.temperatures) * len(cfg.templates) * cfg.repetitions
    exp_start = time.time()
    print(f'[{cfg.experiment_id}] {cfg.description}')
    print(f'  预计 {calls_in_exp} 次调用...')

    records = runner.run_experiment(cfg, templates)

    exp_elapsed = time.time() - exp_start
    valid = sum(1 for r in records if r.choice_status.value == 'VALID')
    refused = sum(1 for r in records if r.choice_status.value == 'REFUSED')
    ambiguous = sum(1 for r in records if r.choice_status.value == 'AMBIGUOUS')
    avg_latency = sum(r.latency_ms for r in records) / len(records) if records else 0

    total_calls += len(records)
    total_valid += valid
    total_refused += refused
    total_ambiguous += ambiguous

    print(f'  完成: {len(records)} 次调用, 耗时 {exp_elapsed:.0f}s')
    print(f'  VALID={valid}, REFUSED={refused}, AMBIGUOUS={ambiguous}, 平均延迟={avg_latency:.0f}ms')
    print(f'  输出: {output_path}')
    print()

elapsed = time.time() - start_time
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)

print('========================================')
print('  全量实验完成')
print('========================================')
print(f'  总调用: {total_calls} 次')
print(f'  VALID: {total_valid}, REFUSED: {total_refused}, AMBIGUOUS: {total_ambiguous}')
print(f'  总耗时: {hours}h {minutes}m {seconds}s')
print(f'  数据目录: {ROOT / \"data\"}')
print('========================================')
"

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') 全量实验结束。"
