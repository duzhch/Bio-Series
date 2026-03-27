#!/bin/bash

set -euo pipefail

CONFIG_FILE="${1:-config/local.yaml}"
DATASETS="${DATASETS:-LargeWhite_Pop1}"
TRAITS="${TRAITS:-BF}"
MODEL="${MODEL:-bio_master_v11}"
OUTPUT_SCRIPT="${OUTPUT_SCRIPT:-submit_cpu_test.sh}"

echo "=== Generating CPU-only test job ==="

python submit_jobs.py \
    --config "$CONFIG_FILE" \
    --datasets "$DATASETS" \
    --traits "$TRAITS" \
    --reps 1 \
    --model "$MODEL" \
    --out-sh "$OUTPUT_SCRIPT" \
    --cpu-only

echo "Generated CPU test script: $OUTPUT_SCRIPT"
echo "To run: ./$OUTPUT_SCRIPT"
