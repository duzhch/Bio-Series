#!/bin/bash

# Generate CPU-only jobs to avoid GPU conflicts
CONFIG_FILE="config/global_config.yaml"
DATASETS="LargeWhite_Pop1"
TRAITS="BF"
OUTPUT_SCRIPT="submit_cpu_test.sh"

echo "=== Generating CPU-only test job ==="

python submit_jobs.py \
    --config "$CONFIG_FILE" \
    --datasets "$DATASETS" \
    --traits "$TRAITS" \
    --reps 1 \
    --out-sh "$OUTPUT_SCRIPT" \
    --cpu-only

echo "Generated CPU test script: $OUTPUT_SCRIPT"
echo "To run: ./$OUTPUT_SCRIPT"