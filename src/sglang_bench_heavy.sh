#!/usr/bin/env bash
set -euo pipefail

API_KEY="EMPTY"
BASE_URL="http://10.255.252.103:30001"
MODEL="Qwen3-235B-A22B-Instruct-2507"

# Heavier sweep (be careful!)
python llm_bench.py   --base-url "${BASE_URL}"   --api-key "${API_KEY}"   --model "${MODEL}"   --endpoint chat.completions   --concurrency "8,16,32,64,96,128"   --requests-per-stage 100   --stream   --max-tokens 128   --prompt-file prompts_chat.json   --outdir "llm_bench_out_heavy"

python generate_report.py --indir "llm_bench_out_heavy"
echo "Open llm_bench_out_heavy/report.html"
