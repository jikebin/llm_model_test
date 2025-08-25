#!/usr/bin/env bash
set -euo pipefail

API_KEY="EMPTY"
# BASE_URL="http://10.255.252.103:30001"
# MODEL="Qwen3-235B-A22B-Instruct-2507"
BASE_URL = "http://10.255.252.24:30000"
MODEL = "Qwen3-235B-A22B-Instruct-2507-AWQ"

# Quick sanity smoke test
python llm_bench.py   --base-url "${BASE_URL}"   --api-key "${API_KEY}"   --model "${MODEL}"   --endpoint chat.completions   --concurrency "1,2,4"   --requests-per-stage 10   --stream   --max-tokens 64   --prompt-file prompts_chat.json   --outdir "llm_bench_out_quick"

python generate_report.py --indir "llm_bench_out_quick"
echo "Open llm_bench_out_quick/report.html"
