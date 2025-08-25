#!/usr/bin/env bash
set -euo pipefail

# -------- SGLang target (OpenAI-compatible) --------
API_KEY="EMPTY"
BASE_URL="http://10.255.252.103:30001"   # IMPORTANT: Do NOT include /v1, the bench adds it.
MODEL="Qwen3-235B-A22B-Instruct-2507"

# -------- Workload knobs --------
CONCURRENCY_SWEEP="1,2,4,8,16,32,64"
REQS_PER_STAGE=60
MAX_TOKENS=128
OUTDIR="llm_bench_out"

# -------- Run bench --------
python llm_bench.py   --base-url "${BASE_URL}"   --api-key "${API_KEY}"   --model "${MODEL}"   --endpoint chat.completions   --concurrency "${CONCURRENCY_SWEEP}"   --requests-per-stage ${REQS_PER_STAGE}   --stream   --max-tokens ${MAX_TOKENS}   --prompt-file prompts_chat.json   --outdir "${OUTDIR}"

# -------- Build report --------
python generate_report.py --indir "${OUTDIR}"

echo
echo "Done. Open ${OUTDIR}/report.html"
