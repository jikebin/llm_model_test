
# LLM Benchmark Pack (OpenAI-compatible)

This pack includes:

- `llm_bench.py` — async load tester that hits your OpenAI-compatible endpoint and measures:
  - QPS and latency (avg/p50/p95/p99)
  - TTFT (with `--stream`)
  - TPS (tokens/sec) using response `usage` fields
  - Basic saturation hint (best QPS and where latency/error start to rise)
  - Outputs: per-request CSVs, per-stage JSON summaries, combined CSV, overall summary

- `generate_report.py` — consumes the CSV/JSON to make **charts** and a **single HTML report**.

## Quick Start

```bash
pip install aiohttp numpy pandas matplotlib
python llm_bench.py \
  --base-url http://localhost:8000 \
  --api-key sk-xxx \
  --model your-model-name \
  --endpoint chat.completions \
  --concurrency 1,2,4,8,16,32 \
  --requests-per-stage 50 \
  --stream \
  --user "Write a one-sentence fact about Oklahoma."

python generate_report.py --indir llm_bench_out
# Open llm_bench_out/report.html
```

## 关键参数说明：

* --concurrency：并发扫参（支持混合：1,2,4-32）
* --requests-per-stage：每个并发等级的请求数
* --stream：开启流式 → 才能测 TTFT（首 token 时间）
* --endpoint：chat.completions（默认）或 responses
* --prompt-file：自定义消息（JSON/JSONL），否则用 --system/--user 生成简单对话
* --max-tokens：限制生成长度（可控 TPS/QPS）

脚本会输出到 ./llm_bench_out/：
* requests_cX.csv：每个并发等级的逐请求明细
* summary_cX.json：每级并发的统计摘要
* summary_all.csv：所有并发的汇总
* overall_summary.json：最佳 QPS、饱和并发位点等总览


## 生成图表 + HTML 报告

```bash
python generate_report.py --indir llm_bench_out
# 打开 llm_bench_out/report.htmls
```

### 报告包含：

- Latency vs Concurrency（p50/p95/p99）
- QPS vs Concurrency
- TTFT P99 vs Concurrency（流式时）
- TPS vs Concurrency（基于 usage.completion_tokens）
- 汇总表格与 JSON 总览（便于机器读取）

### 指标口径一览

* QPS：成功请求数 / 阶段总耗时
* 平均/分位延迟：逐请求总耗时（请求→完整响应）
* TTFT（需要 --stream）：POST 发出到第一块流事件到达的时间
* TPS：Σ(completion_tokens) / Σ(生成时间)
* 生成时间约等于（每次的 latency - TTFT，无流式时用 latency 近似）

注：脚本默认从响应的 usage 字段读取 token 统计（OpenAI 兼容规范）。请确保你的服务返回 usage.total_tokens / prompt_tokens / completion_tokens。若没有，TPS 会不可用。

### 如何判断性能瓶颈 & 找最佳并发

1. 打开 report.html 看两张图：
    * QPS vs Concurrency：QPS 随并发上升到一个峰值后趋平或下降 → 峰值附近就是最佳并发。
    * Latency p99 vs Concurrency：若 p99 随并发剧烈上升，同时 QPS 不再增长 → 已过饱和。
2. 再核对 summary_all.csv / overall_summary.json：
    * best_qps_at_concurrency 给出峰值点
    * 对应 error_rate 是否开始抬头、latency_p99 是否猛升
3. 调参建议（经验法则）：
    * 如果 QPS 没达预期 且 p99 很早就抬头：优先排查服务端瓶颈（GPU/NCCL、批量化、KV Cache、并发队列、限流）。
    * 如果 TPS 偏低：确认模型是否限制 max_tokens，或服务端是否开启了压缩/流控、中间件复制/JSON 序列化开销。
    * 如果 TTFT 高 但生成期很快：检查首包路径（调度、排队、prompt 编码、首轮解码设置、prefill batch）。

### 常见对接提示

* 你的 OpenAI 兼容服务若是 /v1/responses，运行时用 --endpoint responses。
* 流式响应需以 data: {json}\n\n 形式逐块推送并以 data: [DONE] 结尾（多数兼容实现如此）。
* 若你启用了服务端并批（例如 sglang/vLLM 的 continuous batching），并发扫参能很好体现吞吐曲线。
* 建议在服务端同时记录：
    * 入队/出队/调度耗时
    * 首 token 解码时间
    * 生成 token 总数与平均 token/s（服务端视角）
    * GPU 利用率/显存/显存碎片（便于定位瓶颈）

### Notes

- **TTFT** is measured only in streaming mode (`--stream`) as the time until the first streamed chunk arrives.
- **TPS** uses the server's `usage.completion_tokens`. Ensure your server returns `usage`.
- If you use `/v1/responses`, set `--endpoint responses`.
- Put a custom conversation in a JSON/JSONL with `--prompt-file` if desired.


## 测试脚本

### 快速冒烟（1 分钟）

```bash
bash sglang_bench_quick.sh
# 生成 llm_bench_out_quick/report.html
```

### 标准全量

```bash
bash sglang_bench_run.sh
# 并发扫参：1,2,4,8,16,32,64；每级 60 次；max_tokens=128；开启流式测 TTFT
# 生成 llm_bench_out/report.html（含 QPS/延迟/TTFT/TPS 图表与表格）
```

### 重压慎用

```bash
bash sglang_bench_heavy.sh
# 并发：8,16,32,64,96,128；每级 100 次；可能把服务推到瓶颈
```

### 结果解读（与瓶颈判断）

- QPS vs Concurrency：出现峰值后趋平/下降 → 峰值附近就是最佳并发
- Latency p99 vs Concurrency：p99 随并发陡增且 QPS 不再提升 → 已饱和
- TTFT（流式首包时间）高但生成期很快 → 首包路径/排队/编码/首轮解码可能是瓶颈
- TPS 低 → 确认服务端是否返回 usage.completion_tokens；检查生成长度与批处理

报告文件夹里有：逐请求明细 CSV、各并发摘要 JSON、总汇 CSV、report.html 可直接与同事共享。

### 针对 SGLang 的小提示

- 你的 endpoint 走 /v1/chat/completions（脚本默认），流式对 TTFT 的测量更准确
- 若你开启了服务端 continuous batching/batching queue，图里的 QPS 峰值更明显
- 生成长度建议 max_tokens=128 起步，保证可比性；想拉满 TPS 再逐步增大
- 如果你在服务端开启了 KV cache 压缩/低精度或批处理上限，请记下配置，便于对比不同配置下的 QPS/延迟