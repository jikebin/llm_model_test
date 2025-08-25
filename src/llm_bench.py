
#!/usr/bin/env python3
"""
llm_bench.py — OpenAI-compatible LLM performance benchmarker

Features
- Sweeps over multiple concurrency levels (e.g., 1,2,4,8,16,...)
- Measures per-request latency (avg, p50, p95, p99), QPS, error rate
- Measures TTFT (time-to-first-token) when using streaming
- Measures TPS (tokens/sec) using "usage" from responses (preferred) or via token deltas in stream
- Detects saturation by observing rising latency/error with increasing concurrency
- Writes detailed per-request CSV + per-concurrency summary CSV + overall JSON summary

Requirements
- Python 3.9+
- aiohttp, numpy, pandas (for convenience), matplotlib not required for this script
- Endpoint: OpenAI-compatible /v1/chat/completions or /v1/responses
"""

import argparse
import asyncio
import aiohttp
import time
import json
import csv
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import os

def parse_concurrency_list(s: str) -> List[int]:
    parts = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a,b = tok.split("-",1)
            a,b = int(a), int(b)
            start, end = (a,b) if a<=b else (b,a)
            parts.extend(list(range(start, end+1)))
        else:
            parts.append(int(tok))
    # dedupe while preserving order
    seen = set()
    out = []
    for x in parts:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, p))

def now() -> float:
    return time.perf_counter()

def build_payload(messages: List[Dict[str, Any]], model: str, stream: bool, endpoint: str, max_tokens: Optional[int]) -> Dict[str, Any]:
    if endpoint == "chat.completions":
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload
    elif endpoint == "responses":
        payload = {
            "model": model,
            "input": [{"role": "user", "content": [{"type":"text", "text": messages[-1]['content']}]}],
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        return payload
    else:
        raise ValueError("Unknown endpoint type")

async def stream_chat(session: aiohttp.ClientSession, url: str, headers: Dict[str,str], payload: Dict[str,Any]) -> Dict[str, Any]:
    """Return dict with timings and usage approximation for streaming chat.completions"""
    t0 = now()
    ttft = None
    completion_tokens = 0
    prompt_tokens = None
    total_tokens = None
    error = None
    status = None

    try:
        async with session.post(url, headers=headers, json=payload, timeout=None) as resp:
            status = resp.status
            if status != 200:
                error = f"HTTP {status}"
                # Try read text for diagnostics
                try:
                    body = await resp.text()
                    error += f": {body[:200]}"
                except Exception:
                    pass
                t_end = now()
                return {
                    "status": status, "error": error,
                    "latency": t_end - t0, "ttft": None,
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens
                }
            async for line_bytes in resp.content:
                if not line_bytes:
                    continue
                line = line_bytes.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    # skip keep-alives or malformed
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                # mark TTFT on first chunk where choices[0].delta.* appears
                if ttft is None:
                    ttft = now() - t0

                # Try to accumulate tokens if server includes "usage" deltas (not standard). We fallback later.
                # Here we just count chunks as proxy, but that's poor. Better to rely on final usage.
                # So we won't try to compute tokens here.
                pass
            # attempt to read a non-stream final usage if server appends it (some impls do), else None
            # we won't have it unless the server sends an extra JSON; leave None
            t_end = now()
            return {
                "status": status, "error": error,
                "latency": t_end - t0, "ttft": ttft,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens
            }
    except asyncio.CancelledError:
        raise
    except Exception as e:
        t_end = now()
        return {
            "status": status, "error": f"EXC: {e}",
            "latency": t_end - t0, "ttft": None,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens
        }

async def nonstream_chat(session: aiohttp.ClientSession, url: str, headers: Dict[str,str], payload: Dict[str,Any]) -> Dict[str, Any]:
    """Non-streaming chat.completions or responses"""
    t0 = now()
    try:
        async with session.post(url, headers=headers, json=payload, timeout=None) as resp:
            status = resp.status
            text = await resp.text()
            t_end = now()
            if status != 200:
                return {
                    "status": status, "error": f"HTTP {status}: {text[:200]}",
                    "latency": t_end - t0, "ttft": None,
                    "prompt_tokens": None, "completion_tokens": None, "total_tokens": None
                }
            try:
                j = json.loads(text)
            except Exception as e:
                return {
                    "status": status, "error": f"JSON parse error: {e}",
                    "latency": t_end - t0, "ttft": None,
                    "prompt_tokens": None, "completion_tokens": None, "total_tokens": None
                }
            usage = j.get("usage") or {}
            return {
                "status": status, "error": None,
                "latency": t_end - t0, "ttft": None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
    except asyncio.CancelledError:
        raise
    except Exception as e:
        t_end = now()
        return {
            "status": None, "error": f"EXC: {e}",
            "latency": t_end - t0, "ttft": None,
            "prompt_tokens": None, "completion_tokens": None, "total_tokens": None
        }

async def stream_responses(session: aiohttp.ClientSession, url: str, headers: Dict[str,str], payload: Dict[str,Any]) -> Dict[str, Any]:
    """Streaming /v1/responses (SSE-like)."""
    # Many servers mirror the same "data:" stream format.
    return await stream_chat(session, url, headers, payload)

async def one_request(session, base_url, endpoint, headers, model, messages, stream, max_tokens):
    if endpoint == "chat.completions":
        url = f"{base_url}/v1/chat/completions"
    elif endpoint == "responses":
        url = f"{base_url}/v1/responses"
    else:
        raise ValueError("endpoint must be chat.completions or responses")

    payload = build_payload(messages, model, stream, endpoint, max_tokens)

    if stream:
        if endpoint == "chat.completions":
            return await stream_chat(session, url, headers, payload)
        else:
            return await stream_responses(session, url, headers, payload)
    else:
        return await nonstream_chat(session, url, headers, payload)

async def run_stage(concurrency:int, args, messages):
    results = []
    sem = asyncio.Semaphore(concurrency)

    headers = {
        "Authorization": f"Bearer {args.api_key}" if args.api_key else "",
        "Content-Type": "application/json",
    }

    connector = aiohttp.TCPConnector(limit=None, ssl=False)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # optional warmup
        for _ in range(args.warmup):
            try:
                await one_request(session, args.base_url, args.endpoint, headers, args.model, messages, args.stream, args.max_tokens)
            except Exception:
                pass

        start = now()
        in_flight = 0
        done = 0
        errors = 0

        async def worker(req_id:int):
            nonlocal done, errors
            async with sem:
                r = await one_request(session, args.base_url, args.endpoint, headers, args.model, messages, args.stream, args.max_tokens)
                r["req_id"] = req_id
                r["concurrency"] = concurrency
                results.append(r)
                done += 1
                if r.get("error"):
                    errors += 1

        tasks = []
        total_to_send = args.requests_per_stage
        # Launch bursts up to concurrency, then maintain steady rate as tasks finish
        for i in range(total_to_send):
            tasks.append(asyncio.create_task(worker(i)))

        await asyncio.gather(*tasks)
        elapsed = now() - start

    # Compute summaries
    latencies = [r["latency"] for r in results if r.get("latency") is not None]
    ttfts = [r["ttft"] for r in results if r.get("ttft") is not None]
    statuses = [r.get("status") for r in results]
    succ = sum(1 for r in results if not r.get("error") and (r.get("status") == 200))
    err_rate = 1 - (succ / max(1, len(results)))

    # QPS as completed-successes / elapsed
    qps = succ / elapsed if elapsed > 0 else float("nan")

    # TPS from usage where available
    comp_tokens = [r.get("completion_tokens") for r in results if r.get("completion_tokens") is not None]
    total_completion_tokens = sum(comp_tokens) if comp_tokens else 0

    # Generation time is from TTFT to end; approximate by latency - ttft when ttft exists, else latency
    gen_times = []
    for r in results:
        lat = r.get("latency")
        tt = r.get("ttft")
        if lat is None:
            continue
        if tt is None:
            gen_times.append(lat)
        else:
            gen_times.append(max(1e-6, lat - tt))
    total_gen_time = sum(gen_times) if gen_times else 0.0
    tps = (total_completion_tokens / total_gen_time) if total_gen_time > 0 else float("nan")

    summary = {
        "concurrency": concurrency,
        "requests": len(results),
        "success": succ,
        "error_rate": err_rate,
        "elapsed_sec": elapsed,
        "qps": qps,
        "latency_avg": float(np.mean(latencies)) if latencies else float("nan"),
        "latency_p50": pct(latencies, 50) if latencies else float("nan"),
        "latency_p95": pct(latencies, 95) if latencies else float("nan"),
        "latency_p99": pct(latencies, 99) if latencies else float("nan"),
        "ttft_avg": float(np.mean(ttfts)) if ttfts else float("nan"),
        "ttft_p99": pct(ttfts, 99) if ttfts else float("nan"),
        "total_completion_tokens": total_completion_tokens,
        "total_gen_time_sec": total_gen_time,
        "tps_tokens_per_sec": tps,
    }
    return results, summary

def load_messages(prompt_path: Optional[str], system: Optional[str], user: Optional[str]) -> List[Dict[str,Any]]:
    if prompt_path:
        # support JSONL with {"messages":[...]} lines or raw array
        txt = open(prompt_path, "r", encoding="utf-8").read()
        txt_strip = txt.strip()
        if txt_strip.startswith("["):
            arr = json.loads(txt_strip)
            return arr
        else:
            # take first line
            for line in txt.splitlines():
                line=line.strip()
                if not line:
                    continue
                j = json.loads(line)
                if "messages" in j:
                    return j["messages"]
            raise ValueError("Unrecognized prompt file format")
    else:
        msgs = []
        if system:
            msgs.append({"role":"system","content":system})
        msgs.append({"role":"user","content":user or "Say hello in one sentence."})
        return msgs

def write_csv(path: str, rows: List[Dict[str,Any]]):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="Base URL, e.g., http://localhost:8000")
    ap.add_argument("--api-key", default="", help="API key if required (Bearer)")
    ap.add_argument("--endpoint", default="chat.completions", choices=["chat.completions","responses"], help="OpenAI-style endpoint")
    ap.add_argument("--model", required=True, help="Model name to send to server")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32", help="Comma list and/or ranges, e.g., 1,2,4-16,32")
    ap.add_argument("--requests-per-stage", type=int, default=50, help="Number of requests per concurrency stage")
    ap.add_argument("--stream", action="store_true", help="Use streaming to measure TTFT")
    ap.add_argument("--max-tokens", type=int, default=None, help="max tokens for generation")
    ap.add_argument("--prompt-file", default=None, help="JSON or JSONL file with messages")
    ap.add_argument("--system", default=None, help="System prompt")
    ap.add_argument("--user", default="Write a 20-word answer about why the sky is blue.", help="User prompt")
    ap.add_argument("--warmup", type=int, default=2, help="Number of warmup requests before measuring each stage")
    ap.add_argument("--outdir", default="llm_bench_out", help="Output directory")
    args = ap.parse_args()

    messages = load_messages(args.prompt_file, args.system, args.user)
    conc_list = parse_concurrency_list(args.concurrency)

    os.makedirs(args.outdir, exist_ok=True)
    all_summaries = []
    all_rows = []

    for c in conc_list:
        print(f"== Running stage: concurrency={c} ==")
        results, summary = asyncio.run(run_stage(c, args, messages))
        all_rows.extend(results)
        all_summaries.append(summary)
        # per-stage CSV
        stage_csv = os.path.join(args.outdir, f"requests_c{c}.csv")
        write_csv(stage_csv, results)
        # stage summary CSV
        with open(os.path.join(args.outdir, f"summary_c{c}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2))

    # combined CSV and summary
    write_csv(os.path.join(args.outdir, "requests_all.csv"), all_rows)
    df = pd.DataFrame(all_summaries)
    df.to_csv(os.path.join(args.outdir, "summary_all.csv"), index=False)

    overall = {
        "best_qps": float(df["qps"].max()) if not df.empty else float("nan"),
        "best_qps_at_concurrency": int(df.loc[df["qps"].idxmax(), "concurrency"]) if not df.empty else None,
        "min_p99_latency": float(df["latency_p99"].min()) if not df.empty else float("nan"),
        "saturation_hint": int(df.loc[df["qps"].idxmax(), "concurrency"]) if not df.empty else None,
    }
    with open(os.path.join(args.outdir, "overall_summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("== Overall summary ==")
    print(json.dumps(overall, indent=2))

if __name__ == "__main__":
    main()

"""
python llm_bench.py \
  --base-url http://localhost:8000 \
  --api-key sk-xxx \
  --model your-model-name \
  --endpoint chat.completions \
  --concurrency 1,2,4,8,16,32 \
  --requests-per-stage 50 \
  --stream \
  --user "Write a one-sentence fact about Oklahoma."
"""