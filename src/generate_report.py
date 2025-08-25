
#!/usr/bin/env python3
"""
generate_report.py — build charts + HTML report from llm_bench outputs

Inputs:
  --indir llm_bench_out
Outputs:
  report.html, plus PNG charts
Rules:
  - Uses matplotlib (no seaborn)
  - Each chart is a separate figure
  - Does not set specific colors/styles
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

def save_plot(fig, outpath):
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="llm_bench_out", help="Directory with summary_all.csv and request CSVs")
    args = ap.parse_args()

    summary_csv = os.path.join(args.indir, "summary_all.csv")
    if not os.path.exists(summary_csv):
        raise SystemExit(f"Missing {summary_csv}. Run llm_bench.py first.")

    df = pd.read_csv(summary_csv)

    charts = []

    # Latency p50/p95/p99 vs concurrency
    fig1 = plt.figure()
    plt.plot(df["concurrency"], df["latency_p50"], marker="o", label="p50")
    plt.plot(df["concurrency"], df["latency_p95"], marker="o", label="p95")
    plt.plot(df["concurrency"], df["latency_p99"], marker="o", label="p99")
    plt.xlabel("Concurrency")
    plt.ylabel("Latency (s)")
    plt.title("Latency vs Concurrency")
    plt.legend()
    p1 = os.path.join(args.indir, "latency_vs_concurrency.png")
    save_plot(fig1, p1)
    charts.append(("Latency vs Concurrency", p1))

    # QPS vs concurrency
    fig2 = plt.figure()
    plt.plot(df["concurrency"], df["qps"], marker="o")
    plt.xlabel("Concurrency")
    plt.ylabel("QPS")
    plt.title("QPS vs Concurrency")
    p2 = os.path.join(args.indir, "qps_vs_concurrency.png")
    save_plot(fig2, p2)
    charts.append(("QPS vs Concurrency", p2))

    # TTFT p99 vs concurrency (if present)
    if "ttft_p99" in df.columns and df["ttft_p99"].notna().any():
        fig3 = plt.figure()
        plt.plot(df["concurrency"], df["ttft_p99"], marker="o")
        plt.xlabel("Concurrency")
        plt.ylabel("TTFT P99 (s)")
        plt.title("TTFT P99 vs Concurrency")
        p3 = os.path.join(args.indir, "ttft_p99_vs_concurrency.png")
        save_plot(fig3, p3)
        charts.append(("TTFT P99 vs Concurrency", p3))

    # TPS vs concurrency
    if "tps_tokens_per_sec" in df.columns and df["tps_tokens_per_sec"].notna().any():
        fig4 = plt.figure()
        plt.plot(df["concurrency"], df["tps_tokens_per_sec"], marker="o")
        plt.xlabel("Concurrency")
        plt.ylabel("TPS (tokens/sec)")
        plt.title("TPS vs Concurrency")
        p4 = os.path.join(args.indir, "tps_vs_concurrency.png")
        save_plot(fig4, p4)
        charts.append(("TPS vs Concurrency", p4))

    # Build HTML report
    overall_path = os.path.join(args.indir, "overall_summary.json")
    overall = {}
    if os.path.exists(overall_path):
        with open(overall_path, "r", encoding="utf-8") as f:
            overall = json.load(f)

    html = []
    html.append("<html><head><meta charset='utf-8'><title>LLM Benchmark Report</title></head><body>")
    html.append(f"<h1>LLM Benchmark Report</h1>")
    html.append(f"<p>Generated at: {datetime.now().isoformat()}</p>")
    html.append("<h2>Overall Summary</h2>")
    html.append("<pre>" + json.dumps(overall, indent=2, ensure_ascii=False) + "</pre>")

    html.append("<h2>Stage Summaries</h2>")
    html.append("<table border='1' cellpadding='6' cellspacing='0'>")
    html.append("<tr><th>Concurrency</th><th>QPS</th><th>Err rate</th><th>Latency p50</th><th>Latency p95</th><th>Latency p99</th><th>TTFT p99</th><th>TPS</th></tr>")
    for _, row in df.sort_values("concurrency").iterrows():
        html.append("<tr>" +
            f"<td>{int(row['concurrency'])}</td>" +
            f"<td>{row['qps']:.3f}</td>" +
            f"<td>{row['error_rate']:.3f}</td>" +
            f"<td>{row['latency_p50']:.3f}</td>" +
            f"<td>{row['latency_p95']:.3f}</td>" +
            f"<td>{row['latency_p99']:.3f}</td>" +
            f"<td>{'' if pd.isna(row.get('ttft_p99', float('nan'))) else f'{row.get('ttft_p99'):.3f}'}</td>" +
            f"<td>{'' if pd.isna(row.get('tps_tokens_per_sec', float('nan'))) else f'{row.get('tps_tokens_per_sec'):.2f}'}</td>" +
        "</tr>")
    html.append("</table>")

    html.append("<h2>Charts</h2>")
    for title, path in charts:
        html.append(f"<h3>{title}</h3>")
        imgname = os.path.basename(path)
        html.append(f"<img src='{imgname}' style='max-width: 900px; display:block; margin-bottom: 12px;'>")

    html.append("</body></html>")

    out_html = os.path.join(args.indir, "report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"Wrote HTML report: {out_html}")

if __name__ == "__main__":
    main()
