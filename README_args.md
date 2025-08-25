# LLM 性能压测指标说明（Markdown 版）

> 适用于你当前的 SGLang + OpenAI 兼容端点压测脚本（`llm_bench.py`、`generate_report.py`）。下述名词与**脚本输出列名**一一对应，便于阅读 `summary_all.csv` / `report.html`。

---

## 1) 延迟（Latency）与分位数（Percentile）

* **Latency（延迟）**
  单次请求从**发出**到**完整响应结束**的总时长（秒）。脚本中会统计 `latency_avg / latency_p50 / latency_p95 / latency_p99`。

  * `latency_avg`：平均延迟
  * `latency_p50`（**P50**/中位数）：50% 请求的延迟不超过该值
  * `latency_p95`（**P95**）：95% 请求的延迟不超过该值
  * `latency_p99`（**P99**/尾延迟）：99% 请求的延迟不超过该值（更能体现“长尾”）

**解读建议**

* **P50** 看“多数请求”的典型体验；**P95/P99** 看“长尾”与系统抖动。
* 并发升高时，若 **QPS 不再增长**且 **P99 急剧上升**，说明已**接近/进入饱和**（排队显著）。

**统计注意**

* 样本过小会导致分位数不稳：建议每个并发档至少 **200+** 请求看 P95，**1000+** 更适合看 P99（经验法则）。

---

## 2) 首 Token 响应时间（TTFT）

* **TTFT（Time To First Token）**
  仅在**流式**（`--stream`）时测量：从请求发出到**第一块流数据**（`data: {...}`）到达的时间（秒）。脚本输出 `ttft_avg`、`ttft_p99`（若开启 stream）。

  * 反映**排队、预填充（prefill）与首轮解码**等首包路径的开销。
  * 与生成长度关系不大（主要影响的是后续生成阶段）。

---

## 3) 吞吐（QPS / RPS）

* **QPS（Queries Per Second）**
  在某并发档内，**成功请求数 / 阶段耗时**。脚本输出列为 `qps`。

  > 注：有时也写作 **RPS**（Requests Per Second），本脚本里的 QPS 即每秒成功请求处理数。

**解读建议**

* QPS 随并发增长先升后平/降：**峰值附近**就是该配置下的**最佳并发**。
* `overall_summary.json` 给出 `best_qps` 与 `best_qps_at_concurrency`，便于快速定位。

---

## 4) Token 指标（Prompt / Completion / Total）

* **prompt\_tokens**：提示词 token 数
* **completion\_tokens**：生成 token 数（受 `--max-tokens` 上限约束）
* **total\_tokens**：`prompt_tokens + completion_tokens`

> 需要服务端在响应里返回 `usage` 字段（OpenAI 兼容格式）。若没有，某些 token 相关统计可能为空。

---

## 5) 生成时长与 TPS（Tokens Per Second）

* **生成时长（Generation Time）**
  近似：`latency - TTFT`（流式时）；若不流式，则以 `latency` 近似。脚本在每个请求上汇总生成阶段时长，并记录总和 `total_gen_time_sec`。

* **TPS（Tokens Per Second）**
  计算式（脚本列 `tps_tokens_per_sec`）：

  ```
  TPS = Σ(completion_tokens) / Σ(生成时长)
  ```

  反映**模型解码速率**与**服务端生成效率**。

  * **提升 TPS**：开启/优化批处理、启用低精度/更快内核、合理的 `max_tokens`，减少后处理与传输开销。

---

## 6) 错误率（Error Rate）

* **error\_rate**
  某并发档内：`1 - (成功数 / 总请求数)`。脚本将非 200 状态与异常视作失败。

  * 并发升高伴随错误率上升 → 说明**队列溢出/超时/资源不足**，进入饱和区。

---

## 7) 并发与阶段（Concurrency / Stage）

* **concurrency（并发）**
  脚本内部用信号量限制的**同时在飞请求数**上限（不是 CPU 线程数）。
* **requests\_per\_stage**
  每个并发档要完成的**请求总数**。例如 `concurrency=4` 且 `requests_per_stage=10`，表示最多并行 4 个，一共跑完 10 个请求后统计该档数据。
* **warmup**
  每档压测前的**预热请求数**（默认 2），不纳入统计，用于避开冷启动波动。

---

## 8) 饱和与最佳并发（Saturation / Sweet Spot）

* **饱和（Saturation）**
  随并发上升，**QPS 不再提升**且 **P99/错误率**明显抬头，即系统达到瓶颈（调度/排队/GPU 饱和/网络 IO 等）。
* **最佳并发（Sweet Spot）**
  `qps` 的**峰值所在并发**；脚本在 `overall_summary.json` 中给出 `best_qps_at_concurrency` 与 `saturation_hint`（二者通常一致或相近）。

---

## 9) 其他常用术语

* **Throughput（吞吐量）**
  可指 **QPS** 或 **TPS**；在 LLM 场景中二者常并论：

  * QPS：单位时间内**完整请求**处理数量
  * TPS：单位时间内**生成 token**数量
* **SLA / SLO**
  面向用户/业务的服务目标（例如 **P95 ≤ 500ms**、**错误率 ≤ 0.1%** 等）；报告中的分位数与错误率用于对照/校验 SLO。
* **RPS vs QPS**
  在本脚本里可视作一回事；但有的团队会区分“请求”（HTTP 层面）与“查询”（业务层面）。

---

## 10) 脚本输出字段对照表

| 字段名                       | 含义                      | 单位/备注   |
| ------------------------- | ----------------------- | ------- |
| `concurrency`             | 并发上限                    | 个       |
| `requests`                | 本档请求总数                  | 个       |
| `success`                 | 成功请求数（HTTP 200 且无异常）    | 个       |
| `error_rate`              | 错误率                     | 0\~1    |
| `elapsed_sec`             | 本档总耗时                   | 秒       |
| `qps`                     | 吞吐（成功请求数/总耗时）           | 次/秒     |
| `latency_avg`             | 平均延迟                    | 秒       |
| `latency_p50/p95/p99`     | 分位延迟                    | 秒       |
| `ttft_avg/ttft_p99`       | 首 token 时间（平均/99 分位）    | 秒（仅流式）  |
| `total_completion_tokens` | 本档生成 token 总数           | 个       |
| `total_gen_time_sec`      | 本档生成阶段总时长               | 秒       |
| `tps_tokens_per_sec`      | 令牌吞吐（Σ生成 token / Σ生成时长） | token/秒 |

**总体文件 `overall_summary.json`**

* `best_qps`：所有并发档的 QPS 最大值
* `best_qps_at_concurrency`：对应的并发值
* `min_p99_latency`：所有档中最小的 P99
* `saturation_hint`：与峰值并发一致（可视作饱和点附近）

---

## 11) 读数与调参小抄

* **想提 QPS**：减小 `--max-tokens`、加大批处理上限、优化调度/队列、启用持续批（continuous batching）。
* **想提 TPS**：启用更快内核/低精度，减少后处理/传输阻塞；保持较长但适度的生成（避免太短导致统计不稳）。
* **想稳住 P99**：限流/排队隔离、预热实例、降低单请求资源波动、避免极端长 prompt。
* **样本量**：`requests_per_stage` 建议 ≥ `max(200, 30×并发)` 用于看 P95；P99 适当更多。

---

需要的话，我可以把这份指标说明**直接附加到自动化压测报告**里（在 `report.html` 末尾追加“指标解释”章节），或输出一份**企业内审稿模板（含 SLO 协议段）**，方便你发周报/评审。
