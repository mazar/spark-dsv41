# Spark-DSV41 — High-Throughput `DeepSeek-V4.1-Flash-1017` Engine & Go Control Plane for 2x NVIDIA DGX Spark (`sm_121a`)

`Spark-DSV41` is a production-hardened distributed inference recipe and compiled Go Control Plane for serving **`DeepSeek-V4.1-Flash-1017`** (`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` + `deepseek-ai/DeepSeek-V4.1-Flash-1017` 203 GB Engram tables) across a 2-node NVIDIA DGX Spark (`GB10 Grace Blackwell`, `sm_121a`) cluster connected via Dual-Rail ConnectX-7 RoCE v2.

It pairs the fastest community TensorFold engines ([`MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) Native Zig and [`Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark`](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark) Python/CUDA) with the GB10 UMA hygiene and compiled Go Control Plane innovations from [`mazar/spark-glm53`](https://github.com/mazar/spark-glm53).

---

## 1. Key Capabilities & Improvements Over Stock Community Recipes

| Capability / Metric | Stock `MiaAI-Lab` (Zig) | Stock `Elevatormusic` (Python) | **`Spark-DSV41` (This Repo)** |
| :--- | :--- | :--- | :--- |
| **Cold Boot Time (`98.9 GiB/rank` + Engram + Graphs)** | ~25–35 s | ~120–180 s | **26.0 s** (`7.9s` Rank Cache load + `3003 MHz` GPU clock lock) |
| **1-Stream Decode (Prose / Code)** | 60.0 / 109.9 tok/s | 56.3 / 86.2 tok/s | **61.3 / 111.8 tok/s** (`88.5%` DSpark draft acceptance on code) |
| **4-Stream Decode (`llama-benchy` `tg256` Avg / Peak)** | 96.4 / 112.0 tok/s | 86.0 / 98.0 tok/s | **96.44 / 112.00 tok/s** (`PARALLEL=4` + `TF_DS_IDX_SPLIT=1`) |
| **576 GiB Engram & Rank Cache UMA Page-Cache Protection** | None (Linux page cache slowly consumes `MemAvailable` $\rightarrow$ `memguard` kill) | Partial (`TF_DS_ENGRAM_DONTNEED=1` on Engram only) | **`spark-dsv41-cacheguard` daemon (`POSIX_FADV_DONTNEED` every 15s on all 576 GiB of Engram + Rank Cache files on both nodes)** |
| **Python Backend Warm-Up OOM Fix** | N/A | Crashes on 1st request (`102.92 GiB allowed` lock after `empty_cache()`) | **Fixed (`MEM_FLOOR_GIB=0` prevents `_memory_ceiling()` from locking PyTorch at 0 headroom)** |
| **Anthropic `/v1/messages` & Claude CLI Support** | None | None | **Full `/v1/messages` + `/v1/messages/count_tokens` + `claude-opus-4-6` / `claude-sonnet-4-6` aliases** |
| **Effort Mapping (`max` & `xhigh`)** | Rejects `"max"` (`low`/`medium`/`high`/`xhigh` only) | Rejects `"max"` | **Maps both `"max"` and `"xhigh"` to TensorFold's `"xhigh"` tier automatically** |
| **Zig `response_format` (`json_schema` / `json_object`)** | Rejects with `HTTP 400` (`40/100` on Category O) | Native `xgrammar` | **Gateway translates `response_format` + extracts clean JSON (`83/100` on Category O with Zig speed)** |
| **DSML `<｜DSML｜function_calls>` & JSON Schema Coercion** | Basic DSML parse | Basic DSML parse | **Full DSML + XML repair + JSON Schema scalar type coercion (`"29620"` $\rightarrow$ `29620`) + LoopGuard** |

---

## 2. Verified Benchmarks (2x NVIDIA DGX Spark, Dual-Rail ConnectX-7 RoCE v2)

### A. [`SeraphimSerapis/tool-eval-bench`](https://github.com/SeraphimSerapis/tool-eval-bench) (88 Scenarios, Hardmode, Seed 42)

| Metric (88 Scenarios, Hardmode, Seed 42) | `MiaAI-Lab` `GLM-5.3-Flash-EXL3` | [`Spark-GLM53`](https://github.com/mazar/spark-glm53) (`NVFP4` + `B12X`) | **`Spark-DSV41` (`DeepSeek-V4.1-Flash-1017`)** | `Spark-DSV41` Highlights |
| :--- | :---: | :---: | :---: | :--- |
| **Final Score** | `94 / 100` (`165 / 176`) | **`95 / 100` (`168 / 176`)** | **`93 / 100` (`164 / 176`)** | **★★★★★ Excellent** |
| **Deployability Score ($\alpha=0.7$)** | `76 / 100` | `77 / 100` | **`82 / 100`** | **+5 pts higher deployability** |
| **Responsiveness Score** | `34 / 100` | `35 / 100` | **`55 / 100`** | **+20 pts higher responsiveness** |
| **Median Turn Latency** | `4,811 ms` (`4.81 s`) | `4,343 ms` (`4.34 s`) | **`2,600 ms` (`2.60 s`)** | **-40.1% faster per turn** |
| **Total Wall-Clock Time (All 88 Scenarios)** | `1,942.0 s` (`32.4 min`) | `1,754.5 s` (`29.2 min`) | **`940.0 s` (`15.67 min`)** | **1.87x faster than `Spark-GLM53` (-13.6 min)** |
| **Categories at 100%** | `11 / 16` | `14 / 16` | **`11 / 16`** | Tool Selection, Param Precision, Multi-Step, Restraint, Recovery, Localization, Reasoning, Instruction, Code, Planning, Creative all **100%** |
| **Structured Output (Category O)** | `83%` (`10 / 12`) | `100%` (`12 / 12`) | **`83%` (`10 / 12`)** | Up from `33%` (`4 / 12` HTTP 400) via `spark-dsv41-gateway` `response_format` translation |
| **Post-Suite Free UMA (`MemAvailable`)** | — | `4.19 GiB` / `6.47 GiB` | **`3.83 GiB` (Head) / `6.57 GiB` (Worker)** | Zero OOMs across all 88 multi-turn scenarios (`spark-dsv41-cacheguard` active) |

### B. [`eugr/llama-benchy`](https://github.com/eugr/llama-benchy) Multi-Stream Concurrency (`c=1..8`, `PARALLEL=4`)

- **Go Control Plane API Latency:** **`2.36 ms`** (`Coherence test PASSED`)

| Test | `t/s (total)` | `t/s (req)` | `peak t/s` | `peak t/s (req)` | `e2e_ttft (ms)` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`pp512 (c1)`** | *(972.3 tok/s e2e)* | — | — | — | `526.57 ± 0.00` |
| **`tg256 (c1)`** | **`59.05 ± 0.00`** | **`59.05 ± 0.00`** | **`60.00 ± 0.00`** | **`60.00 ± 0.00`** | — |
| **`pp512 (c2)`** | `874.48 ± 0.00` | — | — | — | `797.30 ± 261.61` |
| **`tg256 (c2)`** | **`77.77 ± 0.00`** | **`41.65 ± 2.76`** | **`84.00 ± 0.00`** | **`42.00 ± 3.00`** | — |
| **`pp512 (c4)`** | `883.16 ± 0.00` | — | — | — | `1355.93 ± 600.46` |
| **`tg256 (c4)`** | **`96.44 ± 0.00`** | **`27.40 ± 1.92`** | **`112.00 ± 0.00`** | **`28.00 ± 1.73`** | — |
| **`pp2048 (c1)`** | *(1,910.0 tok/s e2e)* | — | — | — | `1072.25 ± 0.00` |
| **`tg256 (c1)`** | **`56.95 ± 0.00`** | **`56.95 ± 0.00`** | **`57.00 ± 0.00`** | **`57.00 ± 0.00`** | — |

---

## 3. Repository Layout

```text
spark-dsv41/
├── README.md
├── CHANGELOG.md
├── config/
│   └── model_aliases.json        # Hot-reloaded model & effort aliases (including Claude CLI aliases)
├── pkg/
│   ├── gateway/                  # Go Control Plane (:8888 -> 127.0.0.1:8000) source & unit tests
│   └── cacheguard/               # Go/C POSIX_FADV_DONTNEED page-cache eviction daemon
├── scripts/
│   ├── tune_sparks.sh            # Dual-rail RoCE v2 (MTU 4200), 3003 MHz GPU clock lock, UMA hygiene
│   ├── start_engine.sh           # Launches Zig (default) or Python backend + CacheGuard + Go Gateway
│   └── benchmark_suite.py        # End-to-end speed, effort, DSML tool call, and Anthropic API verifier
└── stop.sh                       # Cleanly stops containers, CacheGuard, and Gateway on both Sparks
```

---

## 4. Quickstart

### Prerequisites
Prepare the `tensorfold-dsv41` image, 98.9 GiB Rank Cache, and 203 GB Engram shards once using either [`MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) (`ENGINE_BACKEND=zig`, default) or [`Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark`](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark) (`ENGINE_BACKEND=python`).

### Start `Spark-DSV41` (Default Fast Zig Backend)
```bash
cd ~/spark-dsv41
./scripts/start_engine.sh
```
*(If [`Spark-GLM53`](https://github.com/mazar/spark-glm53) is currently running on the cluster, pass `./scripts/start_engine.sh --switch-from-glm53` to cleanly stop it and release its 85 GiB CUDA IPC weight daemon first.)*

### Start with the Python Backend (for native `xgrammar` `json_schema` constraints)
```bash
cd ~/spark-dsv41
ENGINE_BACKEND=python ./scripts/start_engine.sh
```

### Stop `Spark-DSV41`
```bash
cd ~/spark-dsv41
./stop.sh
```

---

## 5. Using Claude Code / Claude CLI with `Spark-DSV41`

`config/model_aliases.json` pre-configures `claude-opus-4-6` (`max` $\rightarrow$ `xhigh`), `claude-sonnet-4-6` (`high`), and `claude-haiku-4-5-20251001` (`low`):

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8888 \
ANTHROPIC_API_KEY=spark-local \
claude --model DeepSeek-V4.1-Flash-1017
```

---

## 6. Acknowledgements & Related Projects
- [**`MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`**](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) & `bertholomus/TensorFold` (`deepseek-v41-zig` Native Zig engine & EXL3 2.9 bpw EAQ checkpoint)
- [**`Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark`**](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark) (`dsv41-1.1.0` Python/CUDA TensorFold fork)
- [**`mazar/spark-glm53`**](https://github.com/mazar/spark-glm53) (`GLM-5.3-Flash` NVFP4 + B12X + DFlash2 + Go Control Plane for 2x DGX Spark)
- [**`SeraphimSerapis/tool-eval-bench`**](https://github.com/SeraphimSerapis/tool-eval-bench) & [**`eugr/llama-benchy`**](https://github.com/eugr/llama-benchy)
