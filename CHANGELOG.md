# Changelog — `Spark-DSV41`

## [1.0.0] — 2026-10-24

### Added
- **Dedicated `spark-dsv41` Repository:**
  - Dual-backend launcher (`scripts/start_engine.sh`) supporting both `ENGINE_BACKEND=zig` (`tensorfold-dsv41:exl3-tp2-625d420e24a4`, default for maximum throughput) and `ENGINE_BACKEND=python` (`tensorfold-dsv41:730d41abd8f4`, for `xgrammar` `json_schema` guided decoding) behind the compiled Go Control Plane on port `8888`.
  - Safe `--switch-from-glm53` flag to cleanly stop `Spark-GLM53` and release its `85.28 GiB` CUDA IPC weight daemon before loading `DeepSeek-V4.1-Flash-1017` (`98.9 GiB/rank`).
- **Compiled Go/C Page-Cache Guard (`pkg/cacheguard` -> `bin/spark-dsv41-cacheguard`):**
  - Continuously evicts Linux file page-cache pages backed by the `98.9 GiB` Rank Cache and `203 GB` Engram safetensors (`model-00047-of-00048.safetensors` & `model-00048-of-00048.safetensors`) on both Head and Worker via `posix_fadvise(..., POSIX_FADV_DONTNEED)`, preventing GB10 UMA exhaustion during long multi-turn sessions.
- **Compiled Go Control Plane (`pkg/gateway` -> `bin/spark-dsv41-gateway`):**
  - Full OpenAI (`/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/tokenize`) and Anthropic (`/v1/messages`, `/v1/messages/count_tokens`) compatibility for Claude CLI.
  - Automatic `reasoning_effort` normalization mapping `"max"` and `"xhigh"` to TensorFold's native `"xhigh"` tier and stripping historical `<think>...</think>` blocks (`clear_thinking=true`).
  - Automatic `response_format` (`json_schema` / `json_object`) translation and clean JSON extraction (`extractCleanJSONObject`) for the Native Zig engine (`tensorfold-native`), lifting Category O (`Structured Output`) in `tool-eval-bench` from `40/100` (`4/12`) to **`83/100` (`10/12`)**.
  - DeepSeek-V4.1 DSML (`<｜DSML｜function_calls>`, `<｜DSML｜ calls>`) and `<tool_call>` parser/repair with JSON Schema scalar type coercion (`CoerceArgsJSONToSchema`).
  - 256-token Block-Aligned Radix Prefix Router with `O_DIRECT` NVMe spill tier.
- **Verified Benchmarks (`llama-benchy` & `tool-eval-bench --seed 42 --hardmode`):**
  - **`tool-eval-bench` (88 Scenarios, Hardmode, Seed 42, Run `2026-10-24T15-11-28.023171Z_1603170a`):**
    - **Final Score:** **`93 / 100` (`164 / 176` pts, `★★★★★ Excellent`)**, **11/16 categories at 100%**
    - **Deployability Score:** **`82 / 100`** (vs `77` on `Spark-GLM53` and `76` on `MiaAI-Lab GLM-5.3-Flash-EXL3`)
    - **Median Turn Latency:** **`2.6 s`** (vs `4.34 s` on `Spark-GLM53` and `4.81 s` on `MiaAI-Lab`)
    - **Total Wall-Clock Time:** **`940.0 s` (`15.67 min`)** — **1.87x faster** than `Spark-GLM53` (`1,754.5 s` / `29.2 min`) and **2.07x faster** than `MiaAI-Lab GLM-5.3-Flash-EXL3` (`1,942.0 s` / `32.4 min`)!
    - **Post-88-Scenario Free UMA:** `3.83 GiB` (Head) / `6.57 GiB` (Worker) — zero OOMs across all 88 multi-turn scenarios.
  - **`llama-benchy` (`--pp 512 2048 --tg 256 --exact-tg --no-cache --runs 1 --concurrency 1 2 4 8`):**
    - Go Control Plane API Latency: **`2.36 ms`** (`Coherence test PASSED`)
    - `tg256`: **`59.05 tok/s`** (`c=1`), **`77.77 tok/s`** (`c=2`, `84.00 tok/s` peak), **`96.44 tok/s`** (`c=4`, `112.00 tok/s` peak)
- **Python Backend Warm-Up OOM Fix:**
  - Diagnosed and fixed the `OutOfMemoryError: Tried to allocate 12.00 MiB ... 102.92 GiB allowed` crash in `tensorfold-dsv41:730d41abd8f4` (`engine.py:_memory_ceiling()`) by setting `MEM_FLOOR_GIB=0` alongside `spark-dsv41-cacheguard`.
