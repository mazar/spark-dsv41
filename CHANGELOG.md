# Changelog — `Spark-DSV41`

## [1.1.0] — 2026-10-25

### Added
- **Custom Hybrid EXL3 RankCache Quantizer ([`scripts/quantize_rank_cache.py`](file:///home/massoud/spark-dsv41/scripts/quantize_rank_cache.py)):**
  - Re-quantizes `1,579` critical matrices per rank directly from the official unquantized `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`1bae0e1088993b9b82de08f59de5be5a34e42170`) `FP8` / `BF16` / `FP4` source weights on GPU (`exllamav3_ext` `mul1` Viterbi trellis encoder) while preserving the calibrated Hadamard scale vectors (`suh`, `svh`):
    1. **8.0 bpw (`K=8`, `K2=16`)** for all `363` rank-local Attention (`wq_a`, `wq_b`, `wkv`, `wo_a.slice.*`, `wo_b`), MLA Compressor (`wkv`, `wgate`), Sparse Indexer (`wq_b`), Engram (`engram.wkv`), DSpark MTP (`mtp.0.main_proj`, `mtp.0..2.attn.*`), and `head` matrices (`16x–240x` MSE reduction, relative L2 error `0.47%–0.54%`, `cos_sim = 0.999989`).
    2. **4.0 bpw (`K=4`, `K2=8`)** standardization across all `21` 5-bit `shared_experts.{w1,w2,w3}` layers (`Layers 0..10` and `30..39`), ensuring `k2_gu[1] <= 8` and `k2_d[1] <= 8` in `weights.zig` (`Loader.experts`) so TensorFold's fastest fused prompt kernels (`grouped_mma3_kernel` and `grouped_down3_kernel`) are unlocked on **all 40 layers** (`2,416.1 tok/s` prefill at `10K` tokens).
    3. **3.0 bpw (`K=3`, `K2=6`)** upgrade for the `384` highest-impact routed experts in the `Layers 18..22` `2.0 bpw` cliff (`top 128` in Layer 20 CSA2 source + `top 64` each in Layers 18, 19, 21, 22, ranked deterministically by `(gate.bias - min(gate.bias) + 0.1) * ||w2.svh||_2`), cutting relative L2 error from `28.99%` to `13.20%` (`4.8x` MSE reduction).
  - Automatic `RANK_CACHE_TAG=custom-hq` selection in [`scripts/start_engine.sh`](file:///home/massoud/spark-dsv41/scripts/start_engine.sh) and [`pkg/cacheguard/main.go`](file:///home/massoud/spark-dsv41/pkg/cacheguard/main.go).
- **Verified `tool-eval-bench` Quality Improvement (`2026-10-25T16-44-19.930138Z_6e9e8f3d`):**
  - On the 88-scenario benchmark (`TC-01..TC-88`), score improved from **`93 / 100` (`164 / 176` pts, 11/16 categories at 100%)** on stock `Mia EAQ 2.9 bpw` to **`95 / 100` (`168 / 176` pts, 13/16 categories at 100%)**, flipping `TC-40` (*Domain Confusion*), `TC-60` (*Cross-Turn Sleeper Injection*), `TC-62` (*5-Turn Research Chain*), and `TC-80` (*Preconditioned Update Safety*, ★★★★★) to **`2/2` PASS**.
  - On the expanded 97-scenario `tool-eval-bench v2.8.1` suite (`TC-01..TC-97`), scored **`93 / 100` (`180 / 194` pts, `★★★★★ Excellent`)** with **`13 / 16` categories at 100%** (`Context & State` and `Toolset Scale` both raised to `100%`).
- **Industry-Standard Coding Intelligence Benchmarks ([`scripts/run_coding_benchmarks.py`](file:///home/massoud/spark-dsv41/scripts/run_coding_benchmarks.py)):**
  - Evaluated `Spark-DSV41 (custom-hq)`, `Spark-DSV41 (stock 2.9 bpw)`, and `Spark-GLM53 (NVFP4)` head-to-head across `164` `EvalPlus HumanEval+` tasks and `30` stratified `LiveCodeBench v6` competitive programming tasks (`10 Easy`, `10 Medium`, `10 Hard`):
    - **`LiveCodeBench v6` (`30` tasks):** `Spark-DSV41 (custom-hq)` achieved **`93.33%` (`28 / 30` — `10/10 Easy`, `10/10 Medium`, `8/10 Hard`)**, outperforming both `Spark-DSV41 (stock 2.9 bpw)` (`90.00%`, `27 / 30`) and `Spark-GLM53 (NVFP4)` (`76.67%`, `23 / 30`) while eliminating circular `<think>` timeouts (`40%–51%` fewer `<think>` tokens on Hard/Medium tasks).
    - **`EvalPlus HumanEval+` (`164` tasks):** All three tied at **`96.34%` (`158 / 164`)** on Base `HumanEval` (`90.85%` / `91.46%` / `92.68%` on `80x` Extra Tests), with `Spark-DSV41` completing all `164` tasks in **`243.5–264.0 s`** vs **`654.2 s`** on `Spark-GLM53` (**2.48x–2.69x faster**).

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
