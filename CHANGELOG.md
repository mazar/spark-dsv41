# Changelog — `Spark-DSV41`

## [1.0.0] — 2026-10-24

### Added
- **Dedicated `/home/massoud/spark-dsv41` Repository:**
  - Dual-backend launcher (`scripts/start_engine.sh`) supporting both `ENGINE_BACKEND=zig` (`tensorfold-dsv41:exl3-tp2-625d420e24a4`, default for maximum throughput) and `ENGINE_BACKEND=python` (`tensorfold-dsv41:730d41abd8f4`, for `xgrammar` `json_schema` guided decoding) behind the compiled Go Control Plane on port `8888`.
  - Safe `--switch-from-glm53` flag to cleanly stop `Spark-GLM53` and release its `85.28 GiB` CUDA IPC weight daemon before loading `DeepSeek-V4.1-Flash-1017` (`98.9 GiB/rank`).
- **Compiled Go/C Page-Cache Guard (`pkg/cacheguard` -> `bin/spark-dsv41-cacheguard`):**
  - Continuously evicts Linux file page-cache pages backed by the `98.9 GiB` Rank Cache and `203 GB` Engram safetensors (`model-00047-of-00048.safetensors` & `model-00048-of-00048.safetensors`) on both Head and Worker via `posix_fadvise(..., POSIX_FADV_DONTNEED)`, preventing GB10 UMA exhaustion during long multi-turn sessions.
- **Compiled Go Control Plane (`pkg/gateway` -> `bin/spark-dsv41-gateway`):**
  - Full OpenAI (`/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/tokenize`) and Anthropic (`/v1/messages`, `/v1/messages/count_tokens`) compatibility for Claude CLI.
  - Automatic `reasoning_effort` normalization mapping `"max"` and `"xhigh"` to TensorFold's native `"xhigh"` tier and stripping historical `<think>...</think>` blocks (`clear_thinking=true`).
  - DeepSeek-V4.1 DSML (`<｜DSML｜function_calls>`, `<｜DSML｜ calls>`) and `<tool_call>` parser/repair with JSON Schema scalar type coercion (`CoerceArgsJSONToSchema`).
  - 256-token Block-Aligned Radix Prefix Router with `O_DIRECT` NVMe spill tier.
- **Python Backend Warm-Up OOM Fix:**
  - Diagnosed and fixed the `OutOfMemoryError: Tried to allocate 12.00 MiB ... 102.92 GiB allowed` crash in `tensorfold-dsv41:730d41abd8f4` (`engine.py:_memory_ceiling()`) by setting `MEM_FLOOR_GIB=0` alongside `spark-dsv41-cacheguard`.
