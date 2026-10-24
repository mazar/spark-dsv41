# Spark-DSV41 — High-Throughput `DeepSeek-V4.1-Flash-1017` Engine for 2x NVIDIA DGX Spark (`sm_121a`)

`Spark-DSV41` is a dedicated, production-hardened recipe and Go Control Plane for serving **`DeepSeek-V4.1-Flash-1017`** (`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` + `deepseek-ai/DeepSeek-V4.1-Flash-1017` 203 GB Engram tables) across a 2-node NVIDIA DGX Spark (`GB10`, `sm_121a`) cluster connected via Dual-Rail ConnectX-7 RoCE v2.

It combines the fastest community TensorFold engines (`tensorfold-dsv41:exl3-tp2-625d420e24a4` Native Zig and `tensorfold-dsv41:730d41abd8f4` Python/CUDA) with all hardware, UMA hygiene, and Go Control Plane innovations developed in `Spark-GLM53`.

---

## 1. What `Spark-DSV41` Improves Over Stock Community Recipes

| Capability / Metric | Stock `MiaAI-Lab` (Zig) | Stock `Elevatormusic` (Python) | **`Spark-DSV41` (This Repo)** |
| :--- | :--- | :--- | :--- |
| **1-Stream Decode (Prose / Code)** | 74.9 / 109.9 tok/s | 56.3 / 86.2 tok/s | **77.5 / 113.5 tok/s** (`ENGINE_BACKEND=zig` + 3003 MHz GPU clock lock) |
| **4-Stream Decode (Code)** | 155.5 tok/s | 124.2 tok/s | **161.0+ tok/s** (`PARALLEL=4` + `TF_DS_IDX_SPLIT=1`) |
| **Long-Context Prefill (8K–400K)** | 2,175 tok/s (decays over time) | ~1,300 tok/s | **2,250+ tok/s** (`TF_DS_IDX_SPLIT=1` + `TF_DS_PREFILL_INTERLEAVE_MS=250`) |
| **203 GB Engram & 98.9 GiB Rank Cache UMA Protection** | None (Linux page cache slowly eats `MemAvailable` -> `memguard` kill) | Partial (`TF_DS_ENGRAM_DONTNEED=1` on Engram only) | **`spark-dsv41-cacheguard` daemon (`POSIX_FADV_DONTNEED` every 15s on all 576 GiB of Engram + Rank Cache files on both nodes)** |
| **Python Backend Warm-Up OOM Fix (`r0-20261021T153736Z.log`)** | N/A | Crashes on 1st request (`102.92 GiB allowed` lock after `empty_cache()`) | **Fixed (`MEM_FLOOR_GIB=0` prevents `_memory_ceiling()` from locking PyTorch at 0 headroom)** |
| **Anthropic `/v1/messages` & Claude CLI Support** | None | None | **Full `/v1/messages` + `/v1/messages/count_tokens` + `claude-opus-4-6` / `claude-sonnet-4-6` aliases** |
| **Effort Mapping (`max` & `xhigh`)** | Rejects `"max"` (`low`/`medium`/`high`/`xhigh` only) | Rejects `"max"` | **Maps both `"max"` and `"xhigh"` to TensorFold's `"xhigh"` tier automatically** |
| **DSML `<｜DSML｜function_calls>` & JSON Schema Coercion** | Basic DSML parse | Basic DSML parse | **Full DSML + XML repair + JSON Schema scalar type coercion (`"8080"` -> `8080`) + LoopGuard** |

---

## 2. Repository Layout

```text
/home/massoud/spark-dsv41/
├── README.md
├── CHANGELOG.md
├── config/
│   └── model_aliases.json        # Hot-reloaded model & effort aliases (including Claude CLI aliases)
├── bin/
│   ├── spark-dsv41-gateway       # Compiled Go Control Plane (:8888 -> 127.0.0.1:8000)
│   └── spark-dsv41-cacheguard    # Compiled Go/C POSIX_FADV_DONTNEED page-cache eviction daemon
├── pkg/
│   ├── gateway/                  # Go Control Plane source & unit tests (main.go, main_test.go)
│   └── cacheguard/               # Go/C page-cache guard source (main.go)
├── scripts/
│   ├── tune_sparks.sh            # Dual-rail RoCE v2 (MTU 4200), 3003 MHz GPU clock lock, UMA hygiene
│   ├── start_engine.sh           # Launches Zig (default) or Python backend + CacheGuard + Go Gateway
│   └── benchmark_suite.py        # End-to-end speed, effort, DSML tool call, and Anthropic API verifier
└── stop.sh                       # Cleanly stops containers, CacheGuard, and Gateway on both Sparks
```

---

## 3. Quickstart & Switching Between `Spark-GLM53` and `Spark-DSV41`

Because `Spark-GLM53` pins `85.28 GiB/node` in `spark-cuda-ipc-weights` and `Spark-DSV41` loads `98.9 GiB/node` of `DeepSeek-V4.1-Flash-1017` weights, only one model should reside in GPU memory at a time.

### Switch from `Spark-GLM53` to `Spark-DSV41` (Default Fast Zig Backend)
```bash
cd /home/massoud/spark-dsv41
./scripts/start_engine.sh --switch-from-glm53
```

### Start with the Python Backend (for strict `xgrammar` `json_schema` constraints)
```bash
cd /home/massoud/spark-dsv41
ENGINE_BACKEND=python ./scripts/start_engine.sh --switch-from-glm53
```

### Stop `Spark-DSV41`
```bash
cd /home/massoud/spark-dsv41
./stop.sh
```

---

## 4. Using Claude Code / Claude CLI with `Spark-DSV41`

`config/model_aliases.json` pre-configures `claude-opus-4-6` (`max` -> `xhigh`), `claude-sonnet-4-6` (`high`), and `claude-haiku-4-5-20251001` (`low`):

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8888 \
ANTHROPIC_API_KEY=spark-local \
claude --model DeepSeek-V4.1-Flash-1017
```
