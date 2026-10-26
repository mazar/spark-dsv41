#!/usr/bin/env bash
# ==============================================================================
# Spark-DSV41: End-to-End Bare-Metal 2x DGX Spark Cluster Bootstrap & Launch
# ==============================================================================
# Prepares a fresh 2x NVIDIA DGX Spark (GB10, sm_121a) cluster from scratch:
#   1. Tunes ConnectX-7 Dual-Rail RoCE v2 fabric (MTU 4200, GID index 3), locks
#      GB10 GPU clocks at 3003 MHz, and reclaims UMA memory across both nodes.
#   2. Builds/pulls `tensorfold-dsv41:exl3-tp2-625d420e24a4` (from `zig/Dockerfile`
#      + `zig/patches/`), downloads `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`
#      (196 GiB) + `deepseek-ai/DeepSeek-V4.1-Flash-1017` Engram tables (203 GB,
#      shards 47 & 48) + `kit-abe07c7d`, configures the Worker NFS volume, and
#      generates the baseline `45ee5d0f` Rank 0 and Rank 1 caches.
#   3. Streams ~12 GiB of unquantized `FP8`/`BF16`/`FP4` source weights from
#      `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`1bae0e10`) and runs parallel GPU
#      EXL3 Viterbi trellis quantization (`scripts/build_custom_hq.sh`) on both
#      Sparks to build the `custom-hq` (8.0 / 4.0 / 3.0 bpw) RankCache (~31 min).
#   4. Compiles the Go `spark-dsv41-gateway` & `spark-dsv41-cacheguard` binaries,
#      pins the `100.81 GiB` RankCache in UMA, and starts the server on port 8888.
#
# Usage:
#   ./scripts/setup_from_scratch.sh
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WORKER_IP="${WORKER_IP:-${WORKER:-10.100.240.2}}"

echo -e "\033[1;36m[setup-from-scratch] Step 1/4: Tuning 2x DGX Spark RoCE v2 Fabric & UMA Hygiene...\033[0m"
WORKER_IP="$WORKER_IP" "$ROOT_DIR/scripts/tune_sparks.sh" --verify

echo -e "\033[1;36m[setup-from-scratch] Step 2/4: Preparing Engine Image, Checkpoints, Kit, NFS & Base RankCache...\033[0m"
WORKER="$WORKER_IP" "$ROOT_DIR/scripts/prepare.sh"

echo -e "\033[1;36m[setup-from-scratch] Step 3/4: Building Custom Hybrid EXL3 RankCache (custom-hq) on GPU...\033[0m"
WORKER_IP="$WORKER_IP" "$ROOT_DIR/scripts/build_custom_hq.sh"

echo -e "\033[1;36m[setup-from-scratch] Step 4/4: Starting Spark-DSV41 Engine + Go Control Plane on :8888...\033[0m"
WORKER_IP="$WORKER_IP" "$ROOT_DIR/scripts/start_engine.sh"
