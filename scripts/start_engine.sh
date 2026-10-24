#!/usr/bin/env bash
# ==============================================================================
# Spark-DSV41: High-Throughput Distributed DeepSeek-V4.1-Flash-1017 Engine
# Target: 2x NVIDIA DGX Spark (GB10, sm_121a), TP=2 over Dual-Rail ConnectX-7 RoCE v2
#
# Supports two backends behind the Spark-DSV41 Go Control Plane (:8888 -> :8000):
#   - ENGINE_BACKEND=zig    (default: tensorfold-dsv41:exl3-tp2-625d420e24a4,
#                            74.9 tok/s prose, 109.9 tok/s code, 155.5 tok/s 4-stream)
#   - ENGINE_BACKEND=python (tensorfold-dsv41:730d41abd8f4 with MEM_FLOOR_GIB=0 OOM fix
#                            and full xgrammar json_schema / regex structured decoding)
#
# Usage:
#   ./scripts/start_engine.sh                         # Start with defaults
#   ./scripts/start_engine.sh --switch-from-glm53     # Stop Spark-GLM53 & free 85 GiB IPC daemon first
#   ENGINE_BACKEND=python ./scripts/start_engine.sh   # Start Python backend (json_schema grammar support)
#   DRY_RUN=1 ./scripts/start_engine.sh               # Print commands without starting
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENGINE_BACKEND="${ENGINE_BACKEND:-zig}"
PORT="${PORT:-8888}"
UPSTREAM_PORT="${UPSTREAM_PORT:-8000}"
CONTEXT="${CONTEXT:-1048576}"
PARALLEL="${PARALLEL:-4}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
THINKING_BUDGET="${THINKING_BUDGET:-}"
DRAFTS="${DRAFTS:-1}"
MTP_DRAFTS="${MTP_DRAFTS:-5}"
PREFILL_INTERLEAVE_MS="${PREFILL_INTERLEAVE_MS:-250}"
TF_DS_IDX_SPLIT="${TF_DS_IDX_SPLIT:-1}"
TF_DS_IMAGE_CACHE="${TF_DS_IMAGE_CACHE:-1}"
RDMA_IDLE_S="${RDMA_IDLE_S:-2}"
RDMA_NAP_US="${RDMA_NAP_US:-500}"
GPU_LOCK_MHZ="${GPU_LOCK_MHZ:-3003}"
SWITCH_FROM_GLM53="${SWITCH_FROM_GLM53:-0}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"

HEAD_IP="${HEAD_IP:-10.100.240.1}"
WORKER_IP="${WORKER_IP:-10.100.240.2}"
MASTER_PORT="${MASTER_PORT:-29620}"
ROCE_HCAS="${ROCE_HCAS:-rocep1s0f1,roceP2p1s0f1}"
SOCKET_IFNAME="${SOCKET_IFNAME:-enp1s0f1np1}"
GID_INDEX="${GID_INDEX:-3}"

for arg in "$@"; do
    case "$arg" in
        --switch-from-glm53) SWITCH_FROM_GLM53=1 ;;
        --zig) ENGINE_BACKEND="zig" ;;
        --python) ENGINE_BACKEND="python" ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
    esac
done

log()  { echo -e "\033[1;36m[spark-dsv41]\033[0m $*"; }
warn() { echo -e "\033[1;33m[spark-dsv41 WARN]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[spark-dsv41 ERROR]\033[0m $*" >&2; exit 1; }

# 1. Build Go Control Plane & CacheGuard binaries if needed
if [[ ! -x "$ROOT_DIR/bin/spark-dsv41-gateway" ]] || [[ ! -x "$ROOT_DIR/bin/spark-dsv41-cacheguard" ]]; then
    log "Building Spark-DSV41 Go Gateway and CacheGuard binaries..."
    mkdir -p "$ROOT_DIR/bin"
    (cd "$ROOT_DIR/pkg/gateway" && go build -o "$ROOT_DIR/bin/spark-dsv41-gateway" .)
    (cd "$ROOT_DIR/pkg/cacheguard" && go build -o "$ROOT_DIR/bin/spark-dsv41-cacheguard" .)
    ssh -o BatchMode=yes "$WORKER_IP" "mkdir -p /home/massoud/spark-dsv41/bin"
    scp -q "$ROOT_DIR/bin/spark-dsv41-cacheguard" "$WORKER_IP:/home/massoud/spark-dsv41/bin/spark-dsv41-cacheguard"
fi

# 2. Check for running Spark-GLM53 containers or 85 GiB CUDA IPC weight daemon
if [[ "$DRY_RUN" != "1" ]]; then
    if docker ps --format '{{.Names}}' | grep -qE '^spark-glm53' || pgrep -f "[s]park-cuda-ipc-weights|[s]park-shm-weights" >/dev/null 2>&1; then
        if [[ "$SWITCH_FROM_GLM53" == "1" ]]; then
            log "Stopping active Spark-GLM53 engine and freeing 85 GiB weight daemon (--switch-from-glm53)..."
            if [[ -x /home/massoud/spark-glm53/stop.sh ]]; then
                /home/massoud/spark-glm53/stop.sh --purge-shm
            fi
        else
            die "Spark-GLM53 is currently running or holding 85 GiB in spark-cuda-ipc-weights. DeepSeek-V4.1-Flash-1017 requires ~104 GiB/node. Run with '--switch-from-glm53' to cleanly stop Spark-GLM53 first."
        fi
    fi

    # Stop any previous Spark-DSV41 instance
    "$ROOT_DIR/stop.sh"

    # Run cluster RoCE v2 + 3003 MHz GPU clock + UMA hygiene tuning
    GPU_LOCK_MHZ="$GPU_LOCK_MHZ" "$ROOT_DIR/scripts/tune_sparks.sh"

    # Verify >= 111 GiB MemAvailable on both nodes before loading 98.9 GiB Rank Cache
    head_avail=$(awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo)
    worker_avail=$(ssh -o BatchMode=yes "$WORKER_IP" "awk '/^MemAvailable:/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo")
    log "Pre-launch Free UMA: Head=${head_avail} GiB, Worker=${worker_avail} GiB"
    if awk -v a="$head_avail" 'BEGIN {exit !(a < 110.0)}'; then
        die "Head has only ${head_avail} GiB MemAvailable (need >= 110 GiB for 98.9 GiB weights + KV pool)."
    fi
    if awk -v a="$worker_avail" 'BEGIN {exit !(a < 110.0)}'; then
        die "Worker has only ${worker_avail} GiB MemAvailable (need >= 110 GiB for 98.9 GiB weights + KV pool)."
    fi
fi

mkdir -p "$ROOT_DIR/logs"

# 3. Launch Backend Engine on internal port $UPSTREAM_PORT (8000)
if [[ "$ENGINE_BACKEND" == "zig" ]]; then
    ZIG_IMAGE="${ZIG_IMAGE:-tensorfold-dsv41:exl3-tp2-625d420e24a4}"
    CONTAINER_NAME="dsv41-1017-exl3-tf"
    HF_CACHE="${HF_CACHE:-/home/massoud/.cache/huggingface}"
    MODEL_REV="45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b"
    ENGRAM_REV="1bae0e1088993b9b82de08f59de5be5a34e42170"
    DATA_SUB="tensorfold-dsv41/45ee5d0f"
    KIT_SUB="tensorfold-dsv41/kit-abe07c7d/kit"
    MODEL_IN_CONTAINER="/hf/hub/models--Mia-AiLab--DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/snapshots/${MODEL_REV}"
    SERVED_IN_CONTAINER="/hf/${DATA_SUB}/model"
    ENGRAM_IN_CONTAINER="/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/${ENGRAM_REV}"

    COMMON_ARGS="--label tf.recipe.ablit=0 --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
    COMMON_ENV="-e TF_DS_REPLAY_FLOOR=1 -e TF_DS_STREAMS=${PARALLEL} -e TF_RDMA_IDLE_S=${RDMA_IDLE_S} -e TF_RDMA_NAP_US=${RDMA_NAP_US} -e HF_HUB_OFFLINE=1 -e TF_DS_PREFILL_INTERLEAVE_MS=${PREFILL_INTERLEAVE_MS} -e TF_DS_IDX_SPLIT=${TF_DS_IDX_SPLIT} -e TF_DS_IMAGE_CACHE=${TF_DS_IMAGE_CACHE} -e TF_RDMA_GID_INDEX=${GID_INDEX} -e NCCL_SOCKET_IFNAME=${SOCKET_IFNAME} -e NCCL_IB_HCA=${ROCE_HCAS} -e NCCL_IB_GID_INDEX=${GID_INDEX}"

    WORKER_CMD="docker run -d --name ${CONTAINER_NAME} ${COMMON_ARGS} -v dsv41-1017-exl3-hf:/hf:ro -v \$HOME/.cache/dsv41-tensorfold/engram-1bae0e10:/engram:ro -v \$HOME/.cache/dsv41-tensorfold/rank-cache-45ee5d0f:/rank-cache:ro ${COMMON_ENV} ${ZIG_IMAGE} tf-dsv41-lanes ${MODEL_IN_CONTAINER} /rank-cache 1 2 ${HEAD_IP} ${MASTER_PORT} /hf/${KIT_SUB} --engram /engram --token-map /hf/${DATA_SUB}/dsv41_token_map.json --pool ${CONTEXT} --streams ${PARALLEL} --drafts ${DRAFTS} --graphs 1 --side 1 --prefetch 1 --aio 1 --rdma ${ROCE_HCAS}"

    THINK_FLAG="--thinking"
    [[ "$THINKING" == "0" ]] && THINK_FLAG="--no-thinking"
    EFFORT_ARGS=""
    [[ -n "$REASONING_EFFORT" ]] && EFFORT_ARGS="--reasoning-effort ${REASONING_EFFORT}"
    BUDGET_ARGS=""
    [[ -n "$THINKING_BUDGET" ]] && BUDGET_ARGS="--thinking-budget ${THINKING_BUDGET}"

    HEAD_CMD="docker run -d --name ${CONTAINER_NAME} ${COMMON_ARGS} -v ${HF_CACHE}:/hf:ro -v ${ROOT_DIR}:${ROOT_DIR}:ro -e TF_TP_WORLD=2 -e TF_TP_MASTER=${HEAD_IP} -e TF_TP_PORT=${MASTER_PORT} -e TF_DS_RANK_CACHE=/hf/${DATA_SUB}/rank-cache -e TF_DS_KIT=/hf/${KIT_SUB} -e TENSORFOLD_CUDA_KERNELS=/hf/${KIT_SUB}/aot -e TF_DS_ENGRAM=${ENGRAM_IN_CONTAINER} -e TF_DS_TOKEN_MAP=/hf/${DATA_SUB}/dsv41_token_map.json -e TF_DS_GRAPHS=1 -e TF_DS_HC_SIDE=1 -e TF_DS_L2_PREFETCH=1 -e TF_DS_ENGRAM_AIO=1 -e TF_RDMA_DEVICES=${ROCE_HCAS} ${COMMON_ENV} ${ZIG_IMAGE} tensorfold-native serve ${SERVED_IN_CONTAINER} --backend cuda --host 127.0.0.1 --port ${UPSTREAM_PORT} --name DeepSeek-V4.1-Flash-1017 --context ${CONTEXT} --parallel ${PARALLEL} --max-tokens ${MAX_TOKENS} --temperature 1.0 --top-p 0.95 --top-k 20 ${THINK_FLAG} ${EFFORT_ARGS} ${BUDGET_ARGS}"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run:zig] WORKER ($WORKER_IP):"
        echo "  $WORKER_CMD"
        echo "[dry-run:zig] HEAD ($HEAD_IP):"
        echo "  $HEAD_CMD"
        exit 0
    fi

    log "Starting Zig Rank 1 container ($CONTAINER_NAME) on Worker ($WORKER_IP)..."
    ssh -o BatchMode=yes "$WORKER_IP" "$WORKER_CMD" >/dev/null

    log "Starting Zig Rank 0 container ($CONTAINER_NAME) on Head ($HEAD_IP -> 127.0.0.1:$UPSTREAM_PORT)..."
    eval "$HEAD_CMD" >/dev/null

elif [[ "$ENGINE_BACKEND" == "python" ]]; then
    PY_IMAGE="${PY_IMAGE:-tensorfold-dsv41:730d41abd8f4}"
    MODEL_VIEW="${MODEL_VIEW:-/home/massoud/models/deepseek-v41-1017-exl3-2.9bpw}"
    CACHE_DIR="${CACHE_DIR:-/home/massoud/tfv41-cache}"
    # CRITICAL FIX: MEM_FLOOR_GIB=0 disables the buggy static set_per_process_memory_fraction()
    # ceiling lock in _memory_ceiling() that caused OutOfMemoryError on 12 MiB allocations after warm().
    MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-0}"

    run_py_args() {
        local r="$1"
        cat <<EOF
-d --name tfv41-r${r} --gpus all --network host --ipc host --shm-size 32g --oom-score-adj 1000
--ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_PTRACE
-v ${MODEL_VIEW}/model:/model:ro -v ${MODEL_VIEW}/engram:/engram:ro -v ${CACHE_DIR}:/cache
-e TF_DS_REPLAY=1 -e TF_DS_REPLAY_FLOOR=1 -e TF_DS_PREFILL_CHUNK=2048
-e TF_DS_RANK_CACHE=/cache/tf-ds-rank -e TF_DS_RANK_CACHE_READERS=32
-e TF_DS_WARM_LENGTHS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,33,131,514,1024,2113 -e TF_DS_ENGRAM=/engram
-e TF_DS_SPILL_GIB=64 -e TF_DS_SPILL_DIR=/cache/tf-ds-spill
-e TF_DS_ENGRAM_DONTNEED=1 -e TF_DS_MEM_FLOOR_GIB=${MEM_FLOOR_GIB}
-e TF_DROP_UNOFFERED_TOOL_CALLS=1
-e TRITON_CACHE_DIR=/cache/triton -e CUDA_CACHE_PATH=/cache/nv
-e TF_RDMA_DEVICES=${ROCE_HCAS} -e TF_RDMA_GID_INDEX=${GID_INDEX}
-e NCCL_IB_HCA=${ROCE_HCAS} -e NCCL_IB_GID_INDEX=${GID_INDEX} -e NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
-e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e HF_HUB_OFFLINE=1
--entrypoint tensorfold ${PY_IMAGE} serve /model --tp 2 --rank ${r} --master ${HEAD_IP}
--host 127.0.0.1 --port ${UPSTREAM_PORT} --name DeepSeek-V4.1-Flash-1017
--context ${CONTEXT} --vision --parallel ${PARALLEL} --mtp-drafts ${MTP_DRAFTS}
--temperature 1.0 --top-p 1.0 $( [[ "$THINKING" == "1" ]] && echo "--thinking" || echo "--no-thinking" ) --no-update-check
EOF
    }

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run:python] HEAD (rank 0):"
        echo "  docker run $(run_py_args 0 | tr '\n' ' ')"
        echo "[dry-run:python] WORKER (rank 1):"
        echo "  docker run $(run_py_args 1 | tr '\n' ' ')"
        exit 0
    fi

    log "Starting Python Rank 0 container (tfv41-r0) on Head (MEM_FLOOR_GIB=$MEM_FLOOR_GIB)..."
    # shellcheck disable=SC2046
    docker run $(run_py_args 0 | tr '\n' ' ') >/dev/null
    for _ in $(seq 1 120); do ss -ltn | grep -q ":29551 " && break; sleep 2; done

    log "Starting Python Rank 1 container (tfv41-r1) on Worker ($WORKER_IP)..."
    ssh -o BatchMode=yes "$WORKER_IP" "docker run $(run_py_args 1 | tr '\n' ' ')" >/dev/null
else
    die "Unsupported ENGINE_BACKEND='$ENGINE_BACKEND' (must be 'zig' or 'python')"
fi

# 4. Wait for Upstream Engine Readiness
log "Waiting for DeepSeek-V4.1-Flash-1017 ($ENGINE_BACKEND backend) to finish loading ~98.9 GiB/rank..."
start_ts=$SECONDS
until curl -sf --max-time 5 "http://127.0.0.1:${UPSTREAM_PORT}/v1/models" >/dev/null 2>&1; do
    if (( SECONDS - start_ts >= WAIT_TIMEOUT )); then
        die "Timed out waiting for upstream engine after ${WAIT_TIMEOUT}s. Check docker logs."
    fi
    sleep 5
done

# 5. Start Spark-DSV41 CacheGuard Daemon & Go Control Plane inside the persistent container scope
HEAD_CNAME="dsv41-1017-exl3-tf"
WORKER_CNAME="dsv41-1017-exl3-tf"
if [[ "$ENGINE_BACKEND" == "python" ]]; then
    HEAD_CNAME="tfv41-r0"
    WORKER_CNAME="tfv41-r1"
fi

log "Starting spark-dsv41-cacheguard daemon on Head and Worker..."
docker cp "$ROOT_DIR/bin/spark-dsv41-cacheguard" "${HEAD_CNAME}:/usr/local/bin/spark-dsv41-cacheguard"
docker exec -d "$HEAD_CNAME" /usr/local/bin/spark-dsv41-cacheguard -interval 15s /hf/tensorfold-dsv41/45ee5d0f/rank-cache /hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 /cache/tf-ds-rank /engram
ssh -o BatchMode=yes "$WORKER_IP" "docker cp /home/massoud/spark-dsv41/bin/spark-dsv41-cacheguard ${WORKER_CNAME}:/usr/local/bin/spark-dsv41-cacheguard && docker exec -d ${WORKER_CNAME} /usr/local/bin/spark-dsv41-cacheguard -interval 15s /rank-cache /engram /cache/tf-ds-rank"

log "Starting Spark-DSV41 Go Control Plane on :${PORT} -> http://127.0.0.1:${UPSTREAM_PORT}..."
docker cp "$ROOT_DIR/bin/spark-dsv41-gateway" "${HEAD_CNAME}:/usr/local/bin/spark-dsv41-gateway"
docker exec "$HEAD_CNAME" mkdir -p /etc/spark-dsv41
docker cp "$ROOT_DIR/config/model_aliases.json" "${HEAD_CNAME}:/etc/spark-dsv41/model_aliases.json"
docker exec -d "$HEAD_CNAME" /usr/local/bin/spark-dsv41-gateway \
    -listen ":${PORT}" \
    -upstream "http://127.0.0.1:${UPSTREAM_PORT}" \
    -aliases-config "/etc/spark-dsv41/model_aliases.json"
sleep 1

# Run immediate cacheguard sweep right after 98.9 GiB weight load completes
"$ROOT_DIR/bin/spark-dsv41-cacheguard" --once >/dev/null 2>&1 || true
ssh -o BatchMode=yes "$WORKER_IP" "/home/massoud/spark-dsv41/bin/spark-dsv41-cacheguard --once >/dev/null 2>&1 || true"

head_after=$(awk '/^MemAvailable:/ {printf "%.2f", $2/1048576}' /proc/meminfo)
worker_after=$(ssh -o BatchMode=yes "$WORKER_IP" "awk '/^MemAvailable:/ {printf \"%.2f\", \$2/1048576}' /proc/meminfo")
log "Spark-DSV41 ($ENGINE_BACKEND backend) is LIVE on port ${PORT} in $((SECONDS - start_ts))s!"
log "Post-load Free UMA: Head=${head_after} GiB, Worker=${worker_after} GiB"
