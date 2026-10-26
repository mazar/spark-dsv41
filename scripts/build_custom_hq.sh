#!/usr/bin/env bash
# ==============================================================================
# Spark-DSV41: One-Command Custom Hybrid EXL3 RankCache Builder (custom-hq)
# ==============================================================================
# Upgrades both Rank 0 (Head) and Rank 1 (Worker) TP=2 RankCache files in parallel
# directly from the official unquantized `deepseek-ai/DeepSeek-V4.1-Flash-1017`
# (`1bae0e1088993b9b82de08f59de5be5a34e42170`) source weights over HTTP Range
# requests (~12 GiB streamed per rank, encoded on GPU via exllamav3_ext in ~31 min):
#   - 8.0 bpw (K=8, K2=16): All 363 Rank-local Attention / MLA Compressor /
#     Sparse Indexer / Engram-wkv / DSpark MTP / LM-Head matrices
#   - 4.0 bpw (K=4, K2=8): All 21 5-bit shared_experts layers (0..10, 30..39)
#     to unlock grouped_mma3_kernel & grouped_down3_kernel across all 40 layers
#   - 3.0 bpw (K=3, K2=6): Top 384 highest-impact routed experts across the
#     Layers 18..22 2.0-bpw cliff (top 128 in Layer 20 + top 64 in Layers 18,19,21,22)
#
# Usage:
#   ./scripts/build_custom_hq.sh
#   ./scripts/build_custom_hq.sh --smoke-test   # Verify 5 matrices & exit
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WORKER_IP="${WORKER_IP:-10.100.240.2}"
QUANT_IMAGE="${QUANT_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
STOCK_REV="1bae0e1088993b9b82de08f59de5be5a34e42170"
SMOKE_TEST=0

for arg in "$@"; do
    case "$arg" in
        --smoke-test) SMOKE_TEST=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
    esac
done

log() { echo -e "\033[1;36m[build-custom-hq]\033[0m $*"; }
die() { echo -e "\033[1;31m[build-custom-hq ERROR]\033[0m $*" >&2; exit 1; }

HEAD_IN=$(ls "$HOME"/.cache/huggingface/tensorfold-dsv41/45ee5d0f/rank-cache/rank0of2-*.bin 2>/dev/null | head -n 1 || true)
WORKER_IN=$(ssh -o BatchMode=yes "$WORKER_IP" 'ls $HOME/.cache/dsv41-tensorfold/rank-cache-45ee5d0f/rank1of2-*.bin 2>/dev/null | head -n 1' || true)

if [[ -z "$HEAD_IN" || -z "$WORKER_IN" ]]; then
    log "Baseline 45ee5d0f RankCache not found yet; running ./scripts/prepare.sh first..."
    WORKER="$WORKER_IP" "$ROOT_DIR/scripts/prepare.sh"
    HEAD_IN=$(ls "$HOME"/.cache/huggingface/tensorfold-dsv41/45ee5d0f/rank-cache/rank0of2-*.bin 2>/dev/null | head -n 1 || true)
    WORKER_IN=$(ssh -o BatchMode=yes "$WORKER_IP" 'ls $HOME/.cache/dsv41-tensorfold/rank-cache-45ee5d0f/rank1of2-*.bin 2>/dev/null | head -n 1' || true)
fi

[[ -n "$HEAD_IN" ]] || die "Head baseline RankCache not found in ~/.cache/huggingface/tensorfold-dsv41/45ee5d0f/rank-cache/"
[[ -n "$WORKER_IN" ]] || die "Worker baseline RankCache not found on $WORKER_IP:~/.cache/dsv41-tensorfold/rank-cache-45ee5d0f/"

if ! docker image inspect "$QUANT_IMAGE" >/dev/null 2>&1; then
    log "Pulling EXL3 GPU quantizer image ($QUANT_IMAGE) on Head..."
    docker pull "$QUANT_IMAGE"
fi
if ! ssh -o BatchMode=yes "$WORKER_IP" "docker image inspect '$QUANT_IMAGE' >/dev/null 2>&1"; then
    log "Ensuring EXL3 GPU quantizer image ($QUANT_IMAGE) on Worker ($WORKER_IP)..."
    ssh -o BatchMode=yes "$WORKER_IP" "docker pull '$QUANT_IMAGE'" || {
        log "Transferring $QUANT_IMAGE from Head to Worker via docker save | docker load..."
        docker save "$QUANT_IMAGE" | ssh -o BatchMode=yes "$WORKER_IP" "docker load"
    }
fi

HEAD_ENGRAM_DIR="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/${STOCK_REV}"
mkdir -p "$HEAD_ENGRAM_DIR"
if [[ ! -f "$HEAD_ENGRAM_DIR/model.safetensors.index.json" ]]; then
    log "Downloading model.safetensors.index.json (${STOCK_REV:0:8})..."
    curl -sSL "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/${STOCK_REV}/model.safetensors.index.json" \
        -o "$HEAD_ENGRAM_DIR/model.safetensors.index.json"
fi

ssh -o BatchMode=yes "$WORKER_IP" "mkdir -p \$HOME/spark-dsv41/scripts \$HOME/.cache/dsv41-tensorfold/engram-${STOCK_REV:0:8}"
scp -q "$ROOT_DIR/scripts/quantize_rank_cache.py" "$WORKER_IP:~/spark-dsv41/scripts/quantize_rank_cache.py"
scp -q "$HEAD_ENGRAM_DIR/model.safetensors.index.json" "$WORKER_IP:~/.cache/dsv41-tensorfold/engram-${STOCK_REV:0:8}/model.safetensors.index.json"

if [[ "$SMOKE_TEST" == "1" ]]; then
    log "Running 5-matrix GPU quantization smoke test on Head..."
    docker run --rm --gpus all --entrypoint python3 \
        -v "$ROOT_DIR:$ROOT_DIR:ro" \
        -v "$HOME/.cache:$HOME/.cache" \
        "$QUANT_IMAGE" \
        "$ROOT_DIR/scripts/quantize_rank_cache.py" \
            --input-cache "$HEAD_IN" \
            --output-dir "$HOME/.cache/huggingface/tensorfold-dsv41/custom-hq/rank-cache" \
            --stock-index "$HEAD_ENGRAM_DIR/model.safetensors.index.json" \
            --engram-dir "$HEAD_ENGRAM_DIR" \
            --limit-quant 5
    exit 0
fi

log "Launching parallel GPU EXL3 RankCache quantization on Head (quant-r0) and Worker (quant-r1)..."
docker rm -f quant-r0 >/dev/null 2>&1 || true
ssh -o BatchMode=yes "$WORKER_IP" "docker rm -f quant-r1 >/dev/null 2>&1 || true"

docker run -d --name quant-r0 --gpus all --entrypoint python3 \
    -v "$ROOT_DIR:$ROOT_DIR:ro" \
    -v "$HOME/.cache:$HOME/.cache" \
    "$QUANT_IMAGE" \
    "$ROOT_DIR/scripts/quantize_rank_cache.py" \
        --input-cache "$HEAD_IN" \
        --output-dir "$HOME/.cache/huggingface/tensorfold-dsv41/custom-hq/rank-cache" \
        --stock-index "$HEAD_ENGRAM_DIR/model.safetensors.index.json" \
        --engram-dir "$HEAD_ENGRAM_DIR" >/dev/null

ssh -o BatchMode=yes "$WORKER_IP" "docker run -d --name quant-r1 --gpus all --entrypoint python3 \
    -v \$HOME/spark-dsv41:\$HOME/spark-dsv41:ro \
    -v \$HOME/.cache:\$HOME/.cache \
    $QUANT_IMAGE \
    \$HOME/spark-dsv41/scripts/quantize_rank_cache.py \
        --input-cache $WORKER_IN \
        --output-dir \$HOME/.cache/dsv41-tensorfold/rank-cache-custom-hq \
        --stock-index \$HOME/.cache/dsv41-tensorfold/engram-${STOCK_REV:0:8}/model.safetensors.index.json \
        --engram-dir \$HOME/.cache/dsv41-tensorfold/engram-${STOCK_REV:0:8} >/dev/null"

log "Monitoring quant-r0 (Head) and quant-r1 (Worker)..."
while true; do
    r0_state=$(docker inspect -f '{{.State.Running}}' quant-r0 2>/dev/null || echo "false")
    r1_state=$(ssh -o BatchMode=yes "$WORKER_IP" "docker inspect -f '{{.State.Running}}' quant-r1 2>/dev/null || echo false")
    echo "--- $(date '+%H:%M:%S') ---"
    docker logs --tail 2 quant-r0 2>&1 || true
    ssh -o BatchMode=yes "$WORKER_IP" "docker logs --tail 2 quant-r1 2>&1 || true"
    if [[ "$r0_state" != "true" && "$r1_state" != "true" ]]; then
        break
    fi
    sleep 30
done

r0_rc=$(docker inspect -f '{{.State.ExitCode}}' quant-r0)
r1_rc=$(ssh -o BatchMode=yes "$WORKER_IP" "docker inspect -f '{{.State.ExitCode}}' quant-r1")
[[ "$r0_rc" == "0" ]] || die "quant-r0 failed with exit code $r0_rc"
[[ "$r1_rc" == "0" ]] || die "quant-r1 failed with exit code $r1_rc"

docker rm -f quant-r0 >/dev/null
ssh -o BatchMode=yes "$WORKER_IP" "docker rm -f quant-r1 >/dev/null"

ln -sfn ../45ee5d0f/model "$HOME/.cache/huggingface/tensorfold-dsv41/custom-hq/model"
ln -sfn ../45ee5d0f/dsv41_token_map.json "$HOME/.cache/huggingface/tensorfold-dsv41/custom-hq/dsv41_token_map.json"

if [[ -x "$ROOT_DIR/bin/spark-dsv41-cacheguard" ]]; then
    "$ROOT_DIR/bin/spark-dsv41-cacheguard" --once || true
    ssh -o BatchMode=yes "$WORKER_IP" "\$HOME/spark-dsv41/bin/spark-dsv41-cacheguard --once || true"
fi

log "SUCCESS! Custom Hybrid EXL3 RankCache (custom-hq) is ready on both Sparks."
log "Run ./scripts/start_engine.sh to serve with RANK_CACHE_TAG=custom-hq."
