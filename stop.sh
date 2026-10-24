#!/usr/bin/env bash
# ==============================================================================
# Spark-DSV41: Clean Cluster Shutdown Script (Head + Worker)
# Stops spark-dsv41-gateway, spark-dsv41-cacheguard, and TensorFold containers
# (`dsv41-1017-exl3-tf`, `tfv41-r0`, `tfv41-r1`) on both DGX Sparks.
# ==============================================================================
set -euo pipefail

WORKER_HOST="${WORKER_HOST:-10.100.240.2}"
STOP_TIMEOUT="${STOP_TIMEOUT:-20}"

log() {
    echo -e "\033[1;36m[spark-dsv41:stop]\033[0m $*"
}

log "Stopping Spark-DSV41 Go Gateway and CacheGuard on Head..."
pkill -f "spark-dsv41-gateway" 2>/dev/null || true
pkill -f "spark-dsv41-cacheguard" 2>/dev/null || true
pkill -f "dsv41-memguard" 2>/dev/null || true

for cname in dsv41-1017-exl3-tf tfv41-r0 spark-dsv41-r0; do
    if docker ps -a --format '{{.Names}}' | grep -qx "$cname"; then
        log "Stopping container $cname on Head..."
        docker stop -t "$STOP_TIMEOUT" "$cname" >/dev/null 2>&1 || true
        docker rm -f "$cname" >/dev/null 2>&1 || true
    fi
done

if ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" true 2>/dev/null; then
    log "Stopping Spark-DSV41 CacheGuard and TensorFold containers on Worker ($WORKER_HOST)..."
    ssh -o BatchMode=yes "$WORKER_HOST" "
        pkill -f 'spark-dsv41-cacheguard' 2>/dev/null || true
        pkill -f 'dsv41-memguard' 2>/dev/null || true
        for cname in dsv41-1017-exl3-tf tfv41-r1 spark-dsv41-r1; do
            if docker ps -a --format '{{.Names}}' | grep -qx \"\$cname\"; then
                docker stop -t $STOP_TIMEOUT \"\$cname\" >/dev/null 2>&1 || true
                docker rm -f \"\$cname\" >/dev/null 2>&1 || true
            fi
        done
    "
fi

# Flush any lingering Engram/Rank Cache pages from Linux page cache
if [ -x /home/massoud/spark-dsv41/bin/spark-dsv41-cacheguard ]; then
    /home/massoud/spark-dsv41/bin/spark-dsv41-cacheguard --once >/dev/null 2>&1 || true
fi
ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "/home/massoud/spark-dsv41/bin/spark-dsv41-cacheguard --once >/dev/null 2>&1 || true" 2>/dev/null || true

head_avail=$(awk '/^MemAvailable:/ {printf "%.2f", $2/1048576}' /proc/meminfo)
worker_avail=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "awk '/^MemAvailable:/ {printf \"%.2f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo "?")
log "Spark-DSV41 stopped cleanly. Free UMA: Head=${head_avail} GiB, Worker=${worker_avail} GiB."
