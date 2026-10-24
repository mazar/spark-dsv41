#!/usr/bin/env bash
# ==============================================================================
# Spark-DSV41: Dual-Rail ConnectX-7 RoCE v2, 3003 MHz GPU Clock Lock & GB10 UMA Hygiene
# Target Hardware: 2x NVIDIA DGX Spark (GB10 Grace Blackwell, sm_121a)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD_R0_IP="${HEAD_R0_IP:-10.100.240.1}"
HEAD_R1_IP="${HEAD_R1_IP:-10.100.241.1}"
WORKER_R0_IP="${WORKER_R0_IP:-10.100.240.2}"
WORKER_R1_IP="${WORKER_R1_IP:-10.100.241.2}"
WORKER_HOST="${WORKER_HOST:-$WORKER_R0_IP}"
GPU_LOCK_MHZ="${GPU_LOCK_MHZ:-3003}"

log() {
    echo -e "\033[1;36m[spark-dsv41:tune]\033[0m $*"
}

apply_node_tuning() {
    local role="$1"
    local r0_ip="$2"
    local r1_ip="$3"

    # 1. Configure ConnectX-7 Twin Interfaces (MTU 4200, IPv4, IPv6 link-local off for deterministic GID index 3)
    for dev in enp1s0f1np1 enP2p1s0f1np1; do
        if ip link show "$dev" >/dev/null 2>&1; then
            sudo -n ip link set dev "$dev" mtu 4200 up 2>/dev/null || true
            sudo -n sysctl -qw "net.ipv6.conf.${dev}.disable_ipv6=1" 2>/dev/null || true
        fi
    done

    if ip link show enp1s0f1np1 >/dev/null 2>&1; then
        if ! ip -4 addr show dev enp1s0f1np1 | grep -q "$r0_ip"; then
            sudo -n ip addr add "${r0_ip}/24" dev enp1s0f1np1 2>/dev/null || true
        fi
    fi
    if ip link show enP2p1s0f1np1 >/dev/null 2>&1; then
        if ! ip -4 addr show dev enP2p1s0f1np1 | grep -q "$r1_ip"; then
            sudo -n ip addr add "${r1_ip}/24" dev enP2p1s0f1np1 2>/dev/null || true
        fi
    fi

    # 2. GB10 UMA Hygiene: disable THP, disable swap, aggressively reclaim VFS cache, drop clean caches
    if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
        echo never | sudo -n tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
    elif sudo -n true 2>/dev/null; then
        echo never | sudo -n tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
    fi
    sudo -n swapoff -a 2>/dev/null || true
    sudo -n sysctl -qw vm.swappiness=0 2>/dev/null || true
    sudo -n sysctl -qw vm.vfs_cache_pressure=200 2>/dev/null || true
    sudo -n sysctl -qw vm.overcommit_memory=1 2>/dev/null || true
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

    # 3. Lock GB10 GPU Clocks at 3003 MHz (+3% to +5% sustained prefill and decode throughput)
    if [[ "$GPU_LOCK_MHZ" != "0" ]]; then
        sudo -n nvidia-smi -lgc "${GPU_LOCK_MHZ},${GPU_LOCK_MHZ}" >/dev/null 2>&1 || true
    fi

    # 4. Evict Rank Cache & Engram safetensors from OS page cache
    if [ -x /home/massoud/spark-dsv41/bin/spark-dsv41-cacheguard ]; then
        /home/massoud/spark-dsv41/bin/spark-dsv41-cacheguard --once 2>/dev/null || true
    fi

    local mem_avail
    mem_avail=$(awk '/^MemAvailable:/ {printf "%.2f", $2/1048576}' /proc/meminfo)
    local clk
    clk=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")
    echo "[$role] Tuned: RoCE MTU=4200 (${r0_ip}, ${r1_ip}), GPU Clock=${clk}, MemAvailable=${mem_avail} GiB"
}

log "Tuning Head Node (local)..."
apply_node_tuning "HEAD" "$HEAD_R0_IP" "$HEAD_R1_IP"

log "Tuning Worker Node ($WORKER_HOST)..."
ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" "$(declare -f apply_node_tuning); GPU_LOCK_MHZ='$GPU_LOCK_MHZ' apply_node_tuning WORKER '$WORKER_R0_IP' '$WORKER_R1_IP'"

if [[ "${1:-}" == "--verify" ]]; then
    log "Verifying RoCE v2 Dual-Rail connectivity (MTU 4200)..."
    ping -c 2 -W 2 -I enp1s0f1np1 -M do -s 4172 "$WORKER_R0_IP" >/dev/null
    ping -c 2 -W 2 -I enP2p1s0f1np1 -M do -s 4172 "$WORKER_R1_IP" >/dev/null
    log "Both ConnectX-7 RoCE v2 rails (rocep1s0f1 + roceP2p1s0f1) verified at MTU 4200."
fi
