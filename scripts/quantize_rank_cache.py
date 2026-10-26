#!/usr/bin/env python3
"""
Spark-DSV41 Custom Hybrid Rank-Cache Quantizer
==============================================
Upgrades a TP=2 RankCache file (`rank0of2-*.bin` or `rank1of2-*.bin`) directly from
the unquantized `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`1bae0e1088993b9b82de08f59de5be5a34e42170`)
FP8/BF16/FP4 source weights using the GPU `exllamav3_ext` `mul1` Viterbi trellis encoder:

1. All 363 Rank-local Attention / Compressor / Indexer / Engram-wkv / DSpark-main_proj / LM-head
   matrices (`wq_a`, `wq_b`, `wkv`, `wo_a.slice.*`, `wo_b`, `compressor.wkv/wgate`, `indexer.wq_b`,
   `engram.wkv`, `mtp.0.main_proj`, `head`) -> upgraded from 4b/5b/6b to **8.0 bpw (`K=8`, `K2=16`)**.
2. All 21 5-bit `shared_experts.{w1,w2,w3}` groups (Layers 0..10 and 30..39, 63 matrices) ->
   standardized to **4.0 bpw (`K=4`, `K2=8`)** so every layer's fused `experts` pack satisfies
   `k2_gu[1] <= 8` and `k2_d[1] <= 8`, unlocking `grouped_mma3_kernel` and `grouped_down3_kernel`
   on ALL 40 layers!
3. Layers 18..22 Routed Experts (`2.0 bpw` cliff in Mia EAQ):
   - Layer 20 (CSA2 candidate source): top 128 experts upgraded from `2.0 bpw` to **3.0 bpw (`K=3`, `K2=6`)**
   - Layers 18, 19, 21, 22: top 64 experts per layer upgraded from `2.0 bpw` to **3.0 bpw (`K=3`, `K2=6`)**
   (384 total experts = 1,152 expert matrices per rank upgraded to 3.0 bpw, ranked deterministically
   by `(gate.bias - min(gate.bias) + 0.1) * ||w2.suh||_2 * ||w2.svh||_2`).
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
import torch

# Bypass exllamav3/__init__.py (which imports optional flash_attn/kbnf) and load the compiled CUDA extension directly
for mod in ["exllamav3", "exllamav3.modules", "exllamav3.modules.quant", "exllamav3.modules.quant.exl3_lib"]:
    if mod not in sys.modules:
        pkg = types.ModuleType(mod)
        pkg.__path__ = ["/usr/local/lib/python3.12/dist-packages/" + mod.replace(".", "/")]
        sys.modules[mod] = pkg

from exllamav3.modules.quant.exl3_lib.quantize import (
    blockwise_preapply_had_l_,
    blockwise_preapply_had_r_,
    pack_trellis,
    quantize_tiles,
    tensor_core_perm,
)

REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REV = "1bae0e1088993b9b82de08f59de5be5a34e42170"
MAGIC = b"TFDSRK01"

FP4_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def parse_range(text: str):
    if text == "None":
        return None
    assert text.startswith("(") and text.endswith(")"), f"Bad range: {text}"
    a, b = text[1:-1].split(", ")
    return int(a), int(b)


class StockSource:
    def __init__(self, index_json_path: str, local_engram_dir: str | None = None):
        with open(index_json_path, "r") as f:
            self.weight_map = json.load(f)["weight_map"]
        self.local_dirs = [Path(index_json_path).parent]
        if local_engram_dir:
            self.local_dirs.append(Path(local_engram_dir))
        self.headers = {}
        self.cdn_urls = {}
        self.hdr_lock = threading.Lock()
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _local_shard(self, shard: str) -> Path | None:
        for d in self.local_dirs:
            p = d / shard
            if p.exists():
                return p
        return None

    def get_header(self, shard: str):
        if shard in self.headers:
            return self.headers[shard]
        with self.hdr_lock:
            if shard in self.headers:
                return self.headers[shard]
            loc = self._local_shard(shard)
            if loc is not None:
                with open(loc, "rb") as f:
                    hlen = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(hlen))
                self.headers[shard] = (8 + hlen, hdr)
                return self.headers[shard]

            url = f"https://huggingface.co/{REPO_ID}/resolve/{REV}/{shard}"
            for attempt in range(5):
                try:
                    resp = self.session.get(url, headers={"Range": "bytes=0-7"}, timeout=30)
                    resp.raise_for_status()
                    cdn_url = resp.url
                    hlen = struct.unpack("<Q", resp.content)[0]
                    resp2 = self.session.get(cdn_url, headers={"Range": f"bytes=8-{8+hlen-1}"}, timeout=30)
                    resp2.raise_for_status()
                    hdr = json.loads(resp2.content)
                    self.cdn_urls[shard] = cdn_url
                    self.headers[shard] = (8 + hlen, hdr)
                    return self.headers[shard]
                except Exception as e:
                    if attempt == 4:
                        raise
                    time.sleep(1.5 * (attempt + 1))

    def fetch_bytes(self, shard: str, rel_start: int, rel_end: int) -> bytes:
        base, _ = self.get_header(shard)
        loc = self._local_shard(shard)
        if loc is not None:
            with open(loc, "rb") as f:
                f.seek(base + rel_start)
                return f.read(rel_end - rel_start)

        url = self.cdn_urls.get(shard, f"https://huggingface.co/{REPO_ID}/resolve/{REV}/{shard}")
        abs_start = base + rel_start
        abs_end = base + rel_end - 1
        for attempt in range(6):
            try:
                resp = self.session.get(url, headers={"Range": f"bytes={abs_start}-{abs_end}"}, timeout=60)
                if resp.status_code in (401, 403):
                    # Refresh signed CDN URL if expired
                    url = f"https://huggingface.co/{REPO_ID}/resolve/{REV}/{shard}"
                    resp = self.session.get(url, headers={"Range": f"bytes={abs_start}-{abs_end}"}, timeout=60)
                    self.cdn_urls[shard] = resp.url
                resp.raise_for_status()
                data = resp.content
                if len(data) != (rel_end - rel_start):
                    raise IOError(f"Short HTTP range read on {shard}: got {len(data)}, expected {rel_end - rel_start}")
                return data
            except Exception as e:
                if attempt == 5:
                    raise
                time.sleep(1.5 * (attempt + 1))

    def prefetch_batch(self, prefixes_and_slices: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], torch.Tensor]:
        """
        Fetch and dequantize a batch of (prefix, cols_str, rows_str) in parallel using a ThreadPoolExecutor!
        """
        import concurrent.futures

        def _one(item):
            p, c, r = item
            return item, self.load_sliced_weight(p, c, r)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            return dict(ex.map(_one, prefixes_and_slices))

    def load_sliced_weight(self, prefix: str, cols_str: str, rows_str: str) -> torch.Tensor:
        """
        Reconstruct the exact FP32 [K_slice, N_slice] weight matrix (in EXL3 [in_features, out_features] orientation)
        from the unquantized stock 1017 tensor (which is stored in PyTorch [out_features, in_features] orientation).
        """
        cols = parse_range(cols_str)
        rows = parse_range(rows_str)

        if ".slice." in prefix:
            stock_prefix, _, sidx = prefix.partition(".slice.")
            g = int(sidx)
            assert cols is None and rows is None
            cols = (g * 1024, (g + 1) * 1024)
        else:
            stock_prefix = prefix

        w_key = f"{stock_prefix}.weight"
        s_key = f"{stock_prefix}.scale"
        shard = self.weight_map[w_key]
        _, hdr = self.get_header(shard)

        w_info = hdr[w_key]
        w_dt = w_info["dtype"]
        w_sh = w_info["shape"]  # [N_full, K_full] (or [N_full, K_full // 2] for I8 FP4)
        w_lo, w_hi = w_info["data_offsets"]

        # Note: in PyTorch [N, K] layout:
        # - `cols` (EXL3 N slice) slices dim 0 (rows of stock weight)!
        # - `rows` (EXL3 K slice) slices dim 1 (columns of stock weight)!
        if w_dt == "BF16":
            N_full, K_full = w_sh
            n0, n1 = cols if cols is not None else (0, N_full)
            k0, k1 = rows if rows is not None else (0, K_full)
            row_bytes = K_full * 2
            raw = self.fetch_bytes(shard, w_lo + n0 * row_bytes, w_lo + n1 * row_bytes)
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(n1 - n0, K_full)[:, k0:k1]
            return t.float().t().contiguous()

        elif w_dt == "F8_E4M3":
            N_full, K_full = w_sh
            n0, n1 = cols if cols is not None else (0, N_full)
            k0, k1 = rows if rows is not None else (0, K_full)
            row_bytes = K_full
            w_raw = self.fetch_bytes(shard, w_lo + n0 * row_bytes, w_lo + n1 * row_bytes)
            w_u8 = np.frombuffer(w_raw, dtype=np.uint8).reshape(n1 - n0, K_full)[:, k0:k1]
            w_f32 = torch.from_numpy(w_u8.copy()).view(torch.float8_e4m3fn).float().numpy()

            s_info = hdr[s_key]
            s_sh = s_info["shape"]
            s_lo, s_hi = s_info["data_offsets"]
            s_raw = self.fetch_bytes(shard, s_lo, s_hi)
            s_u8 = np.frombuffer(s_raw, dtype=np.uint8).reshape(s_sh)[n0 // 32 : n1 // 32, k0 // 32 : k1 // 32]
            s_f32 = np.ldexp(1.0, s_u8.astype(np.int32) - 127).astype(np.float32)
            s_exp = np.repeat(np.repeat(s_f32, 32, axis=0), 32, axis=1)
            return torch.from_numpy((w_f32 * s_exp).T).contiguous()

        elif w_dt == "I8":
            # Packed FP4 e2m1: w_sh is [N_full, K_full // 2], scale is [N_full, K_full // 32]
            N_full, K_half = w_sh
            K_full = K_half * 2
            n0, n1 = cols if cols is not None else (0, N_full)
            k0, k1 = rows if rows is not None else (0, K_full)
            row_bytes = K_half
            w_raw = self.fetch_bytes(shard, w_lo + n0 * row_bytes, w_lo + n1 * row_bytes)
            w_u8 = np.frombuffer(w_raw, dtype=np.uint8).reshape(n1 - n0, K_half)[:, k0 // 2 : k1 // 2]
            w_f32 = np.stack([FP4_LUT[w_u8 & 0xF], FP4_LUT[w_u8 >> 4]], axis=-1).reshape(n1 - n0, k1 - k0)

            s_info = hdr[s_key]
            s_sh = s_info["shape"]
            s_lo, s_hi = s_info["data_offsets"]
            s_row_bytes = s_sh[1]
            s_raw = self.fetch_bytes(shard, s_lo + n0 * s_row_bytes, s_lo + n1 * s_row_bytes)
            s_u8 = np.frombuffer(s_raw, dtype=np.uint8).reshape(n1 - n0, s_sh[1])[:, k0 // 32 : k1 // 32]
            s_f32 = np.ldexp(1.0, s_u8.astype(np.int32) - 127).astype(np.float32)
            s_exp = np.repeat(s_f32, 32, axis=1)
            return torch.from_numpy((w_f32 * s_exp).T).contiguous()

        else:
            raise ValueError(f"Unsupported stock weight dtype {w_dt} for {w_key}")


def select_upgraded_experts(in_file, idx_map: dict) -> dict[int, set[int]]:
    """
    Deterministically select the routed experts in Layers 18..22 to upgrade from 2.0 bpw to 3.0 bpw:
    - Layer 20 (CSA2 candidate source): top 128 experts
    - Layers 18, 19, 21, 22: top 64 experts per layer
    Note: `w2` in the RankCache has `svh` of length `5120` (replicated across ranks!) and `suh` of length `1152`
    (unit sign vector +-1, so `||suh||_2 == sqrt(1152)` on both ranks!).
    To guarantee 100% bitwise identical expert selection on Rank 0 and Rank 1 without depending on rank-sliced `suh`,
    we rank by `(gate.bias - min(gate.bias) + 0.1) * ||w2.svh||_2` (both `gate.bias` and `w2.svh` are replicated across ranks!).
    """
    selected = {}
    for l in [18, 19, 20, 21, 22]:
        b_row = idx_map[f"layers.{l}.ffn.gate.bias|torch.float32"]
        in_file.seek(b_row[3])
        bias = np.frombuffer(in_file.read(b_row[4]), dtype=np.float32)
        svh_norms = np.zeros(384, dtype=np.float64)
        for e in range(384):
            # Find w2 svh key for either rank 0 or rank 1
            k0 = f"layers.{l}.ffn.experts.{e}.w2|None|(0, 1152)|svh"
            k1 = f"layers.{l}.ffn.experts.{e}.w2|None|(1152, 2304)|svh"
            svh_row = idx_map.get(k0) or idx_map[k1]
            in_file.seek(svh_row[3])
            svh = np.frombuffer(in_file.read(svh_row[4]), dtype=np.float16).astype(np.float64)
            svh_norms[e] = np.linalg.norm(svh)
        score = (bias - bias.min() + 0.1) * svh_norms
        top_k = 128 if l == 20 else 64
        top_set = set(int(x) for x in np.argsort(-score)[:top_k])
        selected[l] = top_set
        print(f"[quantizer] Layer {l}: upgrading {len(top_set)} experts from 2.0b -> 3.0b (first 8: {sorted(top_set)[:8]})", flush=True)
    return selected


def quantize_matrix_on_gpu(w_true_cpu: torch.Tensor, suh_bytes: bytes, svh_bytes: bytes, target_bits: int, device: torch.device) -> bytes:
    """
    Encode `w_true_cpu` [K, N] into an EXL3 `mul1` trellis at `target_bits` on `device` using the calibrated `suh` and `svh`.
    Returns the raw `int16 [K/16, N/16, 16 * target_bits]` bytes.
    """
    wr = w_true_cpu.to(device=device, dtype=torch.float32)
    suh = torch.frombuffer(bytearray(suh_bytes), dtype=torch.float16).to(device=device, dtype=torch.float32).unsqueeze(1)
    svh = torch.frombuffer(bytearray(svh_bytes), dtype=torch.float16).to(device=device, dtype=torch.float32).unsqueeze(0)

    wr /= svh
    blockwise_preapply_had_r_(wr, 128)
    wr /= suh
    blockwise_preapply_had_l_(wr, 128)

    tk, tn = wr.shape[0] // 16, wr.shape[1] // 16
    perm = tensor_core_perm(device)
    tiles = wr.reshape(tk, 16, tn, 16).permute(0, 2, 1, 3).reshape(tk * tn, 256)[:, perm].contiguous()
    quant_args = {"K": target_bits, "devices": [device], "mul1": True}
    _, qi = quantize_tiles(tiles, quant_args)
    trellis = pack_trellis(qi.view(tk, tn, 256), quant_args)
    return trellis.cpu().numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", required=True, help="Path to existing rank0of2-*.bin or rank1of2-*.bin")
    parser.add_argument("--output-dir", required=True, help="Directory to write new rank{0,1}of2-<hash>.bin")
    parser.add_argument("--stock-index", required=True, help="Path to stock 1017 model.safetensors.index.json")
    parser.add_argument("--engram-dir", default=None, help="Optional path to local Engram shards 47 & 48")
    parser.add_argument("--limit-quant", type=int, default=0, help="For smoke testing: stop after N quantized matrices")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    # Warm up CUDA extension
    dummy = torch.zeros((16, 256), dtype=torch.float32, device=device)
    quantize_tiles(dummy, {"K": 8, "devices": [device], "mul1": True})
    torch.cuda.synchronize()

    in_path = Path(args.input_cache)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_name = in_path.name
    assert m_name.startswith("rank") and "of" in m_name
    rank_prefix = m_name.split("-")[0]  # e.g. rank0of2 or rank1of2

    stock = StockSource(args.stock_index, args.engram_dir)

    with open(in_path, "rb") as f_in:
        f_in.seek(-16, 2)
        footer_pos = f_in.tell()
        idx_off, magic = struct.unpack("<Q8s", f_in.read(16))
        assert magic == MAGIC, f"Bad magic in {in_path}: {magic}"
        f_in.seek(idx_off)
        old_index = json.loads(f_in.read(footer_pos - idx_off))
        idx_map = {row[0]: row for row in old_index}

        selected_experts = select_upgraded_experts(f_in, idx_map)

        # Determine target bitwidth for every |tr entry that should be re-quantized
        targets = {}  # key -> new_bits (int)
        for key, dt, sh, off, b in old_index:
            if not key.endswith("|tr"):
                continue
            prefix, cols_s, rows_s, _ = key.split("|")
            old_bits = sh[2] // 16
            if ".experts." in prefix:
                # Check if in Layers 18..22 selected_experts
                parts = prefix.split(".")
                if parts[0] == "layers" and int(parts[1]) in selected_experts:
                    lay_idx = int(parts[1])
                    exp_idx = int(parts[4])
                    if exp_idx in selected_experts[lay_idx]:
                        targets[key] = 3
            else:
                if "shared_experts" in prefix:
                    if old_bits != 4:
                        targets[key] = 4
                elif "indexer.wk" in prefix:
                    # Already 8-bit
                    pass
                else:
                    targets[key] = 8

        print(
            f"[quantizer] {rank_prefix}: {len(old_index)} total entries, {len(targets)} matrices scheduled for GPU EXL3 re-quantization",
            flush=True,
        )
        import concurrent.futures
        target_keys = list(targets.keys())
        prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        prefetched_futures = {}
        next_enqueue_idx = 0

        def enqueue_up_to(idx_limit: int):
            nonlocal next_enqueue_idx
            limit = min(idx_limit, len(target_keys))
            while next_enqueue_idx < limit:
                tk = target_keys[next_enqueue_idx]
                p, cs, rs, _ = tk.split("|")
                prefetched_futures[tk] = prefetch_pool.submit(stock.load_sliced_weight, p, cs, rs)
                next_enqueue_idx += 1

        # Pre-enqueue first 24 target matrices
        enqueue_up_to(24)

        if args.limit_quant > 0:
            # Test mode only
            t0 = time.time()
            for idx_c, (key, new_bits) in enumerate(list(targets.items())[: args.limit_quant]):
                prefix, cols_s, rows_s, _ = key.split("|")
                suh_row = idx_map[f"{prefix}|{cols_s}|{rows_s}|suh"]
                svh_row = idx_map[f"{prefix}|{cols_s}|{rows_s}|svh"]
                f_in.seek(suh_row[3])
                suh_b = f_in.read(suh_row[4])
                f_in.seek(svh_row[3])
                svh_b = f_in.read(svh_row[4])
                enqueue_up_to(idx_c + 24)
                w_cpu = prefetched_futures.pop(key).result()
                tr_b = quantize_matrix_on_gpu(w_cpu, suh_b, svh_b, new_bits, device)
                print(
                    f"  [{idx_c+1}/{args.limit_quant}] {key} -> {new_bits}b ({len(tr_b):,} bytes, {time.time()-t0:.1f}s)",
                    flush=True,
                )
            prefetch_pool.shutdown(wait=False)
            return

        partial_path = out_dir / f".{rank_prefix}.partial"
        new_index = []
        at = 0
        quant_done = 0
        tiles_done = 0
        total_tiles = sum(idx_map[k][2][0] * idx_map[k][2][1] for k in targets)
        t_start = time.time()

        with open(partial_path, "wb", buffering=16 * 1024 * 1024) as f_out:
            for i, (key, dt, sh, off, b) in enumerate(old_index):
                if key in targets:
                    new_bits = targets[key]
                    prefix, cols_s, rows_s, _ = key.split("|")
                    suh_row = idx_map[f"{prefix}|{cols_s}|{rows_s}|suh"]
                    svh_row = idx_map[f"{prefix}|{cols_s}|{rows_s}|svh"]
                    f_in.seek(suh_row[3])
                    suh_b = f_in.read(suh_row[4])
                    f_in.seek(svh_row[3])
                    svh_b = f_in.read(svh_row[4])

                    enqueue_up_to(quant_done + 24)
                    w_cpu = prefetched_futures.pop(key).result()
                    tr_bytes = quantize_matrix_on_gpu(w_cpu, suh_b, svh_b, new_bits, device)
                    new_sh = [sh[0], sh[1], 16 * new_bits]
                    assert len(tr_bytes) == sh[0] * sh[1] * 16 * new_bits * 2

                    f_out.write(tr_bytes)
                    new_index.append([key, dt, new_sh, at, len(tr_bytes)])
                    at += len(tr_bytes)
                    quant_done += 1
                    tiles_done += sh[0] * sh[1]

                    if quant_done % 25 == 0 or quant_done == len(targets):
                        elapsed = time.time() - t_start
                        tile_rate = tiles_done / max(elapsed, 1e-3)
                        eta = (total_tiles - tiles_done) / max(tile_rate, 1e-6)
                        pct = 100.0 * tiles_done / max(total_tiles, 1)
                        print(
                            f"[quantizer] {rank_prefix}: {quant_done}/{len(targets)} matrices, "
                            f"{pct:.1f}% tiles ({at / (1024**3):.2f} GiB written, "
                            f"elapsed={elapsed/60:.1f}m, ETA={eta/60:.1f}m) - last: {key} -> {new_bits}b",
                            flush=True,
                        )
                else:
                    f_in.seek(off)
                    rem = b
                    while rem > 0:
                        chunk = f_in.read(min(rem, 16 * 1024 * 1024))
                        if not chunk:
                            raise IOError(f"Short read on {in_path} at {key}")
                        f_out.write(chunk)
                        rem -= len(chunk)
                    new_index.append([key, dt, sh, at, b])
                    at += b

            # Format index JSON identically to rank_cache.zig: `[[...], [...]]`
            index_json_bytes = json.dumps(new_index).encode("utf-8")
            f_out.write(index_json_bytes)
            f_out.write(struct.pack("<Q8s", at, MAGIC))
            f_out.flush()
            os.fsync(f_out.fileno())

        prefetch_pool.shutdown(wait=False)

        digest = hashlib.sha256(index_json_bytes).hexdigest()[:20]
        final_name = f"{rank_prefix}-{digest}.bin"
        final_path = out_dir / final_name
        partial_path.rename(final_path)
        print(
            f"[quantizer] SUCCESS! Wrote {final_path} ({at / (1024**3):.3f} GiB data, "
            f"{len(new_index)} entries, sha256={digest}) in {(time.time() - t_start)/60:.1f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
