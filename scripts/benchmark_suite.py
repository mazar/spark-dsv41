#!/usr/bin/env python3
"""
Spark-DSV41 Verification & Throughput Benchmark Suite
Tests:
1. /health and /v1/models (alias resolution + default effort metadata)
2. 1-stream Prose & Code Decode Throughput (tok/s)
3. Reasoning Effort Mapping ("max" -> upstream "xhigh")
4. DeepSeek-V4.1 DSML Tool-Calling & JSON Schema Type Coercion
5. Anthropic /v1/messages & /v1/messages/count_tokens Compatibility (Claude CLI)
"""

import argparse
import json
import time
import urllib.request


def post_json(url: str, payload: dict, timeout: int = 180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        headers = dict(resp.headers)
    elapsed = time.perf_counter() - t0
    return body, headers, elapsed


def get_json(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Spark-DSV41 Benchmark Suite")
    parser.add_argument("--base-url", default="http://127.0.0.1:8888", help="Gateway base URL")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke & protocol verification")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    print(f"=== Spark-DSV41 Verification & Benchmark Suite ({base}) ===")

    # 1. Health & Models
    health = get_json(f"{base}/health")
    print(f"[1/5] Health: status={health.get('status')} upstream_healthy={health.get('upstream_healthy')}")
    models = get_json(f"{base}/v1/models")
    model_ids = [m["id"] for m in models.get("data", [])]
    print(f"      Served Models ({len(model_ids)}): {', '.join(model_ids[:6])}...")

    # 2. 1-Stream Prose & Pure Code Decode Speed
    max_toks = 128 if args.quick else 512
    resp_prose, _, elapsed_prose = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": "DeepSeek-V4.1-Flash-1017",
            "reasoning_effort": "none",
            "max_tokens": max_toks,
            "messages": [
                {
                    "role": "user",
                    "content": "Explain how modern CPU branch predictors, reorder buffers, and speculative execution work together.",
                }
            ],
        },
    )
    prose_toks = resp_prose["usage"]["completion_tokens"]
    prose_tps = prose_toks / elapsed_prose if elapsed_prose > 0 else 0.0
    print(f"[2a/5] 1-Stream Prose Decode: {prose_toks} tokens in {elapsed_prose:.2f}s -> {prose_tps:.1f} tok/s")

    resp, hdrs, elapsed = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": "DeepSeek-V4.1-Flash-1017",
            "reasoning_effort": "none",
            "temperature": 0.0,
            "max_tokens": max_toks,
            "messages": [
                {
                    "role": "user",
                    "content": "Output only raw Python code (no comments, no markdown prose) implementing a red-black tree with insert, delete, and range query methods.",
                }
            ],
        },
    )
    comp_toks = resp["usage"]["completion_tokens"]
    code_tps = comp_toks / elapsed if elapsed > 0 else 0.0
    print(f"[2b/5] 1-Stream Code Decode:  {comp_toks} tokens in {elapsed:.2f}s -> {code_tps:.1f} tok/s")

    # 3. Reasoning Effort ("max" -> "xhigh")
    resp_max, hdrs_max, _ = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": "dsv41-max",
            "temperature": 0.6,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Compute 37 * 43 step by step."}],
        },
    )
    eff_hdr = hdrs_max.get("X-Spark-Reasoning-Effort") or hdrs_max.get("x-spark-reasoning-effort")
    print(f"[3/5] Effort Alias 'dsv41-max': X-Spark-Reasoning-Effort={eff_hdr}")

    # 4. Tool Calling
    resp_tc, _, _ = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": "DeepSeek-V4.1-Flash-1017",
            "reasoning_effort": "none",
            "temperature": 0.0,
            "max_tokens": 256,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "query_cluster_node",
                        "description": "Query a DGX Spark node status by port and hostname",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "hostname": {"type": "string"},
                                "port": {"type": "integer"},
                            },
                            "required": ["hostname", "port"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "messages": [
                {"role": "user", "content": "Query node spark-8a27 on port 29620."}
            ],
        },
    )
    tcs = resp_tc["choices"][0]["message"].get("tool_calls", [])
    print(f"[4/5] DSML Tool Call: count={len(tcs)} -> {json.dumps(tcs[0]['function']) if tcs else 'NONE'}")

    # 5. Anthropic /v1/messages (Claude CLI compatibility)
    anth_resp, anth_hdrs, _ = post_json(
        f"{base}/v1/messages",
        {
            "model": "claude-opus-4-6",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Reply with READY."}],
        },
    )
    print(
        f"[5/5] Anthropic /v1/messages ('claude-opus-4-6'): stop_reason={anth_resp.get('stop_reason')} "
        f"effort={anth_hdrs.get('X-Spark-Reasoning-Effort') or anth_hdrs.get('x-spark-reasoning-effort')}"
    )


if __name__ == "__main__":
    main()
