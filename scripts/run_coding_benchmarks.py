#!/usr/bin/env python3
"""
Industry-Standard Coding Intelligence Benchmark Runner
======================================================
Evaluates an OpenAI-compatible endpoint (`http://127.0.0.1:8888/v1`) on:
1. EvalPlus `HumanEval+` (all 164 problems, Base HumanEval + 80x stricter EvalPlus edge-case tests)
2. `LiveCodeBench v6` (`livecodebench/code_generation_lite` `test6.jsonl`, stratified Easy / Medium / Hard
   competitive programming problems from AtCoder & LeetCode with full public + private test execution).
"""

import argparse
import ast
import base64
import concurrent.futures
import io
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path

from evalplus.data import get_human_eval_plus, write_jsonl
from evalplus.evaluate import evaluate as evalplus_evaluate
from huggingface_hub import hf_hub_download


def extract_python_code(text: str, entry_point: str | None = None) -> str:
    """Extract Python code from markdown code fences (after stripping any <think>...</think> block)."""
    if not text:
        return ""
    # Strip any residual <think>...</think> tags if present in content
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        if entry_point:
            for b in reversed(blocks):
                if f"def {entry_point}" in b or "class Solution" in b:
                    return b.strip()
        return blocks[-1].strip()

    # Fallback: if no closing fence or raw code
    if text.startswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:]).strip()
    return text


def chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    enable_thinking: bool,
    reasoning_effort: str | None = None,
    thinking_budget: int = 4096,
    timeout: float = 300.0,
) -> tuple[str, str, int, float]:
    """
    Send a deterministic (temperature=0.0) chat completion request.
    Returns (content, reasoning_content, completion_tokens, elapsed_sec).
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    ctk = {"enable_thinking": enable_thinking}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": ctk,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    t0 = time.time()
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            msg = out["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            comp_toks = out.get("usage", {}).get("completion_tokens", 0)
            if enable_thinking and ("```" not in content and "def " not in content and "import " not in content):
                # If the backend exhausted max_tokens inside <think> without emitting a code block, retry directly without thinking
                c2, r2, t2, e2 = chat_completion(
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    max_tokens=4096,
                    enable_thinking=False,
                    reasoning_effort=None,
                    thinking_budget=0,
                    timeout=timeout,
                )
                return c2, reasoning + "\n" + content + "\n" + r2, comp_toks + t2, elapsed + e2
            return content, reasoning, comp_toks, elapsed
        except Exception as e:
            if attempt == 3:
                return "", f"ERROR: {e}", 0, time.time() - t0
            time.sleep(2.0 * (attempt + 1))
    return "", "", 0, time.time() - t0


# ==============================================================================
# Part 1: EvalPlus HumanEval+ (164 Problems)
# ==============================================================================
def run_humaneval_plus(
    base_url: str,
    model: str,
    out_dir: Path,
    tag: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    max_tokens: int = 2048,
) -> dict:
    hep = get_human_eval_plus()
    tasks = list(hep.items())
    samples_path = out_dir / f"humaneval_plus_{tag}.jsonl"

    # Resume support if partial samples exist
    done_map = {}
    if samples_path.exists():
        with open(samples_path, "r") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("solution", "").strip():
                        done_map[row["task_id"]] = row

    print(
        f"[HumanEval+] Running {len(tasks)} problems (already cached={len(done_map)}, "
        f"concurrency={concurrency}, thinking={enable_thinking})...",
        flush=True,
    )

    t_start = time.time()
    total_tokens = 0

    def _solve_one(item):
        task_id, prob = item
        if task_id in done_map:
            return done_map[task_id], 0, 0.0
        entry = prob["entry_point"]
        prompt = (
            "Complete the following Python 3 function so that it passes all edge-case unit tests.\n"
            "Return ONLY the complete Python code (including any needed imports and the full function definition) "
            "inside a ```python ... ``` code block.\n\n"
            f"```python\n{prob['prompt']}\n```"
        )
        content, _, comp_toks, elapsed = chat_completion(
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            reasoning_effort="low" if enable_thinking else None,
        )
        code = extract_python_code(content, entry_point=entry)
        # If the model only returned the body without `def entry`, prepend prob['prompt']
        if f"def {entry}" not in code:
            solution = prob["prompt"] + "\n" + code
        else:
            # Ensure standard imports from prompt are preserved if not in code
            prompt_imports = "\n".join(
                line for line in prob["prompt"].splitlines() if line.startswith(("import ", "from "))
            )
            solution = (prompt_imports + "\n" + code).strip() if prompt_imports else code

        return {"task_id": task_id, "solution": solution}, comp_toks, elapsed

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(_solve_one, item) for item in tasks]
        for idx, fut in enumerate(futs, 1):
            row, toks, _ = fut.result()
            results.append(row)
            total_tokens += toks
            if idx % 20 == 0 or idx == len(tasks):
                elapsed = time.time() - t_start
                print(
                    f"  [HumanEval+] {idx}/{len(tasks)} completed ({total_tokens:,} tokens, {elapsed:.1f}s)",
                    flush=True,
                )

    write_jsonl(str(samples_path), results)

    # Run official EvalPlus harness
    print(f"[HumanEval+] Executing official EvalPlus test suite (Base + 80x Extra tests)...", flush=True)
    eval_results_path = out_dir / f"humaneval_plus_{tag}_eval_results.json"
    if eval_results_path.exists():
        eval_results_path.unlink()

    evalplus_evaluate(dataset="humaneval", samples=str(samples_path), parallel=2)

    with open(eval_results_path, "r") as f:
        eval_data = json.load(f)["eval"]

    base_pass = 0
    plus_pass = 0
    failures = []
    for tid, res_list in eval_data.items():
        r0 = res_list[0]
        b_ok = r0["base_status"] == "pass"
        p_ok = b_ok and (r0["plus_status"] == "pass")
        if b_ok:
            base_pass += 1
        if p_ok:
            plus_pass += 1
        else:
            failures.append({"task_id": tid, "base": r0["base_status"], "plus": r0["plus_status"]})

    n = len(tasks)
    summary = {
        "total": n,
        "base_pass": base_pass,
        "base_pass_at_1": round(100.0 * base_pass / n, 2),
        "plus_pass": plus_pass,
        "plus_pass_at_1": round(100.0 * plus_pass / n, 2),
        "total_tokens": total_tokens,
        "wall_seconds": round(time.time() - t_start, 2),
        "failures": failures,
    }
    print(
        f"[HumanEval+ Summary ({tag})] Base: {base_pass}/{n} ({summary['base_pass_at_1']}%), "
        f"Plus (80x tests): {plus_pass}/{n} ({summary['plus_pass_at_1']}%)",
        flush=True,
    )
    return summary


# ==============================================================================
# Part 2: LiveCodeBench v6 (Competitive Algorithmic Coding)
# ==============================================================================
LCB_RUNNER_SCRIPT = r"""
import io, json, math, sys, time, traceback
import heapq, bisect, itertools, collections, functools
from typing import *
from collections import *
from functools import *
from itertools import *
from heapq import *
from bisect import *

sys.setrecursionlimit(10**6)

def compare_values(actual, expected):
    if actual == expected:
        return True
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-5)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(compare_values(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(compare_values(actual[k], expected[k]) for k in expected)
    return False

def compare_stdout(actual_str: str, expected_str: str) -> bool:
    a_toks = actual_str.strip().split()
    e_toks = expected_str.strip().split()
    if a_toks == e_toks:
        return True
    if len(a_toks) != len(e_toks):
        return False
    for a, e in zip(a_toks, e_toks):
        if a == e:
            continue
        try:
            if not math.isclose(float(a), float(e), rel_tol=1e-5, abs_tol=1e-5):
                return False
        except Exception:
            return False
    return True

def main():
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    tests = payload["tests"]
    func_name = payload.get("func_name")

    if func_name:
        ns = dict(globals())
        try:
            exec(compile(code, "<solution>", "exec"), ns)
        except Exception as e:
            print(json.dumps({"passed": False, "reason": f"compile_error: {e}", "passed_count": 0, "total": len(tests)}))
            return
        if "Solution" not in ns:
            print(json.dumps({"passed": False, "reason": "missing_Solution_class", "passed_count": 0, "total": len(tests)}))
            return
        passed_c = 0
        for idx, tc in enumerate(tests):
            try:
                args = [json.loads(line) for line in tc["input"].strip().splitlines() if line.strip()]
                expected = json.loads(tc["output"].strip())
                sol = ns["Solution"]()
                fn = getattr(sol, func_name)
                actual = fn(*args)
                if not compare_values(actual, expected):
                    print(json.dumps({"passed": False, "reason": f"wrong_answer_tc_{idx}", "passed_count": passed_c, "total": len(tests)}))
                    return
                passed_c += 1
            except Exception as e:
                print(json.dumps({"passed": False, "reason": f"runtime_error_tc_{idx}: {e}", "passed_count": passed_c, "total": len(tests)}))
                return
        print(json.dumps({"passed": True, "reason": "ok", "passed_count": passed_c, "total": len(tests)}))
    else:
        # Stdin / Stdout program
        compiled = None
        try:
            compiled = compile(code, "<solution>", "exec")
        except Exception as e:
            print(json.dumps({"passed": False, "reason": f"compile_error: {e}", "passed_count": 0, "total": len(tests)}))
            return
        passed_c = 0
        for idx, tc in enumerate(tests):
            inp = tc["input"]
            if not inp.endswith("\n"):
                inp += "\n"
            expected = tc["output"]
            old_stdin, old_stdout = sys.stdin, sys.stdout
            sys.stdin = io.StringIO(inp)
            sys.stdout = io.StringIO()
            ns = dict(globals())
            ns["__name__"] = "__main__"
            try:
                exec(compiled, ns)
                if "main" in ns and callable(ns["main"]) and sys.stdout.getvalue() == "":
                    ns["main"]()
                out_str = sys.stdout.getvalue()
            except SystemExit:
                out_str = sys.stdout.getvalue()
            except Exception as e:
                sys.stdin, sys.stdout = old_stdin, old_stdout
                print(json.dumps({"passed": False, "reason": f"runtime_error_tc_{idx}: {e}", "passed_count": passed_c, "total": len(tests)}))
                return
            finally:
                sys.stdin, sys.stdout = old_stdin, old_stdout

            if not compare_stdout(out_str, expected):
                print(json.dumps({"passed": False, "reason": f"wrong_answer_tc_{idx}", "passed_count": passed_c, "total": len(tests)}))
                return
            passed_c += 1
        print(json.dumps({"passed": True, "reason": "ok", "passed_count": passed_c, "total": len(tests)}))

if __name__ == "__main__":
    main()
"""


def decode_lcb_tests(row: dict) -> list[dict]:
    pub = json.loads(row["public_test_cases"])
    priv_raw = row["private_test_cases"]
    try:
        priv = json.loads(priv_raw)
    except Exception:
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(priv_raw.encode("utf-8")))))
    return pub + priv


def evaluate_lcb_solution(code: str, tests: list[dict], func_name: str | None, timeout: float = 20.0) -> dict:
    if not code.strip():
        return {"passed": False, "reason": "empty_code", "passed_count": 0, "total": len(tests)}
    payload = json.dumps({"code": code, "tests": tests, "func_name": func_name}).encode("utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", LCB_RUNNER_SCRIPT],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        lines = [ln for ln in proc.stdout.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        if lines:
            return json.loads(lines[-1])
        return {
            "passed": False,
            "reason": f"subprocess_crash: {proc.stderr.decode('utf-8', errors='replace')[:200]}",
            "passed_count": 0,
            "total": len(tests),
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "reason": "TLE", "passed_count": 0, "total": len(tests)}
    except Exception as e:
        return {"passed": False, "reason": f"eval_exception: {e}", "passed_count": 0, "total": len(tests)}


def load_stratified_lcb(per_tier: int = 15) -> list[dict]:
    path = hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(path)]
    by_diff = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    selected = []
    for diff in ["easy", "medium", "hard"]:
        items = sorted(by_diff[diff], key=lambda x: x["question_id"])
        selected.extend(items[:per_tier])
    return selected


def run_livecodebench(
    base_url: str,
    model: str,
    out_dir: Path,
    tag: str,
    per_tier: int = 15,
    concurrency: int = 4,
    enable_thinking: bool = True,
    reasoning_effort: str = "medium",
    max_tokens: int = 8192,
) -> dict:
    problems = load_stratified_lcb(per_tier=per_tier)
    ckpt_path = out_dir / f"lcb_v6_{tag}.jsonl"
    done_map = {}
    if ckpt_path.exists():
        kept_lines = []
        with open(ckpt_path, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    reason = r.get("reason", "")
                    if reason != "empty_code" and not (reason.startswith("compile_error") and r.get("completion_tokens", 0) >= 6144):
                        done_map[r["question_id"]] = r
                        kept_lines.append(json.dumps(r) + "\n")
        with open(ckpt_path, "w") as f:
            f.writelines(kept_lines)

    print(
        f"[LiveCodeBench v6] Running {len(problems)} problems ({per_tier}/tier Easy/Medium/Hard, "
        f"already cached={len(done_map)}, concurrency={concurrency}, thinking={enable_thinking}, effort={reasoning_effort})...",
        flush=True,
    )

    t_start = time.time()

    def _solve_lcb(prob: dict) -> dict:
        qid = prob["question_id"]
        if qid in done_map:
            return done_map[qid]

        starter = (prob.get("starter_code") or "").strip()
        meta = json.loads(prob["metadata"]) if isinstance(prob["metadata"], str) and prob["metadata"].strip() else (prob.get("metadata") or {})
        func_name = meta.get("func_name")

        if starter:
            prompt = (
                "Solve the following competitive programming problem in Python 3.\n"
                "Implement the `Solution` class method shown in the starter code below.\n"
                "Return ONLY your complete Python 3 solution inside a ```python ... ``` code block.\n\n"
                f"### Problem: {prob['question_title']}\n"
                f"{prob['question_content']}\n\n"
                f"### Starter Code:\n```python\n{starter}\n```"
            )
        else:
            prompt = (
                "Solve the following competitive programming problem in Python 3.\n"
                "Read input from standard input (`sys.stdin`) and write the answer to standard output (`print`).\n"
                "Return ONLY your complete Python 3 program inside a ```python ... ``` code block.\n\n"
                f"### Problem: {prob['question_title']}\n"
                f"{prob['question_content']}"
            )

        content, _, comp_toks, elapsed = chat_completion(
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort if enable_thinking else None,
        )
        code = extract_python_code(content, entry_point=func_name)
        tests = decode_lcb_tests(prob)
        verdict = evaluate_lcb_solution(code, tests, func_name=func_name)
        return {
            "question_id": qid,
            "question_title": prob["question_title"],
            "platform": prob["platform"],
            "difficulty": prob["difficulty"],
            "passed": verdict["passed"],
            "reason": verdict["reason"],
            "passed_count": verdict["passed_count"],
            "total_tests": verdict["total"],
            "completion_tokens": comp_toks,
            "elapsed_sec": round(elapsed, 2),
        }

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        fut_to_prob = {pool.submit(_solve_lcb, p): p for p in problems}
        for idx, fut in enumerate(concurrent.futures.as_completed(fut_to_prob), 1):
            rec = fut.result()
            records.append(rec)
            with open(ckpt_path, "a") as f_ck:
                if rec["question_id"] not in done_map:
                    f_ck.write(json.dumps(rec) + "\n")
            status_icon = "PASS" if rec["passed"] else f"FAIL({rec['reason']})"
            print(
                f"  [LCB {idx}/{len(problems)}] {rec['question_id']} ({rec['difficulty']}, {rec['platform']}): "
                f"{status_icon} [{rec['passed_count']}/{rec['total_tests']} tests, {rec['completion_tokens']} toks, {rec['elapsed_sec']}s]",
                flush=True,
            )

    # Sort deterministically by question_id
    records.sort(key=lambda r: (["easy", "medium", "hard"].index(r["difficulty"]), r["question_id"]))
    by_tier = defaultdict(list)
    for r in records:
        by_tier[r["difficulty"]].append(r)

    tier_summary = {}
    for diff in ["easy", "medium", "hard"]:
        t_list = by_tier[diff]
        p_cnt = sum(1 for r in t_list if r["passed"])
        tier_summary[diff] = {
            "passed": p_cnt,
            "total": len(t_list),
            "pass_at_1": round(100.0 * p_cnt / max(len(t_list), 1), 2),
        }

    tot_pass = sum(1 for r in records if r["passed"])
    tot_toks = sum(r["completion_tokens"] for r in records)
    summary = {
        "total": len(records),
        "passed": tot_pass,
        "pass_at_1": round(100.0 * tot_pass / max(len(records), 1), 2),
        "by_difficulty": tier_summary,
        "total_tokens": tot_toks,
        "wall_seconds": round(time.time() - t_start, 2),
        "records": records,
    }
    print(
        f"[LiveCodeBench v6 Summary ({tag})] Overall: {tot_pass}/{len(records)} ({summary['pass_at_1']}%) | "
        f"Easy: {tier_summary['easy']['passed']}/{tier_summary['easy']['total']} ({tier_summary['easy']['pass_at_1']}%) | "
        f"Medium: {tier_summary['medium']['passed']}/{tier_summary['medium']['total']} ({tier_summary['medium']['pass_at_1']}%) | "
        f"Hard: {tier_summary['hard']['passed']}/{tier_summary['hard']['total']} ({tier_summary['hard']['pass_at_1']}%)",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--tag", required=True, help="Run tag (e.g. dsv41_custom_hq, dsv41_stock_29bpw, glm53_nvfp4)")
    parser.add_argument("--out-dir", default="/home/massoud/spark-dsv41/logs/coding_eval")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--lcb-per-tier", type=int, default=15)
    parser.add_argument("--lcb-effort", default="medium")
    parser.add_argument("--lcb-max-tokens", type=int, default=8192)
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument("--skip-lcb", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"tag": args.tag, "model": args.model, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if not args.skip_humaneval:
        report["humaneval_plus"] = run_humaneval_plus(
            base_url=args.base_url,
            model=args.model,
            out_dir=out_dir,
            tag=args.tag,
            concurrency=args.concurrency,
            enable_thinking=False,
            max_tokens=2048,
        )

    if not args.skip_lcb:
        report["livecodebench_v6"] = run_livecodebench(
            base_url=args.base_url,
            model=args.model,
            out_dir=out_dir,
            tag=args.tag,
            per_tier=args.lcb_per_tier,
            concurrency=args.concurrency,
            enable_thinking=True,
            reasoning_effort=args.lcb_effort,
            max_tokens=args.lcb_max_tokens,
        )

    summary_path = out_dir / f"summary_{args.tag}.json"
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Done] Wrote full coding benchmark summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
