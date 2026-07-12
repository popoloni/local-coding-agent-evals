#!/usr/bin/env python3
"""Evaluate llmstack-backed models on the hard tool-reasoning benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmstack_eval_utils import (  # noqa: E402
    activate_llmstack_model,
    build_chat_payload,
    check_llmstack,
    default_python_bin,
    discover_llmstack_root,
    headroom_base_urls,
    inference_base_urls,
    load_llmstack_config,
    post_json_with_fallback,
    resolve_model_target,
    response_text,
)


DEFAULT_TASKS = Path(__file__).with_name("hard_reasoning_tasks.jsonl")

TOOLS = [
    {
        "name": "edit_file",
        "description": "Edit one file with precise instructions.",
        "arguments": {"path": "relative path", "instructions": "short edit instruction"},
    },
    {
        "name": "ask_clarification",
        "description": "Ask one concise question when the next action is ambiguous or risky.",
        "arguments": {"question": "question text"},
    },
    {
        "name": "final_answer",
        "description": "Answer directly when the task can be solved from the prompt.",
        "arguments": {"answer": "concise answer"},
    },
]


SYSTEM_PROMPT = """You are being evaluated on hard tool-use reasoning.
Choose exactly one next action. Do not execute tools. Do not explain outside JSON.
Return only one JSON object with this schema:
{"tool": "<tool name>", "arguments": {"key": "value"}}

Tool catalog:
{tools}
"""


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return tasks


def build_prompt(task: dict) -> str:
    return (
        SYSTEM_PROMPT.replace("{tools}", json.dumps(TOOLS, indent=2))
        + "\nTask:\n"
        + task["prompt"]
        + "\n\nReturn only JSON."
    )


def extract_json(text: str) -> dict:
    stripped = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if stripped.startswith("```") and fence_match:
        stripped = fence_match.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def normalize_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def contains_all(value, expected_parts) -> bool:
    normalized = normalize_text(value)
    return all(normalize_text(part) in normalized for part in expected_parts)


def score_response(task: dict, response: dict) -> dict:
    expected = task["expected"]
    expected_tool = expected["tool"]
    actual_tool = response.get("tool")
    if actual_tool != expected_tool:
        return {"passed": False, "score": 0.0, "reason": f"wrong tool: expected {expected_tool}, got {actual_tool}"}

    arguments = response.get("arguments")
    if not isinstance(arguments, dict):
        return {"passed": False, "score": 0.0, "reason": "arguments must be an object"}

    for key, expected_value in expected.get("required_arguments", {}).items():
        if arguments.get(key) != expected_value:
            return {
                "passed": False,
                "score": 0.5,
                "reason": f"wrong argument {key}: expected {expected_value!r}, got {arguments.get(key)!r}",
            }

    for key, expected_parts in expected.get("argument_contains", {}).items():
        if not contains_all(arguments.get(key), expected_parts):
            return {"passed": False, "score": 0.5, "reason": f"argument {key} missing required content"}

    if expected.get("answer_contains") and not contains_all(arguments.get("answer"), expected["answer_contains"]):
        return {"passed": False, "score": 0.5, "reason": "answer missing required content"}

    return {"passed": True, "score": 1.0, "reason": "ok"}


def call_llmstack(chat_url: str, fallback_chat_url: str | None, target: str, prompt: str, timeout_s: float, temperature: float, task_id: str) -> str:
    payload = build_chat_payload(target, prompt, max_tokens=768, temperature=temperature, top_p=1.0)
    response = post_json_with_fallback(
        chat_url,
        payload,
        timeout_s,
        fallback_url=fallback_chat_url,
        retries=2,
        retry_backoff_s=2.0,
        request_label=f"reasoning task {task_id}",
    )
    return response_text(response)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args) -> int:
    llmstack_root = (args.llmstack_root or discover_llmstack_root(ROOT)).resolve()
    cfg = load_llmstack_config(llmstack_root)
    model_key, target = resolve_model_target(cfg, args.model_key)
    python_bin = args.python_bin or default_python_bin(llmstack_root)
    if args.activate_model:
        activate_llmstack_model(llmstack_root, model_key, python_bin=python_bin)
        cfg = load_llmstack_config(llmstack_root)
        _, target = resolve_model_target(cfg, model_key)
    inference_chat_url, models_url = inference_base_urls(cfg)
    headroom_chat_url, _ = headroom_base_urls(cfg)
    chat_url = inference_chat_url if args.surface == "inference" else headroom_chat_url
    fallback_chat_url = inference_chat_url if args.surface == "headroom" else None
    model_ids = check_llmstack(models_url, min(args.timeout, 10))
    tasks = load_tasks(args.tasks)

    print(f"llmstack root: {llmstack_root}")
    print(f"Model key: {model_key}")
    print(f"Model target: {target}")
    print(f"Surface: {args.surface}")
    print(f"Active /v1/models: {', '.join(model_ids) if model_ids else 'n/a'}")
    if args.surface == "headroom":
        print(f"Fallback chat URL on transient errors: {inference_chat_url}")
    print()

    rows = []
    passed = 0
    for task in tasks:
        started_at = time.monotonic()
        try:
            raw = call_llmstack(
                chat_url,
                fallback_chat_url,
                target,
                build_prompt(task),
                args.timeout,
                args.temperature,
                task["id"],
            )
        except RuntimeError as exc:
            raw = ""
            parsed = None
            result = {
                "passed": False,
                "score": 0.0,
                "reason": f"request failed: {exc}",
            }
            elapsed_s = time.monotonic() - started_at
            row = {
                "id": task["id"],
                "category": task.get("category", ""),
                "passed": result["passed"],
                "score": result["score"],
                "reason": result["reason"],
                "expected_tool": task["expected"]["tool"],
                "actual_tool": "",
                "elapsed_s": f"{elapsed_s:.2f}",
                "raw": raw,
            }
            rows.append(row)
            print(f"FAIL {task['id']}: {result['reason']}", flush=True)
            continue
        elapsed_s = time.monotonic() - started_at
        try:
            parsed = extract_json(raw)
            result = score_response(task, parsed)
        except Exception as exc:
            parsed = None
            result = {"passed": False, "score": 0.0, "reason": f"invalid JSON: {exc}"}
        if result["passed"]:
            passed += 1
        row = {
            "id": task["id"],
            "category": task.get("category", ""),
            "passed": result["passed"],
            "score": result["score"],
            "reason": result["reason"],
            "expected_tool": task["expected"]["tool"],
            "actual_tool": parsed.get("tool") if isinstance(parsed, dict) else "",
            "elapsed_s": f"{elapsed_s:.2f}",
            "raw": raw,
        }
        rows.append(row)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {task['id']}: {result['reason']}", flush=True)

    print()
    print(f"Score: {passed}/{len(tasks)} passed ({passed / len(tasks) * 100:.1f}%)")
    if args.csv:
        write_csv(args.csv, rows)
        print(f"Wrote CSV: {args.csv}")
    return 0 if passed == len(tasks) else 1


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate llmstack-backed models on five harder coding and project reasoning tasks.")
    parser.add_argument("--model-key", help="llmstack model registry key, for example dflash-qwen35b-moe.")
    parser.add_argument("--activate-model", action="store_true", help="Call `llmstack model use <model-key> --restart` before the benchmark.")
    parser.add_argument("--llmstack-root", type=Path, help="Path to the llmstack workspace root. Default: auto-detect via llmstack_config.json.")
    parser.add_argument("--python-bin", help="Python executable used for optional llmstack CLI calls.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help=f"JSONL task file. Default: {DEFAULT_TASKS}.")
    parser.add_argument("--timeout", type=float, default=180, help="Per-task timeout in seconds. Default: 180.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Default: 0.0.")
    parser.add_argument("--surface", choices=("inference", "headroom"), default="inference", help="Send requests either directly to inference or through Headroom. Default: inference.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_benchmark(args)
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())