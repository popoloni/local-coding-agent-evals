#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_TASKS = Path(__file__).with_name("hard_reasoning_tasks.jsonl")

JSON_FORMAT = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["tool", "arguments"],
}


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


def load_tasks(path):
    tasks = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return tasks


def build_prompt(task):
    return (
        SYSTEM_PROMPT.replace("{tools}", json.dumps(TOOLS, indent=2))
        + "\nTask:\n"
        + task["prompt"]
        + "\n\nReturn only JSON."
    )


def extract_json(text):
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


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def contains_all(value, expected_parts):
    normalized = normalize_text(value)
    return all(normalize_text(part) in normalized for part in expected_parts)


def score_response(task, response):
    expected = task["expected"]
    expected_tool = expected["tool"]
    actual_tool = response.get("tool")
    if actual_tool != expected_tool:
        return {
            "passed": False,
            "score": 0.0,
            "reason": f"wrong tool: expected {expected_tool}, got {actual_tool}",
        }

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
            return {
                "passed": False,
                "score": 0.5,
                "reason": f"argument {key} missing required content",
            }

    if expected.get("answer_contains"):
        if not contains_all(arguments.get("answer"), expected["answer_contains"]):
            return {
                "passed": False,
                "score": 0.5,
                "reason": "answer missing required content",
            }

    return {"passed": True, "score": 1.0, "reason": "ok"}


def post_json(url, payload, timeout_s):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not connect to Ollama at {url}: {exc}") from exc


def call_ollama(host, model, prompt, timeout_s, temperature):
    response = post_json(
        f"{host.rstrip('/')}/api/chat",
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": JSON_FORMAT,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": temperature,
                "num_predict": 768,
            },
        },
        timeout_s,
    )
    return response.get("message", {}).get("content", "") or response.get("response", "")


def run_benchmark(args):
    tasks = load_tasks(args.tasks)
    rows = []
    passed = 0
    for task in tasks:
        started_at = time.monotonic()
        raw = call_ollama(args.host, args.model, build_prompt(task), args.timeout, args.temperature)
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


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Ollama on five harder coding and project reasoning tasks."
    )
    parser.add_argument("--model", required=True, help="Ollama model name.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host. Default: {DEFAULT_HOST}.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help=f"JSONL task file. Default: {DEFAULT_TASKS}.")
    parser.add_argument("--timeout", type=float, default=180, help="Per-task timeout in seconds. Default: 180.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Default: 0.0.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_benchmark(args)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
