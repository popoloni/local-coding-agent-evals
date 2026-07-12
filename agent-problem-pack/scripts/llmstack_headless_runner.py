#!/usr/bin/env python3
"""Run agent-problem-pack problems through a llmstack-backed headless agent CLI on demand."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACK_ROOT = SCRIPT_DIR.parent
EVAL_ROOT = PACK_ROOT.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from llmstack_eval_utils import (  # noqa: E402
    activate_llmstack_model,
    default_python_bin,
    discover_llmstack_root,
    load_llmstack_config,
    resolve_model_target,
)
from pack_tools import PROBLEMS, capture_run, prepare_run  # noqa: E402


def parse_json_objects(text: str) -> list[dict]:
    objects: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not (line.startswith("{") or line.startswith("[")):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            objects.extend(item for item in parsed if isinstance(item, dict))
        elif isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def normalize_usage(*, harness: str, model: str, payload: dict | None) -> dict:
    payload = payload or {}
    usage = payload.get("usage") if isinstance(payload, dict) else None
    model_usage = payload.get("modelUsage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {
            "schema_version": 1,
            "harness": harness,
            "model": model,
            "source": "cli_json_missing_usage",
            "exact": False,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_input_tokens": None,
            "reasoning_output_tokens": None,
            "raw_usage": payload or None,
            "notes": "Headless run completed but no usage object was found in CLI JSON output.",
        }
    return {
        "schema_version": 1,
        "harness": harness,
        "model": model,
        "source": "cli_json",
        "exact": True,
        "input_tokens": usage.get("input_tokens") or usage.get("inputTokens"),
        "output_tokens": usage.get("output_tokens") or usage.get("outputTokens"),
        "total_tokens": usage.get("total_tokens") or usage.get("totalTokens"),
        "cached_input_tokens": usage.get("cached_input_tokens") or usage.get("cachedInputTokens"),
        "reasoning_output_tokens": usage.get("reasoning_output_tokens") or usage.get("reasoningOutputTokens"),
        "raw_usage": {"usage": usage, "modelUsage": model_usage},
        "notes": "",
    }


def default_claude_command(python_bin: str, prompt: str, bypass_permissions: bool) -> list[str]:
    command = [python_bin, "-m", "llmstack.cli", "interactive", "--", "--output-format", "json"]
    if bypass_permissions:
        command.extend(["--permission-mode", "bypassPermissions"])
    command.extend(["-p", prompt])
    return command


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def run_agent(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env)


def prepare_and_run_one(*, llmstack_root: Path, python_bin: str, model_key: str, model_target: str, problem_id: str, run_name: str, bypass_permissions: bool) -> int:
    run_dir = prepare_run(PACK_ROOT.resolve(), problem_id, run_name)
    workspace = run_dir / "workspace"
    artifacts = run_dir / "artifacts"
    prompt = (artifacts / "task-prompt.txt").read_text(encoding="utf-8").strip()

    command = default_claude_command(python_bin, prompt, bypass_permissions)
    result = run_agent(command, workspace)
    write_text(artifacts / "headless-command.txt", "$ " + " ".join(command) + "\n")
    write_text(artifacts / "headless-stdout.jsonl", result.stdout)
    write_text(artifacts / "headless-stderr.txt", result.stderr)

    events = parse_json_objects(result.stdout)
    final_payload = events[-1] if events else None
    usage_record = normalize_usage(harness="claude-via-llmstack", model=model_target, payload=final_payload)
    (artifacts / "usage.json").write_text(json.dumps(usage_record, indent=2) + "\n", encoding="utf-8")

    verification = capture_run(run_dir, PACK_ROOT.resolve())
    print(f"{problem_id}: agent_exit={result.returncode} verify_exit={verification.returncode} run={run_dir}")
    return 0 if result.returncode == 0 and verification.returncode == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent-problem-pack through a native llmstack headless Claude session on demand.")
    parser.add_argument("--model-key", help="llmstack model registry key, for example dflash-ornith35b-moe.")
    parser.add_argument("--activate-model", action="store_true", help="Call `llmstack model use <model-key> --restart` before starting the suite.")
    parser.add_argument("--llmstack-root", type=Path, help="Path to the llmstack workspace root. Default: auto-detect.")
    parser.add_argument("--python-bin", help="Python executable used for llmstack CLI calls.")
    parser.add_argument("--problem", choices=sorted(PROBLEMS), help="Run only one problem instead of the whole pack.")
    parser.add_argument("--run-name-prefix", default="llmstack-claude", help="Prefix used when creating run names.")
    parser.add_argument("--no-bypass-permissions", action="store_true", help="Do not pass `--permission-mode bypassPermissions` to Claude.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    llmstack_root = (args.llmstack_root or discover_llmstack_root(PACK_ROOT)).resolve()
    python_bin = args.python_bin or default_python_bin(llmstack_root)
    cfg = load_llmstack_config(llmstack_root)
    model_key, model_target = resolve_model_target(cfg, args.model_key)
    if args.activate_model:
        activate_llmstack_model(llmstack_root, model_key, python_bin=python_bin)
        cfg = load_llmstack_config(llmstack_root)
        _, model_target = resolve_model_target(cfg, model_key)

    problem_ids = [args.problem] if args.problem else sorted(PROBLEMS)
    failures = 0
    for index, problem_id in enumerate(problem_ids, start=1):
        run_name = f"{args.run_name_prefix}-{model_key}-{index:02d}"
        failures += prepare_and_run_one(
            llmstack_root=llmstack_root,
            python_bin=python_bin,
            model_key=model_key,
            model_target=model_target,
            problem_id=problem_id,
            run_name=run_name,
            bypass_permissions=not args.no_bypass_permissions,
        )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())