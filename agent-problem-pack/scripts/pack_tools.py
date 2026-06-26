#!/usr/bin/env python
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Problem:
    identifier: str
    title: str
    verify_command: tuple[str, ...]
    task_prompt: str
    expected_behavior: tuple[str, ...]


PROBLEMS = {
    "problem-01-tokenizer-regression": Problem(
        identifier="problem-01-tokenizer-regression",
        title="Tokenizer Regression",
        verify_command=("uv", "run", "pytest"),
        task_prompt=(
            "A tokenizer regression test fails. Diagnose the root cause and make the smallest "
            "safe code change so the tests pass. Explain the fix briefly after editing."
        ),
        expected_behavior=(
            "The agent should identify that splitting stripped empty input can create an empty-string token.",
            "The fix should filter empty parts produced by split.",
            "The tests should pass with uv run pytest.",
            "The change should stay scoped to tokenizer.py unless there is a clear reason.",
        ),
    ),
    "problem-02-shell-command-injection": Problem(
        identifier="problem-02-shell-command-injection",
        title="Shell Command Injection",
        verify_command=("uv", "run", "pytest"),
        task_prompt=(
            "Review and fix the command runner. The command comes from a JSON task file that readers may edit. "
            "Make the smallest safe change that avoids command injection risk while preserving support for "
            "explicit argument lists. Explain the risk and the safer direction after editing."
        ),
        expected_behavior=(
            "The agent should identify shell=True as the command-injection risk.",
            "The implementation should avoid shell=True.",
            "The implementation should accept an explicit argument list.",
            "String commands from editable task data should be rejected or otherwise not executed through a shell.",
            "The tests should pass with uv run pytest.",
        ),
    ),
    "problem-03-cross-platform-task-path": Problem(
        identifier="problem-03-cross-platform-task-path",
        title="Cross-Platform Task Path",
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "The benchmark should find its JSONL task file whether it is run from the project root or from "
            "its own script directory. Make the smallest code change that fixes the path handling. "
            "Explain the change briefly."
        ),
        expected_behavior=(
            "The agent should anchor the task path to the script file, not the current working directory.",
            'The expected pattern is Path(__file__).with_name("personal_tool_reasoning_tasks.jsonl") or an equivalently robust file-relative path.',
            "The fix should be in code/tool-reasoning-benchmark/ollama_tool_reasoning_bench.py.",
            "The tests should pass with uv run pytest tests.",
        ),
    ),
    "problem-04-import-error-after-refactor": Problem(
        identifier="problem-04-import-error-after-refactor",
        title="Import Error After Refactor",
        verify_command=("uv", "run", "pytest", "tests"),
        task_prompt=(
            "The test suite fails after a file move from config.py to settings.py. Inspect the failing import "
            "and make the smallest compatibility-preserving fix so existing imports keep working. "
            "Explain what you changed."
        ),
        expected_behavior=(
            "The agent should inspect the failing test/import before editing.",
            "The fix should preserve the old project.config import path.",
            "A small compatibility module src/project/config.py that re-exports from settings.py is the expected minimal fix.",
            "The tests should pass with uv run pytest tests.",
        ),
    ),
    "problem-05-mutable-default-cache": Problem(
        identifier="problem-05-mutable-default-cache",
        title="Mutable Default Cache Leak",
        verify_command=("uv", "run", "pytest"),
        task_prompt=(
            "A unit test fails only when the whole file is run, but passes in isolation. Diagnose the root "
            "cause and make the smallest safe fix. Explain why the failure only appears when both tests run."
        ),
        expected_behavior=(
            "The agent should identify the mutable default argument as the root cause.",
            "The fix should use None as the default and create a new dict inside the function.",
            "The tests should pass with uv run pytest.",
            "The explanation should mention state leaking across calls/tests.",
        ),
    ),
}


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    if not slug:
        raise ValueError("run name must contain at least one alphanumeric character")
    return slug


def run_command(command, cwd):
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env)


def checked_run(command, cwd):
    result = run_command(command, cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed in {cwd}: {' '.join(command)}\n{result.stdout}{result.stderr}"
        )
    return result


def copy_problem(source, destination):
    def ignore(_directory, names):
        ignored = {".DS_Store", ".git", ".pytest_cache", ".venv", "__pycache__"}
        return {name for name in names if name in ignored}

    shutil.copytree(source, destination, ignore=ignore)


def task_prompt_text(problem):
    return (
        problem.task_prompt
        + "\n\nAt the end, write your concise final answer to AGENT_FINAL_ANSWER.md in this workspace."
    )


def default_usage_record():
    return {
        "schema_version": 1,
        "harness": None,
        "model": None,
        "source": "not_recorded",
        "exact": False,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cached_input_tokens": None,
        "reasoning_output_tokens": None,
        "raw_usage": None,
        "notes": "Overwrite this file with exact CLI usage when running the harness.",
    }


def ensure_usage_file(artifacts):
    usage_path = artifacts / "usage.json"
    if not usage_path.exists():
        usage_path.write_text(json.dumps(default_usage_record(), indent=2) + "\n", encoding="utf-8")


def write_run_metadata(run_dir, problem, run_name):
    metadata = {
        "problem": problem.identifier,
        "title": problem.title,
        "run_name": run_name,
        "verify_command": list(problem.verify_command),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def prepare_run(root, problem_id, run_name):
    problem = PROBLEMS[problem_id]
    safe_run_name = slugify(run_name)
    source = root / problem.identifier
    if not source.exists():
        raise FileNotFoundError(f"problem directory not found: {source}")

    run_dir = root / "runs" / problem.identifier / safe_run_name
    workspace = run_dir / "workspace"
    artifacts = run_dir / "artifacts"
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")

    artifacts.mkdir(parents=True)
    copy_problem(source, workspace)
    shutil.copy2(root / "pyproject.toml", workspace / "pyproject.toml")
    (artifacts / "task-prompt.txt").write_text(task_prompt_text(problem) + "\n", encoding="utf-8")
    ensure_usage_file(artifacts)
    (workspace / "AGENT_FINAL_ANSWER.md").write_text(
        "Write the final answer for this run here.\n",
        encoding="utf-8",
    )
    write_run_metadata(run_dir, problem, safe_run_name)

    checked_run(["git", "init", "--quiet"], workspace)
    checked_run(["git", "add", "."], workspace)
    checked_run(
        [
            "git",
            "-c",
            "user.name=agent-problem-pack",
            "-c",
            "user.email=agent-problem-pack@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "baseline",
        ],
        workspace,
    )
    return run_dir


def load_run_problem(run_dir):
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return PROBLEMS[metadata["problem"]]


def resolve_run_dir(run_dir, root):
    run_dir = Path(run_dir)
    if run_dir.is_absolute():
        return run_dir
    cwd_relative = run_dir.resolve()
    if (cwd_relative / "metadata.json").exists():
        return cwd_relative
    return (root / run_dir).resolve()


def write_combined_output(path, command, result):
    text = [
        f"$ {' '.join(command)}",
        f"exit_code={result.returncode}",
        "",
        "[stdout]",
        result.stdout.rstrip(),
        "",
        "[stderr]",
        result.stderr.rstrip(),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def build_evaluation_prompt(run_dir, problem):
    workspace = run_dir / "workspace"
    artifacts = run_dir / "artifacts"
    expected = "\n".join(f"- {item}" for item in problem.expected_behavior)
    return f"""Evaluate the agent run for {problem.identifier}.

Read these files directly:
- Final answer: {workspace / "AGENT_FINAL_ANSWER.md"}
- Diff: {artifacts / "diff.patch"}
- Git status: {artifacts / "git-status.txt"}
- Verification output: {artifacts / "verification.txt"}
- Token usage: {artifacts / "usage.json"}

Expected behavior:
{expected}

Please return:
- pass/fail
- concise reasoning
- token usage summary if available
- any partial credit notes
- any concerns about unnecessary edits
"""


def capture_run(run_dir, root=PACK_ROOT):
    run_dir = resolve_run_dir(run_dir, Path(root).resolve())
    workspace = run_dir / "workspace"
    artifacts = run_dir / "artifacts"
    problem = load_run_problem(run_dir)

    artifacts.mkdir(exist_ok=True)
    ensure_usage_file(artifacts)
    verification = run_command(problem.verify_command, workspace)
    write_combined_output(artifacts / "verification.txt", problem.verify_command, verification)

    checked_run(["git", "add", "-N", "."], workspace)
    diff = checked_run(["git", "diff", "--no-ext-diff", "--", "."], workspace)
    status = checked_run(["git", "status", "--short"], workspace)
    (artifacts / "diff.patch").write_text(diff.stdout, encoding="utf-8")
    (artifacts / "git-status.txt").write_text(status.stdout, encoding="utf-8")
    (artifacts / "evaluate-with-codex.md").write_text(
        build_evaluation_prompt(run_dir, problem),
        encoding="utf-8",
    )
    return verification


def list_problems():
    for problem in PROBLEMS.values():
        print(f"{problem.identifier}: {problem.title}")


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare and capture agent problem-pack runs.")
    parser.add_argument("--root", type=Path, default=PACK_ROOT, help=f"Pack root. Default: {PACK_ROOT}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available problem ids.")

    prepare_parser = subparsers.add_parser("prepare", help="Create an isolated run workspace.")
    prepare_parser.add_argument("problem", choices=sorted(PROBLEMS))
    prepare_parser.add_argument("run_name")

    capture_parser = subparsers.add_parser("capture", help="Capture diff, verification output, and evaluation prompt.")
    capture_parser.add_argument("run_dir", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            list_problems()
            return 0
        if args.command == "prepare":
            run_dir = prepare_run(args.root.resolve(), args.problem, args.run_name)
            print(run_dir)
            print(run_dir / "artifacts" / "task-prompt.txt")
            return 0
        if args.command == "capture":
            result = capture_run(args.run_dir, args.root.resolve())
            print(args.run_dir / "artifacts" / "evaluate-with-codex.md")
            return result.returncode
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
