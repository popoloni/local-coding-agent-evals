#!/usr/bin/env python3
"""On-demand llmstack evaluation matrix runner across configured model/backend pairs."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmstack_eval_utils import (  # noqa: E402
    default_python_bin,
    discover_llmstack_root,
    headroom_base_urls,
    inference_base_urls,
    load_llmstack_config,
    model_entries,
    resolve_model_target,
    run_shell_command,
    start_script_for_type,
    stop_all_services,
    stop_script_for_type,
    wait_for_headroom,
    wait_for_served_model,
)


def iter_selected_models(cfg: dict, include: set[str] | None, backend_filter: set[str] | None):
    for model_key, model_cfg in model_entries(cfg):
        backend_type = str(model_cfg.get("type") or "").lower()
        if include and model_key not in include:
            continue
        if backend_filter and backend_type not in backend_filter:
            continue
        yield model_key, model_cfg


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _is_transient_failure_text(text: str) -> bool:
    haystack = (text or "").lower()
    markers = (
        "http 502",
        "proxy_error",
        "connection refused",
        "timed out",
        "could not connect to llmstack",
        "could not reach llmstack",
        "remote end closed connection",
        "closed the connection without a response",
        "headroom failed to become healthy",
    )
    return any(marker in haystack for marker in markers)


def _restart_headroom_and_wait(llmstack_root: Path, headroom_health_url: str) -> None:
    subprocess.run(["bash", str(llmstack_root / "bin" / "stop_headroom_server.bash")], cwd=llmstack_root)
    started = run_shell_command(["bash", str(llmstack_root / "bin" / "start_headroom_server.bash")], llmstack_root)
    if started.stdout.strip():
        print(started.stdout.strip())
    if started.returncode != 0:
        if started.stderr.strip():
            print(started.stderr.strip(), file=sys.stderr)
        raise RuntimeError("failed to restart headroom")
    wait_for_headroom(headroom_health_url, timeout_s=90.0)


def _run_benchmark_with_retries(*, cmd: list[str], llmstack_root: Path, label: str, transient_retries: int, headroom_health_url: str, models_url: str, target: str) -> int:
    attempts = max(1, transient_retries + 1)
    retries_used = 0
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, cwd=llmstack_root, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        if result.returncode == 0:
            return 0, retries_used

        combined = "\n".join(filter(None, [result.stdout, result.stderr]))
        if attempt >= attempts or not _is_transient_failure_text(combined):
            return result.returncode, retries_used

        backoff_s = attempt * 3
        retries_used += 1
        print(
            f"warning: {label} transient failure (attempt {attempt}/{attempts}); restarting headroom and retrying in {backoff_s}s",
            file=sys.stderr,
        )
        _restart_headroom_and_wait(llmstack_root, headroom_health_url)
        wait_for_served_model(models_url, target, timeout_s=180.0)
        time.sleep(backoff_s)

    return 1, retries_used


def _annotate_csv_retry_metadata(csv_path: Path, retry_count: int) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    if not rows:
        return

    if "retry_count" not in fieldnames:
        fieldnames.append("retry_count")
    if "retry_used" not in fieldnames:
        fieldnames.append("retry_used")

    for row in rows:
        row["retry_count"] = str(retry_count)
        row["retry_used"] = "true" if retry_count > 0 else "false"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_for_model(*, llmstack_root: Path, python_bin: str, model_key: str, model_cfg: dict, output_root: Path, include_speed: bool, include_reasoning: bool, include_agent_pack: bool, surface: str, bypass_permissions: bool) -> int:
    cfg = load_llmstack_config(llmstack_root)
    _, target = resolve_model_target(cfg, model_key)
    backend_type = str(model_cfg.get("type") or "").lower()
    _, models_url = inference_base_urls(cfg)
    _, headroom_health_url = headroom_base_urls(cfg)

    stop_all_services(llmstack_root)
    start_script = start_script_for_type(llmstack_root, backend_type)
    stop_script = stop_script_for_type(llmstack_root, backend_type)

    print(f"\n=== [{model_key}] backend={backend_type} target={target} ===")
    start_result = run_shell_command(["bash", str(start_script), model_key], llmstack_root)
    if start_result.stdout.strip():
        print(start_result.stdout.strip())
    if start_result.returncode != 0:
        if start_result.stderr.strip():
            print(start_result.stderr.strip(), file=sys.stderr)
        return 1

    wait_for_served_model(models_url, target, timeout_s=180.0)

    headroom_started = run_shell_command(["bash", str(llmstack_root / "bin" / "start_headroom_server.bash")], llmstack_root)
    if headroom_started.stdout.strip():
        print(headroom_started.stdout.strip())
    if headroom_started.returncode != 0:
        if headroom_started.stderr.strip():
            print(headroom_started.stderr.strip(), file=sys.stderr)
        subprocess.run(["bash", str(stop_script)], cwd=llmstack_root)
        return 1
    try:
        wait_for_headroom(headroom_health_url, timeout_s=90.0)
    except RuntimeError:
        _restart_headroom_and_wait(llmstack_root, headroom_health_url)

    model_output_dir = output_root / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    try:
        if include_speed:
            speed_csv = model_output_dir / "llmstack_speed_memory_results.csv"
            cmd = [
                python_bin,
                str(ROOT / "speed-memory-benchmark" / "llmstack_speed_memory_bench.py"),
                "--llmstack-root",
                str(llmstack_root),
                "--model-key",
                model_key,
                "--surface",
                surface,
                "--csv",
                str(speed_csv),
            ]
            result_code, retry_count = _run_benchmark_with_retries(
                cmd=cmd,
                llmstack_root=llmstack_root,
                label=f"speed benchmark for {model_key}",
                transient_retries=2,
                headroom_health_url=headroom_health_url,
                models_url=models_url,
                target=target,
            )
            _annotate_csv_retry_metadata(speed_csv, retry_count)
            failures += 0 if result_code == 0 else 1

        if include_reasoning:
            reasoning_csv = model_output_dir / "llmstack_hard_reasoning_results.csv"
            cmd = [
                python_bin,
                str(ROOT / "hard-tool-reasoning-benchmark" / "llmstack_hard_reasoning_bench.py"),
                "--llmstack-root",
                str(llmstack_root),
                "--model-key",
                model_key,
                "--surface",
                surface,
                "--csv",
                str(reasoning_csv),
            ]
            result_code, retry_count = _run_benchmark_with_retries(
                cmd=cmd,
                llmstack_root=llmstack_root,
                label=f"reasoning benchmark for {model_key}",
                transient_retries=2,
                headroom_health_url=headroom_health_url,
                models_url=models_url,
                target=target,
            )
            _annotate_csv_retry_metadata(reasoning_csv, retry_count)
            failures += 0 if result_code == 0 else 1

        if include_agent_pack:
            cmd = [
                python_bin,
                str(ROOT / "agent-problem-pack" / "scripts" / "llmstack_headless_runner.py"),
                "--llmstack-root",
                str(llmstack_root),
                "--model-key",
                model_key,
                "--run-name-prefix",
                f"matrix-{model_key}",
                "--run-name-suffix",
                output_root.name,
            ]
            if bool(cfg.get("agent_pack_overwrite_runs", False)):
                cmd.append("--overwrite")
            if not bypass_permissions:
                cmd.append("--no-bypass-permissions")
            result = subprocess.run(cmd, cwd=llmstack_root)
            failures += 0 if result.returncode == 0 else 1
    finally:
        subprocess.run(["bash", str(llmstack_root / "bin" / "stop_headroom_server.bash")], cwd=llmstack_root)
        subprocess.run(["bash", str(stop_script)], cwd=llmstack_root)

    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run llmstack-backed evals across configured model/backend pairs on demand.")
    parser.add_argument("--llmstack-root", type=Path, help="Path to the llmstack workspace root. Default: auto-detect.")
    parser.add_argument("--python-bin", help="Python executable used for llmstack and eval scripts.")
    parser.add_argument("--include-model", action="append", default=[], help="Restrict execution to specific llmstack model keys. Repeatable.")
    parser.add_argument("--backend", action="append", default=[], help="Restrict execution to backend types: dflash, mlx, turboquant. Repeatable.")
    parser.add_argument("--skip-speed", action="store_true", help="Skip the speed-memory benchmark.")
    parser.add_argument("--skip-reasoning", action="store_true", help="Skip the hard reasoning benchmark.")
    parser.add_argument("--include-agent-pack", action="store_true", help="Also run the llmstack agent-problem-pack runner for each model key.")
    parser.add_argument("--surface", choices=("inference", "headroom"), default="headroom", help="Chat surface used by the benchmark scripts. Default: headroom.")
    parser.add_argument("--no-bypass-permissions", action="store_true", help="Pass through to the agent-pack runner when included.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory. Default: results/llmstack-matrix-<timestamp>.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    llmstack_root = (args.llmstack_root or discover_llmstack_root(ROOT)).resolve()
    python_bin = args.python_bin or default_python_bin(llmstack_root)
    cfg = load_llmstack_config(llmstack_root)
    include = set(args.include_model or []) or None
    backend_filter = {item.lower() for item in (args.backend or [])} or None
    output_root = (args.output_dir or (ROOT / "results" / f"llmstack-matrix-{timestamp_slug()}" )).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    selected = list(iter_selected_models(cfg, include, backend_filter))
    if not selected:
        print("error: no models matched the requested filters", file=sys.stderr)
        return 1

    for model_key, model_cfg in selected:
        failures += run_for_model(
            llmstack_root=llmstack_root,
            python_bin=python_bin,
            model_key=model_key,
            model_cfg=model_cfg,
            output_root=output_root,
            include_speed=not args.skip_speed,
            include_reasoning=not args.skip_reasoning,
            include_agent_pack=args.include_agent_pack,
            surface=args.surface,
            bypass_permissions=not args.no_bypass_permissions,
        )

    print(f"\nOutput root: {output_root}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
