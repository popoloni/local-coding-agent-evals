#!/usr/bin/env python3
"""Build comparison graphs from llmstack benchmark CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class SpeedPoint:
    model_key: str
    requested_words: int
    prefill_tps: float | None
    decode_tps: float | None
    mlx_peak_gb: float | None
    wall_s: float | None


@dataclass
class ReasoningPoint:
    model_key: str
    pass_rate: float
    passed: int
    total: int


@dataclass
class RetrySummary:
    model_key: str
    speed_retry_count: int | None
    reasoning_retry_count: int | None


@dataclass
class AgentPackPoint:
    model_key: str
    pass_rate: float
    passed: int
    total: int


def combined_retry_count(speed_retry_count: int | None, reasoning_retry_count: int | None) -> int | None:
    values = [value for value in (speed_retry_count, reasoning_retry_count) if value is not None]
    if not values:
        return None
    return max(values)


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def to_bool(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "pass", "passed"}


def latest_csv_per_model(results_root: Path, filename: str) -> dict[str, Path]:
    candidates = sorted(results_root.glob(f"llmstack-matrix-*/**/{filename}"))
    by_model: dict[str, Path] = {}
    for path in candidates:
        model_key = path.parent.name
        prev = by_model.get(model_key)
        if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
            by_model[model_key] = path
    return by_model


def load_speed_points(results_root: Path) -> list[SpeedPoint]:
    speed_files = latest_csv_per_model(results_root, "llmstack_speed_memory_results.csv")
    points: list[SpeedPoint] = []
    for model_key, csv_path in sorted(speed_files.items()):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        rows.sort(key=lambda row: to_int(row.get("requested_words")) or 0)
        row = rows[-1]
        points.append(
            SpeedPoint(
                model_key=model_key,
                requested_words=to_int(row.get("requested_words")) or 0,
                prefill_tps=to_float(row.get("prefill_real_tps")),
                decode_tps=to_float(row.get("decode_tps")),
                mlx_peak_gb=to_float(row.get("mlx_peak_gb")),
                wall_s=to_float(row.get("wall_s")),
            )
        )
    return points


def load_reasoning_points(results_root: Path) -> list[ReasoningPoint]:
    reasoning_files = latest_csv_per_model(results_root, "llmstack_hard_reasoning_results.csv")
    points: list[ReasoningPoint] = []
    for model_key, csv_path in sorted(reasoning_files.items()):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        total = len(rows)
        passed = sum(1 for row in rows if to_bool(row.get("passed")))
        points.append(
            ReasoningPoint(
                model_key=model_key,
                pass_rate=(passed / total) * 100.0,
                passed=passed,
                total=total,
            )
        )
    return points


def load_retry_summaries(results_root: Path) -> list[RetrySummary]:
    speed_files = latest_csv_per_model(results_root, "llmstack_speed_memory_results.csv")
    reasoning_files = latest_csv_per_model(results_root, "llmstack_hard_reasoning_results.csv")

    def retry_from_file(csv_path: Path) -> int | None:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return None
        value = rows[0].get("retry_count")
        parsed = to_int(value)
        return parsed if parsed is not None else None

    models = sorted(set(speed_files.keys()) | set(reasoning_files.keys()))
    summaries: list[RetrySummary] = []
    for model in models:
        speed_retry = retry_from_file(speed_files[model]) if model in speed_files else None
        reasoning_retry = retry_from_file(reasoning_files[model]) if model in reasoning_files else None
        summaries.append(
            RetrySummary(
                model_key=model,
                speed_retry_count=speed_retry,
                reasoning_retry_count=reasoning_retry,
            )
        )
    return summaries


def _parse_verification_exit_code(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^exit_code=(\d+)\s*$", text, flags=re.MULTILINE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _infer_model_from_run_name(run_name: str, known_models: list[str]) -> str | None:
    for model in sorted(known_models, key=len, reverse=True):
        if model in run_name:
            return model
    return None


def load_agent_pack_points(results_root: Path, known_models: list[str]) -> list[AgentPackPoint]:
    del results_root  # reserved for future correlation with a specific matrix folder
    runs_root = Path(__file__).resolve().parent / "agent-problem-pack" / "runs"
    if not runs_root.exists() or not known_models:
        return []

    latest_by_model_problem: dict[tuple[str, str], tuple[float, bool]] = {}
    for verification_path in runs_root.glob("*/**/artifacts/verification.txt"):
        run_dir = verification_path.parent.parent
        metadata_path = run_dir / "metadata.json"
        run_name = run_dir.name
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                run_name = str(metadata.get("run_name") or run_name)
            except (json.JSONDecodeError, OSError):
                pass

        model_key = _infer_model_from_run_name(run_name, known_models)
        if model_key is None:
            continue

        problem_key = run_dir.parent.name
        exit_code = _parse_verification_exit_code(verification_path)
        passed = exit_code == 0
        mtime = verification_path.stat().st_mtime
        key = (model_key, problem_key)
        previous = latest_by_model_problem.get(key)
        if previous is None or mtime > previous[0]:
            latest_by_model_problem[key] = (mtime, passed)

    passed_total: dict[str, tuple[int, int]] = {}
    for (model_key, _problem_key), (_mtime, passed) in latest_by_model_problem.items():
        current_passed, current_total = passed_total.get(model_key, (0, 0))
        passed_total[model_key] = (current_passed + (1 if passed else 0), current_total + 1)

    points: list[AgentPackPoint] = []
    for model_key, (passed, total) in sorted(passed_total.items()):
        if total <= 0:
            continue
        points.append(
            AgentPackPoint(
                model_key=model_key,
                pass_rate=(passed / total) * 100.0,
                passed=passed,
                total=total,
            )
        )
    return points


def barplot(
    ax,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    higher_is_better: bool,
    annotations: list[str] | None = None,
    missing: list[bool] | None = None,
) -> None:
    if not labels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    colors = ["#9ca3af" if (missing and missing[idx]) else "#2563eb" for idx in range(len(values))]
    bars = ax.bar(labels, values, color=colors)
    for idx, (bar, value) in enumerate(zip(bars, values)):
        label = annotations[idx] if annotations and idx < len(annotations) else f"{value:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    if higher_is_better:
        ax.annotate("Higher is better", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9)
    else:
        ax.annotate("Lower is better", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9)


def write_summary(
    output_path: Path,
    speed_points: list[SpeedPoint],
    reasoning_points: list[ReasoningPoint],
    retry_summaries: list[RetrySummary],
    agent_points: list[AgentPackPoint],
) -> Path:
    summary_path = output_path.with_suffix(".md")
    reasoning_map = {point.model_key: point for point in reasoning_points}
    lines = [
        "# LLMStack Comparison Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Latest Speed Snapshot (largest segment per model)",
        "",
        "| Model | Segment words | Prefill tok/s | Decode tok/s | MLX peak GB | Wall s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    if speed_points:
        for point in speed_points:
            lines.append(
                "| "
                f"{point.model_key} | {point.requested_words} | "
                f"{point.prefill_tps if point.prefill_tps is not None else 'n/a'} | "
                f"{point.decode_tps if point.decode_tps is not None else 'n/a'} | "
                f"{point.mlx_peak_gb if point.mlx_peak_gb is not None else 'n/a'} | "
                f"{point.wall_s if point.wall_s is not None else 'n/a'} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend([
        "",
        "## Stability (Retries Used)",
        "",
        "`retry_count` is propagated from the matrix runner and indicates how many transient infra retries were needed for that benchmark run.",
        "",
        "| Model | Speed retry_count | Reasoning retry_count | Total retry_count (max) |",
        "| --- | ---: | ---: | ---: |",
    ])

    if retry_summaries:
        for item in retry_summaries:
            total_retry = combined_retry_count(item.speed_retry_count, item.reasoning_retry_count)
            lines.append(
                f"| {item.model_key} | "
                f"{item.speed_retry_count if item.speed_retry_count is not None else 'n/a'} | "
                f"{item.reasoning_retry_count if item.reasoning_retry_count is not None else 'n/a'} | "
                f"{total_retry if total_retry is not None else 'n/a'} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a |")

    lines.extend([
        "",
        "## Reasoning Score",
        "",
        "| Model | Passed | Total | Pass rate % |",
        "| --- | ---: | ---: | ---: |",
    ])

    all_models = sorted({point.model_key for point in speed_points} | set(reasoning_map.keys()))
    if all_models:
        for model in all_models:
            r = reasoning_map.get(model)
            if r is None:
                lines.append(f"| {model} | n/a | n/a | n/a |")
            else:
                lines.append(f"| {model} | {r.passed} | {r.total} | {r.pass_rate:.1f} |")
    else:
        lines.append("| n/a | n/a | n/a | n/a |")

    lines.extend([
        "",
        "## Agent Problem Pack Score",
        "",
        "| Model | Passed | Total | Pass rate % |",
        "| --- | ---: | ---: | ---: |",
    ])
    agent_map = {point.model_key: point for point in agent_points}
    agent_models = sorted(set(all_models) | set(agent_map.keys()))
    if agent_models:
        for model in agent_models:
            point = agent_map.get(model)
            if point is None:
                lines.append(f"| {model} | n/a | n/a | n/a |")
            else:
                lines.append(f"| {model} | {point.passed} | {point.total} | {point.pass_rate:.1f} |")
    else:
        lines.append("| n/a | n/a | n/a | n/a |")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate comparison graphs from local-coding-agent-evals llmstack CSV results.")
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parent / "results", help="Results folder. Default: local-coding-agent-evals/results.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "llmstack_comparison.png", help="Output graph path (PNG).")
    parser.add_argument("--title", default="LLMStack Model Comparison", help="Chart title.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results_root = args.results_root.resolve()
    output_path = args.output.resolve()

    if not results_root.exists():
        print(f"error: results root not found: {results_root}")
        return 1

    speed_points = load_speed_points(results_root)
    reasoning_points = load_reasoning_points(results_root)
    retry_summaries = load_retry_summaries(results_root)
    known_models = sorted({point.model_key for point in speed_points} | {point.model_key for point in reasoning_points})
    agent_points = load_agent_pack_points(results_root, known_models)
    if not speed_points and not reasoning_points:
        print(f"error: no llmstack CSV files found under {results_root}")
        return 1

    speed_map = {point.model_key: point for point in speed_points}
    reasoning_map = {point.model_key: point for point in reasoning_points}
    agent_map = {point.model_key: point for point in agent_points}
    model_labels = sorted(set(speed_map.keys()) | set(reasoning_map.keys()) | set(agent_map.keys()))

    prefill_missing = [not (label in speed_map and speed_map[label].prefill_tps is not None) for label in model_labels]
    decode_missing = [not (label in speed_map and speed_map[label].decode_tps is not None) for label in model_labels]
    mlx_missing = [not (label in speed_map and speed_map[label].mlx_peak_gb is not None) for label in model_labels]
    wall_missing = [not (label in speed_map and speed_map[label].wall_s is not None) for label in model_labels]

    prefill = [speed_map[label].prefill_tps if label in speed_map and speed_map[label].prefill_tps is not None else 0.0 for label in model_labels]
    decode = [speed_map[label].decode_tps if label in speed_map and speed_map[label].decode_tps is not None else 0.0 for label in model_labels]
    mlx_peak = [speed_map[label].mlx_peak_gb if label in speed_map and speed_map[label].mlx_peak_gb is not None else 0.0 for label in model_labels]
    wall_s = [speed_map[label].wall_s if label in speed_map and speed_map[label].wall_s is not None else 0.0 for label in model_labels]
    pass_rate = [reasoning_map[label].pass_rate if label in reasoning_map else 0.0 for label in model_labels]
    agent_pass_rate = [agent_map[label].pass_rate if label in agent_map else 0.0 for label in model_labels]

    prefill_labels = ["n/a" if missing else f"{value:.2f}" for value, missing in zip(prefill, prefill_missing)]
    decode_labels = ["n/a" if missing else f"{value:.2f}" for value, missing in zip(decode, decode_missing)]
    mlx_labels = ["n/a" if missing else f"{value:.2f}" for value, missing in zip(mlx_peak, mlx_missing)]
    wall_labels = ["n/a" if missing else f"{value:.2f}" for value, missing in zip(wall_s, wall_missing)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    fig.suptitle(args.title, fontsize=16)

    barplot(axes[0][0], model_labels, prefill, "Prefill Throughput", "Tokens/s", higher_is_better=True, annotations=prefill_labels, missing=prefill_missing)
    barplot(axes[0][1], model_labels, decode, "Decode Throughput", "Tokens/s", higher_is_better=True, annotations=decode_labels, missing=decode_missing)
    barplot(axes[1][0], model_labels, mlx_peak, "MLX Peak Memory", "GB", higher_is_better=False, annotations=mlx_labels, missing=mlx_missing)
    barplot(axes[1][1], model_labels, wall_s, "Largest Segment Wall Time", "Seconds", higher_is_better=False, annotations=wall_labels, missing=wall_missing)
    barplot(axes[2][0], model_labels, pass_rate, "Hard Reasoning Pass Rate", "%", higher_is_better=True)
    barplot(axes[2][1], model_labels, agent_pass_rate, "Agent Problem Pack Pass Rate", "%", higher_is_better=True)

    fig.savefig(output_path, dpi=150)
    summary_path = write_summary(output_path, speed_points, reasoning_points, retry_summaries, agent_points)

    print(f"Wrote chart: {output_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
