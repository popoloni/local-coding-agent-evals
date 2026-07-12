#!/usr/bin/env python3
"""Benchmark llmstack long-context performance using the local OpenAI-compatible API and timing CSV."""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmstack_eval_utils import (  # noqa: E402
    MemorySampler,
    activate_llmstack_model,
    build_chat_payload,
    check_llmstack,
    default_python_bin,
    discover_llmstack_root,
    find_process_for_port,
    headroom_base_urls,
    inference_base_urls,
    load_llmstack_config,
    post_json_with_fallback,
    resolve_model_target,
    response_text,
    timings_csv_path,
    wait_for_new_timing_row,
)


DEFAULT_SEGMENTS = "1000,5000,10000,50000"


def parse_segments(value: str) -> list[int]:
    parts = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        segment = int(part)
        if segment <= 0:
            raise ValueError(f"segments must be positive integers, got {segment}")
        parts.append(segment)
    if not parts:
        raise ValueError("at least one segment is required")
    return parts


def make_synthetic_prompt(target_words: int) -> str:
    header = (
        "You are evaluating long-context recall and summarization.\n"
        "Read the numbered records. Preserve important IDs and trends.\n\n"
    )
    footer = (
        "\n\nAnswer the final question in five concise bullets. "
        "Mention the first record ID, the last record ID, and any repeated anomaly."
    )
    vocabulary = [
        "record", "context", "latency", "memory", "throughput", "attention", "window", "retrieval",
        "summary", "signal", "needle", "sequence", "benchmark", "segment", "token", "evaluation",
    ]
    words: list[str] = []
    record_id = 0
    while len(words) < target_words:
        record_id += 1
        anomaly = "ANOMALY_ALPHA" if record_id % 257 == 0 else "normal"
        words.extend([f"ID{record_id:06d}", anomaly, *vocabulary])
    body = " ".join(words[:target_words])
    return f"{header}{body}{footer}"


def make_file_prompt(path: Path, target_words: int) -> str:
    text = path.read_text(encoding="utf-8")
    words = text.split()
    if not words:
        raise ValueError(f"input file is empty: {path}")
    repeated_words: list[str] = []
    while len(repeated_words) < target_words:
        repeated_words.extend(words)
    body = " ".join(repeated_words[:target_words])
    return (
        "Read this long context and answer the final question.\n\n"
        f"{body}\n\n"
        "Answer the final question in five concise bullets. Summarize the key points."
    )


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def run_segment(*, chat_url: str, fallback_chat_url: str | None, target: str, cfg: dict, prompt: str, requested_words: int, max_tokens: int, timeout_s: float, sample_interval_s: float, temperature: float, top_p: float, csv_path: Path) -> dict:
    before_count = 0
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            before_count = sum(1 for _ in handle) - 1
            before_count = max(before_count, 0)

    pid = find_process_for_port(int(cfg.get("inference_port", 8787)))
    sampler = MemorySampler(pid, sample_interval_s)
    started_at = time.monotonic()
    payload = build_chat_payload(target, prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    response = sampler.run_during(
        lambda: post_json_with_fallback(
            chat_url,
            payload,
            timeout_s,
            fallback_url=fallback_chat_url,
            retries=2,
            retry_backoff_s=2.0,
            request_label=f"speed segment {requested_words}",
        )
    )
    wall_s = time.monotonic() - started_at
    memory = sampler.summary()
    timing_row = wait_for_new_timing_row(csv_path, before_count, target)

    row = {
        "model_target": target,
        "requested_words": requested_words,
        "max_tokens": max_tokens,
        "wall_s": round(wall_s, 2),
        "response_chars": len(response_text(response)),
        "backend": timing_row.get("backend") if timing_row else None,
        "served_target": timing_row.get("served_target") if timing_row else None,
        "prompt_tokens": as_int(timing_row.get("prompt_tokens")) if timing_row else None,
        "prefill_time_s": as_float(timing_row.get("prefill_time_s")) if timing_row else None,
        "prefill_real_tps": as_float(timing_row.get("prefill_real_tps")) if timing_row else None,
        "decode_tokens": as_int(timing_row.get("decode_tokens")) if timing_row else None,
        "decode_time_s": as_float(timing_row.get("decode_time_s")) if timing_row else None,
        "decode_tps": as_float(timing_row.get("decode_tps")) if timing_row else None,
        "cache_hit_pct": as_float(timing_row.get("cache_hit_pct")) if timing_row else None,
        "mlx_peak_gb": as_float(timing_row.get("mlx_peak_gb")) if timing_row else None,
        **memory,
    }
    return row


def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_summary(rows: list[dict]) -> None:
    for row in rows:
        print(f"Segment {row['requested_words']} words")
        print(
            "  "
            f"Tokens: prompt {format_value(row.get('prompt_tokens'))}, "
            f"generated {format_value(row.get('decode_tokens'))}"
        )
        print(
            "  "
            f"Speed: prefill {format_value(row.get('prefill_real_tps'))} tok/s, "
            f"decode {format_value(row.get('decode_tps'))} tok/s"
        )
        print(
            "  "
            f"Time: wall {format_value(row.get('wall_s'))} s, "
            f"prefill {format_value(row.get('prefill_time_s'))} s, "
            f"decode {format_value(row.get('decode_time_s'))} s"
        )
        print(
            "  "
            f"Memory: RSS {format_value(row.get('server_rss_peak_mb'))} MB, "
            f"GPU {format_value(row.get('nvidia_gpu_peak_mb'))} MB, "
            f"MLX peak {format_value(row.get('mlx_peak_gb'))} GB"
        )
        print()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark llmstack model speed and memory over long-context segments.")
    parser.add_argument("--model-key", help="llmstack model registry key, for example dflash-qwen35b-moe.")
    parser.add_argument("--activate-model", action="store_true", help="Call `llmstack model use <model-key> --restart` before the benchmark.")
    parser.add_argument("--llmstack-root", type=Path, help="Path to the llmstack workspace root. Default: auto-detect via llmstack_config.json.")
    parser.add_argument("--python-bin", help="Python executable used for optional llmstack CLI calls.")
    parser.add_argument("--segments", default=DEFAULT_SEGMENTS, help=f"Comma-separated approximate prompt word counts. Default: {DEFAULT_SEGMENTS}.")
    parser.add_argument("--input-file", type=Path, help="Optional text file to repeat and truncate instead of synthetic text.")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Generated tokens per segment. Default: 8000.")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="Memory sampling interval in seconds. Default: 1.0.")
    parser.add_argument("--timeout", type=float, default=7200, help="Per-segment HTTP timeout in seconds. Default: 7200.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Default: 0.0.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Accepted for CLI compatibility; llmstack path currently uses the default server-side sampling behavior.")
    parser.add_argument("--surface", choices=("inference", "headroom"), default="inference", help="Send requests either directly to inference or through Headroom. Default: inference.")
    parser.add_argument("--csv", type=Path, help="Optional output CSV path.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        segments = parse_segments(args.segments)
    except ValueError as exc:
        parser.error(str(exc))

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")

    try:
        llmstack_root = (args.llmstack_root or discover_llmstack_root(ROOT)).resolve()
        cfg = load_llmstack_config(llmstack_root)
        model_key, target = resolve_model_target(cfg, args.model_key)
        python_bin = args.python_bin or default_python_bin(llmstack_root)
        if args.activate_model:
            activate_llmstack_model(llmstack_root, model_key, python_bin=python_bin)
            cfg = load_llmstack_config(llmstack_root)
        inference_chat_url, models_url = inference_base_urls(cfg)
        headroom_chat_url, headroom_health_url = headroom_base_urls(cfg)
        model_ids = check_llmstack(models_url, min(args.timeout, 10))
        chat_url = inference_chat_url if args.surface == "inference" else headroom_chat_url
        fallback_chat_url = inference_chat_url if args.surface == "headroom" else None
        csv_path = timings_csv_path(llmstack_root, cfg)

        print(f"Platform: {platform.platform()}")
        print(f"llmstack root: {llmstack_root}")
        print(f"Model key: {model_key}")
        print(f"Model target: {target}")
        print(f"Surface: {args.surface}")
        print(f"Active /v1/models: {', '.join(model_ids) if model_ids else 'n/a'}")
        print(f"Chat URL: {chat_url}")
        if args.surface == "headroom":
            print(f"Headroom health: {headroom_health_url}")
            print(f"Fallback chat URL on transient errors: {inference_chat_url}")
        print(f"Timing CSV: {csv_path}")
        print("Memory: RSS is the llmstack inference process RAM; GPU is nvidia-smi memory when available.")
        print("Timing: prefill/decode metrics are read from llmstack's timing CSV after each request.")
        print()

        rows = []
        for segment in segments:
            print(f"Running segment {segment} requested words...", flush=True)
            prompt = make_file_prompt(args.input_file, segment) if args.input_file else make_synthetic_prompt(segment)
            try:
                row = run_segment(
                    chat_url=chat_url,
                    fallback_chat_url=fallback_chat_url,
                    target=target,
                    cfg=cfg,
                    prompt=prompt,
                    requested_words=segment,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout,
                    sample_interval_s=args.sample_interval,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    csv_path=csv_path,
                )
            except RuntimeError as exc:
                row = {
                    "model_target": target,
                    "requested_words": segment,
                    "max_tokens": args.max_tokens,
                    "wall_s": None,
                    "response_chars": 0,
                    "backend": None,
                    "served_target": None,
                    "prompt_tokens": None,
                    "prefill_time_s": None,
                    "prefill_real_tps": None,
                    "decode_tokens": None,
                    "decode_time_s": None,
                    "decode_tps": None,
                    "cache_hit_pct": None,
                    "mlx_peak_gb": None,
                    "server_rss_peak_mb": None,
                    "server_rss_end_mb": None,
                    "nvidia_gpu_peak_mb": None,
                    "nvidia_gpu_end_mb": None,
                    "error": str(exc),
                }
                print(f"warning: segment {segment} failed: {exc}", file=sys.stderr, flush=True)
            rows.append(row)
            print(
                "  "
                f"wall={format_value(row['wall_s'])}s, "
                f"prefill_tps={format_value(row.get('prefill_real_tps'))}, "
                f"decode_tps={format_value(row.get('decode_tps'))}, "
                f"mlx_peak_gb={format_value(row.get('mlx_peak_gb'))}",
                flush=True,
            )

        print()
        print_summary(rows)

        if args.csv:
            write_csv(args.csv, rows)
            print(f"\nWrote CSV: {args.csv}")
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())