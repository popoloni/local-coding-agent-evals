#!/usr/bin/env python3
"""
Benchmark Ollama long-context prompt evaluation across several prompt sizes.

Works with the Python standard library only. On macOS it samples Ollama process
RSS. On Linux/NVIDIA systems, including DGX Spark style setups, it also samples
GPU memory via nvidia-smi when available.
"""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_SEGMENTS = "1000,5000,10000,50000"
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class MemorySample:
    def __init__(self, elapsed_s, ollama_rss_mb, nvidia_gpu_used_mb):
        self.elapsed_s = elapsed_s
        self.ollama_rss_mb = ollama_rss_mb
        self.nvidia_gpu_used_mb = nvidia_gpu_used_mb


class MemorySampler:
    def __init__(self, interval_s):
        self.interval_s = interval_s
        self.samples = []
        self._started_at = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2))
        if not self.samples:
            self.samples.append(self._sample())
        return self.samples

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval_s)

    def _sample(self):
        return MemorySample(
            elapsed_s=time.monotonic() - self._started_at,
            ollama_rss_mb=sample_ollama_rss_mb(),
            nvidia_gpu_used_mb=sample_nvidia_gpu_used_mb(),
        )


def parse_segments(value):
    segments = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            segment = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid segment {part!r}") from exc
        if segment <= 0:
            raise ValueError(f"segments must be positive integers, got {segment}")
        segments.append(segment)
    if not segments:
        raise ValueError("at least one segment is required")
    return segments


def make_synthetic_prompt(target_words):
    header = (
        "You are evaluating long-context recall and summarization.\n"
        "Read the numbered records. Preserve important IDs and trends.\n\n"
    )
    footer = (
        "\n\nAnswer the final question in five concise bullets. "
        "Mention the first record ID, the last record ID, and any repeated anomaly."
    )
    vocabulary = [
        "record",
        "context",
        "latency",
        "memory",
        "throughput",
        "attention",
        "window",
        "retrieval",
        "summary",
        "signal",
        "needle",
        "sequence",
        "benchmark",
        "segment",
        "token",
        "evaluation",
    ]
    words = []
    record_id = 0
    while len(words) < target_words:
        record_id += 1
        anomaly = "ANOMALY_ALPHA" if record_id % 257 == 0 else "normal"
        words.extend(
            [
                f"ID{record_id:06d}",
                anomaly,
                *vocabulary,
            ]
        )
    body = " ".join(words[:target_words])
    return f"{header}{body}{footer}"


def make_file_prompt(path, target_words):
    text = path.read_text(encoding="utf-8")
    words = text.split()
    if not words:
        raise ValueError(f"input file is empty: {path}")
    repeated_words = []
    while len(repeated_words) < target_words:
        repeated_words.extend(words)
    body = " ".join(repeated_words[:target_words])
    return (
        "Read this long context and answer the final question.\n\n"
        f"{body}\n\n"
        "Answer the final question in five concise bullets. Summarize the key points."
    )


def ns_to_seconds(value):
    if value is None:
        return None
    return float(value) / 1_000_000_000


def tokens_per_second(count, duration_ns):
    duration_s = ns_to_seconds(duration_ns)
    if count is None or duration_s is None or duration_s <= 0:
        return None
    return count / duration_s


def sample_ollama_rss_mb():
    pids = find_ollama_pids()
    rss_kb_total = 0
    for pid in pids:
        try:
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in output.splitlines():
            line = line.strip()
            if line:
                try:
                    rss_kb_total += int(line)
                except ValueError:
                    continue
    if rss_kb_total == 0:
        return None
    return rss_kb_total / 1024


def find_ollama_pids():
    if shutil.which("pgrep"):
        try:
            output = subprocess.check_output(
                ["pgrep", "-f", "ollama"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return sorted({int(line) for line in output.splitlines() if line.strip().isdigit()})
        except (OSError, subprocess.CalledProcessError):
            pass

    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []

    pids = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and "ollama" in parts[1].lower():
            try:
                pids.append(int(parts[0]))
            except ValueError:
                continue
    return sorted(set(pids))


def sample_nvidia_gpu_used_mb():
    if not shutil.which("nvidia-smi"):
        return None

    ollama_pids = set(find_ollama_pids())
    if ollama_pids:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            process_memory = parse_nvidia_compute_app_memory_mb(output, ollama_pids)
            if process_memory is not None:
                return process_memory
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    values = [value for value in (parse_nvidia_memory_mb(line) for line in output.splitlines()) if value is not None]
    if not values:
        return None
    return sum(values)


def parse_nvidia_memory_mb(value):
    if not value:
        return None
    lowered = value.lower()
    if "n/a" in lowered or "not supported" in lowered:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
    if not match:
        return None
    return float(match.group(1))


def parse_nvidia_compute_app_memory_mb(output, target_pids):
    values = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if target_pids and pid not in target_pids:
            continue
        memory = parse_nvidia_memory_mb(parts[-1])
        if memory is not None:
            values.append(memory)
    if not values:
        return None
    return sum(values)


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
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not connect to Ollama at {url}: {exc}") from exc
    return json.loads(response_body)


def post_streaming_json(url, payload, timeout_s):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    final_response = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                if chunk.get("done"):
                    final_response = chunk
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not connect to Ollama at {url}: {exc}") from exc

    if final_response is None:
        raise RuntimeError("Ollama stream ended without final metrics")
    return final_response


def stop_ollama_model(host, model):
    if shutil.which("ollama"):
        try:
            result = subprocess.run(
                ["ollama", "stop", model],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if result.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass

    try:
        post_json(f"{host.rstrip('/')}/api/generate", {"model": model, "keep_alive": 0}, 10)
        return True
    except RuntimeError:
        return False


def check_ollama(host, timeout_s):
    url = f"{host.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            response.read(1)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach Ollama at {host}. Start it with `ollama serve` or set OLLAMA_HOST."
        ) from exc


def summarize_memory(samples):
    rss_values = [sample.ollama_rss_mb for sample in samples if sample.ollama_rss_mb is not None]
    gpu_values = [
        sample.nvidia_gpu_used_mb for sample in samples if sample.nvidia_gpu_used_mb is not None
    ]
    return {
        "ollama_rss_peak_mb": max(rss_values) if rss_values else None,
        "ollama_rss_end_mb": rss_values[-1] if rss_values else None,
        "nvidia_gpu_peak_mb": max(gpu_values) if gpu_values else None,
        "nvidia_gpu_end_mb": gpu_values[-1] if gpu_values else None,
    }


def run_segment(
    *,
    host,
    model,
    prompt,
    requested_words,
    num_predict,
    num_ctx,
    timeout_s,
    sample_interval_s,
    keep_alive,
    temperature,
    top_p,
):
    effective_num_ctx = num_ctx if num_ctx is not None else max(2048, requested_words + num_predict + 512)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": effective_num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    sampler = MemorySampler(sample_interval_s)
    sampler.start()
    started_at = time.monotonic()
    try:
        response = post_streaming_json(f"{host.rstrip('/')}/api/generate", payload, timeout_s)
    finally:
        wall_s = time.monotonic() - started_at
        memory = summarize_memory(sampler.stop())

    prompt_eval_count = response.get("prompt_eval_count")
    prompt_eval_duration = response.get("prompt_eval_duration")
    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")

    return {
        "model": model,
        "requested_words": requested_words,
        "num_ctx": effective_num_ctx,
        "num_predict": num_predict,
        "wall_s": wall_s,
        "total_s": ns_to_seconds(response.get("total_duration")),
        "load_s": ns_to_seconds(response.get("load_duration")),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_s": ns_to_seconds(prompt_eval_duration),
        "prompt_tokens_per_s": tokens_per_second(prompt_eval_count, prompt_eval_duration),
        "eval_count": eval_count,
        "eval_s": ns_to_seconds(eval_duration),
        "eval_tokens_per_s": tokens_per_second(eval_count, eval_duration),
        **memory,
    }


def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_segment_size(words):
    if words >= 1000 and words % 1000 == 0:
        return f"{words // 1000}k"
    return str(words)


def print_summary(rows):
    for row in rows:
        print(f"Segment {format_segment_size(row.get('requested_words'))} words")
        print(
            "  "
            f"Tokens: prompt {format_value(row.get('prompt_eval_count'))}, "
            f"generated {format_value(row.get('eval_count'))}"
        )
        print(
            "  "
            f"Speed: prefill {format_value(row.get('prompt_tokens_per_s'))} tok/s, "
            f"decode {format_value(row.get('eval_tokens_per_s'))} tok/s"
        )
        print(
            "  "
            f"Time: wall {format_value(row.get('wall_s'))} s, "
            f"prefill {format_value(row.get('prompt_eval_s'))} s, "
            f"decode {format_value(row.get('eval_s'))} s"
        )
        print(
            "  "
            f"Memory: RSS {format_value(row.get('ollama_rss_peak_mb'))} MB, "
            f"GPU {format_value(row.get('nvidia_gpu_peak_mb'))} MB"
        )
        print()


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama model speed and memory over long-context segments."
    )
    parser.add_argument("--model", required=True, help="Ollama model name, for example qwen2.5:7b.")
    parser.add_argument(
        "--segments",
        default=DEFAULT_SEGMENTS,
        help=f"Comma-separated approximate prompt word counts. Default: {DEFAULT_SEGMENTS}.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Optional text file to repeat and truncate instead of generating synthetic text.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host. Default: {DEFAULT_HOST}.")
    parser.add_argument(
        "--num-ctx",
        type=int,
        help="Fixed Ollama num_ctx for every run. Default auto-sizes per segment.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=8000,
        help="Generated tokens per segment. Default: 8000.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Memory sampling interval in seconds. Default: 1.0.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200,
        help="Per-segment HTTP timeout in seconds. Default: 7200.",
    )
    parser.add_argument("--keep-alive", default="10m", help="Ollama keep_alive value. Default: 10m.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Default: 0.0.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Default: 1.0.")
    parser.add_argument("--csv", type=Path, help="Optional output CSV path.")
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the initial /api/tags connectivity check.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        segments = parse_segments(args.segments)
    except ValueError as exc:
        parser.error(str(exc))

    if args.num_ctx is not None and args.num_ctx <= 0:
        parser.error("--num-ctx must be positive")
    if args.num_predict <= 0:
        parser.error("--num-predict must be positive")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")

    try:
        if not args.skip_health_check:
            check_ollama(args.host, min(args.timeout, 10))

        print(f"Platform: {platform.platform()}")
        print(f"Model: {args.model}")
        print(f"Ollama host: {args.host}")
        print("Memory: RSS is Ollama process RAM; GPU is nvidia-smi process memory when available.")
        print("Prompt size: requested_words is an approximate word target; Ollama token counts are reported.")
        print()

        rows = []
        for segment in segments:
            print(f"Running segment {segment} requested words...", flush=True)
            if args.input_file:
                prompt = make_file_prompt(args.input_file, segment)
            else:
                prompt = make_synthetic_prompt(segment)
            row = run_segment(
                host=args.host,
                model=args.model,
                prompt=prompt,
                requested_words=segment,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
                timeout_s=args.timeout,
                sample_interval_s=args.sample_interval,
                keep_alive=args.keep_alive,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            rows.append(row)
            print(
                "  "
                f"wall={format_value(row['wall_s'])}s, "
                f"prompt_tps={format_value(row['prompt_tokens_per_s'])}, "
                f"rss_peak_mb={format_value(row['ollama_rss_peak_mb'])}, "
                f"gpu_peak_mb={format_value(row['nvidia_gpu_peak_mb'])}",
                flush=True,
            )

        print()
        print_summary(rows)

        if args.csv:
            write_csv(args.csv, rows)
            print(f"\nWrote CSV: {args.csv}")

        print("\nAsking Ollama to unload the model...", file=sys.stderr)
        if stop_ollama_model(args.host, args.model):
            print("Ollama stop request sent.", file=sys.stderr)
        else:
            print("Could not confirm Ollama stopped the model.", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted. Asking Ollama to stop the active model...", file=sys.stderr)
        if stop_ollama_model(args.host, args.model):
            print("Ollama stop request sent.", file=sys.stderr)
        else:
            print("Could not confirm Ollama stopped the model.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
