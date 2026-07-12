#!/usr/bin/env python3
"""Shared helpers for adapting local-coding-agent-evals to llmstack."""

from __future__ import annotations

import csv
import http.client
import json
import os
import socket
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def discover_llmstack_root(start: Path | None = None) -> Path:
    probe = (start or Path(__file__).resolve()).resolve()
    for path in [probe, *probe.parents]:
        cfg = path / "llmstack_config.json"
        if cfg.exists():
            return path
    raise FileNotFoundError("could not locate llmstack_config.json in current parents")


def load_llmstack_config(root: Path) -> dict:
    cfg_path = root / "llmstack_config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def resolve_model_target(cfg: dict, model_key: str | None = None) -> tuple[str, str]:
    registry = cfg.get("models") or {}
    active_key = model_key or cfg.get("active_model")
    if not active_key:
        raise ValueError("no active model found in llmstack config and no --model-key provided")
    model_cfg = registry.get(active_key)
    if not isinstance(model_cfg, dict):
        raise ValueError(f"unknown llmstack model key: {active_key}")
    target = model_cfg.get("target")
    if not target:
        raise ValueError(f"llmstack model {active_key!r} has no target configured")
    return active_key, str(target)


def inference_base_urls(cfg: dict) -> tuple[str, str]:
    host = cfg.get("local_host", "127.0.0.1")
    port = int(cfg.get("inference_port", 8787))
    base = f"http://{host}:{port}"
    return f"{base}/v1/chat/completions", f"{base}/v1/models"


def headroom_base_urls(cfg: dict) -> tuple[str, str]:
    host = cfg.get("local_host", "127.0.0.1")
    port = int(cfg.get("headroom_port", 8789))
    base = f"http://{host}:{port}"
    return f"{base}/v1/chat/completions", f"{base}/health"


def timings_csv_path(root: Path, cfg: dict) -> Path:
    csv_path = Path(cfg.get("timings_csv", "./logs/dflash_timings.csv"))
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    return csv_path


def post_json(url: str, payload: dict, timeout_s: float) -> dict:
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
        raise RuntimeError(f"llmstack HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not connect to llmstack at {url}: {exc}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"request to {url} timed out after {timeout_s}s") from exc
    except http.client.RemoteDisconnected as exc:
        raise RuntimeError(
            "llmstack closed the connection without a response; this usually means the active backend crashed or rejected the request payload"
        ) from exc


def _is_transient_request_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "http 502",
        "proxy_error",
        "timed out",
        "remote end closed connection",
        "closed the connection without a response",
        "connection reset",
        "temporary failure",
    )
    return any(marker in text for marker in transient_markers)


def _post_json_retry(url: str, payload: dict, timeout_s: float, *, retries: int, retry_backoff_s: float, request_label: str) -> dict:
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return post_json(url, payload, timeout_s)
        except RuntimeError as exc:
            if attempt >= attempts or not _is_transient_request_error(exc):
                raise
            delay = retry_backoff_s * attempt
            print(
                f"warning: {request_label} failed on {url} (attempt {attempt}/{attempts}): {exc}; retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(f"unexpected retry loop exhaustion for {request_label}")


def post_json_with_fallback(primary_url: str, payload: dict, timeout_s: float, *, retries: int = 2, retry_backoff_s: float = 2.0, fallback_url: str | None = None, request_label: str = "request") -> dict:
    try:
        return _post_json_retry(
            primary_url,
            payload,
            timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            request_label=request_label,
        )
    except RuntimeError as exc:
        if not fallback_url or fallback_url == primary_url or not _is_transient_request_error(exc):
            raise
        print(
            f"warning: {request_label} switching from {primary_url} to fallback {fallback_url} after transient failures",
            file=sys.stderr,
        )
        return _post_json_retry(
            fallback_url,
            payload,
            timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            request_label=f"{request_label} (fallback)",
        )


def check_llmstack(models_url: str, timeout_s: float) -> list[str]:
    try:
        with urllib.request.urlopen(models_url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach llmstack at {models_url}: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected /v1/models payload from {models_url}: {payload!r}")
    return [str(entry.get("id")) for entry in data if isinstance(entry, dict) and entry.get("id")]


def activate_llmstack_model(root: Path, model_key: str, python_bin: str | None = None) -> None:
    py = python_bin or str((root / "env" / "bin" / "python").resolve())
    subprocess.run([py, "-m", "llmstack.cli", "model", "use", model_key, "--restart"], cwd=root, check=True)


def model_entries(cfg: dict) -> list[tuple[str, dict]]:
    registry = cfg.get("models") or {}
    return [(key, value) for key, value in registry.items() if isinstance(value, dict)]


def start_script_for_type(root: Path, backend_type: str) -> Path:
    mapping = {
        "dflash": root / "bin" / "start_dflash_server.bash",
        "mlx": root / "bin" / "start_mlx_server.bash",
        "turboquant": root / "bin" / "start_turboquant_server.bash",
    }
    script = mapping.get(str(backend_type or "").lower())
    if script is None or not script.exists():
        raise ValueError(f"no start script for backend type: {backend_type}")
    return script


def stop_script_for_type(root: Path, backend_type: str) -> Path:
    mapping = {
        "dflash": root / "bin" / "stop_dflash_server.bash",
        "mlx": root / "bin" / "stop_mlx_server.bash",
        "turboquant": root / "bin" / "stop_turboquant_server.bash",
    }
    script = mapping.get(str(backend_type or "").lower())
    if script is None or not script.exists():
        raise ValueError(f"no stop script for backend type: {backend_type}")
    return script


def wait_for_served_model(models_url: str, expected_target: str, timeout_s: float = 120.0) -> list[str]:
    deadline = time.monotonic() + timeout_s
    last_ids: list[str] = []
    while time.monotonic() < deadline:
        try:
            last_ids = check_llmstack(models_url, min(timeout_s, 5.0))
        except RuntimeError:
            time.sleep(1.0)
            continue
        if expected_target in last_ids:
            return last_ids
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for llmstack to serve {expected_target}; last ids={last_ids}")


def wait_for_headroom(health_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if response.getcode() == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for headroom health at {health_url}")


def run_shell_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True)


def stop_all_services(root: Path) -> None:
    scripts = [
        root / "bin" / "stop_headroom_server.bash",
        root / "bin" / "stop_dflash_server.bash",
        root / "bin" / "stop_mlx_server.bash",
        root / "bin" / "stop_turboquant_server.bash",
    ]
    for script in scripts:
        if script.exists():
            subprocess.run(["bash", str(script)], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def wait_for_new_timing_row(csv_path: Path, previous_count: int, expected_target: str, timeout_s: float = 6.0) -> dict | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not csv_path.exists():
            time.sleep(0.25)
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) <= previous_count:
            time.sleep(0.25)
            continue
        new_rows = rows[previous_count:]
        matching = [row for row in new_rows if str(row.get("served_target") or "") == expected_target]
        if matching:
            return matching[-1]
        time.sleep(0.25)
    return None


def find_process_for_port(port: int) -> str | None:
    try:
        pid_lines = subprocess.check_output(["lsof", "-ti", f"tcp:{port}"], text=True).strip().splitlines()
    except Exception:
        return None
    if not pid_lines:
        return None
    return pid_lines[0].strip() or None


def sample_process_rss_mb(pid: str | None) -> float | None:
    if not pid:
        return None
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    values = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            continue
    if not values:
        return None
    return sum(values) / 1024.0


def sample_nvidia_gpu_used_mb() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    values = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    if not values:
        return None
    return sum(values)


class MemorySampler:
    def __init__(self, pid: str | None, interval_s: float):
        self.pid = pid
        self.interval_s = interval_s
        self.samples: list[tuple[float | None, float | None]] = []
        self._stop = False

    def run_during(self, func):
        import threading

        def sampler_loop():
            while not self._stop:
                self.samples.append((sample_process_rss_mb(self.pid), sample_nvidia_gpu_used_mb()))
                time.sleep(self.interval_s)

        thread = threading.Thread(target=sampler_loop, daemon=True)
        thread.start()
        try:
            return func()
        finally:
            self._stop = True
            thread.join(timeout=max(1.0, self.interval_s * 2.0))
            if not self.samples:
                self.samples.append((sample_process_rss_mb(self.pid), sample_nvidia_gpu_used_mb()))

    def summary(self) -> dict:
        rss_values = [value for value, _ in self.samples if value is not None]
        gpu_values = [value for _, value in self.samples if value is not None]
        return {
            "server_rss_peak_mb": max(rss_values) if rss_values else None,
            "server_rss_end_mb": rss_values[-1] if rss_values else None,
            "nvidia_gpu_peak_mb": max(gpu_values) if gpu_values else None,
            "nvidia_gpu_end_mb": gpu_values[-1] if gpu_values else None,
        }


def response_text(response: dict) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            return str(message.get("content") or "")
    return str(response.get("response") or "")


def build_chat_payload(model_target: str, prompt: str, max_tokens: int, temperature: float, top_p: float, *, response_format: dict | None = None) -> dict:
    payload = {
        "model": model_target,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # Keep the default payload aligned with llmstack's existing executor client.
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


def default_python_bin(root: Path) -> str:
    candidate = root / "env" / "bin" / "python"
    if candidate.exists():
        # Preserve the venv launcher path; resolving symlinks can bypass venv site-packages.
        return str(candidate)
    return sys.executable