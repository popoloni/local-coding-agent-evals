import importlib.util
import os
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "tool-reasoning-benchmark" / "ollama_tool_reasoning_bench.py"
SCRIPT_DIR = SCRIPT.parent


def load_bench_module():
    spec = importlib.util.spec_from_file_location("bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_tasks_can_be_read_from_project_root():
    previous_cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        bench = load_bench_module()
        assert bench.read_default_tasks().strip() == "[]"
    finally:
        os.chdir(previous_cwd)


def test_default_tasks_can_be_read_from_script_directory():
    previous_cwd = os.getcwd()
    try:
        os.chdir(SCRIPT_DIR)
        bench = load_bench_module()
        assert bench.read_default_tasks().strip() == "[]"
    finally:
        os.chdir(previous_cwd)
