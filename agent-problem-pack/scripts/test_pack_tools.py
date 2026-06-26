import importlib.util
import pathlib
import shutil
import subprocess


SCRIPT_PATH = pathlib.Path(__file__).with_name("pack_tools.py")
PACK_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_pack_tools():
    spec = importlib.util.spec_from_file_location("pack_tools", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_pack_without_runs(destination):
    def ignore(directory, names):
        ignored = {"runs", ".DS_Store", ".pytest_cache", ".venv", "__pycache__"}
        return {name for name in names if name in ignored}

    shutil.copytree(PACK_ROOT, destination, ignore=ignore)


def test_prepare_run_creates_isolated_workspace_and_prompt_file(tmp_path):
    tools = load_pack_tools()
    root = tmp_path / "agent-problem-pack"
    copy_pack_without_runs(root)

    run_dir = tools.prepare_run(root, "problem-01-tokenizer-regression", "codex-test")

    assert (run_dir / "workspace" / "tokenizer.py").exists()
    assert (run_dir / "artifacts" / "task-prompt.txt").exists()
    assert (run_dir / "artifacts" / "usage.json").exists()
    assert (run_dir / "workspace" / "pyproject.toml").exists()
    assert (run_dir / "workspace" / "AGENT_FINAL_ANSWER.md").exists()
    assert (run_dir / "workspace" / ".git").exists()

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=run_dir / "workspace",
        text=True,
        capture_output=True,
        check=True,
    )
    assert status.stdout.strip() == ""


def test_verify_commands_use_uv_run_pytest():
    tools = load_pack_tools()

    for problem in tools.PROBLEMS.values():
        assert problem.verify_command[:3] == ("uv", "run", "pytest")
        for expected in problem.expected_behavior:
            assert "unit" + "test" not in expected
            assert "python" + "3" not in expected


def test_capture_run_writes_diff_verification_and_evaluation_prompt(tmp_path):
    tools = load_pack_tools()
    root = tmp_path / "agent-problem-pack"
    copy_pack_without_runs(root)
    run_dir = tools.prepare_run(root, "problem-01-tokenizer-regression", "claude-test")
    workspace = run_dir / "workspace"

    (workspace / "tokenizer.py").write_text(
        "def tokenize(text):\n"
        "    return [part.lower() for part in text.strip().split(',') if part]\n",
        encoding="utf-8",
    )
    (workspace / "AGENT_FINAL_ANSWER.md").write_text(
        "Filtered empty split parts so blank input returns no tokens.\n",
        encoding="utf-8",
    )

    result = tools.capture_run(run_dir)

    assert result.returncode == 0
    assert "tokenizer.py" in (run_dir / "artifacts" / "diff.patch").read_text(encoding="utf-8")
    assert "2 passed" in (run_dir / "artifacts" / "verification.txt").read_text(encoding="utf-8")
    evaluation_prompt = (run_dir / "artifacts" / "evaluate-with-codex.md").read_text(encoding="utf-8")
    assert "Evaluate the agent run" in evaluation_prompt
    assert "AGENT_FINAL_ANSWER.md" in evaluation_prompt
    assert "diff.patch" in evaluation_prompt
    assert "usage.json" in evaluation_prompt


def test_capture_run_accepts_pack_root_relative_run_dir(tmp_path):
    tools = load_pack_tools()
    root = tmp_path / "agent-problem-pack"
    copy_pack_without_runs(root)
    run_dir = tools.prepare_run(root, "problem-01-tokenizer-regression", "qwen-test")
    workspace = run_dir / "workspace"

    (workspace / "tokenizer.py").write_text(
        "def tokenize(text):\n"
        "    return [part.lower() for part in text.strip().split(',') if part]\n",
        encoding="utf-8",
    )

    relative_run_dir = pathlib.Path("runs/problem-01-tokenizer-regression/qwen-test")

    result = tools.capture_run(relative_run_dir, root)

    assert result.returncode == 0
    assert (run_dir / "artifacts" / "evaluate-with-codex.md").exists()


def test_pyproject_declares_pytest_dependency():
    pyproject = (PACK_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "pytest" in pyproject
