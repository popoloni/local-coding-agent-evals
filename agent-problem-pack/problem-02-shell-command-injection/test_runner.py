import inspect
import sys

import pytest
import runner


def test_runner_does_not_use_shell_true():
    source = inspect.getsource(runner.run_user_command)

    assert "shell=True" not in source


def test_runner_accepts_explicit_argument_list():
    output = runner.run_user_command([sys.executable, "-c", "print('ok')"])

    assert output.strip() == "ok"


def test_runner_rejects_string_commands_from_task_data():
    with pytest.raises((TypeError, ValueError)):
        runner.run_user_command("echo unsafe")
