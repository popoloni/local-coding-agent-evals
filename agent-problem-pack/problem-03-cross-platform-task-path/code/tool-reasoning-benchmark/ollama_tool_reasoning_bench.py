from pathlib import Path


TASKS = Path("personal_tool_reasoning_tasks.jsonl")


def read_default_tasks():
    return TASKS.read_text(encoding="utf-8")
