# Agent Problem Pack

Small coding tasks for testing agents such as Codex, Claude Code, and qwen-code.

Each run gets its own workspace. Give the agent only the generated task prompt. After it finishes, run `capture`. That writes the diff, test output, final-answer file path, token-usage file path, and evaluation prompt to files.



&nbsp;

## Quick Start (Running Benchmarks Manually)

From the `agent-problem-pack` folder, list the problems:

```bash
cd agent-problem-pack
uv run python scripts/pack_tools.py list
```

Prepare a run:

```bash
uv run python scripts/pack_tools.py prepare problem-01-tokenizer-regression run-1
```

The command prints:

- the run directory
- the task prompt file

Open the run workspace in your coding agent:

```bash
cd runs/problem-01-tokenizer-regression/run-1/workspace
```

Paste the generated prompt from:

```bash
../artifacts/task-prompt.txt
```

The workspace includes its own `pyproject.toml`, so the agent can run:

```bash
uv run pytest
```

After the agent finishes, capture the result from the `agent-problem-pack` folder:

```bash
uv run python scripts/pack_tools.py capture \
  runs/problem-01-tokenizer-regression/run-1
```

Then open this file and use it as the evaluation prompt:

```bash
runs/problem-01-tokenizer-regression/run-1/artifacts/evaluate-with-codex.md
```

The evaluation prompt points to the answer, diff, git status, and pytest output. You do not need to paste those manually.

Each run also has:

```bash
runs/problem-01-tokenizer-regression/run-1/artifacts/usage.json
```

For automated runs, overwrite this file with the harness and model token usage. If exact usage is unavailable, leave `exact` as `false` and explain why in `notes`.

## Problems

| ID | Task | Test command |
| --- | --- | --- |
| `problem-01-tokenizer-regression` | Fix empty-token handling. | `uv run pytest` |
| `problem-02-shell-command-injection` | Remove command-injection risk. | `uv run pytest` |
| `problem-03-cross-platform-task-path` | Make task-file loading independent of the current directory. | `uv run pytest tests` |
| `problem-04-import-error-after-refactor` | Preserve an old import path after a refactor. | `uv run pytest tests` |
| `problem-05-mutable-default-cache` | Fix state leaking through a mutable default argument. | `uv run pytest` |

Use a new run name for each agent or repeat attempt, for example `codex-1`, `claude-1`, or `qwen-1`.

&nbsp;



## Headless Agent Modes 

These agents can be driven from scripts or benchmark runners:

| Agent | Headless command shape | Notes |
| --- | --- | --- |
| Codex | `codex exec "<prompt>"` | Codex calls this non-interactive mode. |
| Cline | `cline "<prompt>"`, `cline --json "<prompt>"`, or `cline --yolo "<prompt>"` | Useful for one-shot runs and CI-style automation. |
| Claude Code | `ollama launch claude --model <model> -- -p "<prompt>"` | The `--model` flag is required for headless launcher use. Claude arguments go after `--`. |
| Qwen Code | `qwen --model <model> -p "<prompt>"` | Verify the installed flags with `qwen --help` if needed. |

For this pack, use isolated run workspaces and an explicit run name for each model or harness attempt.

Record token usage by default:

- Codex: run with `--json` and extract the `usage` object from the final `turn.completed` event.
- Qwen Code: run with `--output-format json` and extract the final `result.usage` object.
- Claude Code through Ollama: run with `--output-format json` after `--` and extract `usage` plus `modelUsage`.
- Cline: use JSON output if it exposes usage. Otherwise write `usage.json` with `exact: false`.

Normalize usage into:

```json
{
  "schema_version": 1,
  "harness": "claude",
  "model": "qwen3.6:35b",
  "source": "cli_json",
  "exact": true,
  "input_tokens": 24327,
  "output_tokens": 16,
  "total_tokens": 24343,
  "cached_input_tokens": 0,
  "reasoning_output_tokens": 0,
  "raw_usage": {},
  "notes": ""
}
```

&nbsp;



## Prompt for Automated Headless Evaluation

Use this prompt in Codex, Claude Code, Cline, or another coding agent when you want it to run the pack through a headless agent CLI and then inspect the results. **Danger: only do this in a sandbox environment or separate machine as it has the ability to read and manipulate files**.



Replace the target CLI, model, and run name as needed. For Claude, use this command shape:

```bash
ollama launch claude --model qwen3.6:35b -- -p "<prompt>"
```

For unattended benchmark runs in isolated workspaces, the agent may use:

```bash
ollama launch claude --model qwen3.6:35b -- \
  --permission-mode bypassPermissions \
  -p "<prompt>"
```

```text
In /home/rasbt/Developer/local-coding-agent-evals/agent-problem-pack

Evaluate this problem pack with a headless coding agent.

Work only from the agent-problem-pack folder. Do not use other folders or files outside this folder.

First read:
- README.md
- skills/headless-evaluator/SKILL.md

Target agent:
- CLI: codex
- model: qwen3.6:35b
- run name: qwen36-run-2
- Claude command shape, if CLI is Claude: ollama launch claude --model qwen3.6:35b -- -p "<prompt>"

Use the pack scripts directly:
- list problems with uv run python scripts/pack_tools.py list
- prepare one isolated workspace per problem
- run the target agent in headless mode from each workspace using that problem's artifacts/task-prompt.txt
- use the agent's JSON output mode when available and write token usage to artifacts/usage.json
- capture each result with uv run python scripts/pack_tools.py capture

For each problem, read:
- workspace/AGENT_FINAL_ANSWER.md
- artifacts/diff.patch
- artifacts/git-status.txt
- artifacts/verification.txt
- artifacts/usage.json
- artifacts/evaluate-with-codex.md

Report:
- pass/fail per problem
- token usage per problem and totals across the suite
- the likely reason for each failure
- whether the target agent edited the right files
- any cases where the tests passed but the solution looks suspicious

Do not paste large logs. Summarize the relevant evidence and cite the artifact paths.
```
