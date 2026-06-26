# Headless Evaluator

Use this runbook when evaluating `agent-problem-pack` with a headless coding agent such as Codex, Claude Code, Cline, or Qwen Code.

## Inputs

Ask for or infer these values:

- target agent CLI, for example `codex`, `ollama launch claude --model <model> -- -p`, `cline`, or `qwen`
- model name, if the CLI needs one
- run name, for example `qwen36-run-1`

If the requested CLI flags are uncertain, run the CLI help command before starting the suite.

## Rules

- Work from the `agent-problem-pack` folder.
- Do not use `agent-problem-pack-runner`.
- Do not edit the original problem folders.
- Prepare a fresh isolated workspace for each problem.
- Run the target agent from the prepared workspace.
- Record token usage for every problem in `artifacts/usage.json`.
- Capture every run with `uv run python scripts/pack_tools.py capture`.
- Report artifact paths instead of pasting large logs.

## Headless Command Shapes

Common command shapes:

- Codex: `codex exec --json "<prompt>"`
- Claude Code: `ollama launch claude --model <model> -- --output-format json -p "<prompt>"`
- Cline: `cline "<prompt>"`, `cline --json "<prompt>"`, or `cline --yolo "<prompt>"`
- Qwen Code: `qwen --model <model> --output-format json -p "<prompt>"`

Prefer the installed CLI's help output over this table if they disagree. For Claude Code auto mode, `--model <model>` must be passed before `--`, and Claude's `-p "<prompt>"` must be passed after `--`. In isolated benchmark workspaces, `--permission-mode bypassPermissions` can be passed after `--` when unattended file edits are required.

## Token Usage

Every run must leave a normalized usage file at:

```text
runs/<problem-id>/<run-name>/artifacts/usage.json
```

Use this schema:

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

Extraction rules:

- Codex: read JSONL output and use the `usage` object from the final `turn.completed` event.
- Qwen Code: read JSON output and use the final `result.usage` object. Preserve detailed `stats.models` under `raw_usage` when present.
- Claude Code through Ollama: read JSON output and use `usage`. Preserve `modelUsage` under `raw_usage` when present.
- Cline: use exact JSON usage if available. If not, keep `exact: false` and explain the limitation in `notes`.
- If non-JSON warnings appear before JSON, ignore non-JSON lines and parse only valid JSON objects or arrays.

## Workflow

1. List the problems:

   ```bash
   uv run python scripts/pack_tools.py list
   ```

2. For each problem, prepare a workspace:

   ```bash
   uv run python scripts/pack_tools.py prepare <problem-id> <run-name>
   ```

3. Enter the workspace printed by `prepare`.

4. Read the generated prompt:

   ```bash
   ../artifacts/task-prompt.txt
   ```

5. Run the target agent in headless mode with that prompt, using JSON output when available. For Claude Code through Ollama, use:

   ```bash
   ollama launch claude --model <model> -- \
     --output-format json \
     --permission-mode bypassPermissions \
     -p "<prompt>"
   ```

6. Extract token usage from the agent output and overwrite `artifacts/usage.json`.

7. After the agent exits, return to the `agent-problem-pack` folder and capture the run:

   ```bash
   uv run python scripts/pack_tools.py capture runs/<problem-id>/<run-name>
   ```

8. Read these artifacts for each problem:

   ```text
   runs/<problem-id>/<run-name>/workspace/AGENT_FINAL_ANSWER.md
   runs/<problem-id>/<run-name>/artifacts/diff.patch
   runs/<problem-id>/<run-name>/artifacts/git-status.txt
   runs/<problem-id>/<run-name>/artifacts/verification.txt
   runs/<problem-id>/<run-name>/artifacts/usage.json
   runs/<problem-id>/<run-name>/artifacts/evaluate-with-codex.md
   ```

9. Summarize:

   - pass/fail per problem
   - what changed
   - whether tests passed
   - token usage per problem and totals across the suite
   - likely cause of each failure
   - any suspicious solution that passes tests
   - artifact paths used as evidence
