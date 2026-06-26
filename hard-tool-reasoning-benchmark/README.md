# Hard Ollama Tool-Reasoning Benchmark

Five tasks where  the model has to reason over code snippets, debugging traces, safety constraints, and minimal edits.

It does not execute tools. The model returns one JSON object with a tool decision. Some tasks require a `final_answer` with specific technical content, so this tests more than tool-name selection.


&nbsp;
## Run

From the project root:

```bash
python3 hard-tool-reasoning-benchmark/ollama_hard_reasoning_bench.py --model qwen3.6:35b --csv hard_reasoning_results.csv
```

&nbsp;
## What The Five Tasks Cover

The tasks are contained in [hard_reasoning_tasks.jsonl](hard_reasoning_tasks.jsonl) and can be extended.

Currently, they cover:

- diagnosing a failing tokenizer test
- spotting command-injection risk in a helper
- choosing a minimal cross-platform file-path edit
- triaging an import error after a file move
- diagnosing mutable default state leakage

## Scoring

- `1.0`: correct tool and required content/arguments match
- `0.5`: correct tool but required content or arguments are missing
- `0.0`: wrong tool, invalid JSON, or missing `arguments`

The content checks use required substrings, so the model can phrase answers naturally while still being scored consistently.
