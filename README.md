# Local coding agent evals

This repository contains the small scripts and task packs I used for my article.

The goal is not to provide a comprehensive benchmark. These are practical checks I used while comparing local models and coding-agent harnesses such as Qwen Code, Codex, and Claude Code with Ollama-hosted models.

This fork also includes llmstack-adapted benchmark entry points for environments that expose local models through the llmstack OpenAI-compatible API and model registry.

Important: the llmstack adaptation is on-demand only. Nothing in this repo auto-runs evaluations or tests unless you explicitly launch one of the benchmark scripts or use the agent problem pack manually.


&nbsp;
## Contents

- `speed-memory-benchmark/`: measures Ollama prefill speed, decode speed, wall time, and memory use across longer prompts. This test the LLM + inference engine.
- `hard-tool-reasoning-benchmark/`: asks an Ollama model to return one tool decision for five harder reasoning tasks. It does not execute tools. This tests the LLM base capability.
- `agent-problem-pack/`: contains five small coding tasks for testing an actual coding-agent harness in isolated workspaces based on the reasoning tasks above. This tests the LLM + harness combo.

llmstack additions:

- `speed-memory-benchmark/llmstack_speed_memory_bench.py`: benchmarks the active llmstack model or a chosen llmstack model key via `/v1/chat/completions`, then reads prefill/decode metrics from llmstack's timing CSV.
- `hard-tool-reasoning-benchmark/llmstack_hard_reasoning_bench.py`: runs the hard reasoning tasks through the active llmstack model or a chosen llmstack model key.
- `run_llmstack_eval_matrix.py`: iterates over configured llmstack model/backend pairs, starts one backend plus Headroom, runs the selected evals, stops services, and then moves to the next pair.


&nbsp;
## Setup

Clone the repository:

```bash
git clone https://github.com/rasbt/local-coding-agent-evals.git
cd local-coding-agent-evals
```

Install [Ollama](https://ollama.com/) and pull a model:

```bash
ollama pull qwen3.6:35b
```

or, if you are on a Mac:

```bash
ollama pull qwen3.6:35b-mlx
```

The examples below use `uv` for Python script execution where useful.

If you use the llmstack adaptation, Ollama is not required. The llmstack path assumes your local stack is already configured and reachable through `llmstack_config.json` plus the local `/v1/chat/completions` API.

&nbsp;


![](https://sebastianraschka.com/images/github/local-coding-agent-evals/1.png)



&nbsp;
## Speed and memory benchmark

```bash
uv run speed-memory-benchmark/ollama_speed_memory_bench.py \
  --model qwen3.6:35b \
  --csv ollama_speed_memory_results.csv
```

On macOS with an MLX model:

```bash
uv run speed-memory-benchmark/ollama_speed_memory_bench.py \
  --model qwen3.6:35b-mlx \
  --csv ollama_speed_memory_results.csv
```

See [speed-memory-benchmark/README.md](speed-memory-benchmark/README.md) for details.

llmstack version:

```bash
env/bin/python local-coding-agent-evals/speed-memory-benchmark/llmstack_speed_memory_bench.py \
  --model-key dflash-ornith35b-moe \
  --activate-model \
  --csv llmstack_speed_memory_results.csv
```

&nbsp;
## Hard tool-reasoning benchmark

```bash
python3 hard-tool-reasoning-benchmark/ollama_hard_reasoning_bench.py \
  --model qwen3.6:35b \
  --csv hard_reasoning_results.csv
```
See [hard-tool-reasoning-benchmark/README.md](hard-tool-reasoning-benchmark/README.md) for the task format and scoring.

llmstack version:

```bash
env/bin/python local-coding-agent-evals/hard-tool-reasoning-benchmark/llmstack_hard_reasoning_bench.py \
  --model-key dflash-qwen35b-moe \
  --activate-model \
  --csv llmstack_hard_reasoning_results.csv
```

llmstack matrix run across configured backend/model pairs:

```bash
env/bin/python local-coding-agent-evals/run_llmstack_eval_matrix.py
```

Workspace wrapper script:

```bash
bin/launch_llmstack_evals.bash
```

The wrapper activates `env`, creates a timestamped output directory under `local-coding-agent-evals/results`, and then launches the matrix runner. Pass filters such as `--include-model`, `--backend`, `--skip-speed`, or `--skip-reasoning` through to the underlying Python runner.

The matrix runner is on-demand only. It starts one backend/model pair at a time, starts Headroom, runs the selected evals, stops the services, and then continues to the next configured pair.

Comparison graph script (reuse author script if present, else fallback):

```bash
bin/plot_llmstack_comparison.bash
```

This command searches for a native author plotting script first. If none is found, it generates `local-coding-agent-evals/results/llmstack_comparison.png` plus a markdown summary from the latest llmstack speed/reasoning CSV files for each model.

&nbsp;
## Agent problem pack

The problem pack is for testing a real coding-agent harness, not just a raw model response.

See [agent-problem-pack/README.md](agent-problem-pack/README.md) for the full workflow.

&nbsp;

| Harness   | qwen3.6:35b | north-mini-code-1.0:q4_K_M | gemma4:e2b | nemotron-3-nano:latest |
| --------- | ----------: | -------------------------: | ----------: | ---------------------: |
| claude    |         5/5 |                        5/5 |         3/5 |                    4/5 |
| codex     |         5/5 |                        5/5 |         0/5 |                    5/5 |
| qwen-code |         4/5 |                        4/5 |         1/5 |                    4/5 |

&nbsp;

![](https://sebastianraschka.com/images/github/local-coding-agent-evals/2.png)


&nbsp;
## Notes

Run agent benchmarks (`agent-problem-pack`) in an isolated workspace, separate user account, VM, or separate machine when possible. Coding agents can read files and run commands, which is also what makes them useful.

The result files in this repository are examples from my runs. They are meant as a starting point for comparison, not as final model rankings.
