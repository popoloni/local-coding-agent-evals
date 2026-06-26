# Local coding agent evals

This repository contains the small scripts and task packs I used for my article.

The goal is not to provide a comprehensive benchmark. These are practical checks I used while comparing local models and coding-agent harnesses such as Qwen Code, Codex, and Claude Code with Ollama-hosted models.


&nbsp;
## Contents

- `speed-memory-benchmark/`: measures Ollama prefill speed, decode speed, wall time, and memory use across longer prompts. This test the LLM + inference engine.
- `hard-tool-reasoning-benchmark/`: asks an Ollama model to return one tool decision for five harder reasoning tasks. It does not execute tools. This tests the LLM base capability.
- `agent-problem-pack/`: contains five small coding tasks for testing an actual coding-agent harness in isolated workspaces based on the reasoning tasks above. This tests the LLM + harness combo.


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

&nbsp;
## Hard tool-reasoning benchmark

```bash
python3 hard-tool-reasoning-benchmark/ollama_hard_reasoning_bench.py \
  --model qwen3.6:35b \
  --csv hard_reasoning_results.csv
```

See [hard-tool-reasoning-benchmark/README.md](hard-tool-reasoning-benchmark/README.md) for the task format and scoring.

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
