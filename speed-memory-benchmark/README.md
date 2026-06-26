# Ollama Speed and Memory Benchmark

This script measures Ollama prompt prefill speed, decode speed, wall time, and memory across long-context prompt sizes. It uses deterministic synthetic prompts by default and can optionally use a text file as input.


&nbsp;
## Run

Start Ollama and make sure the model is available:

```bash
ollama pull qwen3.6:35b
```

From the project root:

```bash
uv run speed-memory-benchmark/ollama_speed_memory_bench.py --model qwen3.6:35b --csv ollama_speed_memory_results.csv
```

On macOS with an MLX model:

```bash
uv run speed-memory-benchmark/ollama_speed_memory_bench.py --model qwen3.6:35b-mlx --csv ollama_speed_memory_results.csv
```

Defaults:

- prompt sizes: `1k, 5k, 10k, 50k` words
- max generation: `8000` tokens
- model: no default, `--model` is required


&nbsp;
## Notes

The benchmark uses Ollama's prompt evaluation metrics for prefill speed and output evaluation metrics for decode speed. On NVIDIA systems it samples GPU memory through `nvidia-smi`. On macOS, GPU memory is not reported separately, so Activity Monitor can be more informative for MLX-backed models.
