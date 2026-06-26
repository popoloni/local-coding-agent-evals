import importlib.util
import io
import json
import pathlib
import sys
import unittest
from contextlib import redirect_stderr
from unittest import mock


SCRIPT_PATH = pathlib.Path(__file__).with_name("ollama_speed_memory_bench.py")


def load_module():
    spec = importlib.util.spec_from_file_location("ollama_speed_memory_bench", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OllamaLongContextBenchTests(unittest.TestCase):
    def test_parse_segments_accepts_comma_separated_positive_ints(self):
        bench = load_module()

        self.assertEqual(bench.parse_segments("1000, 5000,20000"), [1000, 5000, 20000])

    def test_parse_segments_rejects_non_positive_values(self):
        bench = load_module()

        with self.assertRaises(ValueError):
            bench.parse_segments("1000,0")

    def test_make_synthetic_prompt_targets_requested_word_count(self):
        bench = load_module()

        prompt = bench.make_synthetic_prompt(25)

        self.assertGreaterEqual(len(prompt.split()), 25)
        self.assertIn("Answer the final question", prompt)

    def test_ns_to_seconds_handles_missing_and_zero_values(self):
        bench = load_module()

        self.assertIsNone(bench.ns_to_seconds(None))
        self.assertEqual(bench.ns_to_seconds(0), 0.0)
        self.assertEqual(bench.ns_to_seconds(2_000_000_000), 2.0)

    def test_parse_nvidia_memory_mb_handles_units_and_unsupported_values(self):
        bench = load_module()

        self.assertEqual(bench.parse_nvidia_memory_mb("12345 MiB"), 12345.0)
        self.assertEqual(bench.parse_nvidia_memory_mb("6789"), 6789.0)
        self.assertIsNone(bench.parse_nvidia_memory_mb("N/A"))
        self.assertIsNone(bench.parse_nvidia_memory_mb("Not Supported"))

    def test_parse_nvidia_compute_app_memory_sums_matching_ollama_pids(self):
        bench = load_module()
        output = (
            "111, /usr/local/lib/ollama/runners/cuda_v12/ollama_llama_server, 20480 MiB\n"
            "222, python, 1024 MiB\n"
            "333, ollama_llama_server, 4096 MiB\n"
        )

        self.assertEqual(bench.parse_nvidia_compute_app_memory_mb(output, {111, 333}), 24576.0)

    def test_parser_defaults_to_standard_intervals_and_8k_prediction(self):
        bench = load_module()

        args = bench.build_parser().parse_args(["--model", "example-model"])

        self.assertEqual(args.segments, "1000,5000,10000,50000")
        self.assertEqual(args.num_predict, 8000)

    def test_print_summary_uses_compact_per_segment_blocks(self):
        bench = load_module()
        output = io.StringIO()
        rows = [
            {
                "requested_words": 1000,
                "prompt_eval_count": 1200,
                "wall_s": 12.34,
                "prompt_tokens_per_s": 100.0,
                "eval_tokens_per_s": 50.0,
                "ollama_rss_peak_mb": 1024.0,
                "nvidia_gpu_peak_mb": 8192.0,
            }
        ]

        with mock.patch("sys.stdout", output):
            bench.print_summary(rows)

        text = output.getvalue()
        self.assertIn("Segment 1k words", text)
        self.assertIn("Speed: prefill 100.00 tok/s, decode 50.00 tok/s", text)
        self.assertIn("Memory: RSS 1024.00 MB, GPU 8192.00 MB", text)
        self.assertNotIn("unified/system", text)
        self.assertNotIn("requested_words  num_ctx", text)

    def test_streaming_generate_request_returns_final_metrics(self):
        bench = load_module()
        captured_payloads = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                chunks = [
                    {"response": "partial", "done": False},
                    {"response": "", "done": True, "eval_count": 3},
                ]
                for chunk in chunks:
                    yield (json.dumps(chunk) + "\n").encode("utf-8")

        def fake_urlopen(request, timeout):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        with mock.patch.object(bench.urllib.request, "urlopen", side_effect=fake_urlopen):
            response = bench.post_streaming_json(
                "http://127.0.0.1:11434/api/generate",
                {"model": "example-model", "stream": True},
                10,
            )

        self.assertTrue(captured_payloads[0]["stream"])
        self.assertEqual(response["eval_count"], 3)

    def test_main_stops_ollama_model_on_keyboard_interrupt(self):
        bench = load_module()

        with mock.patch.object(bench, "run_segment", side_effect=KeyboardInterrupt):
            with mock.patch.object(bench, "stop_ollama_model", return_value=True) as stop_model:
                exit_code = bench.main(
                    [
                        "--model",
                        "example-model",
                        "--segments",
                        "1",
                        "--skip-health-check",
                    ]
                )

        self.assertEqual(exit_code, 130)
        stop_model.assert_called_once_with(bench.DEFAULT_HOST, "example-model")

    def test_main_stops_ollama_model_after_successful_run(self):
        bench = load_module()
        row = {
            "requested_words": 1,
            "prompt_eval_count": 2,
            "eval_count": 3,
            "wall_s": 4.0,
            "prompt_eval_s": 1.0,
            "eval_s": 3.0,
            "prompt_tokens_per_s": 2.0,
            "eval_tokens_per_s": 1.0,
            "ollama_rss_peak_mb": 10.0,
            "nvidia_gpu_peak_mb": None,
        }

        with mock.patch.object(bench, "run_segment", return_value=row):
            with mock.patch.object(bench, "stop_ollama_model", return_value=True) as stop_model:
                exit_code = bench.main(
                    [
                        "--model",
                        "example-model",
                        "--segments",
                        "1",
                        "--skip-health-check",
                    ]
                )

        self.assertEqual(exit_code, 0)
        stop_model.assert_called_once_with(bench.DEFAULT_HOST, "example-model")

    def test_main_reports_runtime_errors_without_traceback(self):
        bench = load_module()
        stderr = io.StringIO()

        with mock.patch.object(bench, "run_segment", side_effect=RuntimeError("model missing")):
            with redirect_stderr(stderr):
                exit_code = bench.main(
                    [
                        "--model",
                        "missing-model",
                        "--segments",
                        "1",
                        "--skip-health-check",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("error: model missing", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
