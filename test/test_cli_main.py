import asyncio
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ragnarok.cli.main import main
from ragnarok.data import RAGExecInfo, Result
from ragnarok.generate.llm import PromptMode


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FakeAgent:
    def __init__(self, reasoning=None):
        self._reasoning = reasoning
        self._model = "gpt-4o"
        self._context_size = 8192
        self._prompt_mode = PromptMode.CHATQA
        self._num_few_shot_examples = 0

    def _build_results(self, requests, topk):
        results = []
        for request in requests:
            results.append(
                Result(
                    query=request.query,
                    references=[
                        candidate.docid for candidate in request.candidates[:topk]
                    ],
                    answer=[
                        type(
                            "Sentence",
                            (),
                            {
                                "text": f"Answer for {request.query.text}.",
                                "citations": [0],
                            },
                        )()
                    ],
                    rag_exec_summary=RAGExecInfo(
                        prompt="prompt",
                        response="response",
                        input_token_count=12,
                        output_token_count=4,
                        reasoning=self._reasoning,
                    ),
                )
            )
        return results

    def answer_batch(
        self, requests, topk, shuffle_candidates=False, logging=False, vllm=False
    ):
        return self._build_results(requests, topk)

    async def async_answer(
        self, request, topk, shuffle_candidates=False, logging=False
    ):
        await asyncio.sleep(0)
        return self._build_results([request], topk)[0]

    async def async_answer_batch(
        self,
        requests,
        topk,
        shuffle_candidates=False,
        logging=False,
        vllm=False,
        max_concurrency=8,
    ):
        await asyncio.sleep(0)
        return self._build_results(requests, topk)


class TestRagnarokCLI(unittest.TestCase):
    def test_no_color_env_suppresses_ansi_codes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                output_path,
                [
                    {
                        "run_id": "demo-run",
                        "topic_id": "q1",
                        "topic": "topic one",
                        "references": ["d1"],
                        "response_length": 10,
                        "answer": [{"text": "answer one", "citations": [0]}],
                    }
                ],
            )

            stdout = StringIO()
            with patch.dict(os.environ, {"NO_COLOR": ""}):
                with redirect_stdout(stdout):
                    exit_code = main(
                        ["view", str(output_path), "--color", "always"]
                    )

            self.assertEqual(exit_code, 0)
            self.assertNotIn("\033[", stdout.getvalue())

    def test_version_flag_prints_version_and_exits(self):
        stdout = StringIO()
        with self.assertRaises(SystemExit) as exc_info:
            with redirect_stdout(stdout):
                main(["--version"])
        self.assertEqual(exc_info.exception.code, 0)
        self.assertIn("ragnarok", stdout.getvalue())

    def test_prompt_list_returns_json_catalog(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["prompt", "list", "--output", "json"])

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["command"], "prompt")
        self.assertEqual(output["artifacts"][0]["name"], "prompt-catalog")
        modes = {entry["prompt_mode"] for entry in output["artifacts"][0]["data"]}
        self.assertIn("chatqa", modes)
        self.assertIn("ragnarok_v4", modes)

    def test_prompt_show_returns_text_definition(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["prompt", "show", "--prompt-mode", "chatqa"])

        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Ragnarok Prompt Mode", rendered)
        self.assertIn("prompt_mode: chatqa", rendered)
        self.assertIn("[instruction]", rendered)
        self.assertIn("[chatqa_system_message]", rendered)

    def test_prompt_show_returns_json_definition(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                ["prompt", "show", "--prompt-mode", "ragnarok_v4", "--output", "json"]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["command"], "prompt")
        view = output["artifacts"][0]["data"]
        self.assertEqual(view["prompt_mode"], "ragnarok_v4")
        self.assertIn(
            "Ensure each sentence has at least one citation.", view["instruction"]
        )

    def test_prompt_render_returns_chat_messages_for_gpt_models(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "prompt",
                    "render",
                    "--prompt-mode",
                    "chatqa",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps(
                        {"query": "what is python", "candidates": ["Python is useful."]}
                    ),
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        view = output["artifacts"][0]["data"]
        self.assertEqual(view["prompt"]["format"], "chat_messages")
        self.assertEqual(view["prompt"]["messages"][0]["role"], "system")
        self.assertIn("Query: what is python", view["prompt"]["user_message"])

    def test_prompt_render_returns_single_string_for_chatqa_models(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "prompt",
                    "render",
                    "--prompt-mode",
                    "chatqa",
                    "--model",
                    "chatqa",
                    "--input-json",
                    json.dumps(
                        {"query": "what is python", "candidates": ["Python is useful."]}
                    ),
                ]
            )

        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("format: single_string", rendered)
        self.assertIn("[prompt]", rendered)
        self.assertIn("Context:", rendered)

    def test_prompt_render_rejects_unsupported_prompt_mode(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "prompt",
                    "render",
                    "--prompt-mode",
                    "cohere",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps(
                        {"query": "what is python", "candidates": ["Python is useful."]}
                    ),
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 5)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["errors"][0]["code"], "unsupported_prompt_mode")

    def test_generate_direct_via_input_json(self):
        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(),
            ),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps(
                        {"query": "what is python", "candidates": ["Python is useful."]}
                    ),
                    "--prompt-mode",
                    "chatqa",
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["command"], "generate")
        self.assertEqual(output["status"], "success")
        self.assertEqual(output["artifacts"][0]["kind"], "data")
        self.assertEqual(output["artifacts"][0]["name"], "generation-results")
        self.assertEqual(output["artifacts"][0]["data"][0]["topic"], "what is python")

    def test_generate_direct_via_stdin(self):
        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch(
                "sys.stdin.read",
                return_value=json.dumps({"query": "q", "candidates": ["passage"]}),
            ),
            patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(),
            ),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--stdin",
                    "--prompt-mode",
                    "chatqa",
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["artifacts"][0]["data"][0]["topic"], "q")

    def test_generate_direct_async_via_input_json(self):
        class AsyncOnlyAgent(FakeAgent):
            def answer_batch(
                self,
                requests,
                topk,
                shuffle_candidates=False,
                logging=False,
                vllm=False,
            ):
                raise AssertionError("sync answer_batch should not run in async mode")

        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=AsyncOnlyAgent(),
            ),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps(
                        {"query": "what is python", "candidates": ["Python is useful."]}
                    ),
                    "--prompt-mode",
                    "chatqa",
                    "--execution-mode",
                    "async",
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["resolved"]["execution_mode"], "async")
        self.assertEqual(output["artifacts"][0]["data"][0]["topic"], "what is python")

    def test_generate_direct_text_output_is_not_empty(self):
        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(),
            ),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps(
                        {"query": "what is python", "candidates": ["Python is useful."]}
                    ),
                    "--prompt-mode",
                    "chatqa",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            "query: what is python\n"
            "answer: Answer for what is python. [1]\n"
            "references: [d0]\n",
        )

    def test_generate_direct_json_output_can_include_redacted_trace(self):
        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(reasoning="Used the only candidate as support."),
            ),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps(
                        {"query": "what is python", "candidates": ["Python is useful."]}
                    ),
                    "--prompt-mode",
                    "chatqa",
                    "--include-trace",
                    "--include-reasoning",
                    "--redact-prompts",
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        record = output["artifacts"][0]["data"][0]
        self.assertEqual(
            record["reasoning_traces"], ["Used the only candidate as support."]
        )
        self.assertIsNone(record["trace"]["prompt"])
        self.assertEqual(record["trace"]["response"], "response")
        self.assertEqual(record["trace"]["input_token_count"], 12)

    def test_generate_direct_model_alias_sets_resolved_model(self):
        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(),
            ),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps({"query": "q", "candidates": ["p"]}),
                    "--prompt-mode",
                    "chatqa",
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["resolved"]["model"], "gpt-4o")

    def test_generate_forwards_use_openrouter_to_openai_compatible_args(self):
        stdout = StringIO()
        fake_agent = FakeAgent()
        with (
            redirect_stdout(stdout),
            patch(
                "ragnarok.generate.api_keys.get_openai_compatible_args",
                return_value={
                    "keys": "router-key",
                    "api_base": "https://openrouter.ai/api/v1",
                },
            ) as patched_args,
            patch("ragnarok.generate.gpt.SafeOpenai", return_value=fake_agent),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--use-openrouter",
                    "--input-json",
                    json.dumps({"query": "q", "candidates": ["p"]}),
                    "--prompt-mode",
                    "chatqa",
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        patched_args.assert_called_once_with("gpt-4o", False, True)

    def test_generate_direct_json_output_includes_reasoning_traces(self):
        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(reasoning="Used the only candidate as support."),
            ),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps(
                        {"query": "what is python", "candidates": ["Python is useful."]}
                    ),
                    "--prompt-mode",
                    "chatqa",
                    "--include-reasoning",
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(
            output["artifacts"][0]["data"][0]["reasoning_traces"],
            ["Used the only candidate as support."],
        )

    def test_generate_direct_text_output_prints_reasoning_traces(self):
        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(reasoning="Used the only candidate as support."),
            ),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps(
                        {"query": "what is python", "candidates": ["Python is useful."]}
                    ),
                    "--prompt-mode",
                    "chatqa",
                    "--include-reasoning",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            "query: what is python\n"
            "answer: Answer for what is python. [1]\n"
            "references: [d0]\n"
            "reasoning: Used the only candidate as support.\n",
        )

    def test_generate_batch_request_file_writes_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "requests.jsonl"
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "query": {"qid": "q1", "text": "what is python"},
                        "candidates": [
                            {
                                "docid": "d1",
                                "score": 1.0,
                                "doc": {
                                    "segment": "Python is used for web development."
                                },
                            }
                        ],
                    }
                ],
            )

            with patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(),
            ):
                exit_code = main(
                    [
                        "generate",
                        "--model",
                        "gpt-4o",
                        "--input-file",
                        str(input_path),
                        "--output-file",
                        str(output_path),
                        "--prompt-mode",
                        "chatqa",
                    ]
                )

            self.assertEqual(exit_code, 0)
            records = read_jsonl(output_path)
            self.assertEqual(records[0]["topic_id"], "q1")
            self.assertEqual(
                records[0]["answer"][0]["text"], "Answer for what is python."
            )

    def test_generate_batch_request_file_is_quiet_in_text_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "requests.jsonl"
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "query": {"qid": "q1", "text": "what is python"},
                        "candidates": [
                            {
                                "docid": "d1",
                                "score": 1.0,
                                "doc": {
                                    "segment": "Python is used for web development."
                                },
                            }
                        ],
                    }
                ],
            )

            stdout = StringIO()
            with (
                redirect_stdout(stdout),
                patch(
                    "ragnarok.cli.operations.create_generation_agent",
                    return_value=FakeAgent(),
                ),
            ):
                exit_code = main(
                    [
                        "generate",
                        "--model",
                        "gpt-4o",
                        "--input-file",
                        str(input_path),
                        "--output-file",
                        str(output_path),
                        "--prompt-mode",
                        "chatqa",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(output_path.exists())

    def test_generate_batch_request_file_async_writes_output(self):
        class AsyncOnlyAgent(FakeAgent):
            def answer_batch(
                self,
                requests,
                topk,
                shuffle_candidates=False,
                logging=False,
                vllm=False,
            ):
                raise AssertionError("sync answer_batch should not run in async mode")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "requests.jsonl"
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "query": {"qid": "q1", "text": "what is python"},
                        "candidates": [
                            {
                                "docid": "d1",
                                "score": 1.0,
                                "doc": {
                                    "segment": "Python is used for web development."
                                },
                            }
                        ],
                    }
                ],
            )

            with patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=AsyncOnlyAgent(),
            ):
                exit_code = main(
                    [
                        "generate",
                        "--model",
                        "gpt-4o",
                        "--input-file",
                        str(input_path),
                        "--output-file",
                        str(output_path),
                        "--prompt-mode",
                        "chatqa",
                        "--execution-mode",
                        "async",
                    ]
                )

            self.assertEqual(exit_code, 0)
            records = read_jsonl(output_path)
            self.assertEqual(records[0]["topic_id"], "q1")

    def test_generate_dataset_mode_calls_existing_orchestration(self):
        captured = {}

        def fake_run_dataset_generation(args, logger):
            captured["dataset"] = args.dataset
            captured["retrieval_method"] = [
                str(method) for method in args.retrieval_method
            ]
            captured["topk"] = args.topk
            return [], {"ok": True}

        with patch(
            "ragnarok.cli.main.run_dataset_generation",
            side_effect=fake_run_dataset_generation,
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--dataset",
                    "rag24.raggy-dev",
                    "--retrieval-method",
                    "bm25,rank_zephyr_rho",
                    "--topk",
                    "100,5",
                    "--prompt-mode",
                    "chatqa",
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["dataset"], "rag24.raggy-dev")
        self.assertEqual(captured["retrieval_method"], ["bm25", "rank_zephyr_rho"])
        self.assertEqual(captured["topk"], [100, 5])

    def test_generate_dataset_mode_rejects_async_execution(self):
        exit_code = main(
            [
                "generate",
                "--model",
                "gpt-4o",
                "--dataset",
                "rag24.raggy-dev",
                "--prompt-mode",
                "chatqa",
                "--execution-mode",
                "async",
                "--output",
                "json",
            ]
        )

        self.assertEqual(exit_code, 2)

    def test_generate_dry_run_returns_zero(self):
        exit_code = main(
            [
                "generate",
                "--model",
                "gpt-4o",
                "--dataset",
                "rag24.raggy-dev",
                "--prompt-mode",
                "chatqa",
                "--dry-run",
                "--output",
                "json",
            ]
        )

        self.assertEqual(exit_code, 0)

    def test_write_policy_conflict_returns_json_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "requests.jsonl"
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "query": {"qid": "q1", "text": "what is python"},
                        "candidates": [],
                    }
                ],
            )
            output_path.write_text("existing\n", encoding="utf-8")

            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--input-file",
                    str(input_path),
                    "--output-file",
                    str(output_path),
                    "--prompt-mode",
                    "chatqa",
                    "--output",
                    "json",
                ]
            )

            self.assertEqual(exit_code, 5)

    def test_write_policy_overwrite_and_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "requests.jsonl"
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "query": {"qid": "q1", "text": "what is python"},
                        "candidates": [],
                    }
                ],
            )
            output_path.write_text("existing\n", encoding="utf-8")

            with patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(),
            ):
                overwrite_exit = main(
                    [
                        "generate",
                        "--model",
                        "gpt-4o",
                        "--input-file",
                        str(input_path),
                        "--output-file",
                        str(output_path),
                        "--prompt-mode",
                        "chatqa",
                        "--overwrite",
                    ]
                )
            self.assertEqual(overwrite_exit, 0)

            with patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=FakeAgent(),
            ):
                resume_exit = main(
                    [
                        "generate",
                        "--model",
                        "gpt-4o",
                        "--input-file",
                        str(input_path),
                        "--output-file",
                        str(output_path),
                        "--prompt-mode",
                        "chatqa",
                        "--resume",
                    ]
                )
            self.assertEqual(resume_exit, 0)

    def test_missing_command_returns_descriptive_error(self):
        with patch("sys.stderr.write") as stderr_write:
            exit_code = main([])
        self.assertEqual(exit_code, 2)
        self.assertTrue(
            any(
                "No command provided." in call.args[0]
                for call in stderr_write.call_args_list
            )
        )
        self.assertTrue(
            any(
                "generate, validate, convert, view, prompt, describe, schema, doctor"
                in call.args[0]
                for call in stderr_write.call_args_list
            )
        )

    def test_top_level_help_includes_command_summaries(self):
        stdout = StringIO()
        with self.assertRaises(SystemExit) as exc_info:
            with redirect_stdout(stdout):
                main(["--help"])
        self.assertEqual(exc_info.exception.code, 0)
        rendered = stdout.getvalue().lower()
        self.assertIn("ragnarok packaged cli", rendered)
        self.assertIn("run generation from direct json input", rendered)
        self.assertIn("inspect an existing generation artifact", rendered)

    def test_view_generate_output_returns_json_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                output_path,
                [
                    {
                        "run_id": "demo-run",
                        "topic_id": "q1",
                        "topic": "What is Python used for? " * 5,
                        "references": ["d1", "d2", "d3"],
                        "response_length": 42,
                        "answer": [
                            {
                                "text": "Python is used for web development. " * 8,
                                "citations": [0, 1],
                            }
                        ],
                    }
                ],
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    ["view", str(output_path), "--records", "1", "--output", "json"]
                )

            self.assertEqual(exit_code, 0)
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["command"], "view")
            self.assertEqual(
                output["artifacts"][0]["data"]["artifact_type"],
                "generate-output-record",
            )
            self.assertEqual(
                output["artifacts"][0]["data"]["summary"]["run_ids"], ["demo-run"]
            )
            self.assertEqual(len(output["artifacts"][0]["data"]["sampled_records"]), 1)
            self.assertEqual(
                output["artifacts"][0]["data"]["sampled_records"][0]["topic"],
                ("What is Python used for? " * 5).strip(),
            )

    def test_view_generate_output_text_respects_record_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                output_path,
                [
                    {
                        "run_id": "demo-run",
                        "topic_id": "q1",
                        "topic": "topic one",
                        "references": ["d1"],
                        "response_length": 10,
                        "answer": [{"text": "answer one", "citations": [0]}],
                    },
                    {
                        "run_id": "demo-run",
                        "topic_id": "q2",
                        "topic": "topic two",
                        "references": ["d2"],
                        "response_length": 12,
                        "answer": [{"text": "answer two", "citations": [0]}],
                    },
                ],
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    ["view", str(output_path), "--records", "1", "--color", "never"]
                )

            self.assertEqual(exit_code, 0)
            text = stdout.getvalue()
            self.assertIn("Ragnarok View", text)
            self.assertIn("topic_id=q1", text)
            self.assertNotIn("topic_id=q2", text)

    def test_view_empty_file_returns_json_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.jsonl"
            path.write_text("", encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["view", str(path), "--output", "json"])

            self.assertEqual(exit_code, 5)
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["errors"][0]["code"], "invalid_view_input")

    def test_view_malformed_file_returns_json_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.jsonl"
            path.write_text("{not-json}\n", encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["view", str(path), "--output", "json"])

            self.assertEqual(exit_code, 5)
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["errors"][0]["code"], "invalid_view_input")

    def test_describe_schema_and_doctor_json_envelopes(self):
        for argv, expected_command in [
            (["describe", "generate", "--output", "json"], "describe"),
            (["schema", "generate-direct-input", "--output", "json"], "schema"),
            (["doctor", "--output", "json"], "doctor"),
        ]:
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(argv)
            self.assertEqual(exit_code, 0)
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["command"], expected_command)

    def test_missing_input_file_returns_json_error(self):
        exit_code = main(
            [
                "generate",
                "--model",
                "gpt-4o",
                "--input-file",
                "/tmp/does-not-exist.jsonl",
                "--prompt-mode",
                "chatqa",
                "--output",
                "json",
            ]
        )
        self.assertEqual(exit_code, 4)

    def test_validate_generate_accepts_and_rejects_inputs(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            valid_exit = main(
                [
                    "validate",
                    "generate",
                    "--input-json",
                    json.dumps({"query": "q", "candidates": ["p"]}),
                    "--output",
                    "json",
                ]
            )
        self.assertEqual(valid_exit, 0)

        stdout = StringIO()
        with redirect_stdout(stdout):
            invalid_exit = main(
                [
                    "validate",
                    "generate",
                    "--input-json",
                    json.dumps({"query": "q"}),
                    "--output",
                    "json",
                ]
            )
        self.assertEqual(invalid_exit, 6)

    def test_validate_rag24_and_rag25_fixture_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            rag24_topics = temp_path / "topics.tsv"
            rag24_run = temp_path / "run.jsonl"
            rag25_topics = temp_path / "topics.jsonl"
            rag25_run = temp_path / "run25.jsonl"

            rag24_topics.write_text("2024-1\tWhat is Python?\n", encoding="utf-8")
            write_jsonl(
                rag24_run,
                [
                    {
                        "run_id": "run",
                        "topic_id": "2024-1",
                        "topic": "What is Python?",
                        "references": ["msmarco_v2.1_doc_1_1#1_1"],
                        "response_length": 2,
                        "answer": [{"text": "A language.", "citations": [0]}],
                    }
                ],
            )
            write_jsonl(
                rag25_topics,
                [{"id": "1", "title": "What is Python?"}],
            )
            write_jsonl(
                rag25_run,
                [
                    {
                        "metadata": {
                            "team_id": "team",
                            "run_id": "run",
                            "narrative_id": "1",
                        },
                        "references": ["msmarco_v2.1_doc_1_1#1_1"],
                        "answer": [{"text": "A language.", "citations": [0]}],
                    }
                ],
            )

            self.assertEqual(
                main(
                    [
                        "validate",
                        "rag24-output",
                        "--topicfile",
                        str(rag24_topics),
                        "--runfile",
                        str(rag24_run),
                        "--output",
                        "json",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "validate",
                        "rag25-output",
                        "--input",
                        str(rag25_run),
                        "--topics",
                        str(rag25_topics),
                        "--output",
                        "json",
                    ]
                ),
                0,
            )

    def test_convert_trec25_format_preserves_prompt_enrichment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "old.jsonl"
            output_path = temp_path / "new.jsonl"
            prompt_path = temp_path / "prompts.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "run_id": "run",
                        "topic_id": "1",
                        "topic": "What is Python?",
                        "references": ["msmarco_v2.1_doc_1_1#1_1"],
                        "answer": [{"text": "A language.", "citations": [0]}],
                    }
                ],
            )
            write_jsonl(
                prompt_path,
                [
                    {
                        "query": {"qid": "1"},
                        "rag_exec_summary": {"prompt": "Tell me about Python."},
                    }
                ],
            )

            exit_code = main(
                [
                    "convert",
                    "trec25-format",
                    "--input-file",
                    str(input_path),
                    "--output-file",
                    str(output_path),
                    "--prompt-file",
                    str(prompt_path),
                    "--output",
                    "json",
                ]
            )
            self.assertEqual(exit_code, 0)
            converted = read_jsonl(output_path)
            self.assertEqual(
                converted[0]["metadata"]["prompt"], "Tell me about Python."
            )

    def test_legacy_run_ragnarok_wrapper_delegates_to_cli(self):
        from ragnarok.scripts.run_ragnarok import cli_compatible_main

        with patch("ragnarok.cli.main.main", return_value=0) as cli_main:
            exit_code = cli_compatible_main(["--model_path", "gpt-4o"])
        self.assertEqual(exit_code, 0)
        cli_main.assert_called_once()


if __name__ == "__main__":
    unittest.main()
