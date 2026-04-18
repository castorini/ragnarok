import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ragnarok.cli.main import main
from ragnarok.data import RAGExecInfo, Result
from ragnarok.generate.llm import PromptMode

pytestmark = pytest.mark.core


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
    def test_batch_json_output_suppresses_progress_bar(self):
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

            import io
            from contextlib import redirect_stderr

            stderr = io.StringIO()
            stdout = StringIO()
            with (
                redirect_stderr(stderr),
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
                        "--output",
                        "json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr.getvalue(), "")

    def test_quiet_flag_suppresses_stderr(self):
        import io
        from contextlib import redirect_stderr

        stderr = io.StringIO()
        stdout = StringIO()
        with (
            redirect_stderr(stderr),
            redirect_stdout(stdout),
            patch("ragnarok.cli.main.doctor_report", return_value={"python_ok": True}),
        ):
            exit_code = main(["--quiet", "doctor", "--output", "json"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")

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
                    exit_code = main(["view", str(output_path), "--color", "always"])

            self.assertEqual(exit_code, 0)
            self.assertNotIn("\033[", stdout.getvalue())

    def test_print_completion_outputs_bash_script(self):
        stdout = StringIO()
        with self.assertRaises(SystemExit) as exc_info:
            with redirect_stdout(stdout):
                main(["--print-completion", "bash"])
        self.assertEqual(exc_info.exception.code, 0)
        output = stdout.getvalue()
        self.assertTrue(
            "complete" in output.lower() or "_ragnarok" in output,
            f"Expected shell completion script, got: {output[:200]}",
        )

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
        data = output["artifacts"][0]["data"]
        modes = {entry["prompt_mode"] for entry in data}
        self.assertIn("chatqa", modes)
        self.assertIn("ragnarok_v4", modes)
        entry = data[0]
        self.assertIn("source_path", entry)
        self.assertIn("method", entry)
        self.assertIn("placeholders", entry)
        self.assertTrue(entry["source_path"].endswith(".yaml"))

    def test_prompt_show_returns_text_definition(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["prompt", "show", "--prompt-mode", "chatqa"])

        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Ragnarok Prompt Template", rendered)
        self.assertIn("prompt_mode: chatqa", rendered)
        self.assertIn("[system]", rendered)
        self.assertIn("[user]", rendered)
        self.assertIn("[instruction]", rendered)

    def test_prompt_show_returns_json_definition(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                ["prompt", "show", "--prompt-mode", "ragnarok_v4", "--output", "json"]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["command"], "prompt")
        self.assertEqual(output["artifacts"][0]["name"], "prompt-template")
        view = output["artifacts"][0]["data"]
        self.assertEqual(view["prompt_mode"], "ragnarok_v4")
        self.assertIn("template", view)
        self.assertIn("source_path", view["template"])
        self.assertIn(
            "Ensure each sentence has at least one citation.",
            view["template"]["instruction"],
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
        self.assertIn("messages", view)
        self.assertIsInstance(view["messages"], list)
        self.assertEqual(view["messages"][0]["role"], "system")
        self.assertIn("Query: what is python", view["messages"][1]["content"])

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
        self.assertIn("Ragnarok Rendered Prompt", rendered)
        self.assertIn("[user]", rendered)
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

    def test_yaml_templates_load_all_modes(self):
        from ragnarok.prompts.template_loader import get_template, list_templates

        templates = list_templates()
        self.assertEqual(len(templates), 8)
        mode_map = {
            PromptMode.CHATQA: "chatqa",
            PromptMode.RAGNAROK_V2: "ragnarok_v2",
            PromptMode.RAGNAROK_V3: "ragnarok_v3",
            PromptMode.RAGNAROK_V4: "ragnarok_v4",
            PromptMode.RAGNAROK_V4_BIOGEN: "ragnarok_v4_biogen",
            PromptMode.RAGNAROK_V4_NO_CITE: "ragnarok_v4_no_cite",
            PromptMode.RAGNAROK_V5_BIOGEN: "ragnarok_v5_biogen",
            PromptMode.RAGNAROK_V5_BIOGEN_NO_CITE: "ragnarok_v5_biogen_no_cite",
        }
        from ragnarok.generate.templates.ragnarok_templates import RagnarokTemplates

        for prompt_mode, yaml_name in mode_map.items():
            tmpl = get_template(yaml_name)
            rt = RagnarokTemplates(prompt_mode)
            self.assertEqual(
                rt.get_instruction(),
                tmpl.instruction,
                f"instruction mismatch for {yaml_name}",
            )
            self.assertTrue(
                tmpl.source_path.endswith(".yaml"),
                f"source_path should end with .yaml for {yaml_name}",
            )
            self.assertEqual(tmpl.method, yaml_name)
            self.assertGreater(len(tmpl.placeholders), 0)

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

    def test_generate_direct_accepts_anserini_rest_payload(self):
        class AssertingAgent(FakeAgent):
            def answer_batch(
                self,
                requests,
                topk,
                shuffle_candidates=False,
                logging=False,
                vllm=False,
            ):
                del topk, shuffle_candidates, logging, vllm
                assert requests[0].query.text == "what is python"
                assert requests[0].query.qid == "q0"
                assert requests[0].candidates[0].docid == "1737459"
                assert (
                    requests[0].candidates[0].doc["segment"]
                    == "Python is widely used for web development."
                )
                return super().answer_batch(requests, 5)

        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch(
                "ragnarok.cli.operations.create_generation_agent",
                return_value=AssertingAgent(),
            ),
        ):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    "gpt-4o",
                    "--input-json",
                    json.dumps(
                        {
                            "api": "v1",
                            "index": "msmarco-v1-passage",
                            "query": {"text": "what is python"},
                            "candidates": [
                                {
                                    "docid": "1737459",
                                    "score": 10.58,
                                    "rank": 1,
                                    "doc": "Python is widely used for web development.",
                                }
                            ],
                        }
                    ),
                    "--prompt-mode",
                    "chatqa",
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["status"], "success")

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
        sys.modules.setdefault("openai", types.ModuleType("openai"))
        sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))
        __import__("ragnarok.generate.gpt")  # ensure submodule is loaded for patching

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
                "generate, serve, validate, convert, view, prompt, describe, schema, doctor"
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

    def test_generate_schema_includes_request_overrides(self):
        from ragnarok.cli.introspection import SCHEMAS

        overrides = SCHEMAS["generate-direct-input"]["properties"]["overrides"][
            "properties"
        ]
        self.assertIn("model", overrides)
        self.assertIn("reasoning_effort", overrides)

    def test_serve_command_starts_uvicorn(self):
        pytest.importorskip("fastapi")
        with patch("uvicorn.run") as uvicorn_run:
            exit_code = main(
                [
                    "serve",
                    "--model",
                    "gpt-4o",
                    "--prompt-mode",
                    "chatqa",
                    "--port",
                    "8083",
                ]
            )
        self.assertEqual(exit_code, 0)
        uvicorn_run.assert_called_once()
        self.assertEqual(uvicorn_run.call_args.kwargs["host"], "0.0.0.0")
        self.assertEqual(uvicorn_run.call_args.kwargs["port"], 8083)

    def test_serve_app_exposes_health_and_generate(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from ragnarok.api.app import create_app
        from ragnarok.api.runtime import ServerConfig

        fake_records = [
            {
                "topic_id": "q0",
                "topic": "q",
                "references": ["d0"],
                "response_length": 8,
                "answer": [{"text": "answer", "citations": [0]}],
            }
        ]

        with patch(
            "ragnarok.api.runtime.run_request_generation",
            return_value=(fake_records, {"request_count": 1}),
        ):
            client = TestClient(
                create_app(
                    ServerConfig(
                        host="127.0.0.1",
                        port=8083,
                        model="gpt-4o",
                        prompt_mode="chatqa",
                    )
                )
            )
            health_response = client.get("/healthz")
            generate_response = client.post(
                "/v1/generate",
                json={
                    "api": "v1",
                    "index": "msmarco-v1-passage",
                    "query": {"text": "q"},
                    "candidates": [
                        {
                            "docid": "d0",
                            "score": 1.0,
                            "rank": 1,
                            "doc": "passage",
                        }
                    ],
                },
            )

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(generate_response.status_code, 200)
        envelope = generate_response.json()
        self.assertEqual(envelope["schema_version"], "castorini.cli.v1")
        self.assertEqual(envelope["command"], "generate")
        self.assertEqual(envelope["artifacts"][0]["name"], "generation-results")

    def test_serve_app_rejects_invalid_payload(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from ragnarok.api.app import create_app
        from ragnarok.api.runtime import ServerConfig

        client = TestClient(
            create_app(
                ServerConfig(
                    host="127.0.0.1",
                    port=8083,
                    model="gpt-4o",
                    prompt_mode="chatqa",
                )
            )
        )

        response = client.post("/v1/generate", json={"query": 1, "candidates": []})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "validation_error")

    def test_serve_app_applies_request_overrides(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from ragnarok.api.app import create_app
        from ragnarok.api.runtime import ServerConfig

        captured: dict[str, object] = {}

        def fake_run_request_generation(requests, args, logger):
            captured["model"] = args.model
            captured["reasoning_effort"] = args.reasoning_effort
            captured["use_openrouter"] = args.use_openrouter
            return (
                [
                    {
                        "topic_id": "q0",
                        "topic": "q",
                        "references": ["d0"],
                        "response_length": 8,
                        "answer": [{"text": "answer", "citations": [0]}],
                    }
                ],
                {"request_count": len(requests)},
            )

        with patch(
            "ragnarok.api.runtime.run_request_generation",
            side_effect=fake_run_request_generation,
        ):
            client = TestClient(
                create_app(
                    ServerConfig(
                        host="127.0.0.1",
                        port=8083,
                        model="gpt-4o",
                        prompt_mode="chatqa",
                    )
                )
            )
            response = client.post(
                "/v1/generate",
                json={
                    "query": {"text": "q"},
                    "candidates": ["passage"],
                    "overrides": {
                        "model": "gpt-4.1-mini",
                        "reasoning_effort": "low",
                        "use_openrouter": True,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["model"], "gpt-4.1-mini")
        self.assertEqual(captured["reasoning_effort"], "low")
        self.assertTrue(captured["use_openrouter"])
        self.assertEqual(response.json()["resolved"]["model"], "gpt-4.1-mini")

    def test_serve_app_rejects_invalid_override_combinations(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from ragnarok.api.app import create_app
        from ragnarok.api.runtime import ServerConfig

        client = TestClient(
            create_app(
                ServerConfig(
                    host="127.0.0.1",
                    port=8083,
                    model="gpt-4o",
                    prompt_mode="chatqa",
                )
            )
        )

        response = client.post(
            "/v1/generate",
            json={
                "query": {"text": "q"},
                "candidates": ["passage"],
                "overrides": {
                    "use_azure_openai": True,
                    "use_openrouter": True,
                },
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "validation_error")

    def test_serve_app_accepts_rank_llm_envelope_payload(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from ragnarok.api.app import create_app
        from ragnarok.api.runtime import ServerConfig

        fake_records = [
            {
                "run_id": "ragnarok",
                "topic_id": "q0",
                "topic": "q",
                "references": ["d0"],
                "response_length": 1,
                "answer": [{"text": "answer", "citations": [0]}],
            }
        ]

        with patch(
            "ragnarok.api.runtime.run_request_generation",
            return_value=(fake_records, {"request_count": 1}),
        ):
            client = TestClient(
                create_app(
                    ServerConfig(
                        host="127.0.0.1",
                        port=8083,
                        model="gpt-4o",
                        prompt_mode="chatqa",
                    )
                )
            )
            response = client.post(
                "/v1/generate",
                json={
                    "schema_version": "castorini.cli.v1",
                    "repo": "rank_llm",
                    "command": "rerank",
                    "artifacts": [
                        {
                            "name": "rerank-results",
                            "kind": "data",
                            "value": [
                                {
                                    "query": {"text": "q", "qid": ""},
                                    "candidates": [
                                        {
                                            "docid": "d0",
                                            "score": 1.0,
                                            "doc": {"contents": "passage"},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["artifacts"][0]["name"], "generation-results")

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

    def test_config_file_sets_default_output_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config" / "ragnarok"
            config_dir.mkdir(parents=True)
            config_file = config_dir / "config.toml"
            config_file.write_text('output = "json"\n', encoding="utf-8")

            stdout = StringIO()
            with (
                patch.dict(
                    os.environ, {"XDG_CONFIG_HOME": str(Path(temp_dir) / "config")}
                ),
                redirect_stdout(stdout),
            ):
                exit_code = main(["doctor"])

            self.assertEqual(exit_code, 0)
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["command"], "doctor")
            self.assertEqual(output["metrics"]["config_file"], str(config_file))

    def test_pipe_generate_json_output_is_valid_jsonl(self):
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
        envelope = json.loads(stdout.getvalue())
        self.assertEqual(envelope["schema_version"], "castorini.cli.v1")
        records = envelope["artifacts"][0]["data"]
        self.assertTrue(all("topic" in r and "answer" in r for r in records))

    def test_legacy_run_ragnarok_wrapper_delegates_to_cli(self):
        from ragnarok.scripts.run_ragnarok import cli_compatible_main

        with patch("ragnarok.cli.main.main", return_value=0) as cli_main:
            exit_code = cli_compatible_main(["--model_path", "gpt-4o"])
        self.assertEqual(exit_code, 0)
        cli_main.assert_called_once()

    def test_legacy_run_ragnarok_wrapper_translates_snake_case_flags(self):
        from ragnarok.scripts.run_ragnarok import cli_compatible_main

        with patch("ragnarok.cli.main.main", return_value=0) as cli_main:
            exit_code = cli_compatible_main(
                [
                    "--model_path=gpt-4o",
                    "--prompt_mode",
                    "chatqa",
                    "--use_azure_openai",
                    "--num_gpus",
                    "2",
                    "--num_few_shot_examples=3",
                    "--reasoning_effort",
                    "high",
                ]
            )

        self.assertEqual(exit_code, 0)
        cli_main.assert_called_once_with(
            [
                "generate",
                "--model=gpt-4o",
                "--prompt-mode",
                "chatqa",
                "--use-azure-openai",
                "--num-gpus",
                "2",
                "--num-few-shot-examples=3",
                "--reasoning-effort",
                "high",
            ]
        )


if __name__ == "__main__":
    unittest.main()
