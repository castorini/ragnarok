import json
import os
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

from ragnarok.data import DataWriter, Query, RAGExecInfo, Result
from ragnarok.generate.llm import LLM, PromptMode


class DummyLLM(LLM):
    def run_llm(self, prompt, logging=False):
        raise NotImplementedError

    def create_prompt(self, request, topk):
        raise NotImplementedError

    def get_num_tokens(self, prompt):
        return 0

    def cost_per_1k_token(self, input_token):
        return 0


class TestReasoningSupport(unittest.TestCase):
    def test_gpt_post_processor_falls_back_without_spacy_or_stanza(self):
        blocked = {"spacy", "stanza"}
        original_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] in blocked:
                raise ModuleNotFoundError(name)
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            if "ragnarok.generate.post_processor" in sys.modules:
                del sys.modules["ragnarok.generate.post_processor"]
            from ragnarok.generate.post_processor import GPTPostProcessor

            answers, rag_exec_response = GPTPostProcessor()(
                "Sentence one [1]. Sentence two [2]."
            )

        self.assertEqual([answer.text for answer in answers], ["Sentence one.", "Sentence two."])
        self.assertEqual([answer.citations for answer in answers], [[0], [1]])
        self.assertEqual(
            rag_exec_response["text"], "Sentence one [1]. Sentence two [2]."
        )

    def test_extract_reasoning_from_think_tags(self):
        llm = DummyLLM(
            model="dummy",
            context_size=1024,
            prompt_mode=PromptMode.CHATQA,
            store_reasoning=True,
        )

        reasoning, cleaned_response = llm._extract_reasoning_from_text(
            "<think>Step 1\nStep 2</think>\nFinal answer [1]."
        )

        self.assertEqual(reasoning, "Step 1\nStep 2")
        self.assertEqual(cleaned_response, "Final answer [1].")

    def test_extract_reasoning_disabled(self):
        llm = DummyLLM(
            model="dummy",
            context_size=1024,
            prompt_mode=PromptMode.CHATQA,
            store_reasoning=False,
        )

        reasoning, cleaned_response = llm._extract_reasoning_from_text(
            "<think>Hidden chain of thought</think>\nVisible answer."
        )

        self.assertIsNone(reasoning)
        self.assertEqual(cleaned_response, "Visible answer.")

    def test_write_rag_exec_summary_includes_reasoning_only_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = os.path.join(temp_dir, "summary.jsonl")
            results = [
                Result(
                    query=Query(text="q1", qid="1"),
                    rag_exec_summary=RAGExecInfo(
                        prompt="prompt-1",
                        response={"text": "answer-1", "citations": []},
                        input_token_count=10,
                        output_token_count=5,
                        reasoning="Model reasoning",
                    ),
                ),
                Result(
                    query=Query(text="q2", qid="2"),
                    rag_exec_summary=RAGExecInfo(
                        prompt="prompt-2",
                        response={"text": "answer-2", "citations": []},
                        input_token_count=11,
                        output_token_count=6,
                    ),
                ),
            ]

            DataWriter(results).write_rag_exec_summary(summary_path)

            with open(summary_path, "r") as handle:
                lines = [json.loads(line) for line in handle]

        self.assertEqual(lines[0]["rag_exec_summary"]["reasoning"], "Model reasoning")
        self.assertNotIn("reasoning", lines[1]["rag_exec_summary"])

    def test_safe_openai_forwards_reasoning_effort(self):
        recorded_kwargs = {}

        class FakeResponse:
            class Choice:
                class Message:
                    content = "Final answer."

                message = Message()

            choices = [Choice()]

        def fake_create(**kwargs):
            recorded_kwargs.update(kwargs)
            return FakeResponse()

        fake_openai = ModuleType("openai")
        fake_openai.proxy = None
        fake_openai.api_key = None
        fake_openai.api_version = None
        fake_openai.api_type = None
        fake_openai.api_base = None
        fake_openai.chat = type(
            "ChatNamespace",
            (),
            {
                "completions": type(
                    "CompletionsNamespace", (), {"create": staticmethod(fake_create)}
                )()
            },
        )()
        fake_openai.Completion = type(
            "CompletionNamespace", (), {"create": staticmethod(fake_create)}
        )

        fake_tiktoken = ModuleType("tiktoken")
        fake_tiktoken.get_encoding = staticmethod(
            lambda _name: type(
                "Encoding", (), {"encode": staticmethod(lambda text: list(text))}
            )()
        )
        fake_post_processor = ModuleType("ragnarok.generate.post_processor")

        class FakeGPTPostProcessor:
            def __call__(self, response):
                return [], {"text": response, "citations": []}

        fake_post_processor.GPTPostProcessor = FakeGPTPostProcessor

        with patch.dict(
            sys.modules,
            {
                "openai": fake_openai,
                "tiktoken": fake_tiktoken,
                "ragnarok.generate.post_processor": fake_post_processor,
            },
        ):
            if "ragnarok.generate.gpt" in sys.modules:
                del sys.modules["ragnarok.generate.gpt"]
            from ragnarok.generate.gpt import SafeOpenai

            model = SafeOpenai(
                model="gpt-5",
                context_size=1024,
                prompt_mode=PromptMode.CHATQA,
                keys=["test-key"],
                reasoning_effort="medium",
            )
            model.run_llm(
                [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "User prompt"},
                ]
            )

        self.assertEqual(recorded_kwargs["reasoning_effort"], "medium")


if __name__ == "__main__":
    unittest.main()
