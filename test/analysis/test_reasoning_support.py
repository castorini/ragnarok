import asyncio
import json
import os
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from ragnarok.data import DataWriter, Query, RAGExecInfo, Result
from ragnarok.generate.generator import RAG
from ragnarok.generate.llm import LLM, PromptMode
from ragnarok.generate.templates.ragnarok_templates import RagnarokTemplates


class DummyLLM(LLM):
    def run_llm(self, prompt, logging=False):
        raise NotImplementedError

    def create_prompt(self, request, topk):
        return "prompt", 1

    def get_num_tokens(self, prompt):
        return 0

    def cost_per_1k_token(self, input_token):
        return 0


class AsyncRecordingLLM(DummyLLM):
    def __init__(self) -> None:
        super().__init__(
            model="dummy",
            context_size=1024,
            prompt_mode=PromptMode.CHATQA,
        )

    def run_llm(self, prompt, logging=False):
        return [], 0

    async def async_run_llm(self, prompt, logging=False):
        delay = (
            0.03 if prompt == "prompt-q1" else 0.01 if prompt == "prompt-q2" else 0.02
        )
        await asyncio.sleep(delay)
        qid = prompt.removeprefix("prompt-")
        return (
            [type("Sentence", (), {"text": f"Answer for {qid}.", "citations": [0]})()],
            RAGExecInfo(
                prompt=prompt,
                response="response",
                input_token_count=12,
                output_token_count=4,
            ),
        )

    def create_prompt(self, request, topk):
        return f"prompt-{request.query.qid}", 1


class TestReasoningSupport(unittest.TestCase):
    def test_create_generation_agent_routes_unknown_model_to_openai_compatible(self):
        fake_safe_openai = MagicMock(return_value="agent")
        fake_gpt_module = ModuleType("ragnarok.generate.gpt")
        fake_gpt_module.SafeOpenai = fake_safe_openai

        args = SimpleNamespace(
            model_path="openrouter/hunter-alpha",
            context_size=8192,
            prompt_mode=PromptMode.CHATQA,
            max_output_tokens=512,
            num_few_shot_examples=0,
            include_reasoning=True,
            reasoning_effort="high",
            use_azure_openai=False,
            num_gpus=1,
        )

        with (
            patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "router-key", "OPENAI_API_KEY": ""},
                clear=False,
            ),
            patch.dict(
                sys.modules,
                {"ragnarok.generate.gpt": fake_gpt_module},
            ),
        ):
            from ragnarok.cli.operations import create_generation_agent

            agent = create_generation_agent(args)

        self.assertEqual(agent, "agent")
        fake_safe_openai.assert_called_once()
        self.assertEqual(fake_safe_openai.call_args.kwargs["model"], args.model_path)
        self.assertEqual(
            fake_safe_openai.call_args.kwargs["api_base"],
            "https://openrouter.ai/api/v1",
        )
        self.assertEqual(fake_safe_openai.call_args.kwargs["keys"], "router-key")

    def test_openrouter_models_use_openrouter_key_even_when_openai_key_exists(self):
        from ragnarok.generate.api_keys import get_openai_compatible_args

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "OPENROUTER_API_KEY": "router-key",
            },
            clear=False,
        ):
            args = get_openai_compatible_args("openrouter/hunter-alpha")

        self.assertEqual(args["keys"], "router-key")
        self.assertEqual(args["api_base"], "https://openrouter.ai/api/v1")

    def test_ragnarok_templates_use_chat_messages_for_openrouter_models(self):
        template = RagnarokTemplates(prompt_mode=PromptMode.CHATQA)

        prompt = template(
            query="What defines Pink Floyds artistic identity?",
            context=["[1] Pink Floyd explored alienation and social critique."],
            model="openrouter/hunter-alpha",
        )

        self.assertIsInstance(prompt, list)
        self.assertEqual(prompt[0]["role"], "system")
        self.assertEqual(prompt[1]["role"], "user")

    def test_rag_answer_accepts_topk_keyword(self):
        rag = RAG(agent=MagicMock())
        expected = Result(query=Query(text="q", qid="1"))
        rag.answer_batch = MagicMock(return_value=[expected])  # type: ignore[method-assign]

        result = rag.answer(
            request=MagicMock(),
            topk=7,
            shuffle_candidates=True,
            logging=True,
        )

        rag.answer_batch.assert_called_once_with(
            requests=[unittest.mock.ANY],
            topk=7,
            shuffle_candidates=True,
            logging=True,
        )
        self.assertIs(result, expected)

    def test_rag_answer_rejects_nonzero_rank_start(self):
        rag = RAG(agent=MagicMock())

        with self.assertRaises(ValueError):
            rag.answer(request=MagicMock(), rank_start=1)

    def test_rag_async_answer_rejects_nonzero_rank_start(self):
        rag = RAG(agent=MagicMock())

        with self.assertRaises(ValueError):
            asyncio.run(rag.async_answer(request=MagicMock(), rank_start=1))

    def test_async_answer_batch_preserves_request_order(self):
        llm = AsyncRecordingLLM()
        requests = [
            type(
                "RequestLike",
                (),
                {
                    "query": Query(text="q1", qid="q1"),
                    "candidates": [
                        type(
                            "CandidateLike",
                            (),
                            {"docid": "d1", "doc": {"segment": "p1"}},
                        )()
                    ],
                },
            )(),
            type(
                "RequestLike",
                (),
                {
                    "query": Query(text="q2", qid="q2"),
                    "candidates": [
                        type(
                            "CandidateLike",
                            (),
                            {"docid": "d2", "doc": {"segment": "p2"}},
                        )()
                    ],
                },
            )(),
            type(
                "RequestLike",
                (),
                {
                    "query": Query(text="q3", qid="q3"),
                    "candidates": [
                        type(
                            "CandidateLike",
                            (),
                            {"docid": "d3", "doc": {"segment": "p3"}},
                        )()
                    ],
                },
            )(),
        ]

        results = asyncio.run(
            llm.async_answer_batch(requests, topk=1, max_concurrency=3)
        )

        self.assertEqual([result.query.qid for result in results], ["q1", "q2", "q3"])
        self.assertEqual(
            [result.answer[0].text for result in results],
            ["Answer for q1.", "Answer for q2.", "Answer for q3."],
        )

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

        self.assertEqual(
            [answer.text for answer in answers], ["Sentence one.", "Sentence two."]
        )
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

    def test_safe_openai_uses_responses_api_for_openai_reasoning_models(self):
        recorded_kwargs = {}

        def fake_create(**kwargs):
            recorded_kwargs.update(kwargs)
            return SimpleNamespace(
                output_text="Final answer.",
                output=[
                    SimpleNamespace(
                        type="reasoning",
                        summary=[SimpleNamespace(type="summary_text", text="chain")],
                    )
                ],
            )

        fake_openai = ModuleType("openai")
        fake_openai.proxy = None
        fake_openai.api_key = None
        fake_openai.api_version = None
        fake_openai.api_type = None
        fake_openai.api_base = None
        fake_openai.OpenAI = lambda **kwargs: SimpleNamespace(
            responses=SimpleNamespace(create=staticmethod(fake_create))
        )
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

        self.assertEqual(
            recorded_kwargs["reasoning"],
            {"effort": "medium", "summary": "auto"},
        )
        self.assertEqual(
            recorded_kwargs["input"],
            [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "System prompt"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "User prompt"}],
                },
            ],
        )

    def test_extract_reasoning_from_model_extra(self):
        llm = DummyLLM(
            model="dummy",
            context_size=1024,
            prompt_mode=PromptMode.CHATQA,
            store_reasoning=True,
        )

        reasoning = llm._extract_reasoning_from_message(
            SimpleNamespace(model_extra={"reasoning": "Provider reasoning"})
        )

        self.assertEqual(reasoning, "Provider reasoning")

    def test_safe_openai_uses_openrouter_extra_body_sync(self):
        recorded_kwargs = {}

        class FakeResponse:
            class Choice:
                class Message:
                    content = "Final answer."
                    model_extra = {"reasoning": "chain"}

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
                model="openrouter/hunter-alpha",
                context_size=1024,
                prompt_mode=PromptMode.CHATQA,
                keys=["test-key"],
                api_base="https://openrouter.ai/api/v1",
                reasoning_effort="high",
                store_reasoning=True,
            )
            answers, rag_exec_info = model.run_llm(
                [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "User prompt"},
                ]
            )

        self.assertEqual(answers, [])
        self.assertEqual(rag_exec_info.reasoning, "chain")
        self.assertEqual(
            recorded_kwargs["extra_body"],
            {"reasoning": {"effort": "high", "exclude": False}},
        )
        self.assertNotIn("reasoning_effort", recorded_kwargs)

    def test_safe_openai_uses_openrouter_extra_body_async(self):
        fake_openai = ModuleType("openai")
        fake_openai.proxy = None
        fake_openai.api_key = None
        fake_openai.api_version = None
        fake_openai.api_type = None
        fake_openai.api_base = None
        fake_openai.chat = type("ChatNamespace", (), {})()
        fake_openai.Completion = type("CompletionNamespace", (), {})()

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
                model="openrouter/hunter-alpha",
                context_size=1024,
                prompt_mode=PromptMode.CHATQA,
                keys=["test-key"],
                api_base="https://openrouter.ai/api/v1",
                reasoning_effort="high",
                store_reasoning=True,
            )

            recorded_kwargs = {}

            async def fake_create(**kwargs):
                recorded_kwargs.update(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Final answer.",
                                model_extra={"reasoning": "chain"},
                            )
                        )
                    ]
                )

            model._async_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            )
            model._async_client_key_id = model._cur_key_id

            answers, rag_exec_info = asyncio.run(
                model.async_run_llm(
                    [
                        {"role": "system", "content": "System prompt"},
                        {"role": "user", "content": "User prompt"},
                    ]
                )
            )

        self.assertEqual(answers, [])
        self.assertEqual(rag_exec_info.reasoning, "chain")
        self.assertEqual(
            recorded_kwargs["extra_body"],
            {"reasoning": {"effort": "high", "exclude": False}},
        )
        self.assertNotIn("reasoning_effort", recorded_kwargs)

    def test_safe_openai_uses_responses_api_for_openrouter_reasoning_models_sync(self):
        recorded_kwargs = {}

        def fake_create(**kwargs):
            recorded_kwargs.update(kwargs)
            return SimpleNamespace(
                output_text="Final answer.",
                output=[SimpleNamespace(type="reasoning", summary=["router chain"])],
            )

        fake_openai = ModuleType("openai")
        fake_openai.proxy = None
        fake_openai.api_key = None
        fake_openai.api_version = None
        fake_openai.api_type = None
        fake_openai.api_base = None
        fake_openai.OpenAI = lambda **kwargs: SimpleNamespace(
            responses=SimpleNamespace(create=staticmethod(fake_create))
        )
        fake_openai.chat = type("ChatNamespace", (), {})()
        fake_openai.Completion = type("CompletionNamespace", (), {})()

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
                model="openrouter/openai/o4-mini",
                context_size=1024,
                prompt_mode=PromptMode.CHATQA,
                keys=["test-key"],
                api_base="https://openrouter.ai/api/v1",
                reasoning_effort="high",
                store_reasoning=True,
            )
            answers, rag_exec_info = model.run_llm(
                [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "User prompt"},
                ]
            )

        self.assertEqual(answers, [])
        self.assertEqual(rag_exec_info.reasoning, "router chain")
        self.assertEqual(
            recorded_kwargs["reasoning"],
            {"effort": "high", "summary": "auto"},
        )

    def test_safe_openai_prefers_direct_reasoning_for_openrouter_responses_sync(self):
        def fake_create(**kwargs):
            del kwargs
            return SimpleNamespace(
                output_text="Final answer.",
                output=[
                    SimpleNamespace(
                        type="reasoning",
                        reasoning="raw router reasoning",
                        summary=["router summary"],
                    )
                ],
            )

        fake_openai = ModuleType("openai")
        fake_openai.proxy = None
        fake_openai.api_key = None
        fake_openai.api_version = None
        fake_openai.api_type = None
        fake_openai.api_base = None
        fake_openai.OpenAI = lambda **kwargs: SimpleNamespace(
            responses=SimpleNamespace(create=staticmethod(fake_create))
        )
        fake_openai.chat = type("ChatNamespace", (), {})()
        fake_openai.Completion = type("CompletionNamespace", (), {})()

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
                model="openrouter/openai/o4-mini",
                context_size=1024,
                prompt_mode=PromptMode.CHATQA,
                keys=["test-key"],
                api_base="https://openrouter.ai/api/v1",
                reasoning_effort="high",
                store_reasoning=True,
            )
            _, rag_exec_info = model.run_llm(
                [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "User prompt"},
                ]
            )

        self.assertEqual(rag_exec_info.reasoning, "raw router reasoning")

    def test_safe_openai_uses_responses_api_for_openrouter_reasoning_models_async(self):
        fake_openai = ModuleType("openai")
        fake_openai.proxy = None
        fake_openai.api_key = None
        fake_openai.api_version = None
        fake_openai.api_type = None
        fake_openai.api_base = None
        fake_openai.chat = type("ChatNamespace", (), {})()
        fake_openai.Completion = type("CompletionNamespace", (), {})()

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
                model="openrouter/openai/o4-mini",
                context_size=1024,
                prompt_mode=PromptMode.CHATQA,
                keys=["test-key"],
                api_base="https://openrouter.ai/api/v1",
                reasoning_effort="high",
                store_reasoning=True,
            )

            recorded_kwargs = {}

            async def fake_create(**kwargs):
                recorded_kwargs.update(kwargs)
                return SimpleNamespace(
                    output_text="Final answer.",
                    output=[
                        SimpleNamespace(type="reasoning", summary=["router chain"])
                    ],
                )

            model._async_client = SimpleNamespace(
                responses=SimpleNamespace(create=fake_create)
            )
            model._async_client_key_id = model._cur_key_id

            answers, rag_exec_info = asyncio.run(
                model.async_run_llm(
                    [
                        {"role": "system", "content": "System prompt"},
                        {"role": "user", "content": "User prompt"},
                    ]
                )
            )

        self.assertEqual(answers, [])
        self.assertEqual(rag_exec_info.reasoning, "router chain")
        self.assertEqual(
            recorded_kwargs["reasoning"],
            {"effort": "high", "summary": "auto"},
        )


if __name__ == "__main__":
    unittest.main()
