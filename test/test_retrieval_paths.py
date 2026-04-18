from __future__ import annotations

import unittest
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ragnarok.data import Query, Result
from ragnarok.generate.llm import PromptMode
from ragnarok.retrieve_and_generate import retrieve_and_generate
from ragnarok.retrieve_and_rerank.retriever import RetrievalMethod, RetrievalMode

pytestmark = pytest.mark.core


class TestRetrievalPaths(unittest.TestCase):
    def _patch_generation_backend(
        self,
    ) -> tuple[Any, Any, Any, MagicMock]:
        fake_agent = object()
        fake_result = Result(query=Query(text="What is Python?", qid="q1"))
        rag_instance = MagicMock()
        rag_instance.answer_batch.return_value = [fake_result]
        rag_instance.write_answer_results.return_value = "results.jsonl"
        fake_gpt_module: Any = ModuleType("ragnarok.generate.gpt")
        fake_gpt_module.SafeOpenai = MagicMock(return_value=fake_agent)
        return (
            patch.dict(
                "sys.modules",
                {"ragnarok.generate.gpt": fake_gpt_module},
            ),
            patch("ragnarok.retrieve_and_generate.RAG", return_value=rag_instance),
            patch(
                "ragnarok.retrieve_and_generate.get_openai_compatible_args",
                return_value={"keys": "test-key"},
            ),
            rag_instance,
        )

    def test_retrieve_and_generate_uses_cached_retriever_by_default(self) -> None:
        safe_openai_patch, rag_patch, key_patch, _rag_instance = (
            self._patch_generation_backend()
        )
        with (
            patch(
                "ragnarok.retrieve_and_generate.Retriever.from_dataset_with_prebuilt_index",
                return_value=["cached-request"],
            ) as cached_retriever,
            patch(
                "ragnarok.retrieve_and_generate.Restriever.from_dataset_with_prebuilt_index"
            ) as service_retriever,
            safe_openai_patch,
            rag_patch,
            key_patch,
        ):
            retrieve_and_generate(
                generator_path="gpt-4o",
                dataset="rag24.raggy-dev",
                retrieval_mode=RetrievalMode.DATASET,
                retrieval_method=[RetrievalMethod.BM25],
                k=[100, 20],
                prompt_mode=PromptMode.CHATQA,
                run_id="demo-run",
            )

        cached_retriever.assert_called_once_with(
            dataset_name="rag24.raggy-dev",
            retrieval_method=[RetrievalMethod.BM25],
            k=[100, 20],
            cache_input_format=unittest.mock.ANY,
        )
        service_retriever.assert_not_called()

    def test_retrieve_and_generate_uses_service_retriever_in_interactive_mode(
        self,
    ) -> None:
        safe_openai_patch, rag_patch, key_patch, _rag_instance = (
            self._patch_generation_backend()
        )
        with (
            patch(
                "ragnarok.retrieve_and_generate.Retriever.from_dataset_with_prebuilt_index"
            ) as cached_retriever,
            patch(
                "ragnarok.retrieve_and_generate.Restriever.from_dataset_with_prebuilt_index",
                return_value="service-request",
            ) as service_retriever,
            safe_openai_patch,
            rag_patch,
            key_patch,
        ):
            retrieve_and_generate(
                generator_path="gpt-4o",
                dataset="rag24.raggy-dev",
                retrieval_mode=RetrievalMode.DATASET,
                retrieval_method=[RetrievalMethod.BM25, RetrievalMethod.RANK_ZEPHYR],
                k=[100, 20],
                prompt_mode=PromptMode.CHATQA,
                interactive=True,
                query="what is python",
                qid=7,
                run_id="demo-run",
            )

        cached_retriever.assert_not_called()
        service_retriever.assert_called_once()
        self.assertEqual(
            service_retriever.call_args.kwargs["dataset_name"], "rag24.raggy-dev"
        )
        self.assertEqual(service_retriever.call_args.kwargs["k"], [100, 20])
        self.assertEqual(
            service_retriever.call_args.kwargs["request"].query.text, "what is python"
        )
        self.assertEqual(service_retriever.call_args.kwargs["request"].query.qid, 7)
