import json
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
