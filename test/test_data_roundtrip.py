from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

from ragnarok.data import (
    CitedSentence,
    DataWriter,
    Query,
    RAGExecInfo,
    Result,
    read_results_from_file,
    result_to_dict,
)

pytestmark = pytest.mark.core


class TestDataRoundTrip(unittest.TestCase):
    def _sample_result(self) -> Result:
        return Result(
            query=Query(text="What is Python?", qid="q1"),
            references=["doc-1", "doc-2"],
            answer=[
                CitedSentence(text="Python is a programming language.", citations=[0]),
                CitedSentence(text="It is used widely.", citations=[1]),
            ],
            rag_exec_summary=RAGExecInfo(
                prompt="prompt",
                response="response",
                input_token_count=12,
                output_token_count=7,
                reasoning="Used the provided passages.",
            ),
        )

    def test_jsonl_round_trip_preserves_output_record_shape(self) -> None:
        result = self._sample_result()
        expected = result_to_dict(result, "demo-run")
        expected.pop("reasoning_traces", None)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            DataWriter(result).write_in_jsonl_format(str(output_path), "demo-run")

            loaded = read_results_from_file(str(output_path))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(result_to_dict(loaded[0], "demo-run"), expected)

    def test_json_round_trip_preserves_output_record_shape(self) -> None:
        result = self._sample_result()
        expected = result_to_dict(result, "demo-run")
        expected.pop("reasoning_traces", None)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.json"
            DataWriter(result).write_in_json_format(str(output_path), "demo-run")

            loaded = read_results_from_file(str(output_path))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(result_to_dict(loaded[0], "demo-run"), expected)
