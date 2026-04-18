from __future__ import annotations

import unittest

import pytest

from ragnarok.scripts.rag25_validation import (
    compute_response_length,
    fix_citations,
)
from ragnarok.scripts.trec25_conversion import convert_record

pytestmark = pytest.mark.core


class TestTrecScriptHelpers(unittest.TestCase):
    def test_compute_response_length_normalizes_and_counts_tokens(self) -> None:
        entry = {
            "answer": [
                {"text": "Cafe\u0301 au lait", "citations": []},
                {"text": "Two  spaces\nkept", "citations": []},
            ]
        }

        self.assertEqual(compute_response_length(entry), 6)

    def test_fix_citations_trims_duplicate_and_out_of_range_index_citations(
        self,
    ) -> None:
        entry = {
            "references": [
                "msmarco_v2.1_doc_1_1#1_1",
                "msmarco_v2.1_doc_1_1#1_1",
            ]
            + [f"msmarco_v2.1_doc_2_{i}#2_{i}" for i in range(2, 105)],
            "answer": [{"text": "Answer.", "citations": [0, 1, 100, 101]}],
        }

        fixed_entry, warnings = fix_citations(entry, count=1, format_type=1)

        self.assertEqual(len(fixed_entry["references"]), 100)
        self.assertEqual(fixed_entry["answer"][0]["citations"], [0, 1])
        self.assertTrue(any("duplicate reference" in warning for warning in warnings))
        self.assertTrue(any("out-of-range" in warning for warning in warnings))

    def test_convert_record_preserves_existing_prompt_enrichment_behavior(self) -> None:
        record = {
            "run_id": "run-1",
            "topic_id": "42",
            "topic": "What is Python?",
            "references": ["msmarco_v2.1_doc_1_1#1_1"],
            "answer": [{"text": "A language.", "citations": [0]}],
        }

        converted = convert_record(record, prompts={"42": "Prompt text"})

        self.assertEqual(converted["metadata"]["prompt"], "Prompt text")
        self.assertEqual(converted["metadata"]["narrative_id"], "42")
        self.assertEqual(converted["answer"], record["answer"])
