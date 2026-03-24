from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from ragnarok.cli.main import main

pytestmark = pytest.mark.integration


class RagnarokCLIIntegrationRegressions(unittest.TestCase):
    def test_convert_validate_and_view_flows_work_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            legacy_input = temp_path / "legacy.jsonl"
            converted_output = temp_path / "converted.jsonl"
            rag24_topics = temp_path / "rag24_topics.txt"
            rag24_run = temp_path / "rag24_run.jsonl"

            legacy_input.write_text(
                json.dumps(
                    {
                        "run_id": "demo-run",
                        "topic_id": "2025-001",
                        "topic": "What is Python used for?",
                        "references": ["msmarco_v2.1_doc_1_1#1_1"],
                        "answer": [
                            {
                                "text": "Python is used for web development.",
                                "citations": [0],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rag24_topics.write_text("2024-001\tTest topic\n", encoding="utf-8")
            rag24_run.write_text(
                json.dumps(
                    {
                        "run_id": "demo-run",
                        "topic_id": "2024-001",
                        "topic": "Test topic",
                        "references": ["msmarco_v2.1_doc_1_1#1_1"],
                        "response_length": 4,
                        "answer": [{"text": "Test answer.", "citations": [0]}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            convert_stdout = StringIO()
            with redirect_stdout(convert_stdout):
                convert_exit = main(
                    [
                        "convert",
                        "trec25-format",
                        "--input-file",
                        str(legacy_input),
                        "--output-file",
                        str(converted_output),
                    ]
                )
            self.assertEqual(convert_exit, 0)

            converted_records = [
                json.loads(line)
                for line in converted_output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(converted_records), 1)
            self.assertEqual(converted_records[0]["metadata"]["run_id"], "demo-run")
            self.assertIn("narrative_id", converted_records[0]["metadata"])
            self.assertIn(
                "Successfully converted: 1 records", convert_stdout.getvalue()
            )

            validate_stdout = StringIO()
            with redirect_stdout(validate_stdout):
                validate_exit = main(
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
                )
            self.assertEqual(validate_exit, 0)
            validation = json.loads(validate_stdout.getvalue())
            self.assertEqual(validation["command"], "validate")
            self.assertEqual(validation["status"], "success")
            self.assertEqual(validation["validation"]["error_count"], 0)

            view_stdout = StringIO()
            with redirect_stdout(view_stdout):
                view_exit = main(["view", str(rag24_run), "--records", "1"])
            self.assertEqual(view_exit, 0)
            rendered = view_stdout.getvalue()
            self.assertIn("Ragnarok View", rendered)
            self.assertIn("topic_id=2024-001", rendered)
