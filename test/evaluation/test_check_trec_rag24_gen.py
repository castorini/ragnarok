import json
import os
import tempfile
import unittest

from src.ragnarok.scripts.check_trec_rag24_gen import Errlog, check_rag_gen_run


class TestCheckRagGenRun(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up the temporary directory after tests
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def create_test_files(self, topic_content, run_content):
        topic_file = os.path.join(self.test_dir, "test_topics.txt")
        run_file = os.path.join(self.test_dir, "test_run.json")

        with open(topic_file, "w") as f:
            f.write(topic_content)

        with open(run_file, "w") as f:
            f.write(run_content)

        return topic_file, run_file

    def test_check_rag_gen_run(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-001",
                    "topic": "Test topic",
                    "references": ["msmarco_v2.1_doc_1_1#1_1"],
                    "response_length": 10,
                    "answer": [{"text": "Test answer.", "citations": [0]}],
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        self.assertEqual(log.error_count, 0)

    def test_check_rag_gen_run_with_topic_errors(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-002",  # Incorrect topic ID
                    "topic": "Wrong topic",
                    "references": ["invalid_reference"],
                    "response_length": 500,
                    "answer": [{"text": "Test answer." * 50, "citations": [1]}],
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        self.assertEqual(log.error_count, 1)

        # Read the error log file
        error_log_file = run_file + ".errlog"
        with open(error_log_file, "r") as f:
            error_log = f.readlines()
            # Check if Unkown topic error is present
            self.assertIn("ERROR Line 1: Unknown topic (2024-002)\n", error_log)
            # Check if Warning WARNING Line 1: No response returned for topic 2024-001 is present
            self.assertIn(
                "WARNING Line 1: No response returned for topic 2024-001\n", error_log
            )

    def test_check_rag_gen_run_missing_fields(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-001",
                    # Missing 'topic' field
                    "references": ["msmarco_v2.1_doc_1_1#1_1"],
                    "response_length": 10,
                    "answer": [{"text": "Test answer.", "citations": [0]}],
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        self.assertEqual(log.error_count, 1)

        error_log_file = run_file + ".errlog"
        with open(error_log_file, "r") as f:
            error_log = f.readlines()
            self.assertIn('ERROR Line 1: Entry is missing "topic" field.\n', error_log)

    def test_check_rag_gen_run_invalid_references(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-001",
                    "topic": "Test topic",
                    "references": ["invalid_reference_1", "invalid_reference_2"],
                    "response_length": 10,
                    "answer": [{"text": "Test answer.", "citations": [0, 1]}],
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        self.assertEqual(log.error_count, 2)

        error_log_file = run_file + ".errlog"
        with open(error_log_file, "r") as f:
            error_log = f.readlines()
            self.assertIn(
                "ERROR Line 1: Invalid reference docno invalid_reference_1\n", error_log
            )
            self.assertIn(
                "ERROR Line 1: Invalid reference docno invalid_reference_2\n", error_log
            )

    def test_check_rag_gen_run_response_length_mismatch(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-001",
                    "topic": "Test topic",
                    "references": ["msmarco_v2.1_doc_1_1#1_1"],
                    "response_length": 5,  # Incorrect length
                    "answer": [
                        {"text": "This is a longer test answer.", "citations": [0]}
                    ],
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        error_log_file = run_file + ".errlog"
        with open(error_log_file, "r") as f:
            error_log = f.readlines()
            self.assertIn(
                "WARNING Line 1: Reported RAG answer (5) is not equal to actual response length (6), maybe you did not NFCK normalize the text or strip characters?\n",
                error_log,
            )

    def test_check_rag_gen_run_too_many_references(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-001",
                    "topic": "Test topic",
                    "references": ["msmarco_v2.1_doc_1_1#1_1"] * 21,  # 21 references
                    "response_length": 10,
                    "answer": [{"text": "Test answer.", "citations": [0]}],
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        self.assertEqual(log.error_count, 21)

        error_log_file = run_file + ".errlog"
        with open(error_log_file, "r") as f:
            error_log = f.readlines()
            self.assertIn(
                "ERROR Line 1: Duplicate document msmarco_v2.1_doc_1_1#1_1 in references\n",
                error_log,
            )
            self.assertIn("ERROR Line 1: Too many references (max 20)\n", error_log)

    def test_check_rag_gen_run_long_answer_fixed(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-001",
                    "topic": "Test topic",
                    "references": ["msmarco_v2.1_doc_1_1#1_1"],
                    "response_length": 450,  # Over the 400 word limit
                    "answer": [{"text": "Test answer is the best. ", "citations": [0]}]
                    * 90,
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        # Check the fixed file
        fixed_file = run_file + ".fixed"
        with open(fixed_file, "r") as f:
            fixed_content = json.loads(f.read())
            self.assertLess(fixed_content["response_length"], 401)
            self.assertLess(
                len(
                    " ".join([sent["text"] for sent in fixed_content["answer"]]).split()
                ),
                401,
            )

        # Check the error log
        error_log_file = run_file + ".errlog"
        with open(error_log_file, "r") as f:
            error_log = f.readlines()
            self.assertIn(
                "WARNING Line 1: Reported response_length is too long\n", error_log
            )
            self.assertIn(
                "WARNING Line 1: Attempting to fix RAG answer of length 450\n",
                error_log,
            )

    def test_check_rag_gen_run_duplicate_citations_fixed(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-001",
                    "topic": "Test topic",
                    "references": [
                        "msmarco_v2.1_doc_1_1#1_1",
                        "msmarco_v2.1_doc_2_2#2_2",
                    ],
                    "response_length": 10,
                    "answer": [{"text": "Test answer.", "citations": [0, 0, 1, 1]}],
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        # Check the fixed file
        fixed_file = run_file + ".fixed"
        with open(fixed_file, "r") as f:
            fixed_content = json.loads(f.read())
            self.assertEqual(fixed_content["answer"][0]["citations"], [0, 1])

        # Check the error log
        error_log_file = run_file + ".errlog"
        with open(error_log_file, "r") as f:
            error_log = f.readlines()
            self.assertIn(
                "WARNING Line 1: Response sentence has duplicate citations\n", error_log
            )

    def test_check_rag_gen_run_out_of_bounds_citation_fixed(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-001",
                    "topic": "Test topic",
                    "references": ["msmarco_v2.1_doc_1_1#1_1"],
                    "response_length": 10,
                    "answer": [
                        {"text": "Test answer.", "citations": [0, 1]}
                    ],  # 1 is out of bounds
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        # Check the fixed file
        fixed_file = run_file + ".fixed"
        with open(fixed_file, "r") as f:
            fixed_content = json.loads(f.read())
            self.assertEqual(fixed_content["answer"][0]["citations"], [0])

        # Check the error log
        error_log_file = run_file + ".errlog"
        with open(error_log_file, "r") as f:
            error_log = f.readlines()
            self.assertIn(
                "WARNING Line 1: Response sentence has a citation that is out of bounds\n",
                error_log,
            )
            self.assertIn(
                "ERROR Line 1: Response sentence has invalid citation 1\n", error_log
            )

    def test_check_rag_gen_run_multiple_errors_fixed(self):
        topic_content = "2024-001\tTest topic\n"
        run_content = (
            json.dumps(
                {
                    "run_id": "test_run",
                    "topic_id": "2024-001",
                    "topic": "Test topic",
                    "references": [
                        "msmarco_v2.1_doc_1_1#1_1",
                        "msmarco_v2.1_doc_2_2#2_2",
                    ],
                    "response_length": 500,  # Over the 400 word limit
                    "answer": [
                        {"text": "Test answer is best.", "citations": [0, 0, 1, 2]}
                    ]
                    * 50
                    + [  # Duplicate and out of bounds citations
                        {"text": "More test answer is okay.", "citations": [1]}
                    ]
                    * 50,
                }
            )
            + "\n"
        )

        topic_file, run_file = self.create_test_files(topic_content, run_content)

        class Args:
            topicfile = topic_file
            runfile = run_file

        args = Args()

        with Errlog(run_file) as log:
            check_rag_gen_run(args, log)

        # Check the fixed file
        fixed_file = run_file + ".fixed"
        with open(fixed_file, "r") as f:
            fixed_content = json.loads(f.read())
            self.assertLess(fixed_content["response_length"], 401)
            self.assertEqual(fixed_content["answer"][0]["citations"], [0, 1])

        # Check the error log
        error_log_file = run_file + ".errlog"
        with open(error_log_file, "r") as f:
            error_log = f.readlines()
            self.assertIn(
                "WARNING Line 1: Reported response_length is too long\n", error_log
            )
            self.assertIn(
                "WARNING Line 1: Attempting to fix RAG answer of length 450\n",
                error_log,
            )
            self.assertIn(
                "WARNING Line 1: Response sentence has duplicate citations\n", error_log
            )
            self.assertIn(
                "WARNING Line 1: Response sentence has a citation that is out of bounds\n",
                error_log,
            )
            self.assertIn(
                "ERROR Line 1: Response sentence has invalid citation 2\n", error_log
            )

        # Test if fixed
        with open(fixed_file, "r") as f:
            fixed_content = json.loads(f.read())
            self.assertEqual(fixed_content["response_length"], 400)
            self.assertEqual(fixed_content["answer"][0]["citations"], [0, 1])
            self.assertEqual(len(fixed_content["answer"]), 90)


if __name__ == "__main__":
    unittest.main()
