from __future__ import annotations

import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO

from ragnarok.cli.main import main


@unittest.skipUnless(
    os.getenv("RAGNAROK_LIVE_OPENAI_SMOKE") == "1",
    "Set RAGNAROK_LIVE_OPENAI_SMOKE=1 to run live OpenAI smoke tests.",
)
class RagnarokLiveOpenAISmokeTests(unittest.TestCase):
    def test_generate_direct_openai_smoke(self) -> None:
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
            self.skipTest("OPENAI_API_KEY or OPENROUTER_API_KEY is required.")

        model = os.getenv("RAGNAROK_LIVE_OPENAI_MODEL", "gpt-4o-mini")
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    model,
                    "--input-json",
                    json.dumps(
                        {
                            "query": "how long is life cycle of flea",
                            "candidates": [
                                "The life cycle of a flea can last anywhere from 20 days to an entire year."
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
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "generate")
        self.assertEqual(payload["status"], "success")
        results = payload["artifacts"][0]["data"]
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["answer"])
