from __future__ import annotations

import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from textwrap import indent

from ragnarok.cli.main import main


@unittest.skipUnless(
    os.getenv("RAGNAROK_LIVE_OPENAI_SMOKE") == "1",
    "Set RAGNAROK_LIVE_OPENAI_SMOKE=1 to run live OpenAI smoke tests.",
)
class RagnarokLiveOpenAISmokeTests(unittest.TestCase):
    def _pretty_render(self, payload: dict[str, object]) -> str:
        result = payload["artifacts"][0]["data"][0]
        lines = [
            "Ragnarok live smoke result",
            f"model: {payload['resolved']['model']}",
            f"query: {result['topic']}",
            "answer:",
        ]
        for index, sentence in enumerate(result["answer"], start=1):
            lines.append(
                f"  {index}. {sentence['text']} citations={sentence.get('citations', [])}"
            )
        lines.extend(
            [
                f"references: {result['references']}",
            ]
        )
        reasoning_traces = result.get("reasoning_traces") or []
        if reasoning_traces:
            lines.append("reasoning:")
            for trace in reasoning_traces:
                lines.append(indent(str(trace), "  "))
        trace = result.get("trace")
        if trace:
            lines.extend(
                [
                    "trace:",
                    indent(json.dumps(trace, indent=2), "  "),
                ]
            )
        return "\n".join(lines)

    def test_direct_generate_openai_smoke(self) -> None:
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
        print(self._pretty_render(payload))

    def test_direct_generate_openai_reasoning_smoke(self) -> None:
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
            self.skipTest("OPENAI_API_KEY or OPENROUTER_API_KEY is required.")

        model = os.getenv("RAGNAROK_LIVE_OPENAI_REASONING_MODEL", "gpt-5-mini")
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "generate",
                    "--model",
                    model,
                    "--include-reasoning",
                    "--reasoning-effort",
                    "medium",
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
        self.assertTrue(results[0].get("reasoning_traces"))
        print(self._pretty_render(payload))
