from __future__ import annotations

import argparse
import importlib.metadata
from typing import Any, cast

from .errors import CLIArgumentParser
from .introspection import COMMAND_DESCRIPTIONS, SCHEMAS
from .operations import parse_retrieval_methods, parse_topk

_shtab: Any | None
try:
    import shtab as _shtab
except ModuleNotFoundError:  # optional dev dependency
    _shtab = None

shtab = cast(Any, _shtab)


def _add_shared_runtime_generation_options(
    parser: argparse.ArgumentParser,
    *,
    topk_default: list[int],
) -> None:
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model or deployment identifier to use for generation.",
    )
    parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI environment settings for OpenAI-compatible generation.",
    )
    parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter for OpenAI-compatible generation.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=8192,
        help="Maximum context window passed to the generation backend.",
    )
    parser.add_argument(
        "--topk",
        type=parse_topk,
        default=topk_default,
        help="Comma-separated retrieval depths; for request input only the last value is used for answer generation.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for local open-weight generation backends.",
    )
    parser.add_argument(
        "--prompt-mode",
        required=True,
        help="Prompt template mode such as chatqa or cohere.",
    )
    parser.add_argument(
        "--shuffle-candidates",
        action="store_true",
        help="Shuffle candidate passages before generation.",
    )
    parser.add_argument(
        "--print-prompts-responses",
        action="store_true",
        help="Print rendered prompts and raw responses during execution.",
    )
    parser.add_argument(
        "--num-few-shot-examples",
        type=int,
        default=0,
        help="Number of few-shot examples to inject into the prompt.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1500,
        help="Maximum generated output tokens.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="ragnarok",
        help="Run identifier stored in generated artifacts.",
    )
    parser.add_argument(
        "--vllm-batched",
        action="store_true",
        help="Use vLLM batch generation where supported.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["sync", "async"],
        default="sync",
        help="Execution mode for direct JSON input or batch JSONL input.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent requests for async generation.",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include backend reasoning or trace fields in sidecar artifacts where supported.",
    )
    parser.add_argument(
        "--include-trace",
        action="store_true",
        help="Include generation trace fields in emitted results where available.",
    )
    parser.add_argument(
        "--redact-prompts",
        action="store_true",
        help="Redact prompt content from emitted trace fields.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for OpenAI-compatible generation backends that support it.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
    )


def build_parser() -> CLIArgumentParser:
    parser = CLIArgumentParser(
        prog="ragnarok",
        description=(
            "Ragnarok packaged CLI for generation, validation, conversion, and "
            "artifact inspection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Common patterns:\n"
            "  ragnarok generate --model gpt-4o --input-json "
            '\'{"query":"q","candidates":["p"]}\' --prompt-mode chatqa --output json\n'
            "  ragnarok serve --model gpt-4o --prompt-mode chatqa --port 8083\n"
            "  ragnarok prompt show --prompt-mode chatqa\n"
            "  ragnarok validate generate --input-json "
            '\'{"query":"q","candidates":["p"]}\' --output json\n'
            "  ragnarok doctor --output json"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('pyragnarok')}",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress all log output (sets log level to CRITICAL).",
    )
    if shtab is not None:
        shtab.add_argument_to(parser, ["--print-completion"])
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=CLIArgumentParser
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run generation from direct JSON input, batch JSONL, or a dataset.",
        description=(
            "Generate RAG answers from direct JSON input, batch JSONL request files, "
            "or dataset-backed retrieval."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ragnarok generate --model gpt-4o --input-json "
            '\'{"query":"q","candidates":["p"]}\' --prompt-mode chatqa --output json\n'
            "  ragnarok generate --model gpt-4o --input-file requests.jsonl "
            "--output-file results.jsonl --prompt-mode chatqa --execution-mode async"
        ),
    )
    generate_inputs = generate_parser.add_mutually_exclusive_group(required=True)
    generate_inputs.add_argument(
        "--dataset",
        type=str,
        help="Dataset identifier for dataset-backed retrieval and generation.",
    )
    generate_inputs.add_argument(
        "--input-file",
        type=str,
        help="Batch JSON or JSONL request file in the shared query-candidate schema.",
    )
    generate_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON request from standard input.",
    )
    generate_inputs.add_argument(
        "--input-json",
        type=str,
        help="Direct JSON request in the shared query-candidate schema.",
    )
    _add_shared_runtime_generation_options(generate_parser, topk_default=[100, 20])
    generate_parser.add_argument(
        "--retrieval-method",
        type=parse_retrieval_methods,
        help="Comma-separated retrieval or reranking stages for dataset-backed generation.",
    )
    generate_parser.add_argument(
        "--output",
        choices=["text", "json", "jsonl"],
        default="text",
        help="Human-readable text, machine-readable JSON envelope, or generated JSONL records.",
    )
    generate_parser.add_argument(
        "--output-file",
        type=str,
        help="Output JSONL path for batch or dataset generation.",
    )
    generate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the command and validate inputs without running models.",
    )
    generate_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the declared contract without running models.",
    )
    generate_parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow writing to an existing output file without truncating it.",
    )
    generate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate an existing output file before writing results.",
    )
    generate_parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Fail if the target output path already exists.",
    )
    generate_parser.add_argument(
        "--manifest-path",
        type=str,
        help="Write the final JSON envelope to a manifest file.",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a FastAPI server for direct Ragnarok generation requests.",
        description=(
            "Start a FastAPI server that exposes direct-input Ragnarok generation "
            "over HTTP."
        ),
    )
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8083)
    _add_shared_runtime_generation_options(serve_parser, topk_default=[20])

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate request payloads or existing output artifacts without running models.",
        description="Validate direct JSON input, batch JSONL requests, or TREC output artifacts without running models.",
    )
    validate_parser.add_argument(
        "target",
        choices=["generate", "rag24-output", "rag25-output"],
        help="Validation target to inspect.",
    )
    validate_inputs = validate_parser.add_mutually_exclusive_group()
    validate_inputs.add_argument(
        "--input-file", type=str, help="Batch JSON or JSONL request file to validate."
    )
    validate_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON payload from standard input.",
    )
    validate_inputs.add_argument(
        "--input-json", type=str, help="Direct JSON payload to validate."
    )
    validate_parser.add_argument(
        "--topicfile",
        type=str,
        help="TREC RAG 2024 topics TSV used for output validation.",
    )
    validate_parser.add_argument(
        "--runfile", type=str, help="TREC RAG 2024 run JSONL to validate."
    )
    validate_parser.add_argument(
        "--input",
        type=str,
        help="TREC RAG 2025 output JSONL to validate, or '-' for stdin.",
    )
    validate_parser.add_argument(
        "--topics", type=str, help="TREC RAG 2025 topic JSONL used for validation."
    )
    validate_parser.add_argument(
        "--format",
        type=int,
        choices=[1, 2],
        default=1,
        help="Citation format for RAG 2025 output validation.",
    )
    validate_parser.add_argument(
        "--apply-fixes",
        action="store_true",
        help="Allow the RAG 2025 validator to write a .fixed artifact when repairable issues are found.",
    )
    validate_parser.add_argument(
        "--fix-length",
        action="store_true",
        default=False,
        help="Trim overly long RAG 2025 answers during validation.",
    )
    validate_parser.add_argument(
        "--fix-citations",
        action="store_true",
        default=False,
        help="Repair duplicate or out-of-range RAG 2025 citations during validation.",
    )
    validate_parser.add_argument(
        "--verbose", action="store_true", help="Print detailed validator progress."
    )
    validate_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable validation summary or JSON envelope.",
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert older Ragnarok artifacts into the newer TREC 2025 output format.",
        description="Convert older Ragnarok JSONL artifacts into the newer TREC 2025 output format.",
    )
    convert_parser.add_argument(
        "target", choices=["trec25-format"], help="Conversion target to produce."
    )
    convert_parser.add_argument(
        "--input-file", required=True, type=str, help="Input JSONL file to convert."
    )
    convert_parser.add_argument(
        "--output-file", required=True, type=str, help="Output JSONL file to write."
    )
    convert_parser.add_argument(
        "--prompt-file",
        type=str,
        help="Optional prompt sidecar JSONL to merge into the converted output.",
    )
    convert_parser.add_argument(
        "--verbose", action="store_true", help="Print detailed conversion progress."
    )
    convert_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable summary or JSON envelope.",
    )

    describe_parser = subparsers.add_parser(
        "describe",
        help="Inspect structured metadata for a public Ragnarok command.",
    )
    describe_parser.add_argument(
        "target",
        choices=sorted(COMMAND_DESCRIPTIONS),
        help="Public command to describe.",
    )
    describe_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable description or JSON envelope.",
    )

    schema_parser = subparsers.add_parser(
        "schema",
        help="Print JSON schemas for supported inputs, outputs, and envelopes.",
    )
    schema_parser.add_argument(
        "target", choices=sorted(SCHEMAS), help="Schema artifact to print."
    )
    schema_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable schema or JSON envelope.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Report environment, dependency, and backend readiness for the packaged CLI.",
    )
    doctor_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable readiness report or JSON envelope.",
    )

    view_parser = subparsers.add_parser(
        "view",
        help="Inspect an existing generation artifact.",
        description="Inspect an existing Ragnarok generation artifact and render a stable summary.",
    )
    view_parser.add_argument("path", type=str, help="Artifact path to inspect.")
    view_parser.add_argument(
        "--type",
        dest="artifact_type",
        type=str,
        help="Explicit artifact type when automatic detection is ambiguous.",
    )
    view_parser.add_argument(
        "--records",
        type=int,
        default=3,
        help="Number of records to sample in the inspection summary.",
    )
    view_parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Color policy for text-mode rendering.",
    )
    view_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable summary or JSON envelope.",
    )

    prompt_parser = subparsers.add_parser(
        "prompt",
        help="Inspect Ragnarok prompt-mode definitions.",
        description="Inspect Ragnarok prompt-mode definitions.",
    )
    prompt_subparsers = prompt_parser.add_subparsers(
        dest="prompt_command", required=True, parser_class=CLIArgumentParser
    )

    prompt_list_parser = prompt_subparsers.add_parser(
        "list",
        help="List supported prompt modes.",
    )
    prompt_list_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable catalog or JSON envelope.",
    )

    prompt_show_parser = prompt_subparsers.add_parser(
        "show",
        help="Show one prompt mode definition.",
    )
    prompt_show_parser.add_argument(
        "--prompt-mode",
        required=True,
        help="Prompt mode to inspect.",
    )
    prompt_show_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable prompt definition or JSON envelope.",
    )

    prompt_render_parser = prompt_subparsers.add_parser(
        "render",
        help="Render one prompt mode against direct input.",
    )
    prompt_render_inputs = prompt_render_parser.add_mutually_exclusive_group(
        required=True
    )
    prompt_render_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON payload from standard input.",
    )
    prompt_render_inputs.add_argument(
        "--input-json",
        type=str,
        help="Direct JSON request in the shared query-candidate schema.",
    )
    prompt_render_parser.add_argument(
        "--prompt-mode",
        required=True,
        help="Prompt mode to render.",
    )
    prompt_render_parser.add_argument(
        "--model",
        required=True,
        help="Model identifier used to choose the runtime prompt shape.",
    )
    prompt_render_parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Maximum number of input candidates to include in the rendered prompt.",
    )
    prompt_render_parser.add_argument(
        "--part",
        choices=["system", "user", "all"],
        default="all",
        help="Rendered prompt section to show in text mode.",
    )
    prompt_render_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable rendered prompt or JSON envelope.",
    )
    return parser
