from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence, cast

from ragnarok.api.runtime import ServerConfig, execute_direct_generate

from .adapters import make_data_artifact, make_file_artifact
from .config import load_config
from .introspection import (
    COMMAND_DESCRIPTIONS,
    SCHEMAS,
    doctor_report,
    validate_generate_batch_file,
    validate_generate_payload,
)
from .logging_utils import setup_logging
from .normalize import normalize_direct_generate_input
from .operations import (
    async_run_request_generation,
    convert_generate_records_to_requests,
    load_request_records,
    parse_retrieval_methods,
    parse_topk,
    run_convert_trec25,
    run_dataset_generation,
    run_request_generation,
    run_validate_rag24,
    run_validate_rag25,
)
from .prompt_view import (
    build_prompt_mode_view,
    build_rendered_prompt_view,
    list_prompt_modes,
    render_prompt_catalog_text,
    render_prompt_mode_text,
    render_rendered_prompt_text,
)
from .responses import CommandResponse
from .spec import EXIT_CODES, KNOWN_COMMANDS, TOP_LEVEL_EXAMPLES
from .view import (
    ViewError,
    build_view_summary,
    detect_artifact_type,
    load_records,
    render_view_summary,
)

_shtab: Any | None
try:
    import shtab as _shtab
except ModuleNotFoundError:  # optional dev dependency
    _shtab = None

shtab = cast(Any, _shtab)


class CLIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        exit_code: int,
        status: str,
        error_code: str,
        command: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        self.status = status
        self.error_code = error_code
        self.command = command or "unknown"
        self.details = details or {}


class CLIArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        if message == "the following arguments are required: command":
            raise CLIError(
                _build_missing_command_message(),
                exit_code=EXIT_CODES["invalid_arguments"],
                status="validation_error",
                error_code="missing_command",
                details={
                    "available_commands": list(KNOWN_COMMANDS),
                    "examples": list(TOP_LEVEL_EXAMPLES),
                    "help_hint": "Run `ragnarok --help` for full usage.",
                },
            )
        raise CLIError(
            message,
            exit_code=EXIT_CODES["invalid_arguments"],
            status="validation_error",
            error_code="invalid_arguments",
            command=_detect_command(sys.argv[1:]),
        )


def _detect_command(argv: Sequence[str]) -> str:
    for token in argv:
        if token in KNOWN_COMMANDS:
            return token
    return "unknown"


def _build_missing_command_message() -> str:
    command_list = ", ".join(KNOWN_COMMANDS)
    examples = "\n".join(f"  {example}" for example in TOP_LEVEL_EXAMPLES)
    return (
        "No command provided. Choose one of: "
        f"{command_list}\n"
        "Examples:\n"
        f"{examples}\n"
        "Run `ragnarok --help` for full usage."
    )


def _wants_json(argv: Sequence[str]) -> bool:
    for index, token in enumerate(argv):
        if token == "--output" and index + 1 < len(argv):
            return argv[index + 1] == "json"
    return False


def _emit_json(data: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(data) + "\n")


def _build_error_response(error: CLIError) -> CommandResponse:
    return CommandResponse(
        command=error.command,
        status=error.status,
        exit_code=error.exit_code,
        errors=[
            {
                "code": error.error_code,
                "message": error.message,
                "details": error.details,
                "retryable": False,
            }
        ],
    )


def _build_runtime_error_response(command: str, error: Exception) -> CommandResponse:
    return CommandResponse(
        command=command,
        status="runtime_error",
        exit_code=EXIT_CODES["runtime_error"],
        errors=[
            {
                "code": "runtime_error",
                "message": str(error),
                "details": {},
                "retryable": False,
            }
        ],
    )


def _ensure_file_exists(path: str, *, command: str, field_name: str) -> None:
    if not Path(path).exists():
        raise CLIError(
            f"{field_name} does not exist: {path}",
            exit_code=EXIT_CODES["missing_resource"],
            status="validation_error",
            error_code="missing_input",
            command=command,
            details={"field": field_name, "path": path},
        )


def _resolve_write_policy(args: argparse.Namespace) -> str:
    if getattr(args, "resume", False):
        return "resume"
    if getattr(args, "overwrite", False):
        return "overwrite"
    if getattr(args, "fail_if_exists", False):
        return "fail_if_exists"
    return "default_fail_if_exists"


def _prepare_output_path(args: argparse.Namespace, *, command: str) -> str:
    output_path = getattr(args, "output_file", None)
    if output_path is None:
        raise CLIError(
            f"{command} requires --output-file",
            exit_code=EXIT_CODES["invalid_arguments"],
            status="validation_error",
            error_code="missing_output_file",
            command=command,
        )
    output_file = Path(cast(str, output_path))
    write_policy = _resolve_write_policy(args)
    if output_file.exists():
        if write_policy == "resume":
            return str(output_file)
        if write_policy == "overwrite":
            output_file.write_text("", encoding="utf-8")
            return str(output_file)
        raise CLIError(
            f"Output file already exists: {output_file}",
            exit_code=EXIT_CODES["validation_error"],
            status="validation_error",
            error_code="write_policy_conflict",
            command=command,
            details={"path": str(output_file), "write_policy": write_policy},
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return str(output_file)


def _write_manifest(manifest_path: str | None, response: CommandResponse) -> None:
    if manifest_path is None:
        return
    Path(manifest_path).write_text(
        json.dumps(response.to_envelope(), indent=2) + "\n", encoding="utf-8"
    )


def _read_direct_payload(args: argparse.Namespace) -> dict[str, Any]:
    try:
        if args.stdin:
            return cast(dict[str, Any], json.loads(sys.stdin.read()))
        if args.input_json is not None:
            return cast(dict[str, Any], json.loads(args.input_json))
    except json.JSONDecodeError as exc:
        raise CLIError(
            "Input payload is not valid JSON",
            exit_code=EXIT_CODES["invalid_arguments"],
            status="validation_error",
            error_code="invalid_json",
            command=args.command,
            details={"error": str(exc)},
        ) from exc
    raise CLIError(
        "Direct input requires --stdin or --input-json",
        exit_code=EXIT_CODES["invalid_arguments"],
        status="validation_error",
        error_code="missing_direct_input",
        command=args.command,
    )


def _resolve_model_name(args: argparse.Namespace) -> str:
    model = getattr(args, "model", None)
    if model:
        args.model_path = model
        return cast(str, model)
    raise CLIError(
        "generate requires --model",
        exit_code=EXIT_CODES["invalid_arguments"],
        status="validation_error",
        error_code="missing_model",
        command="generate",
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
            "  ragnarok serve --model gpt-4o --prompt-mode chatqa --port 8084\n"
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
    generate_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model or deployment identifier to use for generation.",
    )
    generate_parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI environment settings for OpenAI-compatible generation.",
    )
    generate_parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter for OpenAI-compatible generation.",
    )
    generate_parser.add_argument(
        "--context-size",
        type=int,
        default=8192,
        help="Maximum context window passed to the generation backend.",
    )
    generate_parser.add_argument(
        "--topk",
        type=parse_topk,
        default=[100, 20],
        help="Comma-separated retrieval depths; for request input only the last value is used for answer generation.",
    )
    generate_parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for local open-weight generation backends.",
    )
    generate_parser.add_argument(
        "--retrieval-method",
        type=parse_retrieval_methods,
        help="Comma-separated retrieval or reranking stages for dataset-backed generation.",
    )
    generate_parser.add_argument(
        "--prompt-mode",
        required=True,
        help="Prompt template mode such as chatqa or cohere.",
    )
    generate_parser.add_argument(
        "--shuffle-candidates",
        action="store_true",
        help="Shuffle candidate passages before generation.",
    )
    generate_parser.add_argument(
        "--print-prompts-responses",
        action="store_true",
        help="Print rendered prompts and raw responses during execution.",
    )
    generate_parser.add_argument(
        "--num-few-shot-examples",
        type=int,
        default=0,
        help="Number of few-shot examples to inject into the prompt.",
    )
    generate_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1500,
        help="Maximum generated output tokens.",
    )
    generate_parser.add_argument(
        "--run-id",
        type=str,
        default="ragnarok",
        help="Run identifier stored in generated artifacts.",
    )
    generate_parser.add_argument(
        "--vllm-batched",
        action="store_true",
        help="Use vLLM batch generation where supported.",
    )
    generate_parser.add_argument(
        "--execution-mode",
        choices=["sync", "async"],
        default="sync",
        help="Execution mode for direct JSON input or batch JSONL input.",
    )
    generate_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent requests for async generation.",
    )
    generate_parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include backend reasoning or trace fields in sidecar artifacts where supported.",
    )
    generate_parser.add_argument(
        "--include-trace",
        action="store_true",
        help="Include generation trace fields in emitted results where available.",
    )
    generate_parser.add_argument(
        "--redact-prompts",
        action="store_true",
        help="Redact prompt content from emitted trace fields.",
    )
    generate_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for OpenAI-compatible generation backends that support it.",
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
    generate_parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
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
    serve_parser.add_argument("--port", type=int, default=8084)
    serve_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model or deployment identifier to use for generation.",
    )
    serve_parser.add_argument(
        "--prompt-mode",
        required=True,
        help="Prompt template mode such as chatqa or cohere.",
    )
    serve_parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI environment settings for OpenAI-compatible generation.",
    )
    serve_parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter for OpenAI-compatible generation.",
    )
    serve_parser.add_argument("--context-size", type=int, default=8192)
    serve_parser.add_argument(
        "--topk",
        type=parse_topk,
        default=[20],
        help="Comma-separated candidate depths used for direct request generation.",
    )
    serve_parser.add_argument("--num-gpus", type=int, default=1)
    serve_parser.add_argument("--shuffle-candidates", action="store_true")
    serve_parser.add_argument("--print-prompts-responses", action="store_true")
    serve_parser.add_argument("--num-few-shot-examples", type=int, default=0)
    serve_parser.add_argument("--max-output-tokens", type=int, default=1500)
    serve_parser.add_argument("--run-id", type=str, default="ragnarok")
    serve_parser.add_argument("--vllm-batched", action="store_true")
    serve_parser.add_argument(
        "--execution-mode", choices=["sync", "async"], default="sync"
    )
    serve_parser.add_argument("--max-concurrency", type=int, default=8)
    serve_parser.add_argument("--include-reasoning", action="store_true")
    serve_parser.add_argument("--include-trace", action="store_true")
    serve_parser.add_argument("--redact-prompts", action="store_true")
    serve_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    serve_parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
    )

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


def _run_generate_command(args: argparse.Namespace) -> CommandResponse:
    from ragnarok.generate.llm import PromptMode

    model_name = _resolve_model_name(args)
    args.prompt_mode = PromptMode(args.prompt_mode)
    if args.dataset is None and args.retrieval_method is None:
        args.retrieval_method = [parse_retrieval_methods("bm25")[0]]
    if args.output_file is not None and not args.dry_run and not args.validate_only:
        args.output_file = _prepare_output_path(args, command="generate")

    response = CommandResponse(
        command="generate",
        inputs={
            "dataset": args.dataset,
            "input_file": args.input_file,
            "stdin": args.stdin,
            "output_file": args.output_file,
        },
        resolved={
            "model": model_name,
            "model_path": model_name,
            "topk": args.topk,
            "prompt_mode": str(args.prompt_mode),
            "retrieval_method": [str(method) for method in args.retrieval_method or []],
            "run_id": args.run_id,
            "execution_mode": args.execution_mode,
            "max_concurrency": args.max_concurrency,
            "use_openrouter": args.use_openrouter,
        },
    )

    if args.dataset is not None:
        if args.execution_mode == "async":
            raise CLIError(
                "--execution-mode async is not yet supported with --dataset",
                exit_code=EXIT_CODES["invalid_arguments"],
                status="validation_error",
                error_code="unsupported_execution_mode",
                command="generate",
            )
        response.validation = {"valid": True, "mode": "dataset"}
        if args.dry_run or args.validate_only:
            response.mode = "validate" if args.validate_only else "dry-run"
            return response
        logger = setup_logging(args.log_level, quiet=getattr(args, "quiet", False))
        artifacts, metrics = run_dataset_generation(args, logger)
        if artifacts:
            response.artifacts.append(
                make_data_artifact("generation-results", artifacts)
            )
        response.metrics = metrics
        return response

    if args.input_file is not None:
        _ensure_file_exists(
            args.input_file, command="generate", field_name="input_file"
        )
        validation = validate_generate_batch_file(args.input_file)
        response.validation = validation
        if not validation["valid"]:
            raise CLIError(
                "Batch generate input file does not match the expected request shape",
                exit_code=EXIT_CODES["validation_error"],
                status="validation_error",
                error_code="invalid_input_file",
                command="generate",
            )
        if args.dry_run or args.validate_only:
            response.mode = "validate" if args.validate_only else "dry-run"
            return response
        requests = convert_generate_records_to_requests(
            load_request_records(args.input_file)
        )
        logger = setup_logging(args.log_level, quiet=getattr(args, "quiet", False))
        if args.execution_mode == "async":
            records, metrics = asyncio.run(
                async_run_request_generation(requests, args, logger)
            )
        else:
            records, metrics = run_request_generation(requests, args, logger)
        if args.output == "json":
            response.artifacts.append(make_data_artifact("generation-results", records))
        else:
            response.artifacts.append(
                make_file_artifact(
                    "generation-results-jsonl", cast(str, args.output_file)
                )
            )
        response.metrics = metrics
        return response

    payload = _read_direct_payload(args)
    request = normalize_direct_generate_input(payload)
    response.validation = {"valid": True, "record_count": 1}
    if args.dry_run or args.validate_only:
        response.mode = "validate" if args.validate_only else "dry-run"
        response.artifacts.append(
            make_data_artifact(
                "validated-request",
                {
                    "query": {"qid": request.query.qid, "text": request.query.text},
                    "candidate_count": len(request.candidates),
                },
            )
        )
        return response
    return execute_direct_generate(payload, args=args)


def _run_serve_command(args: argparse.Namespace) -> CommandResponse:
    try:
        import uvicorn

        from ragnarok.api.app import create_app
    except ModuleNotFoundError as error:
        raise CLIError(
            "serve requires FastAPI dependencies; install the `api` extra",
            exit_code=EXIT_CODES["missing_resource"],
            status="validation_error",
            error_code="missing_api_dependencies",
            command="serve",
            details={"missing_dependencies": ["fastapi", "uvicorn"]},
        ) from error

    app = create_app(
        ServerConfig(
            host=args.host,
            port=args.port,
            model=args.model,
            prompt_mode=args.prompt_mode,
            use_azure_openai=args.use_azure_openai,
            use_openrouter=args.use_openrouter,
            context_size=args.context_size,
            topk=args.topk,
            num_gpus=args.num_gpus,
            execution_mode=args.execution_mode,
            max_concurrency=args.max_concurrency,
            shuffle_candidates=args.shuffle_candidates,
            print_prompts_responses=args.print_prompts_responses,
            num_few_shot_examples=args.num_few_shot_examples,
            max_output_tokens=args.max_output_tokens,
            run_id=args.run_id,
            vllm_batched=args.vllm_batched,
            include_reasoning=args.include_reasoning,
            include_trace=args.include_trace,
            redact_prompts=args.redact_prompts,
            reasoning_effort=args.reasoning_effort,
            log_level=args.log_level,
            quiet=getattr(args, "quiet", False),
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return CommandResponse(
        command="serve",
        resolved={"host": args.host, "port": args.port},
    )


def _run_validate_command(args: argparse.Namespace) -> CommandResponse:
    response = CommandResponse(command="validate", mode="validate")
    if args.target == "generate":
        if args.input_file is not None:
            _ensure_file_exists(
                args.input_file, command="validate", field_name="input_file"
            )
            response.validation = validate_generate_batch_file(args.input_file)
        else:
            payload = _read_direct_payload(args)
            response.validation = validate_generate_payload(payload)
        return response
    if args.target == "rag24-output":
        if args.topicfile is None or args.runfile is None:
            raise CLIError(
                "rag24-output validation requires --topicfile and --runfile",
                exit_code=EXIT_CODES["invalid_arguments"],
                status="validation_error",
                error_code="missing_validation_target_args",
                command="validate",
            )
        _ensure_file_exists(args.topicfile, command="validate", field_name="topicfile")
        _ensure_file_exists(args.runfile, command="validate", field_name="runfile")
        summary, stdout = run_validate_rag24(args)
        response.validation = summary
        response.metrics = {"stdout": stdout}
        return response
    if args.input is None or args.topics is None:
        raise CLIError(
            "rag25-output validation requires --input and --topics",
            exit_code=EXIT_CODES["invalid_arguments"],
            status="validation_error",
            error_code="missing_validation_target_args",
            command="validate",
        )
    if args.input != "-":
        _ensure_file_exists(args.input, command="validate", field_name="input")
    _ensure_file_exists(args.topics, command="validate", field_name="topics")
    if args.apply_fixes and not (args.fix_length or args.fix_citations):
        args.fix_length = True
        args.fix_citations = True
    summary, stdout = run_validate_rag25(args)
    response.validation = summary
    response.metrics = {"stdout": stdout}
    return response


def _run_convert_command(args: argparse.Namespace) -> CommandResponse:
    _ensure_file_exists(args.input_file, command="convert", field_name="input_file")
    if args.prompt_file is not None:
        _ensure_file_exists(
            args.prompt_file, command="convert", field_name="prompt_file"
        )
    summary, stdout = run_convert_trec25(args)
    return CommandResponse(
        command="convert",
        artifacts=[make_file_artifact("converted-output", args.output_file)],
        metrics={"stdout": stdout, **summary},
    )


def _run_view_command(args: argparse.Namespace) -> CommandResponse:
    try:
        records = load_records(args.path)
        artifact_type = detect_artifact_type(records, args.artifact_type)
    except ViewError as error:
        raise CLIError(
            str(error),
            exit_code=EXIT_CODES["validation_error"],
            status="validation_error",
            error_code="invalid_view_input",
            command="view",
            details={"path": args.path, "artifact_type": args.artifact_type},
        ) from error

    view_summary = build_view_summary(
        args.path, records, artifact_type, record_limit=args.records
    )
    return CommandResponse(
        command="view",
        mode="inspect",
        inputs={"path": args.path},
        resolved={
            "artifact_type": artifact_type,
            "records": args.records,
            "color": args.color,
        },
        artifacts=[make_data_artifact("view-summary", view_summary)],
        metrics=view_summary["summary"],
    )


def _run_prompt_command(args: argparse.Namespace) -> CommandResponse:
    if args.prompt_command == "list":
        return CommandResponse(
            command="prompt",
            mode="inspect",
            artifacts=[make_data_artifact("prompt-catalog", list_prompt_modes())],
        )

    from ragnarok.generate.llm import PromptMode

    prompt_mode = PromptMode(args.prompt_mode)
    return CommandResponse(
        command="prompt",
        mode="inspect",
        inputs={"prompt_mode": args.prompt_mode},
        resolved={"prompt_command": "show"},
        artifacts=[
            make_data_artifact("prompt-template", build_prompt_mode_view(prompt_mode))
        ],
    )


def _run_prompt_render_command(args: argparse.Namespace) -> CommandResponse:
    from ragnarok.generate.llm import PromptMode
    from ragnarok.generate.templates.ragnarok_templates import RagnarokTemplates

    prompt_mode = PromptMode(args.prompt_mode)
    if prompt_mode in {PromptMode.UNSPECIFIED, PromptMode.COHERE}:
        raise CLIError(
            f"prompt render does not support prompt mode {args.prompt_mode}",
            exit_code=EXIT_CODES["validation_error"],
            status="validation_error",
            error_code="unsupported_prompt_mode",
            command="prompt",
        )
    payload = _read_direct_payload(args)
    request = normalize_direct_generate_input(payload)
    topk = max(1, min(args.topk, len(request.candidates)))
    context = [
        f"[{index}] {candidate.doc['segment']}"
        for index, candidate in enumerate(request.candidates[:topk], start=1)
    ]
    rendered = RagnarokTemplates(prompt_mode).render(
        request.query.text,
        context,
        args.model,
    )
    return CommandResponse(
        command="prompt",
        mode="inspect",
        inputs={"prompt_mode": args.prompt_mode, "model": args.model},
        resolved={"prompt_command": "render", "topk": topk, "part": args.part},
        artifacts=[
            make_data_artifact(
                "rendered-prompt",
                build_rendered_prompt_view(
                    rendered,
                    query=request.query.text,
                    context_count=len(context),
                    topk=topk,
                ),
            )
        ],
    )


def _format_text_response(response: CommandResponse) -> str:
    if response.command == "generate":
        if not response.artifacts:
            return ""
        data = response.artifacts[0].get("data", [])
        if not data:
            return ""
        rendered_records: list[str] = []
        for record in data:
            answer_lines: list[str] = []
            for answer in record.get("answer", []):
                text = answer.get("text", "")
                if not text:
                    continue
                citations = answer.get("citations", [])
                citation_suffix = ""
                if citations:
                    human_citations = ",".join(
                        str(int(citation) + 1) for citation in citations
                    )
                    citation_suffix = f" [{human_citations}]"
                answer_lines.append(f"{text}{citation_suffix}")
            lines = [
                f"query: {record.get('topic', '')}",
                f"answer: {' '.join(answer_lines).strip()}",
                f"references: [{', '.join(str(reference) for reference in record.get('references', []))}]",
            ]
            reasoning_traces = [
                trace.strip()
                for trace in record.get("reasoning_traces", [])
                if isinstance(trace, str) and trace.strip()
            ]
            if reasoning_traces:
                lines.append(f"reasoning: {reasoning_traces[0]}")
            rendered_records.append("\n".join(lines).rstrip())
        return "\n\n".join(rendered_records).rstrip()
    if response.command in ("describe", "schema"):
        return json.dumps(response.artifacts[0]["data"], indent=2)
    if response.command == "doctor":
        return json.dumps(response.metrics, indent=2)
    if response.command == "validate":
        return json.dumps(response.validation, indent=2)
    if response.command == "view":
        return render_view_summary(
            cast(dict[str, Any], response.artifacts[0]["data"]),
            color=cast(str, response.resolved["color"]),
        )
    if response.command == "prompt":
        if response.artifacts[0]["name"] == "prompt-catalog":
            return render_prompt_catalog_text(
                cast(list[dict[str, Any]], response.artifacts[0]["data"])
            )
        if response.artifacts[0]["name"] == "rendered-prompt":
            return render_rendered_prompt_text(
                cast(dict[str, Any], response.artifacts[0]["data"]),
                part=cast(str, response.resolved["part"]),
            )
        return render_prompt_mode_text(
            cast(dict[str, Any], response.artifacts[0]["data"])
        )
    if response.command == "serve":
        return ""
    return json.dumps(response.to_envelope(), indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    config, config_path = load_config()
    wants_json = _wants_json(argv)
    try:
        args = parser.parse_args(argv)
        args._config_path = config_path
        for key, value in config.items():
            flag = f"--{key.replace('_', '-')}"
            if flag not in argv:
                setattr(args, key, value)
        if args.command == "generate":
            response = _run_generate_command(args)
        elif args.command == "serve":
            response = _run_serve_command(args)
        elif args.command == "validate":
            response = _run_validate_command(args)
        elif args.command == "convert":
            response = _run_convert_command(args)
        elif args.command == "view":
            response = _run_view_command(args)
        elif args.command == "prompt":
            if args.prompt_command == "render":
                response = _run_prompt_render_command(args)
            else:
                response = _run_prompt_command(args)
        elif args.command == "describe":
            response = CommandResponse(
                command="describe",
                mode="inspect",
                artifacts=[
                    make_data_artifact(args.target, COMMAND_DESCRIPTIONS[args.target])
                ],
                inputs={"target": args.target},
                resolved={"target_command": args.target},
            )
        elif args.command == "schema":
            response = CommandResponse(
                command="schema",
                mode="inspect",
                artifacts=[make_data_artifact(args.target, SCHEMAS[args.target])],
                inputs={"target": args.target},
                resolved={"schema": args.target},
            )
        else:
            report = doctor_report()
            report["config_file"] = str(config_path) if config_path else None
            response = CommandResponse(
                command="doctor",
                mode="inspect",
                metrics=report,
                validation={"python_ok": report["python_ok"]},
            )

        _write_manifest(getattr(args, "manifest_path", None), response)
        if getattr(args, "output", "text") == "json":
            _emit_json(response.to_envelope())
        else:
            text = _format_text_response(response)
            if text:
                sys.stdout.write(text + "\n")
        return response.exit_code
    except CLIError as error:
        response = _build_error_response(error)
        if wants_json:
            _emit_json(response.to_envelope())
        else:
            sys.stderr.write(error.message + "\n")
        return error.exit_code
    except Exception as error:
        command = _detect_command(argv)
        response = _build_runtime_error_response(command, error)
        if wants_json:
            _emit_json(response.to_envelope())
        else:
            sys.stderr.write(f"{error}\n")
        return response.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
