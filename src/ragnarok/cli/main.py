from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence, cast

from ragnarok.generate.llm import PromptMode

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
from .responses import CommandResponse
from .spec import EXIT_CODES, KNOWN_COMMANDS, TOP_LEVEL_EXAMPLES
from .view import (
    ViewError,
    build_view_summary,
    detect_artifact_type,
    load_records,
    render_view_summary,
)


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


def build_parser() -> CLIArgumentParser:
    parser = CLIArgumentParser(prog="ragnarok")
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=CLIArgumentParser
    )

    generate_parser = subparsers.add_parser("generate")
    generate_inputs = generate_parser.add_mutually_exclusive_group(required=True)
    generate_inputs.add_argument("--dataset", type=str)
    generate_inputs.add_argument("--input-file", type=str)
    generate_inputs.add_argument("--stdin", action="store_true")
    generate_inputs.add_argument("--input-json", type=str)
    generate_parser.add_argument("--model-path", type=str, required=True)
    generate_parser.add_argument("--use-azure-openai", action="store_true")
    generate_parser.add_argument("--context-size", type=int, default=8192)
    generate_parser.add_argument("--topk", type=parse_topk, default=[100, 20])
    generate_parser.add_argument("--num-gpus", type=int, default=1)
    generate_parser.add_argument("--retrieval-method", type=parse_retrieval_methods)
    generate_parser.add_argument("--prompt-mode", required=True)
    generate_parser.add_argument("--shuffle-candidates", action="store_true")
    generate_parser.add_argument("--print-prompts-responses", action="store_true")
    generate_parser.add_argument("--num-few-shot-examples", type=int, default=0)
    generate_parser.add_argument("--max-output-tokens", type=int, default=1500)
    generate_parser.add_argument("--run-id", type=str, default="ragnarok")
    generate_parser.add_argument("--vllm-batched", action="store_true")
    generate_parser.add_argument(
        "--execution-mode", choices=["sync", "async"], default="sync"
    )
    generate_parser.add_argument("--max-concurrency", type=int, default=8)
    generate_parser.add_argument("--include-reasoning", action="store_true")
    generate_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    generate_parser.add_argument(
        "--output", choices=["text", "json", "jsonl"], default="text"
    )
    generate_parser.add_argument("--output-file", type=str)
    generate_parser.add_argument("--dry-run", action="store_true")
    generate_parser.add_argument("--validate-only", action="store_true")
    generate_parser.add_argument("--resume", action="store_true")
    generate_parser.add_argument("--overwrite", action="store_true")
    generate_parser.add_argument("--fail-if-exists", action="store_true")
    generate_parser.add_argument("--manifest-path", type=str)
    generate_parser.add_argument("--log-level", type=int, default=0, choices=[0, 1, 2])

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "target", choices=["generate", "rag24-output", "rag25-output"]
    )
    validate_inputs = validate_parser.add_mutually_exclusive_group()
    validate_inputs.add_argument("--input-file", type=str)
    validate_inputs.add_argument("--stdin", action="store_true")
    validate_inputs.add_argument("--input-json", type=str)
    validate_parser.add_argument("--topicfile", type=str)
    validate_parser.add_argument("--runfile", type=str)
    validate_parser.add_argument("--input", type=str)
    validate_parser.add_argument("--topics", type=str)
    validate_parser.add_argument("--format", type=int, choices=[1, 2], default=1)
    validate_parser.add_argument("--fix-length", action="store_true", default=True)
    validate_parser.add_argument("--fix-citations", action="store_true", default=True)
    validate_parser.add_argument("--verbose", action="store_true")
    validate_parser.add_argument("--output", choices=["text", "json"], default="text")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("target", choices=["trec25-format"])
    convert_parser.add_argument("--input-file", required=True, type=str)
    convert_parser.add_argument("--output-file", required=True, type=str)
    convert_parser.add_argument("--prompt-file", type=str)
    convert_parser.add_argument("--verbose", action="store_true")
    convert_parser.add_argument("--output", choices=["text", "json"], default="text")

    describe_parser = subparsers.add_parser("describe")
    describe_parser.add_argument("target", choices=sorted(COMMAND_DESCRIPTIONS))
    describe_parser.add_argument("--output", choices=["text", "json"], default="text")

    schema_parser = subparsers.add_parser("schema")
    schema_parser.add_argument("target", choices=sorted(SCHEMAS))
    schema_parser.add_argument("--output", choices=["text", "json"], default="text")

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--output", choices=["text", "json"], default="text")

    view_parser = subparsers.add_parser("view")
    view_parser.add_argument("path", type=str)
    view_parser.add_argument("--type", dest="artifact_type", type=str)
    view_parser.add_argument("--records", type=int, default=3)
    view_parser.add_argument(
        "--color", choices=["auto", "always", "never"], default="auto"
    )
    view_parser.add_argument("--output", choices=["text", "json"], default="text")

    return parser


def _run_generate_command(args: argparse.Namespace) -> CommandResponse:
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
            "model_path": args.model_path,
            "topk": args.topk,
            "prompt_mode": str(args.prompt_mode),
            "retrieval_method": [str(method) for method in args.retrieval_method or []],
            "run_id": args.run_id,
            "execution_mode": args.execution_mode,
            "max_concurrency": args.max_concurrency,
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
        logger = setup_logging(args.log_level)
        artifacts, metrics = run_dataset_generation(args, logger)
        if artifacts:
            response.artifacts.append({"kind": "data", "data": artifacts})
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
        logger = setup_logging(args.log_level)
        if args.execution_mode == "async":
            records, metrics = asyncio.run(
                async_run_request_generation(requests, args, logger)
            )
        else:
            records, metrics = run_request_generation(requests, args, logger)
        if args.output == "json":
            response.artifacts.append({"kind": "data", "data": records})
        else:
            response.artifacts.append({"kind": "file", "path": args.output_file})
        response.metrics = metrics
        return response

    payload = _read_direct_payload(args)
    request = normalize_direct_generate_input(payload)
    response.validation = {"valid": True, "record_count": 1}
    if args.dry_run or args.validate_only:
        response.mode = "validate" if args.validate_only else "dry-run"
        response.artifacts.append(
            {
                "kind": "data",
                "data": {
                    "query": {"qid": request.query.qid, "text": request.query.text},
                    "candidate_count": len(request.candidates),
                },
            }
        )
        return response
    logger = setup_logging(args.log_level)
    if args.execution_mode == "async":
        records, metrics = asyncio.run(
            async_run_request_generation([request], args, logger)
        )
    else:
        records, metrics = run_request_generation([request], args, logger)
    response.metrics = metrics
    response.artifacts.append({"kind": "data", "data": records})
    return response


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
        artifacts=[{"kind": "file", "path": args.output_file}],
        metrics={"stdout": stdout, **summary},
    )


def _run_view_command(args: argparse.Namespace) -> CommandResponse:
    _ensure_file_exists(args.path, command="view", field_name="path")
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
        artifacts=[{"kind": "data", "data": view_summary}],
        metrics=view_summary["summary"],
    )


def _format_text_response(response: CommandResponse) -> str:
    envelope = response.to_envelope()
    if response.command == "generate":
        return ""
    if response.command == "describe" or response.command == "schema":
        return json.dumps(envelope["artifacts"][0]["data"], indent=2)
    if response.command == "doctor":
        return json.dumps(envelope["metrics"], indent=2)
    if response.command == "validate":
        return json.dumps(envelope["validation"], indent=2)
    if response.command == "view":
        return render_view_summary(
            cast(dict[str, Any], envelope["artifacts"][0]["data"]),
            color=cast(str, envelope["resolved"]["color"]),
        )
    return json.dumps(envelope, indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    wants_json = _wants_json(argv)
    try:
        args = parser.parse_args(argv)
        if args.command == "generate":
            response = _run_generate_command(args)
        elif args.command == "validate":
            response = _run_validate_command(args)
        elif args.command == "convert":
            response = _run_convert_command(args)
        elif args.command == "view":
            response = _run_view_command(args)
        elif args.command == "describe":
            response = CommandResponse(
                command="describe",
                mode="inspect",
                artifacts=[{"kind": "data", "data": COMMAND_DESCRIPTIONS[args.target]}],
            )
        elif args.command == "schema":
            response = CommandResponse(
                command="schema",
                mode="inspect",
                artifacts=[{"kind": "data", "data": SCHEMAS[args.target]}],
            )
        else:
            response = CommandResponse(
                command="doctor", mode="inspect", metrics=doctor_report()
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
