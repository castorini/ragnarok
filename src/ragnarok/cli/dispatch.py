from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

from ragnarok.api.runtime import ServerConfig, execute_direct_generate

from .adapters import make_data_artifact, make_file_artifact
from .errors import CLIError
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
)
from .responses import CommandResponse
from .spec import EXIT_CODES
from .view import (
    ViewError,
    build_view_summary,
    detect_artifact_type,
    load_records,
)


def ensure_file_exists(path: str, *, command: str, field_name: str) -> None:
    if not Path(path).exists():
        raise CLIError(
            f"{field_name} does not exist: {path}",
            exit_code=EXIT_CODES["missing_resource"],
            status="validation_error",
            error_code="missing_input",
            command=command,
            details={"field": field_name, "path": path},
        )


def resolve_write_policy(args: argparse.Namespace) -> str:
    if getattr(args, "resume", False):
        return "resume"
    if getattr(args, "overwrite", False):
        return "overwrite"
    if getattr(args, "fail_if_exists", False):
        return "fail_if_exists"
    return "default_fail_if_exists"


def prepare_output_path(args: argparse.Namespace, *, command: str) -> str:
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
    write_policy = resolve_write_policy(args)
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


def write_manifest(manifest_path: str | None, response: CommandResponse) -> None:
    if manifest_path is None:
        return
    Path(manifest_path).write_text(
        json.dumps(response.to_envelope(), indent=2) + "\n", encoding="utf-8"
    )


def read_direct_payload(args: argparse.Namespace) -> dict[str, Any]:
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


def resolve_model_name(args: argparse.Namespace) -> str:
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


def run_generate_command(args: argparse.Namespace) -> CommandResponse:
    from ragnarok.generate.llm import PromptMode

    model_name = resolve_model_name(args)
    args.prompt_mode = PromptMode(args.prompt_mode)
    if args.dataset is None and args.retrieval_method is None:
        args.retrieval_method = [parse_retrieval_methods("bm25")[0]]
    if args.output_file is not None and not args.dry_run and not args.validate_only:
        args.output_file = prepare_output_path(args, command="generate")

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
        ensure_file_exists(args.input_file, command="generate", field_name="input_file")
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

    payload = read_direct_payload(args)
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


def run_serve_command(args: argparse.Namespace) -> CommandResponse:
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


def run_validate_command(args: argparse.Namespace) -> CommandResponse:
    response = CommandResponse(command="validate", mode="validate")
    if args.target == "generate":
        if args.input_file is not None:
            ensure_file_exists(
                args.input_file, command="validate", field_name="input_file"
            )
            response.validation = validate_generate_batch_file(args.input_file)
        else:
            payload = read_direct_payload(args)
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
        ensure_file_exists(args.topicfile, command="validate", field_name="topicfile")
        ensure_file_exists(args.runfile, command="validate", field_name="runfile")
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
        ensure_file_exists(args.input, command="validate", field_name="input")
    ensure_file_exists(args.topics, command="validate", field_name="topics")
    if args.apply_fixes and not (args.fix_length or args.fix_citations):
        args.fix_length = True
        args.fix_citations = True
    summary, stdout = run_validate_rag25(args)
    response.validation = summary
    response.metrics = {"stdout": stdout}
    return response


def run_convert_command(args: argparse.Namespace) -> CommandResponse:
    ensure_file_exists(args.input_file, command="convert", field_name="input_file")
    if args.prompt_file is not None:
        ensure_file_exists(
            args.prompt_file, command="convert", field_name="prompt_file"
        )
    summary, stdout = run_convert_trec25(args)
    return CommandResponse(
        command="convert",
        artifacts=[make_file_artifact("converted-output", args.output_file)],
        metrics={"stdout": stdout, **summary},
    )


def run_view_command(args: argparse.Namespace) -> CommandResponse:
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


def run_prompt_command(args: argparse.Namespace) -> CommandResponse:
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


def run_prompt_render_command(args: argparse.Namespace) -> CommandResponse:
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
    payload = read_direct_payload(args)
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


def dispatch_command(
    args: argparse.Namespace,
    *,
    config_path: Path | None,
) -> CommandResponse:
    if args.command == "generate":
        return run_generate_command(args)
    if args.command == "serve":
        return run_serve_command(args)
    if args.command == "validate":
        return run_validate_command(args)
    if args.command == "convert":
        return run_convert_command(args)
    if args.command == "view":
        return run_view_command(args)
    if args.command == "prompt":
        if args.prompt_command == "render":
            return run_prompt_render_command(args)
        return run_prompt_command(args)
    if args.command == "describe":
        return CommandResponse(
            command="describe",
            mode="inspect",
            artifacts=[
                make_data_artifact(args.target, COMMAND_DESCRIPTIONS[args.target])
            ],
            inputs={"target": args.target},
            resolved={"target_command": args.target},
        )
    if args.command == "schema":
        return CommandResponse(
            command="schema",
            mode="inspect",
            artifacts=[make_data_artifact(args.target, SCHEMAS[args.target])],
            inputs={"target": args.target},
            resolved={"schema": args.target},
        )
    report = doctor_report()
    report["config_file"] = str(config_path) if config_path else None
    return CommandResponse(
        command="doctor",
        mode="inspect",
        metrics=report,
        validation={"python_ok": report["python_ok"]},
    )
