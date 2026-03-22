from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

from ragnarok.cli.adapters import make_data_artifact
from ragnarok.cli.logging_utils import setup_logging
from ragnarok.cli.normalize import normalize_direct_generate_input
from ragnarok.cli.operations import async_run_request_generation, run_request_generation
from ragnarok.cli.responses import CommandResponse
from ragnarok.cli.spec import EXIT_CODES


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    model: str
    prompt_mode: str
    use_azure_openai: bool = False
    use_openrouter: bool = False
    context_size: int = 8192
    topk: list[int] | None = None
    num_gpus: int = 1
    execution_mode: str = "sync"
    max_concurrency: int = 8
    shuffle_candidates: bool = False
    print_prompts_responses: bool = False
    num_few_shot_examples: int = 0
    max_output_tokens: int = 1500
    run_id: str = "ragnarok"
    vllm_batched: bool = False
    include_reasoning: bool = False
    include_trace: bool = False
    redact_prompts: bool = False
    reasoning_effort: str | None = None
    log_level: int = 0
    quiet: bool = False


def _base_args(config: ServerConfig) -> argparse.Namespace:
    return argparse.Namespace(
        command="generate",
        model=config.model,
        model_path=config.model,
        prompt_mode=config.prompt_mode,
        use_azure_openai=config.use_azure_openai,
        use_openrouter=config.use_openrouter,
        context_size=config.context_size,
        topk=config.topk or [20],
        num_gpus=config.num_gpus,
        retrieval_method=None,
        shuffle_candidates=config.shuffle_candidates,
        print_prompts_responses=config.print_prompts_responses,
        num_few_shot_examples=config.num_few_shot_examples,
        max_output_tokens=config.max_output_tokens,
        run_id=config.run_id,
        vllm_batched=config.vllm_batched,
        execution_mode=config.execution_mode,
        max_concurrency=config.max_concurrency,
        include_reasoning=config.include_reasoning,
        include_trace=config.include_trace,
        redact_prompts=config.redact_prompts,
        reasoning_effort=config.reasoning_effort,
        log_level=config.log_level,
        quiet=config.quiet,
        output="json",
        dataset=None,
        input_file=None,
        input_json=None,
        stdin=False,
        output_file=None,
        dry_run=False,
        validate_only=False,
        resume=False,
        overwrite=False,
        fail_if_exists=False,
        manifest_path=None,
    )


def execute_direct_generate(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> CommandResponse:
    request = normalize_direct_generate_input(payload)
    response = CommandResponse(command="generate")
    response.validation = {"valid": True, "record_count": 1}
    response.inputs = {"source": "direct"}
    response.resolved = {
        "input_mode": "direct",
        "execution_mode": args.execution_mode,
        "model": args.model,
        "prompt_mode": getattr(args.prompt_mode, "value", args.prompt_mode),
    }
    logger = setup_logging(args.log_level, quiet=getattr(args, "quiet", False))
    if args.execution_mode == "async":
        records, metrics = asyncio.run(
            async_run_request_generation([request], args, logger)
        )
    else:
        records, metrics = run_request_generation([request], args, logger)
    response.metrics = metrics
    response.artifacts.append(make_data_artifact("generation-results", records))
    return response


def run_generate_request(
    payload: dict[str, Any],
    *,
    config: ServerConfig,
) -> CommandResponse:
    return execute_direct_generate(payload, args=_base_args(config))


def validation_error_response(
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> CommandResponse:
    return CommandResponse(
        command="generate",
        status="validation_error",
        exit_code=EXIT_CODES["validation_error"],
        errors=[
            {
                "code": "validation_error",
                "message": message,
                "details": details or {},
                "retryable": False,
            }
        ],
    )


def runtime_error_response(error: Exception) -> CommandResponse:
    return CommandResponse(
        command="generate",
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
