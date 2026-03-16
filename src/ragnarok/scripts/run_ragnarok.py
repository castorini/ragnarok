"""Legacy compatibility wrapper for ragnarok generate.

Translates snake_case flags to the kebab-case equivalents used by the
packaged ``ragnarok`` CLI and delegates execution.

sample run:
  python src/ragnarok/scripts/run_ragnarok.py \
    --model_path=gpt-4o --topk=20 --dataset=researchy-questions \
    --retrieval_method=bm25,rank_zephyr --prompt_mode=chatqa \
    --context_size=8192 --max_output_tokens=1500
"""

import sys

# Exhaustive snake_case → kebab-case flag map.  The packaged CLI uses
# kebab-case for every flag; this table preserves backward compatibility
# for callers that still use the old snake_case spelling.
_FLAG_TRANSLATIONS: dict[str, str] = {
    "--model_path": "--model",
    "--use_azure_openai": "--use-azure-openai",
    "--context_size": "--context-size",
    "--num_gpus": "--num-gpus",
    "--retrieval_method": "--retrieval-method",
    "--prompt_mode": "--prompt-mode",
    "--shuffle_candidates": "--shuffle-candidates",
    "--print_prompts_responses": "--print-prompts-responses",
    "--num_few_shot_examples": "--num-few-shot-examples",
    "--max_output_tokens": "--max-output-tokens",
    "--run_id": "--run-id",
    "--vllm_batched": "--vllm-batched",
    "--include_reasoning": "--include-reasoning",
    "--reasoning_effort": "--reasoning-effort",
}


def _translate_argv(argv: list[str]) -> list[str]:
    """Prepend ``generate`` and translate legacy snake_case flags."""
    translated: list[str] = ["generate"]
    for token in argv:
        # Handle both --flag value and --flag=value forms.
        for old, new in _FLAG_TRANSLATIONS.items():
            if token == old:
                token = new
                break
            if token.startswith(old + "="):
                token = new + token[len(old) :]
                break
        translated.append(token)
    return translated


def cli_compatible_main(argv: list[str] | None = None) -> int:
    from ragnarok.cli.main import main as cli_main

    argv = sys.argv[1:] if argv is None else argv
    return cli_main(_translate_argv(argv))


if __name__ == "__main__":
    sys.exit(cli_compatible_main())
