import os
from typing import Any, Dict

from dotenv import load_dotenv

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_public_openai_api_key() -> str | None:
    load_dotenv()
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")


def get_openrouter_api_key() -> str | None:
    load_dotenv()
    return os.getenv("OPENROUTER_API_KEY")


def get_openai_api_key() -> str | None:
    load_dotenv()
    return get_public_openai_api_key() or get_openrouter_api_key()


def uses_openrouter(model_name: str, use_openrouter: bool = False) -> bool:
    load_dotenv()
    if use_openrouter:
        return True
    openai_key = get_public_openai_api_key()
    openrouter_key = get_openrouter_api_key()
    return model_name.startswith("openrouter/") or (
        bool(openrouter_key) and not bool(openai_key)
    )


def get_openai_compatible_args(
    model_name: str, use_azure_openai: bool = False, use_openrouter: bool = False
) -> Dict[str, Any]:
    load_dotenv()
    if use_azure_openai:
        args = get_azure_openai_args()
        args["keys"] = get_public_openai_api_key()
        return args

    if uses_openrouter(model_name, use_openrouter=use_openrouter):
        return {
            "keys": get_openrouter_api_key(),
            "api_base": OPENROUTER_BASE_URL,
        }
    return {"keys": get_public_openai_api_key()}


def get_azure_openai_args() -> Dict[str, str]:
    load_dotenv()
    azure_args = {
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
    }

    # Sanity check
    assert all(
        list(azure_args.values())
    ), "Ensure that `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION` are set"
    return azure_args


def get_cohere_api_key() -> str:
    load_dotenv()
    return os.getenv("CO_API_KEY")


def get_anyscale_api_key() -> str:
    load_dotenv()
    return os.getenv("ANYSCALE_API_KEY")
