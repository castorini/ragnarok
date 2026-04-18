from __future__ import annotations

import sys
from collections.abc import Sequence

from . import dispatch as dispatch_module
from .config import load_config
from .dispatch import dispatch_command, write_manifest
from .errors import (
    CLIError,
    build_error_response,
    build_runtime_error_response,
    detect_command,
    emit_json,
    wants_json,
)
from .introspection import doctor_report
from .operations import run_dataset_generation
from .parser import build_parser
from .rendering import format_text_response


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    config, config_path = load_config()
    output_json = wants_json(argv)
    try:
        args = parser.parse_args(argv)
        for key, value in config.items():
            flag = f"--{key.replace('_', '-')}"
            if flag not in argv:
                setattr(args, key, value)
        # Preserve the historical patch surface for tests and external callers
        # that monkeypatch helpers off ragnarok.cli.main.
        dispatch_module.doctor_report = doctor_report
        dispatch_module.run_dataset_generation = run_dataset_generation
        response = dispatch_command(args, config_path=config_path)
        write_manifest(getattr(args, "manifest_path", None), response)
        if getattr(args, "output", "text") == "json":
            emit_json(response.to_envelope())
        else:
            text = format_text_response(response)
            if text:
                sys.stdout.write(text + "\n")
        return response.exit_code
    except CLIError as error:
        response = build_error_response(error)
        if output_json:
            emit_json(response.to_envelope())
        else:
            sys.stderr.write(error.message + "\n")
        return error.exit_code
    except Exception as error:
        command = detect_command(argv)
        response = build_runtime_error_response(command, error)
        if output_json:
            emit_json(response.to_envelope())
        else:
            sys.stderr.write(f"{error}\n")
        return response.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
