from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateStep:
    name: str
    argv: tuple[str, ...]


ROOT = Path(__file__).resolve().parents[3]


def _run_step(step: GateStep) -> None:
    print(f"[quality-gate] {step.name}", flush=True)
    completed = subprocess.run(step.argv, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    steps = (
        GateStep("ruff check", (sys.executable, "-m", "ruff", "check", ".")),
        GateStep(
            "ruff format",
            (sys.executable, "-m", "ruff", "format", "--check", "."),
        ),
        GateStep(
            "core tests",
            (sys.executable, "-m", "pytest", "-q", "-m", "core", "test"),
        ),
        GateStep(
            "integration tests",
            (sys.executable, "-m", "pytest", "-q", "-m", "integration", "test"),
        ),
        GateStep("mypy", (sys.executable, "-m", "mypy", "src", "test")),
    )
    for step in steps:
        _run_step(step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
