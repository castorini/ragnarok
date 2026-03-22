from __future__ import annotations

from typing import Any

from pydantic import RootModel


class GenericPayload(RootModel[dict[str, Any]]):
    pass
