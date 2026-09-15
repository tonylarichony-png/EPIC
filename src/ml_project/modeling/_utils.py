"""Internal helpers shared by the modeling modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any
def _display_value(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    if callable(value) and not isinstance(value, str):
        return getattr(value, "__name__", repr(value))
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if callable(value):
        return _display_value(value)
    return value


def _sklearn_import_error() -> ImportError:
    return ImportError(
        "03_baseline requires scikit-learn and joblib. Install project "
        "dependencies from the project root: "
        "python -m pip install -r requirements.txt"
    )
