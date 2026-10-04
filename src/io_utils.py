"""Reading of JSON input files and writing of the output file."""
import json
from .errors import CallMeError
from .models import FunctionCall
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path of the file to read.

    Returns:
        The parsed content of the file.

    Raises:
        CallMeError: If the file is missing, unreadable, not UTF-8
            or not valid JSON.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise CallMeError(f'file not found: {path}')
    except json.JSONDecodeError as e:
        raise CallMeError(
            f"invalid json in {path} at line {e.lineno}: {e.msg}")
    except (OSError, UnicodeDecodeError) as e:
        raise CallMeError(f"unable to read {path} - {e}")


def write_output(path: Path, calls: list[FunctionCall]) -> None:
    """Write the function calls to a JSON file.

    Missing parent directories are created.

    Args:
        path: Path of the file to write.
        calls: The function calls to save.

    Raises:
        CallMeError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            file = [r.model_dump() for r in calls]
            json.dump(file, f, indent=2, ensure_ascii=False, allow_nan=False)
    except OSError as e:
        raise CallMeError(
            f'cannot write {path} - {e}')
