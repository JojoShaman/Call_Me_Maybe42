import json
from .errors import CallMeError
from .models import FunctionCall
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise CallMeError(f'file not found: {path}')
    except json.JSONDecodeError as e:
        raise CallMeError(
            f"invalid json in {path} at line {e.lineno}: {e.msg}")
    except (OSError, UnicodeDecodeError) as e:
        raise CallMeError(f"unable to read {path}: {e}")


def write_output(path: Path, calls: list[FunctionCall]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            file = [r.model_dump() for r in calls]
            json.dump(file, f, indent=2, ensure_ascii=False, allow_nan=False)
    except OSError as e:
        raise CallMeError(
            f'Cannot write {path}: {e}')
