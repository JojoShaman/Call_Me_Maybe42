from typing import Any
import json
from pathlib import Path
import argparse
from argparse import Namespace
from .models import FunctionCall

def load_json(path: Path) -> Any:
    try:
        with open(path , encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise CallMeError(f'Error - file not found : {path}')
    except json.JSONDecodeError as e:
        raise CallMeError(
            f"Error - invalid json in {path} at line {e.lineno} : {e.msg}")
    except (OSError, UnicodeDecodeError) as e:
        raise CallMeError(f"Error - unable to read {path}: {e}")

def arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        '--functions_definition', type=Path,
        default=Path('data/input/functions_definition.json'),
        help='JSON file with the available functions')
    parser.add_argument(
        '--input', type=Path,
        default=Path('data/input/function_calling_tests.json'),
        help='JSON file with the prompts to process')
    parser.add_argument(
        '--output', type=Path,
        default=Path('data/output/function_calls.json'),
        help='JSON file to write the result to')
    return parser.parse_args()

def write_output(path: Path, calls: list[FunctionCall]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            file = [r.model_dump() for r in calls]
            json.dump(file, f, indent=2, ensure_ascii=False, allow_nan=False)
    except OSError as e:
        raise CallMeError(
            f'Error occurred while trying to write in {path}: {e}')

class CallMeError(Exception):
    ...