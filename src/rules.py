from typing import Callable, Any
import re
import json
from .errors import CallMeError

Rule = tuple[Callable[[str], bool], Callable[[str], bool]]

ESCAPES: set[str] = set('"\\/bfnrt')
BOOLEANS: list[str] = ['true', 'false']
NB_COMPLETE: re.Pattern[str] = re.compile(r"-?\d+(\.\d+)?")
NB_PARTIAL: re.Pattern[str] = re.compile(r"-?\d*(\.\d*)?")


def split_string(text: str) -> tuple[str, str] | None:
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\':
            if i + 1 == len(text):
                return text, ""
            if text[i + 1] not in ESCAPES:
                return None
            i += 2
            continue
        if c == '"':
            return text[:i], text[i:]
        if c < " ":
            return None
        i += 1
    return text, ""


def string_validator(end: str) -> Rule:
    def is_done(text: str) -> bool:
        parts = split_string(text)
        return parts is not None and parts[1] == end

    def is_valid(text: str) -> bool:
        parts = split_string(text)
        content = parts[0] if parts else ""
        if content[:1] == " " and content.strip():
            return False
        return parts is not None and end.startswith(parts[1])
    return (is_done, is_valid)


def number_validator(end: str) -> Rule:
    def is_done(text: str) -> bool:
        return text.endswith(end)

    def is_valid(text: str) -> bool:
        if text.endswith(end):
            return NB_COMPLETE.fullmatch(text[:-len(end)]) is not None
        return NB_PARTIAL.fullmatch(text) is not None
    return (is_done, is_valid)


def bool_validator(end: str) -> Rule:
    def is_done(text: str) -> bool:
        return text.endswith(end)

    def is_valid(text: str) -> bool:
        if text.endswith(end):
            return text[:-len(end)] in BOOLEANS
        return any(word.startswith(text) for word in BOOLEANS)
    return (is_done, is_valid)


def to_string(text: str, _end: str) -> str:
    parts = split_string(text)
    if parts is None:
        raise CallMeError(f"invalid string generated: {text!r}")
    return str(json.loads(f'"{parts[0]}"'))


def to_number(text: str, end: str) -> float:
    return float(text.removesuffix(end))


def to_boolean(text: str, end: str) -> bool:
    return text.removesuffix(end) == 'true'


RULES: dict[str, Callable[[str], Rule]] = {
    'string': string_validator,
    'number': number_validator,
    'boolean': bool_validator
}

CONVERT: dict[str, Callable[[str, str], Any]] = {
    'string': to_string,
    'number': to_number,
    'boolean': to_boolean
}
