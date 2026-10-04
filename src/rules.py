"""Validation rules and conversions for each parameter type."""
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
    """Split a generated string value at its closing quote.

    Args:
        text: The text generated after the opening quote.

    Returns:
        The content and what follows it, starting at the closing
        quote, or None if the text cannot be a valid JSON string.
    """
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
        if c < " " or c == "\ufffd":
            return None
        i += 1
    return text, ""


def string_validator(end: str) -> Rule:
    """Build the rules of a string value.

    Args:
        end: The text that must close the value.

    Returns:
        The ``is_done`` and ``is_valid`` functions, in that order.
    """
    def is_done(text: str) -> bool:
        """Tell whether the string is closed by the expected end."""
        parts = split_string(text)
        return parts is not None and parts[1] == end

    def is_valid(text: str) -> bool:
        """Tell whether the text can still become a valid string."""
        parts = split_string(text)
        content = parts[0] if parts else ""
        if content[:1] == " " and content.strip():
            return False
        return parts is not None and end.startswith(parts[1])
    return (is_done, is_valid)


def number_validator(end: str) -> Rule:
    """Build the rules of a number value.

    Args:
        end: The text that must close the value.

    Returns:
        The ``is_done`` and ``is_valid`` functions, in that order.
    """
    def is_done(text: str) -> bool:
        """Tell whether the number is followed by the expected end."""
        return text.endswith(end)

    def is_valid(text: str) -> bool:
        """Tell whether the text can still become a valid number."""
        if text.endswith(end):
            return NB_COMPLETE.fullmatch(text[:-len(end)]) is not None
        return NB_PARTIAL.fullmatch(text) is not None
    return (is_done, is_valid)


def bool_validator(end: str) -> Rule:
    """Build the rules of a boolean value.

    Args:
        end: The text that must close the value.

    Returns:
        The ``is_done`` and ``is_valid`` functions, in that order.
    """
    def is_done(text: str) -> bool:
        """Tell whether the boolean is followed by the expected end."""
        return text.endswith(end)

    def is_valid(text: str) -> bool:
        """Tell whether the text can still become true or false."""
        if text.endswith(end):
            return text[:-len(end)] in BOOLEANS
        return any(word.startswith(text) for word in BOOLEANS)
    return (is_done, is_valid)


def bool_complete(end: str) -> Callable[[str], str | None]:
    """Build the function completing a boolean value.

    Args:
        end: The text that must close the value.

    Returns:
        A function returning the full value once its first letter is
        known, else None.
    """
    def complete(text: str) -> str | None:
        """Return the full boolean once only one remains possible."""
        if not text:
            return None
        left = [w for w in BOOLEANS if w.startswith(text)]
        return left[0] + end if len(left) == 1 else None
    return complete


def to_string(text: str, _end: str) -> str:
    """Convert a generated string value into a Python string.

    Args:
        text: The generated text, with its closing delimiter.
        _end: The closing delimiter, unused.

    Returns:
        The decoded string, with JSON escapes resolved.

    Raises:
        CallMeError: If the text is not a valid JSON string.
    """
    parts = split_string(text)
    if parts is None:
        raise CallMeError(f"invalid string generated: {text!r}")
    return str(json.loads(f'"{parts[0]}"'))


def to_number(text: str, end: str) -> float:
    """Convert a generated number value into a float.

    Args:
        text: The generated text, with its closing delimiter.
        end: The closing delimiter to remove.

    Returns:
        The number as a float.
    """
    return float(text.removesuffix(end))


def to_boolean(text: str, end: str) -> bool:
    """Convert a generated boolean value into a bool.

    Args:
        text: The generated text, with its closing delimiter.
        end: The closing delimiter to remove.

    Returns:
        True if the value is ``true``, False otherwise.
    """
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

COMPLETE: dict[str, Callable[[str], Callable[[str], str | None]]] = {
    'boolean': bool_complete
}
