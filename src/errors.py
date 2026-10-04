"""Error type and error formatting shared by the whole program."""
from pydantic import ValidationError


class CallMeError(Exception):
    """Expected error carrying a message meant for the user."""

def describe(e: ValidationError) -> str:
    """Turn a validation error into one readable line per problem.

    Args:
        e: The validation error raised by Pydantic.

    Returns:
        The problems, one per line, each with its location.
    """
    lines: list[str] = []
    for err in e.errors():
        where = '.'.join(
            f"entry {p + 1}" if isinstance(p, int) else str(p)
            for p in err['loc'])
        lines.append(f"  - {where or 'root'}: {err['msg']}")
    return "\n".join(lines)
