from pydantic import ValidationError


class CallMeError(Exception):
    ...


def describe(e: ValidationError) -> str:
    lines: list[str] = []
    for err in e.errors():
        where = '.'.join(
            f"entry {p + 1}" if isinstance(p, int) else str(p)
            for p in err['loc'])
        lines.append(f"  - {where or 'root'}: {err['msg']}")
    return "\n".join(lines)
