from .models import FuncDef
import json
from typing import Any
EXAMPLES: list[tuple[str, str, dict[str, Any]]] = [
    ("Say hello to Brian", "fn_say_hello(person: string)",
        {"name": "fn_say_hello", "parameters": {"person": "Brian"}}),
    ("Replace every digit in 'r2d2' with '#'",
        "fn_replace(text: string, regex: string, replacement: string)",
        {"name": "fn_replace", "parameters":
         {"text": "r2d2", "regex": r"\d", "replacement": "#"}}),
    ("Replace uppercase letters in 'HeLLo' with dashes",
        "fn_replace(text: string, regex: string, replacement: string)",
        {"name": "fn_replace", "parameters":
         {"text": "HeLLo", "regex": "[A-Z]", "replacement": "-"}})]


def param_prompt(prompt: str, definition: FuncDef) -> str:
    shots = "\n\n".join(
        f"prompt: {a}\nfunction: {b}\nanswer: {json.dumps(c)}"
        for a, b, c in EXAMPLES)
    p = (', '.join(f'{a}: {b.type}'
                   for a, b in definition.parameters.items()))
    func = (f'{definition.name}' + f'({p})')
    return ("Extract the arguments of the function call from the prompt. "
            "Copy values from the prompt; when a symbol is named "
            "(asterisks, dashes, underscores...), write the "
            "symbol itself.\n\n"
            f"{shots}\n\n"
            f"prompt: {prompt}\n"
            f"function: {func}\n"
            f'answer: {{"name": "{definition.name}", "parameters": {{')


def func_prompt(scope: str, prompt: str) -> str:
    return (f"Target: {scope}\n"
            f"Question: {prompt}\n"
            'Answer: {"name": "')
