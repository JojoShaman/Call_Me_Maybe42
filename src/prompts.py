"""Prompt templates and few-shot examples sent to the model."""
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
    """Build the prompt used to extract the arguments of a function.

    Args:
        prompt: The natural language request.
        definition: The definition of the selected function.

    Returns:
        The few-shot prompt, ending at the opening of the parameters
        object.
    """
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
    """Build the prompt used to choose a function.

    Args:
        scope: The description of every available function.
        prompt: The natural language request.

    Returns:
        The prompt, ending right before the function name.
    """
    return (f"Target: {scope}\n"
            f"Question: {prompt}\n"
            'Answer: {"name": "')
