"""Selection of the function to call and extraction of its arguments."""
from pydantic import BaseModel
from .decoder import Decoder
from .errors import CallMeError
from .prompts import param_prompt
from .models import FuncDef
from typing import Any
from .rules import RULES, CONVERT, COMPLETE


class SelectFunction(BaseModel):
    """Let the model choose one function among the available ones.

    Attributes:
        names: Names of the available functions.
        decoder: The decoder used to generate the name.
    """
    names: list[str]
    decoder: Decoder

    def extraction(self, ids: list[int]) -> str:
        """Generate the name of the function matching the prompt.

        Args:
            ids: Token ids of the selection prompt, extended in place.

        Returns:
            The name of the selected function.

        Raises:
            CallMeError: If the generation cannot complete.
        """
        def complete(text: str) -> str | None:
            """Return the name once a single candidate remains."""
            left = [n for n in self.names if n.startswith(text)]
            return left[0] + '"' if len(left) == 1 else None

        def is_done(text: str) -> bool:
            """Tell whether the closing quote has been written."""
            return '"' in text

        def is_valid(text: str) -> bool:
            """Tell whether the text is, or can become, a known name."""
            name, quote, _ = text.partition('"')
            if quote:
                return name in self.names
            return any(
                n.startswith(text) for n in self.names)
        return (self.decoder.generate(
            ids, is_done, is_valid, complete=complete).partition('"')[0])


class ExtractParameter(BaseModel):
    """Extract the arguments of a function from a prompt.

    Attributes:
        decoder: The decoder used to generate the values.
    """
    decoder: Decoder

    def extraction(
            self, prompt: str,
            definition: FuncDef) -> dict[str, Any]:
        """Generate one typed value for each parameter of the function.

        Args:
            prompt: The natural language request.
            definition: The definition of the selected function.

        Returns:
            The argument values, indexed by parameter name.

        Raises:
            CallMeError: If a type is unsupported or a value cannot
                be generated.
        """
        ret: dict[str, Any] = {}
        p_prompt = param_prompt(prompt, definition)
        ids: list[int] = self.decoder.encode(p_prompt)
        for index, (name, info) in enumerate(definition.parameters.items()):
            sep: str = '' if index == 0 else ' '
            is_last: bool = index == len(definition.parameters) - 1
            if info.type not in RULES:
                raise CallMeError(f"unsupported type: {info.type!r}")
            quote = '"' if info.type == 'string' else ''
            end = quote + ('}' if is_last else ',')
            if info.type in ('number', 'integer'):
                fixed = f'{sep}"{name}":'
            else:
                fixed = f'{sep}"{name}": {quote}'
            maker = COMPLETE.get(info.type)
            limit = len(self.decoder.encode(prompt)) + 20
            ids.extend(self.decoder.encode(fixed))
            text = self.decoder.generate(
                ids, *RULES[info.type](end),
                max_token=limit if info.type == 'string' else 50,
                complete=maker(end) if maker else None)
            ret[name] = CONVERT[info.type](text, end)
        return ret
