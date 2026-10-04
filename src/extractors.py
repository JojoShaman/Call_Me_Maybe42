from pydantic import BaseModel
from .decoder import Decoder
from .errors import CallMeError
from .prompts import param_prompt
from .models import FuncDef
from typing import Any
from .rules import RULES, CONVERT, COMPLETE


class SelectFunction(BaseModel):
    names: list[str]
    decoder: Decoder

    def extraction(self, ids: list[int]) -> str:
        def complete(text: str) -> str | None:
            left = [n for n in self.names if n.startswith(text)]
            return left[0] + '"' if len(left) == 1 else None

        def is_done(text: str) -> bool:
            return '"' in text

        def is_valid(text: str) -> bool:
            name, quote, _ = text.partition('"')
            if quote:
                return name in self.names
            return any(
                n.startswith(text) for n in self.names)
        return (self.decoder.generate(
            ids, is_done, is_valid, complete=complete).partition('"')[0])


class ExtractParameter(BaseModel):
    decoder: Decoder

    def extraction(
            self, prompt: str,
            definition: FuncDef) -> dict[str, Any]:
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
