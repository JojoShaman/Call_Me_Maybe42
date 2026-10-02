from pydantic import BaseModel, PrivateAttr
from .decoder import Decoder
from .errors import CallMeError
from .prompts import param_prompt
from .models import FuncDef
from typing import Any
from .rules import RULES, CONVERT


class SelectFunction(BaseModel):
    names: list[str]
    decoder: Decoder
    _target: list[str] = PrivateAttr(default_factory=list[str])

    def model_post_init(self, context: Any) -> None:
        self._target = [f'{name}"' for name in self.names]

    def extraction(self, ids: list[int]) -> str:
        def is_done(text: str) -> bool:
            return text in self._target

        def is_valid(text: str) -> bool:
            return any(
                t.startswith(text) for t in self._target)
        return (self.decoder.generate(ids, is_done, is_valid).rstrip('"'))


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
            ids.extend(self.decoder.encode(fixed))
            text = self.decoder.generate(
                ids, *RULES[info.type](end))
            ret[name] = CONVERT[info.type](text, end)
        return ret
