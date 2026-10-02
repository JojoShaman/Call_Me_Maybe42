from pydantic import BaseModel, PrivateAttr, ValidationError
from .data import Data
from .decoder import Decoder
from .extractors import ExtractParameter, SelectFunction
from .models import FunctionCall, params_model
from .prompts import func_prompt
from .errors import CallMeError, describe
import sys
from typing import Any


class Pipeline(BaseModel):
    data: Data
    deco: Decoder
    param: ExtractParameter
    function: SelectFunction
    _validators: dict[str, type[BaseModel]] = PrivateAttr(
        default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        self._validators = {
            name: params_model(f) for name, f in self.data.function.items()
        }

    def run(self) -> list[FunctionCall]:
        prompts = self.data.prompts
        scope = '.\n'.join(
            [f"if task is {f.description}, "
             f"scope is {name}"
             for name, f in self.data.function.items()])
        ret: list[FunctionCall] = []
        for i, p in enumerate(prompts, 1):
            try:
                ids: list[int] = self.deco.encode(
                    func_prompt(scope, p))
                func_name = self.function.extraction(ids)
                params = self.param.extraction(
                    p, self.data.function[func_name])
                try:
                    self._validators[func_name].model_validate(params)
                except ValidationError as e:
                    raise CallMeError(
                        f'Invalid arguments for {func_name!r}:\n{describe(e)}')
                func_call = FunctionCall(
                    prompt=p, name=func_name, parameters=params)
                print(f"[{i}/{len(prompts)}] {func_call.prompt} "
                      f"-> {func_call.name}\n{func_call.parameters}")
                print()
                ret.append(func_call)
            except CallMeError as e:
                print(
                    f"[{i}/{len(prompts)}] skipped {p!r}: {e}",
                    file=sys.stderr)
        return ret
