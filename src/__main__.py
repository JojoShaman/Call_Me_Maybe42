import sys

try:
    from llm_sdk import Small_LLM_Model  # type: ignore
    import numpy as np
    from .data import Data
    from pydantic import BaseModel, ConfigDict, PrivateAttr
    from typing import Any, Callable
    import json
    import re
    from pathlib import Path
    from .models import FuncDef, FunctionCall
    from .utils import arguments, CallMeError, write_output
    from time import perf_counter
    from .decoder import Decoder
except ModuleNotFoundError as e:
    missing = getattr(e, "name", None) or str(e)
    print(f'Error - missing dependancy "{missing}."'
          "run 'uv sync' or 'make install'.", file=sys.stderr)
    sys.exit(1)
except ImportError as e:
    print(f"Error - import failed: {e}. "
          "Run the program with 'uv run python -m src'.", file=sys.stderr)
    sys.exit(1)

ESCAPES = set('"\\/bfnrt')
EXAMPLES: list[tuple[str, str, dict[str, Any]]] = [
    ("Say hello to Brian", "fn_say_hello(person: string)",
        {"name": "fn_say_hello", "parameters": {"person": "Brian"}}),
    ("What is 7 times 12?", "fn_multiply(x: number, y: number)",
        {"name": "fn_multiply", "parameters": {"x": 7.0, "y": 12.0}}),
    ("Replace every digit in 'r2d2' with '#'",
        "fn_replace(text: string, regex: string, replacement: string)",
        {"name": "fn_replace", "parameters":
        {"text": "r2d2", "regex": r"\d", "replacement": "#"}}),
    ("Change the word 'red' to 'blue' in 'red car, redwood'",
        "fn_replace(text: string, regex: string, replacement: string)",
        {"name": "fn_replace", "parameters":
        {"text": "red car, redwood", "regex": r"\bred\b",
        "replacement": "blue"}}),
    ("Replace uppercase letters in 'HeLLo' with dashes",
        "fn_replace(text: string, regex: string, replacement: string)",
        {"name": "fn_replace", "parameters":
        {"text": "HeLLo", "regex": "[A-Z]", "replacement": "-"}}),
    ("Replace all letters in 'Ab3Cd' with question marks",
        "fn_replace(text: string, regex: string, replacement: string)",
        {"name": "fn_replace", "parameters":
        {"text": "Ab3Cd", "regex": "[a-zA-Z]", "replacement": "?"}}),
    ("Replace the word 'is' with 'was' in 'This is it'",
        "fn_replace(text: string, regex: string, replacement: string)",
        {"name": "fn_replace", "parameters":
        {"text": "This is it", "regex": r"\bis\b", "replacement": "was"}})]

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

class Function(BaseModel):
    names: list[str]
    decoder: Decoder
    _target: list[str] =  PrivateAttr(default_factory=list[str])

    def model_post_init(self, context) -> None:
        self._target = [f'{name}"' for name in self.names]

    def extraction(self, ids: list[int]) -> str:
        def is_done(text: str) -> bool:
            return text in self._target
        def is_valid(text: str) -> bool:
            return any(
                t.startswith(text) for t in self._target)
        return(self.decoder.generate(ids, is_done, is_valid).rstrip('"'))

class Parameter(BaseModel):
    decoder: Decoder
    def string_validator(self, end: str) -> tuple[
        Callable[[str],bool], Callable[[str], bool]]:
                    
        def is_done(text: str) -> bool:
            parts = split_string(text)
            return parts is not None and parts[1] == end
        def is_valid(text: str) -> bool:
            parts = split_string(text)
            content = parts[0] if parts else ""
            if content[:1] == " " and content.strip():
                return False
            return parts is not None and end.startswith(parts[1])
        return(is_done, is_valid)

    def bool_validator(self, end: str) -> tuple[
        Callable[[str],bool], Callable[[str], bool]]:
        def is_done(text: str) -> bool:
            return text.endswith(end)
        def is_valid(text: str) -> bool:
            words: list[str] = ['true', 'false']
            if text.endswith(end):
                return text[:-len(end)] in words
            return any(word.startswith(text) for word in words)
        return(is_done, is_valid)

    def number_validator(self, end: str) -> tuple[
        Callable[[str],bool], Callable[[str], bool]]:
        def is_done(text: str) -> bool:
            return text.endswith(end)

        def is_valid(text: str) -> bool:
            if text.endswith(end):
                return (re.fullmatch(
                    r"-?\d+(\.\d+)?", text[:-len(end)]) is not None)
            return re.fullmatch(r"-?\d*(\.\d*)?", text) is not None
        return(is_done, is_valid)

    def extraction(
            self, prompt: str,
            definition: FuncDef) -> dict[str, Any]:
        shots = "\n\n".join(
            f"prompt: {a}\nfunction: {b}\nanswer: {json.dumps(c)}"
            for a, b, c in EXAMPLES
        )
        ret: dict[str, Any] = {}
        p = (', '.join(f'{a}: {b.type}'
                       for a, b in definition.parameters.items()))
        func = (f'{definition.name}' + f'({p})')
        request = (
            "Extract the arguments of the function call from the prompt. "
            "Copy values from the prompt; when a symbol is named "
            "(asterisks, dashes, underscores...), write the symbol itself.\n\n"
            f"{shots}\n\n"
            f"prompt: {prompt}\n"
            f"function: {func}\n"
            f'answer: {{"name": "{definition.name}", "parameters": {{'
        )
        ids: list[int] = self.decoder.encode(request)
        for index, (name, info) in enumerate(definition.parameters.items()):
            p_type = info.type
            sep: str = '' if index == 0 else ' '
            is_last: bool = index == len(definition.parameters) - 1
            if p_type == 'string':
                end: str = '"}' if is_last else '",'
                fixed: str = f'{sep}"{name}": "'
                is_done, is_valid = self.string_validator(end)
            elif p_type == 'number' or p_type == 'boolean':
                end = '}' if is_last else ','
                fixed = f'{sep}"{name}": '
                if p_type == 'number':
                    is_done, is_valid = self.number_validator(end)
                else:
                    is_done, is_valid = self.bool_validator(end)
            ids.extend(self.decoder.encode(fixed))
            text = self.decoder.generate(ids, is_done, is_valid)
            if p_type == 'string':
                parts = split_string(text)
                if parts is None:
                    raise CallMeError(f"invalid string generated: {text!r}")
                ret[name] = json.loads(f'"{parts[0]}"')
            elif p_type == 'number':
                ret[name] = float(text.removesuffix(end))
            elif p_type == 'boolean':
                ret[name] = text.removesuffix(end) == 'true'
            else:
                raise CallMeError(f"unsupported type: {p_type!r}")
        return ret


class Pipeline(BaseModel):
    data: Data
    deco: Decoder
    param: Parameter
    function: Function

    def run(self) -> list[FunctionCall]:
        prompts = self.data.prompts
        scope = '.\n'.join(
            [f"if task is {f.description}, "
             f"scope is {name}"
             for name, f in self.data.function.items()])
        ret: list[FunctionCall] = []
        for i, p in enumerate(prompts, 1):
            try:
                prompt = (
                    f"Target: {scope}\n"
                    f"Question: {p}\n"
                    'Answer: {"name": "'
                )
                ids: list[int] = self.deco.encode(prompt)
                func_name = self.function.extraction(ids)
                params = self.param.extraction(
                    p, self.data.function[func_name])
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


def main() -> int:
    try:
        args = arguments()
        d = Data.open_files(args.input, args.functions_definition)
        start_time = perf_counter()
        m = Small_LLM_Model()
        decoder = Decoder(model=m)
        pipeline = Pipeline(
            data=d, deco=decoder,
            function=Function(names=list(d.function), decoder=decoder),
            param=Parameter(decoder=decoder))
        result = pipeline.run()
        end_time = perf_counter()
        write_output(args.output, result)
    except CallMeError as e:
        print(e, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('program interrupted by user', file=sys.stderr)
        return 130
    except Exception as e:
        print(f'Unexpected error occurred: {e}', file=sys.stderr)
        return 1
    minutes, seconds = divmod(round(end_time - start_time), 60)
    print(f'generation ran for {minutes} minutes {seconds} seconds')
    return 0

if __name__ == '__main__':
    sys.exit(main())