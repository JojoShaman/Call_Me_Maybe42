from pydantic import (
    BaseModel,
    TypeAdapter,
    ValidationError)
from .models import PromptItem, FuncDef
from .errors import CallMeError, describe
from .io_utils import load_json
from pathlib import Path
from collections import Counter


class Data(BaseModel):
    prompts: list[str]
    function: dict[str, FuncDef]

    @classmethod
    def open_files(cls, p_path: Path, f_path: Path) -> "Data":
        try:
            items = TypeAdapter(
                list[PromptItem]).validate_python(load_json(p_path))
        except ValidationError as e:
            raise CallMeError(
                f"Invalid structure in {p_path}:\n{describe(e)}")
        try:
            definitions = TypeAdapter(
                list[FuncDef]).validate_python(load_json(f_path))
        except ValidationError as e:
            raise CallMeError(
                f"Invalid structure in {f_path}:\n{describe(e)}")
        if not definitions:
            raise CallMeError(f"No function defined in {f_path}")
        func: dict[str, FuncDef] = {f.name: f for f in definitions}
        names: list[str] = [f.name for f in definitions]
        duplicates: list[str] = sorted(
            n for n, count in Counter(names).items() if count > 1)
        if duplicates:
            raise CallMeError(
                f"Duplicate function names in {f_path}: "
                f"{', '.join(duplicates)}")
        return cls(prompts=[i.prompt for i in items], function=func)
