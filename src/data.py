from pydantic import (
    BaseModel,
    TypeAdapter,
    ValidationError)
from .models import PromptItem, FuncDef
from .utils import load_json, CallMeError
from pathlib import Path

class Data(BaseModel):
    prompts: list[str]
    function: dict[str, FuncDef]

    @classmethod
    def open_files(cls, p_path: Path, f_path: Path) -> "Data":
        try:
            items = TypeAdapter(
                list[PromptItem]).validate_python(load_json(p_path))
            definitions = TypeAdapter(
                list[FuncDef]).validate_python(load_json(f_path))
        except ValidationError as e:
            raise CallMeError(f"Error - Invalid input file structure:\n{e}")
        if not definitions:
            raise CallMeError("No function defined")
        func: dict[str, FuncDef] = {f.name: f for f in definitions}
        if len(func) != len(definitions):
           raise CallMeError("Duplicate function names")
        return cls(prompts=[i.prompt for i in items], function=func)
