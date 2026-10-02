from pydantic import BaseModel, ConfigDict, Field, create_model
from typing import Literal, Any

ParamType = Literal['string', 'number', 'boolean']
PY_TYPES: dict[str, type[Any]] = {
    'string': str,
    'number': float,
    'boolean': bool
}


class ParamDef(BaseModel):
    type: ParamType


class FuncDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamDef]


class PromptItem(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str
    name: str
    parameters: dict[str, Any]


def params_model(fdef: FuncDef) -> type[BaseModel]:
    fields: dict[str, Any] = {
        f'arg_{i}': (PY_TYPES[p.type], Field(alias=name))
        for i, (name, p) in enumerate(fdef.parameters.items())
    }
    return create_model(
        f'{fdef.name}_params',
        __config__=ConfigDict(
            extra='forbid',
            strict=True,
            allow_inf_nan=False),
        **fields
        )
