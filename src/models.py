from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator)
from typing import Literal, Any, Annotated

ParamType = Literal['string', 'number', 'boolean']
PY_TYPES: dict[str, type[Any]] = {
    'string': str,
    'number': float,
    'boolean': bool
}
ID = Annotated[str, Field(pattern='^[A-Za-z_][A-Za-z0-9_]*$')]


class ParamDef(BaseModel):
    type: ParamType


class FuncDef(BaseModel):
    name: ID
    description: str
    parameters: dict[ID, ParamDef]


class PromptItem(BaseModel):
    prompt: str

    @field_validator('prompt')
    @classmethod
    def check_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str
    name: ID
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
