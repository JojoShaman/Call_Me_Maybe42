"""Pydantic models describing the input and output data."""
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator)
from typing import Literal, Any, Annotated

ParamType = Literal['string', 'number', 'integer','boolean']
PY_TYPES: dict[str, type[Any]] = {
    'string': str,
    'number': float,
    'integer': int,
    'boolean': bool
}
ID = Annotated[str, Field(pattern='^[A-Za-z_][A-Za-z0-9_]*$')]


class ParamDef(BaseModel):
    """Definition of one function parameter.

    Attributes:
        type: The JSON type of the parameter.
    """
    type: ParamType


class FuncDef(BaseModel):
    """Definition of one callable function.

    Attributes:
        name: The function name, a valid identifier.
        description: What the function does.
        parameters: The parameter definitions, indexed by name.
    """
    name: ID
    description: str
    parameters: dict[ID, ParamDef]


class PromptItem(BaseModel):
    """One entry of the prompts file.

    Attributes:
        prompt: The natural language request.
    """
    prompt: str

    @field_validator('prompt')
    @classmethod
    def check_prompt(cls, value: str) -> str:
        """Reject a prompt that is empty or only whitespace.

        Args:
            value: The prompt to check.

        Returns:
            The prompt, unchanged.

        Raises:
            ValueError: If the prompt is blank.
        """
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value


class FunctionCall(BaseModel):
    """One entry of the output file.

    Attributes:
        prompt: The original request.
        name: The name of the function to call.
        parameters: The argument values, indexed by parameter name.
    """
    model_config = ConfigDict(extra='forbid')
    prompt: str
    name: ID
    parameters: dict[str, Any]


def params_model(fdef: FuncDef) -> type[BaseModel]:
    """Build the model validating the arguments of one function.

    Each parameter becomes a required field with a neutral internal
    name and the real name as alias, so any parameter name is allowed.

    Args:
        fdef: The definition of the function.

    Returns:
        A strict model class that forbids extra keys and non-finite
        numbers.
    """
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
