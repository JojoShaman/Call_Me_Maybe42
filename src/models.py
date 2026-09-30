from pydantic import BaseModel
from typing import Literal, Any

ParamType = Literal['string', 'number', 'boolean']

class ParamDef(BaseModel):
    type: ParamType

class FuncDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamDef]

class PromptItem(BaseModel):
    prompt: str

class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]