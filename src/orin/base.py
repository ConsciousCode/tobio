'''
Some data models are so simple they remain unchanged regardless of the context
between databases and LLMs, and there's less friction when they use shared
definitions.
'''

from typing import Literal, Optional, TypedDict
from pydantic import BaseModel

type Role = Literal['user', 'agent', 'system', 'tool']
'''The role of the author of a step.'''

class ToolCall(BaseModel):
    '''Data model for a tool step.'''
    tool_id: str
    name: str
    '''Name of the tool.'''
    arguments: dict
    '''Arguments sent to the tool.'''

class BatchCall(BaseModel):
    '''Data model for a group of tool steps.'''
    role: Role
    name: Optional[str]=None
    calls: list[ToolCall]

class ActionResult(BaseModel):
    '''Data model for an action step.'''
    tool_id: str
    name: str
    result: str

class ChatMessage(BaseModel):
    '''Data model for a chat message.'''
    role: Role
    name: Optional[str]=None
    content: str

type Message = ChatMessage | BatchCall | ActionResult

class ConfigToml_Memory(TypedDict):
    database: str
    schema: str
    message_table: str
    history_limit: int
    summary_count: int

class ConfigToml_Models(TypedDict):
    summarize: str
    json: str
    chat: str

class ConfigToml_Openai(TypedDict):
    api_key: str
    base_url: str

class ConfigToml(TypedDict):
    persona: str
    memory: ConfigToml_Memory
    models: dict[str, str]
    openai: ConfigToml_Openai