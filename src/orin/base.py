'''
Some data models are so simple they remain unchanged regardless of the context
between databases and LLMs, and there's less friction when they use shared
definitions.
'''

from typing import Optional
from pydantic import BaseModel

class ToolCall(BaseModel):
    '''Data model for a tool step.'''
    id: str
    name: str
    arguments: dict

class BatchCall(BaseModel):
    '''Data model for a group of tool steps.'''
    role: str
    name: Optional[str]=None
    calls: list[ToolCall]

class ActionResult(BaseModel):
    '''Data model for an action step.'''
    tool_id: str
    name: str
    result: str

class ChatMessage(BaseModel):
    '''Data model for a chat message.'''
    role: str
    name: Optional[str]=None
    content: str

type Message = ChatMessage | BatchCall | ActionResult