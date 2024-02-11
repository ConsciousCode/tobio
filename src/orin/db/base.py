'''
Common types and models for the database, regardless of implementation.
'''

from typing import ClassVar, Literal, Optional
from functools import cached_property

from pydantic import BaseModel, Field, TypeAdapter, root_validator

from ..base import ToolCall, ActionResult, Role

__all__ = [
    "Author",
    "Step",
    "Row"
]

class Author(BaseModel):
    '''Author of a step.'''
    
    Transient: ClassVar
    class Transient(BaseModel):
        role: Role
        name: Optional[str]
    
    id: int
    role: Role
    name: Optional[str]

class Step(BaseModel):
    '''Step in the conversation.'''
    
    Kind: ClassVar
    type Kind = Literal['text', 'tool', 'action']
    '''
    text = Ordinary text
    tool = LLM calling a tool
    action = Tool's response (the action taken)
    '''
    
    Status: ClassVar
    type Status = Literal['atom', 'stream', 'done', 'failed']
    '''Status of a message, primarily for failsafe purposes.'''
    
    Transient: ClassVar
    class Transient(BaseModel):
        parent_id: Optional[int]
        author: Author
        created_at: Optional[float] = None
        status: 'Step.Status' = Field(default=None)
        
        content: str|ToolCall|list[ToolCall]|ActionResult
        
        @root_validator(pre=True)
        def check_content(cls, values):
            content = values['content']
            if isinstance(content, ToolCall):
                values['content'] = [content]
            
            return values
    
    id: int
    parent_id: Optional[int]
    author: Author
    kind: Kind
    status: Status
    created_at: float
    updated_at: float
    content: str
    
    @cached_property
    def data(self):
        if self.kind == "text":
            raise ValueError("Text steps have no data")
        
        match self.kind:
            case "tool":
                return TypeAdapter(list[ToolCall]).validate_json(self.content)
            case "action":
                return ActionResult.model_validate_json(self.content)
        
        raise ValueError(f"Unknown step kind: {self.kind}")
    
    @cached_property
    def text(self):
        if self.kind != "text":
            raise ValueError("Non-text steps have no text")
        return self.content
    
    def __str__(self):
        who = self.author.role
        if self.author.name is not None:
            who += f" {self.author.name}"
        
        return f"[{who}]@{self.id} {self.content!r}"

type Row = Author | Step