'''
Common types and models for the database, regardless of implementation.
'''

from typing import ClassVar, Literal, Optional
from functools import cached_property

from pydantic import BaseModel, Field, root_validator

__all__ = [
    "Author",
    "Step",
    "ToolData",
    "ActionData",
    "Unbound",
    "Row"
]

class Author(BaseModel):
    '''Author of a step.'''
    
    type Role = Literal['user', 'assistant', 'system', 'tool']
    '''The role of the author of a step.'''
    
    Unbound: ClassVar[type['UnboundAuthor']]
    
    id: int
    role: Role
    name: Optional[str]

class UnboundAuthor(BaseModel):
    role: Author.Role
    name: Optional[str]

Author.Unbound = UnboundAuthor

class ToolData(BaseModel):
    '''Data model for a tool step.'''
    tool_id: str
    name: str
    arguments: dict

class ActionData(BaseModel):
    '''Data model for an action step.'''
    tool_id: str
    result: Optional[str]

class Step(BaseModel):
    '''Step in the conversation.'''
    
    type Kind = Literal['text', 'tool', 'action']
    '''
    text = Ordinary text
    tool = LLM calling a tool
    action = Tool's response (the action taken)
    '''

    type Status = Literal['atom', 'stream', 'done', 'failed']
    '''Status of a message, primarily for failsafe purposes.'''
    
    Unbound: ClassVar[type['UnboundStep']]
    
    id: int
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
                return ToolData.model_validate_json(self.content)
            case "action":
                return ActionData.model_validate_json(self.content)
        
        raise ValueError(f"Unknown step kind: {self.kind}")
    
    @cached_property
    def text(self):
        if self.kind != "text":
            raise ValueError("Non-text steps have no text")
        return self.content

class UnboundStep(BaseModel):
    author: Author
    created_at: Optional[float] = None
    kind: Step.Kind = Field(default=None)
    status: Step.Status = Field(default=None)
    
    text: Optional[str] = None
    data: Optional[ToolData|ActionData] = None
    
    @root_validator(pre=True)
    def validate_content(cls, values):
        kind: Optional[Step.Kind] = values.get("kind")
        text: Optional[str] = values.get('text')
        data: Optional[ToolData|ActionData] = values.get('data')
        
        if kind is None:
            match data:
                case None:
                    if text is None:
                        raise ValueError("If kind is omitted, either text or data must be set.")
                    kind = "text"
                case ToolData():
                    kind = "tool"
                case ActionData():
                    kind = "action"
                case _:
                    raise TypeError("Unknown data type")
            values['kind'] = kind
        
        match kind:
            case "text" if text is None:
                raise ValueError("Text steps must have text.")
            
            case "tool":
                if data is None:
                    raise ValueError("Tool steps must have data.")
                
                ToolData.model_validate(data)
            
            case "action":
                if data is None:
                    raise ValueError("Action steps must have data")
                
                ActionData.model_validate(data)
        
        return values

Step.Unbound = UnboundStep

type Unbound = Author.Unbound | Step.Unbound
type Row = Author | Step