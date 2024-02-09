'''
Code for interacting with language models.
'''

from abc import ABC, abstractmethod
import os
from typing import Any, AsyncIterator, Coroutine, Generator, Literal, ClassVar, Optional, Self
from urllib.parse import urlparse, parse_qs

from pydantic import BaseModel

from ..base import Message
from ..tool import ToolBox
from ..util import  filter_dict, unalias_dict

__all__ = [
    "TextDelta",
    "ToolDelta",
    "ActionRequired",
    "Finish",
    "Delta",
    "ModelConfig",
    "Inference",
    "Provider",
    "ChatModel"
]

DEFAULT_PROTO = "openai"

PROMPT_ENSURE_JSON = "The previous messages are successive attempts to produce valid JSON but have at least one error. Respond only with the corrected JSON."

PROMPT_SUMMARY = "You are the summarization agent of Orin. Summarize the conversation thus far."

class TextDelta(BaseModel):
    content: str

class ToolDelta(BaseModel):
    index: int
    tool_id: Optional[str]
    name: Optional[str]
    arguments: Optional[str]

class ActionRequired(BaseModel):
    tool_id: str
    name: str
    arguments: dict

class Finish(BaseModel):
    reason: Literal["stop", "length", "tool_calls", "content_filter"]

type Delta = TextDelta | ToolDelta | ActionRequired | Finish

class ModelConfig(BaseModel):
    _aliases: ClassVar = {
        "T": "temperature",
        "p": "top_p",
        "max_token": "max_tokens"
    }
    _keys: ClassVar = {
        "model",
        "temperature",
        "max_tokens",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop"
    }
    
    proto: str
    model: str
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[str] = None
    
    def __len__(self):
        return len(self._keys)
    
    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        if key not in self._keys:
            raise KeyError(key)
        setattr(self, key, value)
    
    def to_dict(self):
        return filter_dict(vars(self), self._keys)
    
    @classmethod
    def from_uri(cls, uri: str):
        '''Parse a model specification from a URI.'''
        
        u = urlparse(uri)
        proto, *transport = u.scheme.split("+", 1)
        
        if transport:
            scheme = transport[0]
        elif proto in {"http", "https"}:
            scheme = proto
            proto = DEFAULT_PROTO
        else:
            scheme = "http"
        
        path = os.path.dirname(u.path)
        model = os.path.basename(u.path)
        
        return ModelConfig(
            proto=proto,
            model=model,
            base_url=f"{scheme}://{u.netloc}{path}",
            **filter_dict(
                unalias_dict(
                    parse_qs(u.query),
                    cls._aliases
                ),
                cls._keys
            )
        )

class Inference(ABC):
    '''A reified inference, allowing one to choose to stream or await.'''
    
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[Delta]: ...
    
    @abstractmethod
    def __await__(self) -> Generator[Any, Any, str]: ...

class Provider(ABC):
    '''A provider of language models.'''
    
    config: dict
    models: dict[str, 'ChatModel']
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def __aenter__(self) -> Coroutine[Any, Any, Self]: ...
    
    @abstractmethod
    def __aexit__(self, exc_type, exc_value, traceback) -> Coroutine[Any, Any, None]: ...
    
    @abstractmethod
    def chat(self,
            model: ModelConfig,
            messages: list[Message],
            tools: Optional[ToolBox]=None
        ) -> Inference: ...

class ChatModel:
    '''A language model for turn-based chat.'''
    
    config: ModelConfig
    provider: Provider
    
    def __init__(self, config: ModelConfig, provider: Provider):
        self.config = config
        self.provider = provider
    
    def __call__(self, messages: list[Message], toolbox: Optional[ToolBox]=None) -> Inference:
        '''Generate a response to a series of messages.'''
        return self.provider.chat(self.config, messages, toolbox)