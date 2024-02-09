'''
Code for interacting with language models.
'''

from abc import ABC, abstractmethod
import os
from typing import Any, AsyncIterator, Coroutine, Generator, Literal, ClassVar, Optional, Self
from urllib.parse import ParseResult, urlparse, parse_qs
from contextlib import AsyncExitStack

from pydantic import BaseModel

from ..base import ConfigToml, Message
from ..tool import ToolBox
from ..util import logger

__all__ = [
    "TextDelta",
    "ToolDelta",
    "ActionRequired",
    "Finish",
    "Delta",
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

class Inference(ABC):
    '''A reified inference, allowing one to choose to stream or await.'''
    
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[Delta]: ...
    
    @abstractmethod
    def __await__(self) -> Generator[Any, Any, str]: ...

class Provider(ABC):
    '''A provider of language models.'''
    
    config: dict
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def __aenter__(self) -> Coroutine[Any, Any, Self]: ...
    
    @abstractmethod
    def __aexit__(self, exc_type, exc_value, traceback) -> Coroutine[Any, Any, None]: ...
    
    @abstractmethod
    def model(self, model: str, config: dict) -> dict: ...
    
    @abstractmethod
    def chat(self,
            config: dict,
            messages: list[Message],
            tools: Optional[ToolBox]=None
        ) -> Inference: ...

class Connector:
    '''Abstracts providers so they can be used interchangeably.'''
    
    config: ConfigToml
    providers: dict[str, Provider]
    provider_factories: dict[str, type[Provider]]
    models: dict[str, 'ChatModel']
    
    context: AsyncExitStack
    
    def __init__(self, config: ConfigToml):
        self.config = config
        self.providers = {}
        self.provider_factories = {}
        self.models = {}
    
    def put_provider_factory(self, api: str):
        '''
        Get a possibly cached provider factory for a given API. This level
        of indirection allows us to avoid importing all providers at once.
        '''
        match api:
            case "openai":
                from .openai import export_Provider
                return export_Provider
            
            case _:
                raise ValueError(f"Unknown provider API: {api}")
    
    async def put_provider(self, u: ParseResult):
        '''Get a possibly cached provider for a given URI.'''
        
        api, *transport = u.scheme.split("+", 1)
        if transport:
            scheme = transport[0]
            specific = True
        elif api in {"http", "https"}:
            api = DEFAULT_PROTO
            scheme = api
            specific = False
        else:
            scheme = "http"
            specific = False
        
        # A path suggests an API endpoint, so probably a specific base_url
        path = os.path.dirname(u.path)
        if path not in {"", "/"}:
            specific = True
        
        # See if we already have one
        base_url = f"{scheme}://{u.netloc}{path}"
        if provider := self.providers.get(base_url):
            return provider
        
        # Construct a new one
        config = self.config.get(api)
        if config is None:
            logger.warn("No configuration found for provider API: %s", api)
            config = {}
        
        if specific:
            config['base_url'] = base_url
        
        provider = self.put_provider_factory(api)(config)
        await self.context.enter_async_context(provider)
        self.providers[base_url] = provider
        return provider
    
    async def __aenter__(self):
        self.context = AsyncExitStack()
        await self.context.__aenter__()
        for name, model in self.config['models'].items():
            u = urlparse(model)
            model = os.path.basename(u.path)
            provider = await self.put_provider(u)
            
            self.models[name] = ChatModel(
                provider.model(model, parse_qs(u.query)),
                provider
            )
        
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.context.__aexit__(exc_type, exc_value, traceback)
        del self.context

class ChatModel:
    '''A language model for turn-based chat.'''
    
    config: dict[str, Any]
    provider: Provider
    
    def __init__(self, config: dict[str, Any], provider: Provider):
        self.config = config
        self.provider = provider
    
    def __call__(self, messages: list[Message], toolbox: Optional[ToolBox]=None) -> Inference:
        '''Generate a response to a series of messages.'''
        return self.provider.chat(self.config, messages, toolbox)