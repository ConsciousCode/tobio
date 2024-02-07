'''
Code for interacting with language models.
'''

from _typeshed import structseq
from typing import Any, AsyncIterator, Literal, ClassVar, Optional, cast
from urllib.parse import urlparse, parse_qs
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolMessageParam
import json

from pydantic import BaseModel
import httpx
import openai

from orin.db.base import Author

from .tool import ToolBox
from .db import Step
from .util import async_await, filter_dict, unalias_dict

PROMPT_ENSURE_JSON = "The previous messages are successive attempts to produce valid JSON but have at least one error. Respond only with the corrected JSON."

PROMPT_SUMMARY = "You are the summarization agent of Orin. Summarize the conversation thus far."

class TextDelta(BaseModel):
    content: str

class ToolDelta(BaseModel):
    id: Optional[str]
    name: Optional[str]
    arguments: Optional[str]

class ActionRequired(BaseModel):
    tool_id: str
    name: str
    arguments: dict

class Finish(BaseModel):
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"]

type Delta = TextDelta | ToolDelta | ActionRequired | Finish

class PendingToolCall:
    def __init__(self):
        self.id = ""
        self.name = ""
        self.arguments = ""

class ModelConfig(BaseModel):
    proto: str
    
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
    
    model: str
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
        assert u.scheme in {"openai"}
        
        return ModelConfig(
            proto=u.scheme,
            model=u.path,
            **filter_dict(
                unalias_dict(
                    parse_qs(u.query),
                    cls._aliases
                ),
                cls._keys
            )
        )

class ChatMessage(BaseModel):
    role: Author.Role
    name: Optional[str]
    content: str
    
    def to_openai(self) -> ChatCompletionMessageParam:
        msg = {
            "role": self.role,
            "content": self.content
        }
        if self.name:
            msg["name"] = self.name
        
        return msg # type: ignore

class ToolResponse(BaseModel):
    tool_call_id: str
    content: str
    
    def to_openai(self) -> ChatCompletionToolMessageParam:
        return {
            "role": "tool",
            "content": self.content,
            "tool_call_id": self.tool_call_id
        }

type Message = ChatMessage | ToolResponse

class Provider:
    http_client: httpx.AsyncClient
    openai_client: openai.AsyncClient
    
    config: dict
    models: dict[str, 'ChatModel']
    
    def __init__(self, config):
        self.config = config
        self.models = {
            name: ChatModel(ModelConfig.from_uri(uri), self)
            for name, uri in config['models'].items()
        }
    
    async def __aenter__(self):
        self.http_client = httpx.AsyncClient()
        self.openai_client = openai.AsyncClient(
            api_key=self.config['openai']['api_key'],
            http_client=await self.http_client.__aenter__()
        )
        await self.openai_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.http_client.__aexit__(exc_type, exc_value, traceback)
        await self.openai_client.__aexit__(exc_type, exc_value, traceback)
        del self.http_client
        del self.openai_client
    
    async def ensure_json(self, data: str) -> dict:
        tries: list[Message] = []
        for _ in range(3):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                #logger.warn("Correcting JSON")
                tries.append(ChatMessage(
                    role="user",
                    name=None,
                    content=data
                ))
                print("Correcting JSON", tries)
                result = await self.models['json']([
                    *tries, ChatMessage(
                        role="system",
                        name="prompt",
                        content=PROMPT_ENSURE_JSON
                    )
                ])
                data = result
        
        raise ValueError("Failed to ensure JSON")

class Inference:
    '''A reified inference, allowing one to choose to stream or await.'''
    
    model: 'ChatModel'
    messages: list[Message]
    toolbox: Optional[ToolBox]
    
    def __init__(self, model: 'ChatModel', messages: list[Message], tools: Optional[ToolBox]=None):
        self.model = model
        self.messages = messages
        self.toolbox = tools
    
    async def _action(self, call: PendingToolCall) -> ActionRequired:
        '''Utility method to generate an ActionRequired response.'''
        
        print("ActionRequired:", call.name, call.arguments)
        return ActionRequired(
            tool_id=call.id,
            name=call.name,
            arguments=await self.model.provider.ensure_json(call.arguments)
        )
    
    async def __aiter__(self) -> AsyncIterator[Delta]:
        history = [msg.to_openai() for msg in self.messages]
        tools = [] if self.toolbox is None else self.toolbox.render()
        result = cast(
            openai.AsyncStream[ChatCompletionChunk],
            await self.model.provider.openai_client.chat.completions.create(
                **self.model.config.to_dict(),
                messages=history,
                tools=tools,
                stream=True
            )
        )
        
        tool_calls: list[PendingToolCall] = []
        '''Tool calls being streamed.'''
        
        async for chunk in result:
            choice = chunk.choices[0]
            if reason := choice.finish_reason:
                if reason == "function_call":
                    reason = "tool_calls"
                # Yield any pending tool_calls
                if tool_calls:
                    yield await self._action(tool_calls[-1])
                yield Finish(finish_reason=reason)
                break
            
            delta = choice.delta
            if delta is None:
                continue
            
            if delta.tool_calls is None:
                if delta.content is not None:
                    yield TextDelta(content=delta.content)
                continue
            
            # Tool calls also stream in chunks
            for tc in delta.tool_calls:
                # New tool call, append to list and send the last one
                if len(tool_calls) <= tc.index:
                    if tc.index > 0:
                        yield await self._action(tool_calls[tc.index - 1])
                    
                    tool_calls.append(PendingToolCall())
                
                # Current pending tool call
                pending = tool_calls[tc.index]
                
                # Update it
                if tc.id:
                    pending.id += tc.id
                
                if func := tc.function:
                    func = tc.function
                    if func.name:
                        pending.name += func.name
                    if func.arguments:
                        pending.arguments += func.arguments
                    
                    name = func.name
                    arguments = func.arguments
                else:
                    name = None
                    arguments = None
                
                # Yield the delta
                yield ToolDelta(
                    id=tc.id,
                    name=name,
                    arguments=arguments
                )
    
    @async_await
    async def __await__(self):
        result = await self.model.provider.openai_client.chat.completions.create(
            messages=[msg.to_openai() for msg in self.messages],
            **self.model.config.to_dict(),
            stream=False
        )
        
        return cast(str, result.choices[0].message.content)

class ChatModel:
    config: ModelConfig
    provider: Provider
    
    def __init__(self, config: ModelConfig, provider: Provider):
        self.config = config
        self.provider = provider
    
    def __call__(self, messages: list[Message], toolbox: Optional[ToolBox]=None) -> Inference:
        '''Generate a response to a series of messages.'''
        return Inference(self, messages, toolbox)