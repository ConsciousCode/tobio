'''
This file should only contain the parts most core to Orin with near-zero
coupling to other systems (UI, database, etc). It shouldn't even run as
a standalone program, but rather be imported by other systems.
'''

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, ClassVar, Literal, Mapping, Optional, Protocol, cast, override
import sqlite3
import time
from urllib.parse import parse_qs, urlparse
import json
from functools import cache

import openai
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolParam

import httpx
from pydantic import TypeAdapter

from util import logger, filter_dict, unalias_dict

PROMPT_ENSURE_JSON = "The previous messages are successive attempts to produce valid JSON but have at least one error. Respond only with the corrected JSON."

PROMPT_SUMMARY = "You are the summarization agent of Orin. Summarize the conversation thus far."

type Role = Literal['user', 'assistant', 'system', 'tool']

class AsyncGeneratorReturn(Exception):
    def __init__(self, value):
        super().__init__()
        self.value = value

@dataclass
class ModelConfig(Mapping):
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
    
    def __iter__(self):
        return iter(self._keys)
    
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
    
    def to_openai(self):
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

class Message:
    id: int
    role: Role
    name: Optional[str]
    content: str
    
    def __init__(self, role: Role, name: Optional[str], content: str):
        self.role = role
        self.name = name
        self.content = content
    
    def to_openai(self) -> ChatCompletionMessageParam:
        d = {
            'role': self.role,
            'content': self.content
        }
        if self.name:
            d['name'] = self.name
        
        return d # type: ignore

@dataclass
class TextDelta:
    content: str

@dataclass
class ToolDelta:
    id: Optional[str]
    name: Optional[str]
    arguments: Optional[str]

@dataclass
class ActionRequired:
    name: Optional[str]
    arguments: Optional[dict]

@dataclass
class Finish:
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]]

type Delta = TextDelta | ToolDelta | ActionRequired

class PendingToolCall:
    def __init__(self):
        self.id = ""
        self.name = ""
        self.arguments = ""

@dataclass
class ToolResponse:
    id: str
    name: str
    arguments: dict
    output: Any

class Tool:
    __name__: str
    __doc__: Optional[str]
    
    @abstractmethod
    async def __call__(self, **kwargs) -> Any: ...
    
    @abstractmethod
    def to_openai(self) -> ChatCompletionToolParam: ...

class FunctionTool(Tool):
    def __init__(self, func):
        self.__func__ = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
    
    async def __call__(self, **kwargs) -> Any:
        print("Function call", self.__name__, kwargs)
        return await self.__func__(**kwargs)
    
    @override
    def to_openai(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": str(self.__doc__),
                "parameters": TypeAdapter(self.__func__).json_schema()
            }
        }

class Orin:
    models: Mapping[str, ModelConfig]
    config: dict
    tools: dict[str, Tool]
    tool_schema: list[ChatCompletionToolParam]
    
    db: sqlite3.Connection
    http_client: httpx.AsyncClient
    openai_client: openai.AsyncClient
    
    history: list[ChatCompletionMessageParam]
    
    def __init__(self, config: dict):
        self.models = {
            name: ModelConfig.from_uri(uri)
            for name, uri in config['models'].items()
        }
        self.config = config
        self.tools = {}
        self.tool_schema = []
        
        u = urlparse(config['memory']['database'])
        if u.scheme not in {'', 'sqlite'}:
            raise ValueError('Only sqlite databases are supported')
        logger.debug('Connecting to database %s', u.path)
        self.db = sqlite3.connect(u.path)
        self.db.row_factory = sqlite3.Row
        
        with open('src/schema.sql') as f:
            self.db.executescript(f.read())
        
        self.history = self.sql_load_history()
    
    async def __aenter__(self):
        self.http_client = httpx.AsyncClient()
        self.openai_client = openai.AsyncClient(
            api_key=self.config['openai']['api_key'],
            http_client=await self.http_client.__aenter__()
        )
        await self.openai_client.__aenter__()
        
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.openai_client.__aexit__(exc_type, exc_value, traceback)
        await self.http_client.__aexit__(exc_type, exc_value, traceback)
    
    def add_tools(self, *tools: Tool):
        self.tools.update({tool.__name__: tool for tool in tools})
        self.tool_schema.extend(tool.to_openai() for tool in tools)
    
    def sql_load_history(self) -> list[ChatCompletionMessageParam]:
        messages = self.db.execute(f'''
            SELECT * FROM messages ORDER BY created_at DESC LIMIT
            {self.config['memory']['history_limit']}
        ''').fetchall()
        return [
            Message(row['role'], row['name'], row['content']).to_openai()
            for row in reversed(messages)
        ]
    
    def sql_add_memory(self, message: Message) -> int:
        cur = self.db.execute('''
            INSERT INTO messages (role, name, created_at, content) VALUES (?, ?, ?, ?)
        ''', (message.role, message.name, time.time(), message.content))
        self.db.commit()
        return cast(int, cur.lastrowid)
    
    def prompt_summary(self) -> ChatCompletionMessageParam:
        return {
            'role': 'system',
            'name': 'summary',
            'content': PROMPT_SUMMARY
        }
    
    async def llm_summarize(self, history: list[ChatCompletionMessageParam]) -> str:
        result: ChatCompletion = await self.openai_client.chat.completions.create(
            **self.models['summarize'].to_openai(),
            messages=[*history, self.prompt_summary()]
        )
        return result.choices[0].message.content # type: ignore
    
    async def llm_ensure_json(self, data: str) -> dict:
        tries = []
        for _ in range(3):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                logger.warn("Correcting JSON")
                tries.append(data)
                result: ChatCompletion = await self.openai_client.chat.completions.create(
                    **self.models['json'].to_openai(),
                    messages=[
                        *tries,
                        Message('system', 'prompt', PROMPT_ENSURE_JSON).to_openai()
                    ]
                )
                data = str(result.choices[0].message.content)
        
        raise ValueError("Failed to ensure JSON")
    
    async def llm_chat(self) -> AsyncGenerator[Delta, Optional[Any]]:
        print("Tools:", json.dumps(self.tool_schema, indent=2))
        result: AsyncStream[ChatCompletionChunk] = await self.openai_client.chat.completions.create(
            **self.models['chat'].to_openai(),
            messages=self.history,
            tools=self.tool_schema,
            stream=True
        )
        
        tool_calls: list[PendingToolCall] = []
        '''Tool calls being streamed.'''
        
        async for chunk in result:
            choice = chunk.choices[0]
            if reason := choice.finish_reason:
                raise AsyncGeneratorReturn(reason)
            
            delta = choice.delta
            if delta is None:
                continue
            
            if delta.tool_calls is None:
                if delta.content is not None:
                    yield TextDelta(delta.content)
                continue
            
            # Tool calls also stream in chunks
            for chunk in delta.tool_calls:
                if len(tool_calls) <= chunk.index:
                    if chunk.index > 0:
                        finished = tool_calls[chunk.index - 1]
                        args = await self.llm_ensure_json(finished.arguments)
                        output = yield ActionRequired(finished.name, args)
                        await self.add_message(
                            Message('tool', finished.name, json.dumps(output))
                        )
                    
                    tool_calls.append(PendingToolCall())
                
                pending = tool_calls[chunk.index]
                
                if chunk.id:
                    pending.id += chunk.id
                
                if func := chunk.function:
                    func = chunk.function
                    if func.name:
                        pending.name += func.name
                    if func.arguments:
                        pending.arguments += func.arguments
                    
                    yield ToolDelta(chunk.id, func.name, func.arguments)
                else:
                    yield ToolDelta(chunk.id, None, None)
    
    async def add_message(self, message: Message) -> Message:
        '''Add a new message to the memory.'''
        self.history.append(message.to_openai())
        mid = self.sql_add_memory(message)
        message.id = mid
        return message
    
    async def update_message(self, message: Message, content: str):
        '''Update an existing message in the memory.'''
        self.db.execute('''
            UPDATE messages SET content = content || ? WHERE id = ?
        ''', (content, message.id))
        message.content += content
    
    async def truncate(self, limit: int):
        if len(self.history) > limit:
            oldest, newest = self.history[:-limit], self.history[-limit:]
            
            need_summary = any(m.get("name") == "summary" for m in oldest)
            
            # Are we due for a summary based on frequency?
            if len(newest):
                # Find the last summary message
                for i, m in enumerate(reversed(newest)):
                    if m.get("name") == "summary":
                        break
                
                summary_freq = limit//self.config['memory']['summary_count']
                if len(newest) - i <= summary_freq: # type: ignore
                    need_summary = True
            
            if need_summary:
                summary = await self.llm_summarize(self.history)
                logger.info("Summary: %s", summary)
                return await self.add_message(Message('system', 'summary', summary))
            
            self.history = newest
    
    async def get_history(self) -> list[ChatCompletionMessageParam]:
        return self.history
    
    async def chat(self, message: str) -> AsyncGenerator[str, None]:
        await self.add_message(Message('user', None, message))
        
        deltas = []
        try:
            it = aiter(self.llm_chat())
            while True:
                delta = await it.__anext__()
                match delta:
                    case TextDelta(content):
                        deltas.append(content)
                        yield content
                    case ToolDelta(id, name, args):
                        if deltas:
                            await self.add_message(
                                Message('assistant', None, ''.join(deltas))
                            )
                            deltas.clear()
                    case ActionRequired(name, args):
                        print("Action required", name, args)
                        if name in self.tools:
                            result = await self.tools[name](**(args or {}))
                            await self.add_message(
                                Message('tool', name, json.dumps(result))
                            )
                        else:
                            await self.add_message(
                                Message('system', 'error', f"Tool {name!r} not found")
                            )
                    
                    case _:
                        raise NotImplementedError(delta)
        except AsyncGeneratorReturn as e:
            finish_reason = e.value
        
        if deltas:
            await self.add_message(Message('assistant', None, ''.join(deltas)))
        
        await self.truncate(self.config['memory']['history_limit'])
        
        print(len(self.history), "messages thus far")