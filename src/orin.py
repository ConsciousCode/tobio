'''
This file should only contain the parts most core to Orin with near-zero
coupling to other systems (UI, database, etc). It shouldn't even run as
a standalone program, but rather be imported by other systems.
'''

from dataclasses import dataclass
from typing import AsyncGenerator, ClassVar, Literal, Mapping, Optional, cast
import sqlite3
import time
from urllib.parse import parse_qs, urlparse

import openai
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam

import httpx

from util import logger, filter_dict, unalias_dict

type Role = Literal['user', 'assistant', 'system', 'tool']

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

@dataclass
class Message:
    role: Role
    name: Optional[str]
    content: str
    
    def to_openai(self) -> ChatCompletionMessageParam:
        d = {
            'role': self.role,
            'content': self.content
        }
        if self.name:
            d['name'] = self.name
        
        return d # type: ignore

class Orin:
    models: Mapping[str, ModelConfig]
    config: dict
    
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
            'content': "You are the summarization agent of Orin. Summarize the conversation thus far."
        }
    
    async def llm_summarize(self, history: list[ChatCompletionMessageParam]) -> str:
        result: ChatCompletion = await self.openai_client.chat.completions.create(
            **self.models['summarize'].to_openai(),
            messages=[*history, self.prompt_summary()]
        )
        return result.choices[0].message.content # type: ignore
    
    async def llm_chat(self) -> AsyncGenerator[str, None]:
        result: AsyncStream[ChatCompletionChunk] = await self.openai_client.chat.completions.create(
            **self.models['chat'].to_openai(),
            messages=self.history,
            stream=True
        )
        
        async for chunk in result:
            if delta := chunk.choices[0].delta.content:
                yield delta
    
    async def add_message(self, message: Message) -> int:
        self.history.append(message.to_openai())
        return self.sql_add_memory(message)
    
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
        async for delta in self.llm_chat():
            deltas.append(delta)
            yield delta
        
        await self.add_message(Message('assistant', None, ''.join(deltas)))
        await self.truncate(self.config['memory']['history_limit'])
        
        print(len(self.history), "messages thus far")