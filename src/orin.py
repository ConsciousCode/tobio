'''
This file should only contain the parts most core to Orin with near-zero
coupling to other systems (UI, database, etc). It shouldn't even run as
a standalone program, but rather be imported by other systems.
'''

from typing import AsyncGenerator, Literal, cast
import sqlite3
import time
from urllib.parse import urlparse

import openai
from openai.types.chat import ChatCompletionMessageParam

import httpx

from util import logger

type Role = Literal['user', 'assistant', 'system', 'tool']

class Orin:
    history: list[ChatCompletionMessageParam]
    
    def __init__(self, config: dict):
        self.config = config
        
        u = urlparse(config['memory']['database'])
        if u.scheme not in {'', 'sqlite'}:
            raise ValueError('Only sqlite databases are supported')
        logger.debug('Connecting to database %s', u.path)
        self.db = sqlite3.connect(u.path)
        self.db.row_factory = sqlite3.Row
        
        with open('src/schema.sql') as f:
            self.db.executescript(f.read())
        
        messages = self.db.execute(f'''
            SELECT * FROM messages ORDER BY created_at DESC LIMIT
            {self.config['memory']['history_limit']}
        ''').fetchall()
        
        self.history = [
            {'role': row.author, 'content': row.content}
            for row in reversed(messages)
        ]
    
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
    
    def raw_add_message(self, role: Role, content: str) -> int:
        self.history.append({'role': role, 'content': content}) # type: ignore
        cur = self.db.execute('''
            INSERT INTO messages (author, created_at, content) VALUES (?, ?, ?)
        ''', (role, time.time(), content))
        self.db.commit()
        return cast(int, cur.lastrowid)
    
    async def get_history(self) -> list[ChatCompletionMessageParam]:
        return self.history
    
    async def chat(self, message: str) -> AsyncGenerator[str, None]:
        self.raw_add_message('user', message)
        result = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.history,
            stream=True
        )
        
        deltas = []
        async for chunk in result:
            if delta := chunk.choices[0].delta.content:
                deltas.append(delta)
                yield delta
        
        self.raw_add_message('assistant', ''.join(deltas))