'''
This file should only contain the parts most core to Orin with near-zero
coupling to other systems (UI, database, etc). It shouldn't even run as
a standalone program, but rather be imported by other systems.
'''

from typing import AsyncGenerator
import openai
import httpx

class Orin:
    def __init__(self, config: dict):
        self.config = config
    
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
    
    async def chat(self, message: str) -> AsyncGenerator[str, None]:
        result = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": message}],
            stream=True
        )
        
        async for chunk in result:
            if delta := chunk.choices[0].delta.content:
                yield delta