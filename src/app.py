'''
BOTTOM. UP. ONLY BOTTOM UP. DO NOTHING TOP-DOWN. Every change MUST result in a
working program.
'''

from typing import Optional, cast
import chainlit as cl
from orin import load_config
import openai as oai
import httpx

class Context:
    http: httpx.AsyncClient
    openai: oai.AsyncOpenAI
    
    def __init__(self, config):
        self.config = config
        with open(config['persona'], "r") as f:
            self.prompt = f.read()
    
    async def __aenter__(self):
        self.http = await httpx.AsyncClient().__aenter__()
        self.openai = await oai.AsyncOpenAI(
            api_key=self.config['openai']['api_key'],
            http_client=self.http
        ).__aenter__()
        return self
    
    async def __aexit__(self, *args):
        await self.openai.__aexit__(*args)
        await self.http.__aexit__(*args)

@cl.on_chat_start
async def on_chat_start():
    config = load_config("private/config.toml")
    context = await Context(config).__aenter__()
    cl.user_session.set("context", context)

@cl.on_chat_end
async def on_chat_end():
    context: Optional[Context] = cl.user_session.get("context")
    if context:
        await context.__aexit__()

@cl.on_message
async def on_message(message: cl.Message):
    context = cast(Optional[Context], cl.user_session.get("context"))
    if context is None:
        await cl.Message("Context not found").send()
        return
    
    result = await context.openai.chat.completions.create(
        messages=[
            {"role": "system", "content": context.prompt},
            {"role": "user", "content": message.content}
        ],
        model=context.config['models']['chat'].model
    )
    if content := result.choices[0].message.content:
        await cl.Message(content).send()