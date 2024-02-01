'''
BOTTOM. UP. ONLY BOTTOM UP. DO NOTHING TOP-DOWN. Every change MUST result in a
working program.
'''

from typing import Literal, NotRequired, Optional, Required, TypedDict, cast
import chainlit as cl
from openai.types.chat import ChatCompletionMessageParam
from orin import load_config
import openai as oai
import httpx

type Role = Literal["user", "assistant", "system", "tool"]
type Message = ChatCompletionMessageParam

class Context:
    http: httpx.AsyncClient
    openai: oai.AsyncOpenAI
    history: list[Message]
    
    def __init__(self, config):
        self.config = config
        with open(config['persona'], "r") as f:
            self.prompt = {
                "role": "system",
                "name": "prompt",
                "content": f.read()
            }
        
        self.history = []
    
    async def __aenter__(self):
        self.http = await httpx.AsyncClient().__aenter__()
        self.openai = await oai.AsyncOpenAI(
            api_key=self.config['openai']['api_key'],
            http_client=self.http
        ).__aenter__()
        return self
    
    async def __aexit__(self, exc_type=None, exc_value=None, traceback=None):
        await self.openai.__aexit__(exc_type, exc_value, traceback)
        await self.http.__aexit__(exc_type, exc_value, traceback)
    
    async def add(self, role: Role, content: str, name: Optional[str]=None):
        message = {"role": role, "content": content}
        if name is not None:
            message["name"] = name
        self.history.append(message) # type: ignore
        
        if len(self.history) > self.config['history_limit']:
            self.history.pop(0)
    
    def chatlog(self):
        return [self.prompt, *self.history]

@cl.on_chat_start
async def on_chat_start():
    config = load_config("private/config.toml")
    context = await Context(config).__aenter__()
    cl.user_session.set("context", context)

@cl.on_chat_end
async def on_chat_end():
    context = cast(Context, cl.user_session.get("context"))
    await context.__aexit__(None, None, None)

@cl.on_message
async def on_message(message: cl.Message):
    context = cast(Context, cl.user_session.get("context"))
    
    await context.add("user", message.content)
    
    result = await context.openai.chat.completions.create(
        messages=context.chatlog(),
        model=context.config['models']['chat'].model
    )
    if content := result.choices[0].message.content:
        await context.add("assistant", content, "Orin")
        await cl.Message(author="Orin", content=content).send()