from dataclasses import dataclass
from typing import ClassVar, Dict, Mapping, Optional, Union, override
from autogen import Agent, AssistantAgent, UserProxyAgent
import chainlit as cl
import tomllib
from urllib.parse import parse_qs, urlparse

from .util import filter_dict, unalias_dict

TASK = "Plot a chart of NVDA stock price change YTD and save it on disk."

@dataclass
class ModelConfig(Mapping):
    proto: str
    
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

async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res

class ChainlitAssistantAgent(AssistantAgent):
    @override
    async def a_send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        await cl.Message(
            content=f'*Sending message to "{recipient.name}":*\n\n{message}',
            author="AssistantAgent",
        ).send()
        await super().a_send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

class ChainlitUserProxyAgent(UserProxyAgent):
    @override
    async def a_get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            "Provide feedback to assistant. Press enter to skip and use auto-reply"
        ):
            res = await ask_helper(
                cl.AskActionMessage,
                content="Continue or provide feedback?",
                actions=[
                    cl.Action(
                        name="continue",
                        value="continue",
                        label="✅ Continue"
                    ),
                    cl.Action(
                        name="feedback",
                        value="feedback",
                        label="💬 Provide feedback",
                    ),
                    cl.Action(
                        name="exit",
                        value="exit",
                        label="🔚 Exit Conversation"
                    ),
                ],
            )
            if res.get("value") == "continue":
                return ""
            if res.get("value") == "exit":
                return "exit"

        reply = await ask_helper(
            cl.AskUserMessage, content=prompt, timeout=60
        )
        return reply["content"].strip()
    
    @override
    async def a_send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        await cl.Message(
            content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
            author="UserProxyAgent",
        ).send()
        await super().a_send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

def parse_model_uri(uri):
    '''Parse a model specification from a URI.'''
    
    u = urlparse(uri)
    assert u.scheme in {"openai"}
    
    return ModelConfig(
        proto=u.scheme,
        model=u.path,
        **filter_dict(
            unalias_dict(parse_qs(u.query), {
                "T": "temperature",
                "p": "top_p",
                "max_token": "max_tokens"
            }), {
                "temperature",
                "max_tokens",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "stop"
            }
        )
    )

def load_config(file):
    with open(file, "rb") as f:
        config = tomllib.load(f)
    
    config['models'] = {
        name: parse_model_uri(uri)
        for name, uri in config["models"].items()
    }
    
    return config