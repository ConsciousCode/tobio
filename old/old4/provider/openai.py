'''
OpenAI API-based connector.
'''

from typing import Any, AsyncIterator, Generator, override
from contextlib import asynccontextmanager
import openai
import httpx
import json
from openai.types.chat import ChatCompletionMessageParam

from ..util import filter_dict, typename
from .base import ModelMessage, SystemMessage, AgentMessage, UserMessage, Provider, Run, Step, DeltaStep, ActionStep

def to_openai(message: ModelMessage) -> ChatCompletionMessageParam:
    '''Convert a message to the OpenAI API format.'''
    
    match message:
        case SystemMessage():
            role = "system"
        case AgentMessage():
            role = "assistant"
        case UserMessage():
            role = "user"
        
        case _:
            raise NotImplementedError(f"Unknown message type: {typename(message)}")
    
    return {
        "role": role,
        "content": message.content
    } # type: ignore

class OpenAIRun(Run):
    '''OpenAI API-based run.'''
    
    config: dict
    '''Configuration for the run.'''
    
    prompt: list[ModelMessage]
    '''Prompt to complete.'''
    
    provider: 'OpenAIProvider'
    '''Provider which created the run.'''
    
    def __init__(self, config, client, prompt):
        self.config = config
        self.client = client
        self.prompt = prompt
    
    @override
    def __await__(self) -> Generator[None, None, str]:
        '''Await the run.'''
        
        result = yield from self.client.chat.completions.create(
            messages=list(map(to_openai, self.prompt)),
            stream=False,
            **filter_dict(self.config, {
                'model',
                'max_tokens',
                'temperature',
                'presence_penalty',
                'frequency_penalty',
                'stop',
                'top_p'
            })
        ).__await__()
        
        return result.choices[0].text
    
    @override
    async def __aiter__(self) -> AsyncIterator[Step]:
        '''Start the run and await its resolution.'''
        
        run = await self.client.chat.completions.create(
            messages=list(map(to_openai, self.prompt)),
            stream=True,
            **filter_dict(self.config, {
                'model',
                'max_tokens',
                'temperature',
                'presence_penalty',
                'frequency_penalty',
                'stop',
                'top_p'
            })
        )
        
        async for chunk in run:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield DeltaStep(delta.content)
            
            elif delta.tool_calls is not None:
                for tool_call in delta.tool_calls:
                    function = tool_call.function
                    assert function is not None
                    assert function.name is not None
                    params = self.provider.ensure_json(function.params)
                    assert isinstance(params, dict)
                    yield ActionStep(function.name, params)
            
            else:
                raise NotImplementedError("Unknown API delta")

class OpenAIProvider(Provider):
    '''OpenAI API-based provider.'''
    
    config: dict
    '''Configuration for the provider.'''
    
    client: openai.AsyncClient
    '''Open client connection.'''
    
    def __init__(self, config, client):
        self.config = config
        self.client = client
    
    async def ensure_json(self, data: str) -> Any:
        '''Ensure that the data is parsed as valid json.'''
        
        for i in range(3):
            try:
                # TODO: json schema validation
                return json.loads(data)
            
            except ValueError:
                pass
            
            # TODO: Dependency injection rather than hardcoded model
            
            result = await self.client.chat.completions.create(
                model="gpt-3.5-turbo-0613",
                stream=False,
                messages=[
                    {"role": "system", "content": "Respond with the JSON provided by the user fixed."},
                    {"role": "user", "content": data}
                ]
            )
            
            content = result.choices[0].message.content
            assert content is not None
            data = content
        
        raise ValueError(f"Failed to parse JSON: {data}")
    
    @override
    def __call__(self, prompt: list[ModelMessage]) -> Run:
        '''Build a run to be resolved by the provider.'''
        
        return OpenAIRun(self.config, self.client, prompt)
    
    @override
    @classmethod
    @asynccontextmanager
    async def connect(cls, config: dict):
        '''Connect to the provider.'''
        
        async with httpx.AsyncClient() as http_client:
            async with openai.AsyncClient(
                api_key=config["api_key"],
                http_client=http_client
            ) as openai_client:
                yield OpenAIProvider(config, openai_client)