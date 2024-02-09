'''
Code for interacting with language models.
'''

from typing import Any, AsyncIterator, ClassVar, Optional, cast, override
import json

import httpx
import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from ..tool import ToolBox
from ..util import async_await, filter_dict, typename, unalias_dict
from ..base import Message, ChatMessage, BatchCall, ActionResult

from .base import ChatModel, TextDelta, ToolDelta, ActionRequired, Finish, Delta, Provider, Inference

PROMPT_ENSURE_JSON = "The previous messages are successive attempts to produce valid JSON but have at least one error. Respond only with the corrected JSON."

PROMPT_SUMMARY = "You are the summarization agent of Orin. Summarize the conversation thus far."

class PendingToolCall:
    def __init__(self):
        self.id = ""
        self.name = ""
        self.arguments = ""

def format_to_openai(msg: Message) -> ChatCompletionMessageParam:
    match msg:
        case ChatMessage(role=role, name=name, content=content):
            ob = {
                "role": role,
                "content": content
            }
            if name is not None:
                ob["name"] = name
            return ob # type: ignore
        
        case BatchCall(role=role, name=name, calls=calls):
            print("BatchCall", role, name)
            ob = {
                "role": role,
                "tool_calls": [
                    {
                        "id": tool.id,
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            # Raised an error without str()?
                            "arguments": str(tool.arguments)
                        }
                    } for tool in calls
                ]
            }
            if name is not None:
                ob["name"] = name
            return ob # type: ignore
        
        case ActionResult(tool_id=tool_id, name=name, result=result):
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": name,
                "content": result
            } # type: ignore
        
        case _:
            raise NotImplementedError(f"Message type {typename(msg)} not supported")

class OpenAIProvider(Provider):
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
    
    models: dict[str, ChatModel]
    
    http_client: httpx.AsyncClient
    openai_client: openai.AsyncClient
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.models = {}
    
    @override
    async def __aenter__(self):
        self.http_client = httpx.AsyncClient()
        self.openai_client = openai.AsyncClient(
            api_key=self.config['api_key'],
            base_url=self.config.get("base_url"),
            http_client=await self.http_client.__aenter__()
        )
        await self.openai_client.__aenter__()
        return self
    
    @override
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
    
    @override
    def model(self, model: str, config: dict) -> dict:
        return {
            "model": model,
            **filter_dict(
                unalias_dict(config, self._aliases),
                self._keys
            )
        }
    
    @override
    def chat(self, config: dict, messages: list[Message], tools: Optional[ToolBox]=None) -> Inference:
        return OpenAIInference(config, self, messages, tools)

class OpenAIInference(Inference):
    '''A reified inference, allowing one to choose to stream or await.'''
    
    config: dict[str, Any]
    provider: OpenAIProvider
    messages: list[Message]
    toolbox: Optional[ToolBox]
    
    def __init__(self, config: dict[str, Any], provider: OpenAIProvider, messages: list[Message], tools: Optional[ToolBox]=None):
        self.config = config
        self.provider = provider
        self.messages = messages
        self.toolbox = tools
    
    async def _action(self, call: PendingToolCall) -> ActionRequired:
        '''Utility method to generate an ActionRequired response.'''
        
        print("ActionRequired:", call.name, call.arguments)
        return ActionRequired(
            tool_id=call.id,
            name=call.name,
            arguments=await self.provider.ensure_json(call.arguments)
        )
    
    @override
    async def __aiter__(self) -> AsyncIterator[Delta]:
        history = list(map(format_to_openai, self.messages))
        tools = [] if self.toolbox is None else self.toolbox.render()
        result = cast(
            openai.AsyncStream[ChatCompletionChunk],
            await self.provider.openai_client.chat.completions.create(
                **self.config,
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
                yield Finish(reason=reason)
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
                    index=tc.index,
                    tool_id=tc.id,
                    name=name,
                    arguments=arguments
                )
    
    @override
    @async_await
    async def __await__(self):
        result = await self.provider.openai_client.chat.completions.create(
            messages=list(map(format_to_openai, self.messages)),
            **self.config,
            stream=False
        )
        
        return cast(str, result.choices[0].message.content)

export_Provider = OpenAIProvider