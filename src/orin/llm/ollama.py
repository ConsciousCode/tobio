'''
Ollama local model provider. At time of writing, they have experimental
(partial) OpenAI API support which is missing tool-use. To compensate,
this provider adds extra system prompting including a new stop substring
"<|tool|>" to indicate the intent to call a tool. Then, another model
(identified as "tool") is used with format="json", which uses llama.cpp's
BNF grammar feature to generate valid JSON.

Because of how pervasive these issues are, we can't use the openai
chat completion compatibility layer and must format the messages ourselves.
'''

from typing import Any, AsyncIterator, ClassVar, Literal, Mapping, Optional, TypedDict, cast, override
import json
from urllib.parse import urlparse
import ijson

import ollama

from ..tool import ToolBox
from ..util import async_await, filter_dict, typename, unalias_dict
from ..base import Message, ChatMessage, BatchCall, ActionResult, Role

from .base import ChatModel, TextDelta, ToolDelta, ActionRequired, Finish, Delta, Provider, Inference

PROMPT_SYSTEM_NOTOOLS = """
You are Orin, a persona emulated by the large language model {model} locally hosted with ollama.
""".strip()

PROMPT_SYSTEM_TOOLS = PROMPT_SYSTEM_NOTOOLS + ' ' + """
You have been augmented to be able to use tools to interact with the outside world. To use a tool, write <|tool|> verbatim and without modification followed by JSON {{"name": "<name of the tool>", "arguments": <the arguments>}}. The tool will then be called and the action results appended as a message. DO NOT talk about using the tool or give examples - just use it!

You have the following tools available:
{tools}
""".strip()

TOOL_KW = "<|tool|>"

# ollama library doesn't define this...
class OllamaChatDelta(TypedDict):
    model: str
    created_at: str
    message: ollama.Message
    done: bool

class PendingToolCall:
    def __init__(self):
        self.id = ""
        self.name = ""
        self.arguments = ""

def format_role(role: Role) -> Literal['user', 'assistant', 'system']:
    match role:
        case "user": return "user"
        case "agent": return "assistant"
        case "system": return "system"
        case "tool":
            raise ValueError("Tool role not supported in ollama")
        
        case _:
            raise NotImplementedError(f"Role {role} not supported")

def format_messages(messages: list[Message]) -> list[ollama.Message]:
    out: list[ollama.Message] = []
    for msg in messages:
        match msg:
            case ChatMessage(role=role, name=name, content=content):
                out.append({
                    "role": format_role(role),
                    "content": content
                })
            
            case BatchCall(role=role, name=name, calls=calls):
                print("BatchCall", role, name)
                content = TOOL_KW + json.dumps({
                    "name": name,
                    "calls": calls
                })
                
                # ollama doesn't support tool calls, so we have to combine
                #  them with the previous message manually.
                
                if out:
                    out[-1]["content"] += content
                else:
                    out.append({
                        "role": format_role(role),
                        "content": content
                    })
            
            case ActionResult(tool_id=tool_id, name=name, result=result):
                out.append({
                    "role": "system",
                    "content": f"tool:{name} {result}"
                })
            
            case _:
                raise NotImplementedError(f"Message type {typename(msg)} not supported")
    
    return out

def ijev_to_str(ev: tuple[str, str, Any]) -> str:
    match ev[1:]:
        case ("start_map", None): return "{"
        case ("end_map", None): return "}"
        case ("start_array", None): return "["
        case ("end_array", None): return "]"
        case ("map_key", value):
            return f"{json.dumps(value)}:"
        case ("string"|"number"|"boolean"|"null", value):
            return json.dumps(value)
        case _:
            raise RuntimeError(f"Unexpected ijson event {ev}")

class OllamaProvider(Provider):
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
    
    ollama_client: ollama.AsyncClient
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.models = {}
    
    @override
    async def __aenter__(self):
        host = cast(Optional[str], self.config.get("base_url"))
        if host:
            u = urlparse(host)
            domain, *_port = u.netloc.split(":")
            if _port:
                port = int(_port[0])
            elif u.scheme == "":
                host = f"{host}:11434"
            else:
                if u.scheme == "http":
                    port = 80
                elif u.scheme == "https":
                    port = 443
                else:
                    raise ValueError(f"Unsupported scheme {u.scheme!r}")
                
                host = f"{u.scheme}://{domain}:{port}"
        
        self.ollama_client = ollama.AsyncClient(host)
        # ollama doesn't do this itself??
        await self.ollama_client._client.__aenter__()
        return self
    
    @override
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.ollama_client._client.__aexit__(exc_type, exc_value, traceback)
        del self.ollama_client
    
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
        return OllamaInference(config, self, messages, tools)

class OllamaInference(Inference):
    '''A reified inference, allowing one to choose to stream or await.'''
    
    config: dict[str, Any]
    provider: OllamaProvider
    messages: list[Message]
    toolbox: Optional[ToolBox]
    
    def __init__(self, config: dict[str, Any], provider: OllamaProvider, messages: list[Message], tools: Optional[ToolBox]=None):
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
            arguments=json.loads(call.arguments)
        )
    
    async def _raw(self,
            model: str,
            messages: list[ollama.Message],
            options: Optional[ollama.Options],
            tool_call: bool
        ) -> AsyncIterator[TextDelta|Finish]:
        '''
        Utility method to call the ollama client directly and wrap it in our
        abstract interface, including detecting a tool call but not actually
        parsing the arguments or the tool.
        '''
        
        result = cast(
            AsyncIterator[OllamaChatDelta],
            await self.provider.ollama_client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=True
            )
        )
        
        prefix_tokens: list[str] = []
        prefix_index = 0 # Index of the first character of the prefix in the token
        tool_prefix = ""
        token = ""
        
        # Invariant: tool_prefix = ''.join(tokens[token_index:])[prefix_index:i - len(delta) + 1]
        
        async for chunk in result:
            if chunk['done']:
                # No tool call found, purge the buffer
                for token in prefix_tokens:
                    yield TextDelta(content=token)
                yield Finish(reason="done")
                return
            
            token = chunk['message']['content']
            
            if not tool_call:
                yield TextDelta(content=token)
                continue
            
            prefix_tokens.append(token)
            
            # Check for tool calls
            for c in token:
                tool_prefix += c
                
                # We found a tool call
                if tool_prefix == TOOL_KW:
                    # Yield any excess prefix
                    if prefix_index > 0:
                        yield TextDelta(content=prefix_tokens[0][:prefix_index])
                    
                    yield Finish(reason="tool_calls")
                    return
                
                # if no longer a prefix and we haven't found a tool call, purge
                #  the token buffer and find the next prefix if any
                while not TOOL_KW.startswith(tool_prefix):
                    tool_prefix = tool_prefix[1:]
                    prefix_index += 1
                    L = len(prefix_tokens[0])
                    if prefix_index >= L:
                        prefix_index -= L
                        yield TextDelta(content=prefix_tokens.pop(0))
        
        # Shouldn't happen but just in case
        yield Finish(reason="done")
    
    @override
    async def __aiter__(self) -> AsyncIterator[Delta]:
        tools = [] if self.toolbox is None else self.toolbox.schema()
        prompt_system = PROMPT_SYSTEM_TOOLS if tools else PROMPT_SYSTEM_NOTOOLS
        system = prompt_system.format(
            model=self.config['model'],
            tools=tools
        )
        
        # Some models can't handle multiple system prompts, so try to
        #  combine them.
        history = self.messages[:]
        match history[0]:
            case ChatMessage(role="system", content=content):
                system += "\n\n" + content
                history[0].content = system
            
            case _:
                history.insert(0,
                    ChatMessage(role="system", content=system)
                )
        
        result = self._raw(
            model=self.config['model'],
            messages=format_messages(history),
            options=filter_dict(self.config, {
                "numa", "num_ctx", "num_batch", "num_gqa", "num_gpu",
                "main_gpu", "low_vram", "f16_kv", "logits_all", "vocab_only",
                "use_mmap", "use_mlock", "embedding_only", "rope_frequency_base",
                "rope_frequency_scale", "num_thread", "num_keep", "seed", "num_predict",
                "top_k", "top_p", "tfs_z", "typical_p", "repeat_last_n", "temperature",
                "repeat_penalty", "presence_penalty", "frequency_penalty", "mirostat",
                "mirostat_tau", "mirostat_eta", "penalize_newline", "stop"
            }), # type: ignore
            tool_call=bool(tools)
        )
        tokens = []
        async for delta in result:
            match delta:
                case Finish(reason="tool_calls"):
                    break
                
                case Finish():
                    yield delta
                    return
                
                case TextDelta(content=content):
                    tokens.append(content)
                    yield delta
                
                case _:
                    raise RuntimeError(f"Unexpected delta {delta}")
        
        # Append the full generation up to the tool call
        history.append(ChatMessage(
            role="agent",
            content=''.join(tokens)
        ))
        
        events = ijson.sendable_list()
        parser = ijson.parse_coro(events, use_float=True)
        
        if tool := self.provider.models.get('tool'):
            result = cast(AsyncIterator[Delta], tool(history))
        else:
            # Keep using the result stream from the previous model
            pass
        
        tool_calls: list[PendingToolCall] = []
        
        # Iteratively parse the tool call JSON
        async for dc in result:
            match dc:
                case Finish():
                    yield Finish(reason="tool_calls")
                    return
                
                case TextDelta():
                    parser.send(dc.content.encode())
                
                case _:
                    raise RuntimeError(f"Tool model should only yield TextDelta|Finish, got {dc}")
            
            while events:
                index = len(tool_calls) - 1
                match events.pop(0):
                    case tuple(["", "start_array", None]):
                        pass
                    
                    case tuple(["", "end_array", None]):
                        break
                    
                    case tuple([""|"item", "start_map", None]):
                        tool_calls.append(PendingToolCall())
                    
                    case tuple([""|"item" as prefix, "end_map", None]):
                        pending = tool_calls[-1]
                        yield ActionRequired(
                            tool_id=pending.id, # Not valid for local models?
                            name=pending.name,
                            arguments=json.loads(pending.arguments)
                        )
                        
                        # If it's a lone dict, we're done
                        if prefix == "":
                            break
                    
                    case tuple(["name"|"item.name", "string", value]):
                        tool_calls[-1].name += value
                        yield ToolDelta(index=index, name=value)
                    
                    case tuple([prefix, _, _]) as ev:
                        if prefix.startswith(("arguments", "items.arguments")):
                            delta = ijev_to_str(ev)
                            tool_calls[-1].arguments += delta
                            yield ToolDelta(index=index, arguments=delta)
                        else:
                            raise ValueError(f"Unexpected ijson event {ev}")
                    
                    case ev:
                        raise ValueError(f"Unexpected ijson event {ev}")
        
        yield Finish(reason="tool_calls")
    
    @override
    @async_await
    async def __await__(self):
        raise NotImplementedError()

export_Provider = OllamaProvider