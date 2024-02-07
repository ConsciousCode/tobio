'''
This file should only contain the parts most core to Orin with near-zero
coupling to other systems (UI, database, etc). It shouldn't even run as
a standalone program, but rather be imported by other systems.
'''

from typing import AsyncGenerator, Literal
import json

import openai
import httpx

from .db import Author, Database, Step, StepKind
from .tool import ToolBox
from .util import coroutine, logger
from .llm import ActionRequired, Finish, Message, Provider, TextDelta, ToolDelta

type Role = Literal['user', 'assistant', 'system', 'tool']

class Orin:
    config: dict
    
    provider: Provider
    db: Database
    toolbox: ToolBox
    
    http_client: httpx.AsyncClient
    openai_client: openai.AsyncClient
    
    history: list[Step]
    
    def __init__(self, config: dict):
        self.config = config
        
        self.provider = Provider(config)
        self.db = Database(config['memory'])
        self.toolbox = ToolBox()
        
        self.history = self.db.get_history()
    
    async def __aenter__(self):
        await self.provider.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.provider.__aexit__(exc_type, exc_value, traceback)
    
    def _format_step(self, step: Step) -> Message:
        return Message(
            role=step.author.role,
            name=step.author.name,
            content=step.content
        )
    
    @coroutine
    async def stream_step(self, author: Author, kind: StepKind, content: str) -> AsyncGenerator[None, str]:
        '''Iteratively stream a new step to the memory.'''
        
        step = self.db.add_step(author, kind, content)
        self.history.append(step)
        while True:
            self.db.stream_step(step, (yield))
    
    async def add_step(self, author: Author, kind: StepKind, content: str) -> Step:
        '''Append a discrete step to history.'''
        
        step = self.db.add_step(author, kind, content)
        self.history.append(step)
        return step
    
    async def truncate(self, limit: int):
        if len(self.history) > limit:
            oldest, newest = self.history[:-limit], self.history[-limit:]
            
            need_summary = any(m.author.name == "summary" for m in oldest)
            
            # Are we due for a summary based on frequency?
            if len(newest):
                # Find the last summary message
                for i, m in enumerate(reversed(newest)):
                    if m.author.name == "summary":
                        break
                
                summary_freq = limit//self.config['memory']['summary_count']
                if len(newest) - i <= summary_freq: # type: ignore
                    need_summary = True
            
            if need_summary:
                summary = await self.provider.models['summarize'](
                    [self._format_step(step) for step in self.history],
                    self.toolbox
                )
                logger.info("Summary: %s", summary)
                return await self.add_step(
                    self.db.put_author("system", "summary"),
                    "text",
                    summary
                )
            
            self.history = newest
    
    async def get_history(self) -> list[Step]:
        return self.history
    
    async def cmd(self, cmd, args):
        match cmd:
            case "history":
                return '\n'.join(map(str, await self.get_history()))
            
            case "sql":
                return self.db.raw_format(args)
            
            case "select":
                return self.db.raw_format(f"SELECT {args}")
            
            case _:
                return f"Command {cmd!r} not found"
    
    async def chat(self, prompt: str) -> AsyncGenerator[str, None]:
        await self.add_step(self.db.put_author("user", None), "text", prompt)
        
        it = aiter(self.provider.models['chat'](
            [self._format_step(step) for step in self.history],
            self.toolbox
        ))
        step = None
        while True:
            delta = await it.__anext__()
            match delta:
                case TextDelta(content=content):
                    if step is None:
                        step = await self.stream_step(
                            self.db.put_author("assistant", None),
                            "text",
                            content
                        )
                    else:
                        await step.asend(content)
                    
                    yield content
                case ToolDelta(id=id, name=name, arguments=args):
                    # Tools are too fragile to stream, don't want to record
                    #  a partial tool call (ie invalid JSON) in history
                    step = None
                    print("Tool call", id, name, args)
                case ActionRequired(name=name, arguments=args):
                    print("Action required", name, args)
                    await self.add_step(
                        self.db.put_author("system", "tool"),
                        "tool",
                        f'Assistant used {name} with {json.dumps(args)}'
                    )
                    if name in self.toolbox:
                        author = self.db.put_author("tool", name)
                        kind = "tool"
                        content = json.dumps(await self.toolbox[name](**(args or {})))
                    else:
                        author = self.db.put_author("system", "error")
                        kind = "text"
                        content = f"Tool {name!r} not found"
                    
                    await self.add_step(author, kind, content)
                
                case Finish(finish_reason=reason):
                    print("Finish", reason)
                    break
                
                case _:
                    raise NotImplementedError(delta)
        
        await self.truncate(self.config['memory']['history_limit'])
        
        print(len(self.history), "messages thus far")
    
    async def chat_cmd(self, message: str):
        if message.startswith("/") and not message.startswith("//"):
            cmd, *args = message[1:].split(" ", 1)
            output = await self.cmd(cmd, args[0] if args else "")
            for line in output.splitlines():
                yield f"{line}\n"
        else:
            async for delta in self.chat(message):
                yield delta