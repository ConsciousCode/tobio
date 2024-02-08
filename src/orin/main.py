'''
This file should only contain the parts most core to Orin with near-zero
coupling to other systems (UI, database, etc). It shouldn't even run as
a standalone program, but rather be imported by other systems.
'''

from typing import AsyncGenerator, Literal

import openai
import httpx

from orin.db.base import ActionData, ToolData

from .db import Database, Step
from .tool import ToolBox
from .util import coroutine, logger
from .llm import ActionRequired, Finish, Message, ChatMessage, ToolResponse, Provider, OpenAIProvider, TextDelta, ToolDelta

type Role = Literal['user', 'assistant', 'system', 'tool']

def _format_step(step: Step) -> Message:
    if step.kind == "text":
        return ChatMessage(
            role=step.author.role, # type: ignore
            name=step.author.name,
            content=step.content
        )
    elif step.kind == "tool":
        return ChatMessage(
            role=step.author.role, # type: ignore
            name=step.author.name,
            content=step.content
        )
    elif step.kind == "action":
        return ToolResponse(
            tool_id=step.data.tool_id,
            content=step.data
        )

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
        
        self.provider = OpenAIProvider(config)
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
    async def stream_step(self, step: Step.Unbound) -> AsyncGenerator[None, str]:
        '''Iteratively stream a new step to the memory.'''
        
        assert step.status in {None, "stream"}
        step.status = "stream"
        row = self.db.add_step(step)
        self.history.append(row)
        while True:
            self.db.set_step_content(row, (yield))
    
    async def add_step(self, step: Step.Unbound) -> Step:
        '''Append a discrete step to history.'''
        
        row = self.db.add_step(step)
        self.history.append(row)
        return row
    
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
                return await self.add_step(Step.Unbound(
                    author=self.db.put_author("system", "summary"),
                    kind="text",
                    text=summary
                ))
            
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
        await self.add_step(Step.Unbound(
            author=self.db.put_author("user", None),
            text=prompt
        ))
        
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
                        step = await self.stream_step(Step.Unbound(
                            author=self.db.put_author("assistant", None),
                            text=content
                        ))
                    else:
                        await step.asend(content)
                    
                    yield content
                case ToolDelta(id=id, name=name, arguments=args):
                    # Tools are too fragile to stream, don't want to record
                    #  a partial tool call (ie invalid JSON) in history
                    step = None
                    print("Tool call", id, name, args)
                case ActionRequired(tool_id=tool_id, name=name, arguments=args):
                    print("Action required", name, args)
                    await self.add_step(Step.Unbound(
                        author=self.db.put_author("system", "action"),
                        data=ToolData(
                            tool_id=tool_id,
                            name=name,
                            arguments=args
                        )
                    ))
                    if name in self.toolbox:
                        await self.add_step(Step.Unbound(
                            author=self.db.put_author("system", "action"),
                            data=ActionData(
                                tool_id=tool_id,
                                result=await self.toolbox[name](**(args or {}))
                            )
                        ))
                    else:
                        await self.add_step(Step.Unbound(
                            author=self.db.put_author("system", "error"),
                            text=f"Tool {name!r} not found"
                        ))
                
                case Finish(finish_reason=reason):
                    print("Finish", reason)
                    break
                
                case _:
                    raise NotImplementedError(delta)
        
        await self.truncate(self.config['memory']['history_limit'])
        
        print(len(self.history), "messages thus far")
    