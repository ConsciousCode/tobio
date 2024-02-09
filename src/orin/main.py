'''
This file should only contain the parts most core to Orin with near-zero
coupling to other systems (UI, database, etc). It shouldn't even run as
a standalone program, but rather be imported by other systems.
'''

from typing import AsyncGenerator, Iterator, Optional, cast

from pydantic import TypeAdapter

from .base import ConfigToml, Message, ChatMessage, ActionResult, BatchCall, ToolCall
from .db import Database, Step
from .tool import ToolBox
from .util import coroutine, logger
from .llm import ActionRequired, Finish, Connector, TextDelta, ToolDelta, Delta

def format_steps(steps: list[Step]) -> Iterator[Message]:
    for step in steps:
        match step.kind:
            case "text":
                yield ChatMessage(
                    role=step.author.role,
                    name=step.author.name,
                    content=step.content
                )
            
            case "tool":
                yield BatchCall(
                    role=step.author.role,
                    name=step.author.name,
                    calls=cast(list[ToolCall], step.data)
                )
            
            case "action":
                yield cast(ActionResult, step.data)

class Orin:
    config: ConfigToml
    
    db: Database
    toolbox: ToolBox
    
    history: list[Step]
    persona: str
    
    def __init__(self, config: dict):
        self.config = TypeAdapter(ConfigToml).validate_python(config)
        
        self.connector = Connector(self.config)
        self.db = Database(config['memory'])
        self.toolbox = ToolBox()
        
        self.history = self.db.get_history()
        
        with open(config['persona']) as f:
            self.persona = f.read()
    
    async def __aenter__(self):
        await self.connector.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.connector.__aexit__(exc_type, exc_value, traceback)
    
    @coroutine
    async def stream_step(self, step: Step.Unbound) -> AsyncGenerator[None, Delta]:
        '''Iteratively stream a new step to the memory.'''
        
        assert step.status in {None, "stream"}
        step.status = "stream"
        row = self.db.add_step(step)
        self.history.append(row)
        
        if not isinstance(step.content, str):
            raise TypeError(f"Can only stream text steps")
        
        content = step.content
        while True:
            match delta := (yield):
                case TextDelta():
                    content += delta.content
                    self.db.set_step_content(row, content)
                
                case Finish():
                    self.db.finalize_step(row)
                    break
    
    async def atomic_step(self, step: Step.Unbound) -> Step:
        '''Append a discrete step to history.'''
        
        assert step.status in {None, "atom"}
        step.status = "atom"
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
                summary = await self.connector.models['summarize'](
                    self.prompt(), self.toolbox
                )
                logger.info("Summary: %s", summary)
                return await self.atomic_step(Step.Unbound(
                    parent_id=newest[-1].id,
                    author=self.db.put_author("system", "summary"),
                    content=summary
                ))
            
            self.history = newest
    
    def get_history(self) -> list[Step]:
        return self.history
    
    def prompt(self):
        return [
            ChatMessage(role="system", name="prompt", content=self.persona),
            *format_steps(self.history)
        ]
    
    async def cmd(self, cmd, args):
        match cmd:
            case "history":
                for msg in self.get_history():
                    yield f"{msg}\n"
            
            case "sql":
                result = self.db.raw_format(args)
                for line in result.splitlines():
                    yield f"{line}\n"
            
            case "select":
                result = self.db.raw_format(f"SELECT {args}")
                for line in result.splitlines():
                    yield f"{line}\n"
            
            case "poke":
                async for delta in self.chat(None):
                    yield delta
            
            case _:
                yield f"Command {cmd!r} not found"
    
    async def chat(self, prompt: Optional[str]) -> AsyncGenerator[str, None]:
        last_id = self.history[-1].id if self.history else None
        
        if prompt is not None:
            await self.atomic_step(Step.Unbound(
                parent_id=last_id,
                author=self.db.put_author("user", None),
                content=prompt
            ))
            last_id = self.history[-1].id
        
        it = aiter(self.connector.models['chat'](
            self.prompt(), self.toolbox
        ))
        step = None
        while True:
            match delta := await anext(it):
                case TextDelta(content=content):
                    if step is None:
                        step = await self.stream_step(Step.Unbound(
                            parent_id=last_id,
                            author=self.db.put_author("assistant", None),
                            content=content
                        ))
                        last_id = self.history[-1].id
                    else:
                        await step.asend(delta)
                    
                    yield content
                
                case ToolDelta(tool_id=tool_id, name=name, arguments=args):
                    # TODO: Maybe eventually stream tools? The value is pretty
                    #  low since it can always regenerate.
                    if step is not None:
                        await step.asend(Finish(reason="tool_calls"))
                    step = None
                
                case ActionRequired(tool_id=tool_id, name=name, arguments=args):
                    print("Action required", name, args)
                    await self.atomic_step(Step.Unbound(
                        parent_id=last_id,
                        author=self.db.put_author("assistant", None),
                        content=ToolCall(
                            id=tool_id,
                            name=name,
                            arguments=args
                        )
                    ))
                    last_id = self.history[-1].id
                    
                    # Since the streaming step is already in history, we can
                    #  simply append the responses as they come in.
                    if name in self.toolbox:
                        result = await self.toolbox[name](**(args or {}))
                    else:
                        result = f"Tool {name!r} not found"
                    
                    await self.atomic_step(Step.Unbound(
                        parent_id=last_id,
                        author=self.db.put_author("system", "action"),
                        content=ActionResult(
                            tool_id=tool_id,
                            name=name,
                            result=result
                        )
                    ))
                    last_id = self.history[-1].id
                
                case Finish(reason=reason):
                    break
                
                case _:
                    raise NotImplementedError(delta)
        
        await self.truncate(self.config['memory']['history_limit'])
        
        print(len(self.history), "messages thus far")
    