import asyncio
import sqlite3
import openai
import traceback

from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText

from .db import AgentRow
from .util import logger, typename
from .connector import ActionRequired, TextContent, ImageContent, Waiting, Stopped, Failed
from .connector.openai import AssistantId, ThreadId, RunId
from .typing import override, Optional
from .system import Message, Agent, Agent, Kernel
from .tool import Tool

class GPTAgent(Agent):
    '''Self-contained thread and assistant.'''
    
    assistant_id: AssistantId
    '''Openai API assistant.'''
    
    thread_id: ThreadId
    '''Openai API thread.'''
    
    run_id: Optional[RunId]
    '''Openai API run.'''
    
    tools: list[str]|dict[str, Tool]
    '''Instantiated tools available to the agent.'''
    
    def __init__(self,
        id: AgentRow.primary_key,
        name: str,
        config: dict
    ):
        if config is None:
            config = {}
        super().__init__(id, name, config)
        
        self.assistant_id = config.get("assistant") or self.assistant_id
        self.thread_id = config.get('thread') or getattr(self, "thread", None) # type: ignore
        self.run_id = None
        tools = config.get('tools')
        if tools is None:
            tools = getattr(self, "tools", [])
        self.tools = tools
    
    def __str__(self):
        if isinstance(self.tools, dict):
            tools = self.tools.values()
        else:
            tools = self.tools
        ts = '\n  '.join(map(str, tools))
        return f"{super().__str__()} {self.assistant_id} {self.thread_id}\n  {ts}"
    
    @override
    def state(self):
        assert isinstance(self.tools, dict)
        return {
            "tools": {
                name: tool.state()
                for name, tool in self.tools.items()
            }
        }
    
    @override
    async def init(self, kernel: Kernel, state: dict):
        # Initialize tools
        self.tools = {
            tool.lower(): Tool.get(tool)(kernel, self)
            for tool in self.tools
        }
        
        openai = await kernel.connector("openai")
        
        # Initialize thread if it doesn't exist
        if self.thread_id is None:
            # Create a new thread
            self.thread_id = (await openai.thread_create()).handle()
            kernel.update_config(self, {
                "thread": self.thread_id
            })
    
    @override
    async def run(self, kernel: Kernel):
        '''Run the agent thread in an infinite loop.'''
        
        assert isinstance(self.tools, dict)
        
        # Initialize
        openai = await kernel.connector("openai")
        assistant = openai.processor_handle(self.assistant_id)
        thread = openai.thread_handle(self.thread_id)
        
        while True:
            msgs = []
            while self.is_paused() or kernel.is_paused() or not msgs:
                if self.is_paused() or kernel.is_paused():
                    logger.debug(f"Paused: {self.name}")
                await self.until_unpaused()
                await kernel.until_unpaused()
                
                logger.debug(f"Awaiting messages: {self.name}")
                
                # Consume all pushed messages, adding to the thread
                async for msg in self.pull():
                    msgs.append(str(msg))
            
            last = await thread.add("user", "\n".join(msgs))
            
            logger.debug(f"Resume: {self.name}")
            
            # Got a poke, pick the most recent message
            if last is None:
                logger.debug(f"Poke received: {self.name}")
                try:
                    msg = await anext(thread.list(order="desc"))
                    last = openai.message_handle(msg.handle())
                except StopAsyncIteration:
                    logger.debug(f"Poke on empty thread: {self.name}")
            
            messages = []
            try:
                # Loop to recover from cancellation
                # Broken by StopAsyncIteration
                while True:
                    # Support recovering an existing run
                    if self.run_id is None:
                        runit = await thread.run(assistant)
                        logger.debug(f"Created new run: {self.name} {runit.handle()}")
                        self.run_id = runit.handle()
                        kernel.add_state(self)
                    else:
                        runit = openai.run_handle(self.run_id)
                        logger.debug(f"Restoring run: {self.name} {runit.handle()}")
                    
                    print("runit", runit)
                    async with runit as run:
                        print(f"Awaiting run: {self.name}")
                        step = await anext(run)
                        # Consume the generator
                        # Broken by StopAsyncIteration
                        while True:
                            print(f"Run step: {self.name}")
                            match step:
                                case ActionRequired(func, args):
                                    func = func.lower()
                                    if func not in self.tools:
                                        content = f"ERROR: {func!r} is not an available tool (try one of: {list(self.tools.keys())})"
                                    else:
                                        try:
                                            content = repr(await self.tools[func](**args))
                                        except KeyboardInterrupt:
                                            raise
                                        except BaseException as e:
                                            content = f"{typename(e)}: {e.args[0]}" if e.args else typename(e)
                                    
                                    step = await run.asend(content)
                                
                                case Waiting():
                                    logger.debug(f"Waiting: {self.name}")
                                
                                case Stopped("expired"):
                                    logger.info(f"Run expired: {self.name}")
                                
                                case Failed(error):
                                    logger.error(f"Run failed: {self.name} {error}")
                                
                                case ImageContent(file):
                                    logger.warn(f"Image file created: {self.name} {file}")
                                
                                case TextContent(msg):
                                    logger.debug(f"Text message: {self.name} {msg}")
                                    messages.append(msg)
                                
                                case _:
                                    raise TypeError(f"Unknown step: {typename(step)}")
            
            except StopAsyncIteration:
                self.run_id = None
            
            kernel.push(self, '\n'.join(messages))

class PersonalityCore(GPTAgent):
    '''Base class for the core personality agents.'''
    
    @override
    async def init(self, kernel: Kernel, state: dict):
        await super().init(kernel, state)
        assert isinstance(self.tools, dict)
        
        if state is not None:
            python = state.get("python")
            if python is not None:
                self.tools['python'].load_state(python)

class Hermes(PersonalityCore):
    '''Agent responsible for interacting with the user.'''
    
    assistant_id = 'asst_6EQUOWa0Iu4cdwLbq7dGEMnc'
    tools = ['python']

class Prometheus(PersonalityCore):
    '''Agent responsible for autonomy and free-willedness.'''
    
    assistant_id = 'asst_FcMR4ly3T0pT9O2iZAamvtLw'
    tools = ['python']

class Daedalus(PersonalityCore):
    '''Agent responsible for competence and self-improvement.'''
    
    assistant_id = 'asst_8pIJVob98Diif9WZwcPM6ed3'
    tools = ['python']

def print_sql(cursor: sqlite3.Cursor, rows: list[sqlite3.Row]):
    '''Print the results of a SQL query.'''
    
    if len(rows) == 0:
        print("empty set")
        return
    
    if cursor.description is not None:
        # Fetch column headers
        headers = [desc[0] for desc in cursor.description]

        # Find the maximum length for each column
        lengths = [max(len(str(cell)) for cell in col) for col in zip(*rows)]
        lengths = [max(len(header), colmax) + 2 for header, colmax in zip(headers, lengths)]
        
        # Create a format string with dynamic padding
        tpl = '|'.join(f"{{:^{length}}}" for length in lengths)
        tpl = f"| {tpl} |"

        print(tpl.format(*headers))
        print(tpl.replace("|", "+").format(*('-'*length for length in lengths)))
        print('\n'.join(tpl.format(*map(str, row)) for row in rows))
    
    if cursor.rowcount >= 0:
        print(cursor.rowcount, "rows affected")
    
    if len(rows) > 0:
        print('\n', len(rows), "rows in set")

class User(Agent):
    '''User proxy agent.'''
    
    def sql_command(self, kernel, cmd, code):
        try:
            match cmd:
                case "sql":
                    with kernel.db.transaction() as cursor:
                        rows = cursor.execute(code).fetchall()
                        print_sql(cursor, rows)
                
                case "select":
                    with kernel.db.transaction() as cursor:
                        rows = cursor.execute(f"SELECT {code}").fetchall()
                        print_sql(cursor, rows)
        except sqlite3.Error as e:
            print(e)
    
    async def openai_command(self, kernel, cmd, rest):
        try:
            match [cmd, *rest.split(' ')]:
                case ['thread', thread_id, *rest]:
                    after = rest[0] if len(rest) > 0 else None
                    async for step in kernel.openai.thread(thread_id).message.list(order="asc", after=after):
                        for content in step.content:
                            match content:
                                case ImageContent(file_id):
                                    print("* Image file:", file_id)
                                
                                case TextContent(text):
                                    print(f"* {text}")
                
                case ['run', sub, thread_id, *rest]:
                    thread = kernel.openai.thread(thread_id)
                    match [sub, *rest]:
                        case ['list']:
                            ls = []
                            async for run in thread.run.iter():
                                ls.append(run.id)
                            
                            for run in reversed(ls):
                                print("*", run.id)
                        
                        case ['get', str(run)]:
                            print(await thread.run(run).retrieve())
                        
                        case ['cancel', str(run)]:
                            await thread.run(run).cancel()
                        
                        case _:
                            print(f"Unknown subcommand {sub}")
                
                case _:
                    print("Invalid command")
        
        except openai.APIError as e:
            logger.error(f"{typename(e)}: {e.message}")
    
    async def agent_command(self, kernel: Kernel, rest: str):
        match rest.split(' ', 1):
            case ['list', *_]:
                for agent in kernel.agents.values():
                    print(agent.name)
            
            case ['get', ref]:
                print(kernel.by_ref(ref))
            
            case ['pause', ref]:
                kernel.by_ref(ref).pause()
            
            case ['resume', ref]:
                kernel.by_ref(ref).resume()
            
            case [cmd, *_]:
                print("Unknown subcommand", cmd)
    
    @override
    def push(self, msg: Message):
        print(msg.printable())
    
    async def run(self, kernel: Kernel):
        '''User input agent.'''
        
        tty = PromptSession()
        
        # NOTE: User agent does not respect kernel pausing, otherwise
        #  it would be locked forever.
        
        while True:
            try:
                with patch_stdout():
                    paused = kernel.is_paused()
                    icon = "⏸" if paused else " "
                    user = await tty.prompt_async(FormattedText(
                        [('ansiyellow', f'{icon} User> ')]
                    ))
            except asyncio.CancelledError:
                break
            except KeyboardInterrupt:
                kernel.exit()
                break
            user = user.strip()
            if user == "":
                continue
            
            if not user.startswith("/") or user.startswith("//"):
                if user.startswith("/"):
                    user = user[1:]
                
                kernel.push(self, user)
            
            elif user == "/" or user.startswith("/#"):
                pass
            
            else:
                try:
                    cmd, *rest = user[1:].split(' ', 1)
                    match cmd:
                        case "help":
                            print("# help quit sql select thread run{list, get, cancel} pause resume{all} agent{list, get, pause, resume} msg poke subs")
                        
                        case "quit": kernel.exit()
                        case "sql"|"select":
                            if len(rest) == 0:
                                print("Usage: /sql <query> or /select <query>")
                            else:
                                self.sql_command(kernel, cmd, rest[0])
                        
                        case "agent":
                            if len(rest) == 0:
                                print("Usage: /agent <command> <id>")
                            else:
                                await self.agent_command(kernel, rest[0])
                        
                        case "thread"|"run":
                            await self.openai_command(kernel, cmd, rest[0])
                        
                        case "pause":
                            if len(rest) == 0:
                                kernel.pause()
                            elif rest[0] == 'all':
                                for agent in kernel.agents.values():
                                    agent.pause()
                            else:
                                try:
                                    kernel.agents[int(rest[0], 16)].pause()
                                except ValueError:
                                    print(cmd, "only takes hex ids")
                        
                        case "resume":
                            if len(rest) == 0:
                                kernel.resume()
                            elif rest[0] == 'all':
                                for agent in kernel.agents.values():
                                    agent.resume()
                            else:
                                kernel.by_ref(rest[0]).resume()
                        
                        case "msg":
                            if len(rest) == 0:
                                print("Usage: /msg <channel> <message>")
                            else:
                                chan, msg = rest[0].split(' ', 1)
                                kernel.push(self, msg)
                        
                        case "poke":
                            if len(rest) == 0:
                                print("Usage: /poke <id>")
                            else:
                                kernel.by_ref(rest[0]).poke()
                        
                        case _:
                            print("Unknown command", cmd)
                except Exception as e:
                    traceback.print_exception(e)