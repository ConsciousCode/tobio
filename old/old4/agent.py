'''
Agents are the main components of the system. They act within the system by
pushing messages to each other. A typical agent will use a model for
inference.
'''

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Protocol, override

from .provider import ModelMessage, SystemMessage, AgentMessage, UserMessage, ActionStep, DeltaStep
from .util import read_file
from .db import Message

if TYPE_CHECKING:
    from .kernel import Kernel, Model

MAX_UNSUMMARIZED = 20
'''Maximum number of messages before summarizing.'''
REST_UNSUMMARIZED = 4
'''Number of messages to keep after summarizing.'''

PERSONA_FILE = 'persona.md'
'''File containing the persona for the LLM.'''
THREAD_ID = 1
'''ID of the thread to use.'''

class Agent(Protocol):
    '''Agent interface.'''
    
    id: int
    '''ID of the agent to be added by the kernel.'''
    
    name: str
    '''Name of the agent to be added by the kernel.'''
    
    config: dict
    '''Configuration for the agent.'''
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
    
    def state_dict(self) -> dict:
        '''Return a dictionary of the agent state.'''
        ...
    
    def push(self, channel: str, message: Optional[Message]):
        '''
        Push a message to the agent. If the message is None, the agent will
        resume and produce a new message, even if there are no new messages.
        '''
    
    def thread(self, channel: str) -> Thread:
        pass
    
    async def run(self, kernel: 'Kernel'):
        '''Run the agent.'''
    
    async def command(self, command: str) -> str:
        '''Run a command on the agent.'''
        ...

class BaseAgent(ABC, Agent):
    '''Base agent class implementation.'''
    
    mq: asyncio.Queue[Optional[Message]]
    '''Message queue.'''
    
    registry: dict[str, type['BaseAgent']] = {}
    '''Registry of agent types.'''
    
    def __init_subclass__(cls):
        cls.registry[cls.__name__] = cls
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.mq = asyncio.Queue()
    
    async def pull(self):
        '''
        Wait for at least one message to be pushed to the queue, then yield
        all of them.
        '''
        
        if msg := await self.mq.get():
            yield msg
        
        while not self.mq.empty():
            if msg := self.mq.get_nowait():
                yield msg
    
    @override
    def state_dict(self) -> dict:
        return {}
    
    @override
    def push(self, channel: str, message: Optional[Message]):
        self.mq.put_nowait(message)
    
    @override
    @abstractmethod
    async def run(self, kernel: 'Kernel'): ...
    
    @override
    async def command(self, command: str) -> str:
        return "Agent has no commands."

class Broca(BaseAgent):
    '''Main agent for verbal communication and seat of consciousness.'''
    
    config: dict
    '''Configuration for the agent.'''
    
    buffer: list[ModelMessage]
    '''Buffer of messages seen by the LLM.'''
    
    chat_model: 'Model'
    '''Chat model.'''
    
    summarize_model: 'Model'
    '''Summarization model.'''
    
    kernel: 'Kernel'
    '''Kernel the agent is running in.'''
    
    def __init__(self, config):
        super().__init__(config)
        
        self.persona = read_file(config.get('persona', PERSONA_FILE))
        
        self.max_unsummarized = config.get('max_unsummarized', MAX_UNSUMMARIZED)
        self.rest_unsummarized = config.get('rest_unsummarized', REST_UNSUMMARIZED)
        
        self.buffer = []
        
        self.thread_id = config.get('thread_id', THREAD_ID)
    
    def format(self, message: Message) -> ModelMessage:
        '''Format a message for the LLM.'''
        
        st = time.strftime('%Y-%m%dT%H:%M:%S',
            time.localtime(message.created_at)
        )
        agent = message.agent
        name = agent.name
        steps = message.steps
        content = f"[{st}]\t{name}\t" + '\n'.join(
            step.content for step in steps
        )
        
        if message.agent_id == self.id:
            return AgentMessage(content)
        else:
            return UserMessage(content)
    
    def prompt(self) -> list[ModelMessage]:
        '''Return the prompt for the LLM.'''
        
        return [
            SystemMessage(self.persona),
            *self.buffer
        ]
    
    def add(self, message: Message):
        '''Add a message to the buffer.'''
        
        self.buffer.append(self.format(message))
    
    async def summarize(self):
        '''Summarize the conversation and truncate the oldest messages.'''
        
        summary = await self.summarize_model(self.buffer)
        self.kernel.db.add_summary(self.thread_id, summary)
        self.buffer = self.buffer[-self.rest_unsummarized:]
        self.buffer.append(SystemMessage(summary))
    
    @override
    async def run(self, kernel: 'Kernel'):
        self.chat_model = kernel.model("chat")
        self.summarize_model = kernel.model("summarize")
        
        # Load the conversation buffer
        pending = False
        for msg in self.kernel.db.get_thread_buffer(self.thread_id):
            pending = (msg.agent_id != self.id)
            self.buffer.append(self.format(msg))
        
        # if the conversation was interrupted, respond
        if pending:
            self.push(None)
        
        while True:
            async for msg in self.pull():
                self.buffer.append(UserMessage(msg.content))
            
            content = ""
            async for step in self.chat_model(self.buffer):
                match step:
                    case ActionStep(name, params):
                        kernel.db.add_action(self.id, name, params)
                    
                    case DeltaStep(delta):
                        content += delta
                    
                    case _:
                        raise NotImplementedError(type(step))
            
            self.buffer.append(AgentMessage(content))
            if len(self.buffer) > self.max_unsummarized:
                await self.summarize()

class UserAgent(BaseAgent):
    '''User proxy agent.'''
    
    description = str(__doc__)
    
    uq: asyncio.Queue[str]
    '''User input queue.'''
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.uq = asyncio.Queue()
        
    async def sendMessage(self, msg: str):
        '''Send a message to the user.'''
        self.uq.put_nowait(msg)
    
    @override
    def push(self, message: Optional[Message]):
        print(message)

    async def run(self, kernel: Kernel):
        '''User input agent.'''
        
        while True:
            msg = await self.uq.get()
            kernel.publish(self, "*", msg)
            if not user.startswith("/") or user.startswith("//"):
                if user.startswith("/"):
                    user = user[1:]

                kernel.publish(self, "*", user)

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
                                kernel.publish(self, chan, msg)

                        case "poke":
                            if len(rest) == 0:
                                print("Usage: /poke <id>")
                            else:
                                kernel.by_ref(rest[0]).poke()

                        case "subs":
                            if len(rest) == 0:
                                for chan, subs in kernel.subs.items():
                                    print(chan, ":", ' / '.join(sub.name for sub in subs))
                            else:
                                for chan in kernel.subs_of(kernel.by_ref(rest[0])):
                                    print(chan)

                        case _:
                            print("Unknown command", cmd)
                except Exception as e:
                    traceback.print_exception(e)