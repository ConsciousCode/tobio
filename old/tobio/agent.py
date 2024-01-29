'''
Agents are the main components of the system. They act within the system by
pushing messages to each other. A typical agent will use a model for
inference.
'''

import asyncio
import time
import traceback
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Protocol, override
from urllib.parse import parse_qs, urlparse

from .memory import MessageMemory, StepMemory
from .provider import ModelMessage, SystemMessage, AgentMessage, UserMessage, ActionStep, DeltaStep
from .util import load_module, read_file, now

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
    
    def pause(self):
        '''Pause the agent.'''
        ...
    
    def resume(self):
        '''Resume the agent.'''
        ...
    
    def state_dict(self) -> dict:
        '''Return a dictionary of the agent state.'''
        ...
    
    def poke(self):
        '''Poke the agent to make it produce a message.'''
        ...
    
    def push(self, message: MessageMemory):
        '''
        Push a message to the agent. If the message is None, the agent will
        resume and produce a new message, even if there are no new messages.
        '''
    
    async def run(self, kernel: 'Kernel'):
        '''Run the agent.'''
    
    async def command(self, command: str) -> str:
        '''Run a command on the agent.'''
        ...

class BaseAgent(ABC, Agent):
    '''Base agent class implementation.'''
    
    mq: asyncio.Queue[Optional[MessageMemory]]
    '''Message queue.'''
    
    registry: dict[str, type['BaseAgent']] = {}
    '''Registry of agent types.'''
    
    def __init_subclass__(cls):
        cls.registry[cls.__name__] = cls
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.mq = asyncio.Queue()
    
    @classmethod
    def from_uri(cls, uri) -> tuple[type["BaseAgent"], dict]:
        '''Create an agent from a URI.'''
    
        u = urlparse(uri)
        
        if u.scheme not in {"", "file"}:
            raise ValueError(f"Invalid agent URI scheme: {u.scheme}")
        
        path = u.path
        frag = u.fragment
        config = parse_qs(u.query)
        
        if path == "" or frag == "":
            return cls.registry[path or frag], config
        else:
            return getattr(load_module(path), frag), config
    
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
    def poke(self):
        self.mq.put_nowait(None)
    
    @override
    def push(self, message: MessageMemory):
        self.mq.put_nowait(message)
    
    @override
    @abstractmethod
    async def run(self, kernel: 'Kernel'): ...

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
    
    def format(self, message: MessageMemory) -> ModelMessage:
        '''Format a message for the LLM.'''
        
        st = time.strftime('%Y-%m-%dT%H:%M:%S',
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
    
    def add(self, message: MessageMemory):
        '''Add a message to the buffer.'''
        self.buffer.append(self.format(message))
    
    @override
    async def run(self, kernel: 'Kernel'):
        self.chat_model = kernel.model("chat")
        self.summarize_model = kernel.model("summarize")
        
        # Load the conversation buffer
        pending = False
        for msg in self.kernel.memory.get_thread_buffer('user'):
            pending = (msg.agent_id != self.id)
            self.buffer.append(self.format(msg))
        
        # if the conversation was interrupted, respond
        if pending:
            self.poke()
        
        while True:
            async for msg in self.pull():
                self.buffer.append(self.format(msg))
            
            dbmsg = MessageMemory(None, self.id, "###", now())
            
            content = ""
            text = ""
            async for step in self.chat_model(self.buffer):
                match step:
                    case ActionStep(name, params):
                        if text:
                            dbmsg.steps.append(
                                StepMemory(dbmsg.id, 'text', text)
                            )
                            content += text
                            text = ""
                        dbmsg.steps.append(
                            StepMemory(dbmsg.id, 'action',
                                json.dumps({"name": name, "parameters": params})
                            )
                        )
                    
                    case DeltaStep(delta):
                        text += delta
                    
                    case _:
                        raise NotImplementedError(type(step))
            
            if text:
                dbmsg.steps.append(
                    StepMemory(dbmsg.id, 'text', text)
                )
                content += text
            
            kernel.memory.add(dbmsg)
            self.buffer.append(AgentMessage(content))

class UserAgent(BaseAgent):
    '''User proxy agent.'''
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    async def receive(self, content: str):
        '''Receive a message from the user.'''
        
        # Store the user messager
        msg = self.kernel.memory.add(MessageMemory("", self.id, "user", now()))
        self.kernel.memory.add(StepMemory(msg.id, 'text', content))
        
        if not content.startswith("/") or content.startswith("//"):
            if content.startswith("/"):
                content = content[1:]
            
            #self.kernel.publish("*", self, msg)

        elif content == "/" or content.startswith("/#"):
            pass

        else:
            try:
                result = await self.kernel.command(content[1:])
            except Exception as e:
                traceback.print_exception(e)
    
    async def run(self, kernel: 'Kernel'):
        '''
        Agent is solely a proxy for the agent, so it does nothing on its own.
        '''
        self.kernel = kernel