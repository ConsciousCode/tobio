import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import time
from prompt_toolkit.patch_stdout import patch_stdout
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass

from orin.connector.base import Connector

from .typing import Optional, Any, json_value, AsyncIterator, Awaitable
from .config import DUMB_MODEL, MODEL
from .util import PausableMixin, logger, read_file, typename, Registrant
from .connector import Connector
from .db import AgentRow, Database, MessageRow

class Tool(Registrant):
    kernel: 'Kernel'
    agent: 'Agent'
    parameters: dict
    
    def __init__(self, kernel: 'Kernel', agent: 'Agent'):
        self.kernel = kernel
        self.agent = agent
    
    def state(self) -> object:
        '''Build the state object for the tool.'''
        return None
    
    async def __call__(self, **kwargs):
        ...
    
    @classmethod
    def to_schema(cls):
        return {
            "type": "function",
            "function": {
                "name": cls.__name__.lower(),
                "description": inspect.getdoc(cls),
                "parameters": {
                    "type": "object",
                    "properties": cls.parameters,
                    "required": list(cls.parameters.keys())
                }
            }
        }
    
    def load_state(self, state: object):
        pass

@dataclass
class Message:
    id: MessageRow.primary_key
    src: 'Agent'
    content: str
    created_at: float
    
    def printable(self):
        ts = time.strftime("%H:%M:%S", time.localtime(self.created_at))
        return f"[{ts}] {self.src.name}\n{self.content}"
    
    def __str__(self):
        ts = time.strftime("%H:%M:%S", time.localtime(self.created_at))
        content = json.dumps(self.content)
        
        return f"[{ts}]\t{content}"

type MaybeAwaitable[T] = T|Awaitable[T]

async def maybe_await[T](x: MaybeAwaitable[T]) -> Optional[T]:
    if inspect.isawaitable(x):
        x = await x
    return x # type: ignore

class Agent(Registrant, PausableMixin):
    '''Abstract agent participating in the consortium.'''
    
    id: AgentRow.primary_key
    '''Id of the agent in the database.'''
    
    name: str
    '''Name of the agent - functions as a short description.'''
    
    config: Any
    '''Local configuration for the agent.'''
    
    msg: asyncio.Queue[MaybeAwaitable[Optional[Message]]]
    '''Pending messages to be processed.'''
    
    def __init__(self,
        id: AgentRow.primary_key,
        name: Optional[str]=None,
        config: Any=None
    ):
        super().__init__()
        
        self.id = id
        # Note: Uses agent defaults if available, otherwise AttributeError
        self.name = name or getattr(self, 'name', None) or typename(self)
        self.config = config
        
        self.msgq = asyncio.Queue()
    
    def __str__(self):
        pause = " (paused)" if self.is_paused() else ""
        return f"{self.name}{pause}"
    
    def poke(self):
        '''Poke the agent to generate without an input message.'''
        # If the message queue isn't empty, then we don't need to poke
        if self.msgq.empty():
            logger.debug(f"Poke pushed: {self.name}")
            self.msgq.put_nowait(None)
        
    def push(self, msg: Message):
        '''Push a message to the agent.'''
        logger.debug(f"Push received for {self.name}")
        self.msgq.put_nowait(msg)
    
    async def pull(self) -> AsyncIterator[Message]:
        '''Pull all pending messages from the message queue.'''
        
        if msg := await maybe_await(self.msgq.get()):
            yield msg
        
        while not self.msgq.empty():
            if msg := await maybe_await(self.msgq.get()):
                yield msg
    
    async def init(self, kernel: "Kernel", state: object):
        '''Initialize the agent.'''
        
        if state is not None:
            raise NotImplementedError(f"{typename(self)} agent load_state got non-None state")
    
    def state(self) -> object:
        '''Build the state object for the agent.'''
        return None
    
    @abstractmethod
    async def run(self, kernel: 'Kernel'):
        '''Run the agent.'''

class Kernel(PausableMixin):
    '''System architecture object.'''
    
    taskgroup: asyncio.TaskGroup
    '''TaskGroup for all running agents.'''
    
    db: Database
    '''Database connection.'''
    
    context: AsyncExitStack
    '''Context manager for misc resources.'''
    
    agents: dict[AgentRow.primary_key, Agent]
    '''Map of locally instantiated agents.'''
    
    connectors: dict[str, Connector]
    '''Instantiated connectors.'''
    
    config: dict
    '''System configuration.'''
    
    def __init__(self, taskgroup: asyncio.TaskGroup, db: Database, context: AsyncExitStack, config: dict):
        super().__init__()
        
        self.taskgroup = taskgroup
        self.db = db
        self.context = context
        self.agents = {}
        self.connectors = {}
        self.config = config
    
    def add_state(self, agent: Agent):
        '''Add the agent's state to the database.'''
        self.db.add_state(agent.id, agent.state())
    
    def by_name(self, name: str):
        '''Get all agents with the given name.'''
        
        name = name.lower()
        for agent in self.agents.values():
            if agent.name.lower() == name:
                yield agent
    
    def by_id(self, id: AgentRow.primary_key):
        '''Get the agent with the given id.'''
        return self.agents[id]
    
    def by_ref(self, ref: str|AgentRow.primary_key):
        '''Get the agent with the given reference.'''
        if isinstance(ref, int):
            return self.by_id(ref)
        
        if ref.startswith("@"):
            return self.by_id(int(ref[1:], 16))
        
        try:
            return self.by_id(int(ref, 16))
        except ValueError:
            pass
        
        last = None
        for agent in self.by_name(ref):
            if last is not None:
                raise ValueError(f"Multiple agents with name {ref!r}")
            last = agent
        
        if last is None:
            raise KeyError(ref)
        return last
    
    async def connector(self, name: str):
        '''Get the connector with the given name.'''
        
        conn = self.connectors.get(name)
        if conn is None:
            conn = await self.context.enter_async_context(Connector.get(name).connect(self.config[name]))
            self.connectors[name] = conn
        return conn
    
    def update_config(self, agent, config):
        '''Update the agent's configuration.'''
        
        if config:
            self.db.set_config(agent.id, {**agent.config, **config})
    
    async def create_agent(self,
        type: str,
        name: Optional[str]=None,
        config: json_value=None
    ):
        '''Create a new agent.'''
        
        logger.info(f"create_agent({type!r}, {name!r}, {config!r})")
        
        AgentType = Agent.get(type)
        agent_id = self.db.create_agent(type, name or AgentType.__name__, config)
        agent = AgentType(agent_id, name, config)
        await agent.init(self, None)
        self.agents[agent_id] = agent
        self.taskgroup.create_task(agent.run(self))
        return agent
    
    def push(self, agent: Agent, content: str):
        '''Push a message to the other agents.'''
        
        content = content.strip()
        if not content:
            logger.debug("push({agent.name!r}): pass)")
            return
        
        logger.info(f"push({agent.name!r})")
        
        created_at = int(time.time())
        row = self.db.message(agent.id, content, created_at)
        
        msg = Message(
            row.rowid, self.by_id(row.agent_id),
            row.content, row.created_at
        )
        
        pushes: list[tuple[int, int]] = []
        for agent_id, agent in self.agents.items():
            pushes.append((agent_id, msg.id))
            if agent_id != row.agent_id:
                agent.push(msg)
        self.db.push_many(pushes)
        
        logger.debug(f"push(): {pushes}")
    
    async def load_agents(self):
        '''Reload all agents from the database.'''
        
        logger.info("Loading agents...")
        self.agents.clear()
        for row in self.db.load_agents():
            config = json.loads(row.config)
            agent = Agent.get(row.type)(row.rowid, row.name, config)
            await agent.init(self, self.db.load_state(agent.id))
            self.agents[agent.id] = agent
    
    async def init(self):
        await self.load_agents()
        
        for agent in self.agents.values():
            self.taskgroup.create_task(agent.run(self))
    
    def exit(self):
        current = None
        for task in self.taskgroup._tasks:
            if task == asyncio.current_task():
                current = task
            if task.done() or task == asyncio.current_task():
                continue
            task.cancel()
        
        if current:
            current.cancel()
    
    @asynccontextmanager
    @staticmethod
    async def start(config: dict):
        '''
        Entrypoint for the kernel, initializes everything so we don't have to
        constantly check if things are initialized.
        '''
        
        dbpath = config['global']['database']
        
        with patch_stdout():
            async with AsyncExitStack() as context:
                with Database.connect(dbpath) as db:
                    async with asyncio.TaskGroup() as tg:
                        system = Kernel(tg, db, context, config)
                        await system.init()
                        yield system