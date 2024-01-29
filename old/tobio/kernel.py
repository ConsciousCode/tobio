import asyncio
from collections import defaultdict
from dataclasses import dataclass
from contextlib import AsyncExitStack
from typing import Awaitable, Callable
from urllib.parse import urlparse, parse_qs

from .gateway import Matrix
from .memory import AgentMemory, Memory, MessageMemory, PushMemory
from .provider import ModelMessage, Provider, Run
from .util import logger, PausableMixin, filter_dict, load_module, typename, unalias_dict, now
from .agent import BaseAgent, Agent, UserAgent

@dataclass
class Model:
    '''Model which can be used for inference.'''
    
    provider: Provider
    '''Provider of the model.'''
    
    config: dict
    '''Configuration of the model.'''
    
    def __call__(self, prompt: list[ModelMessage]) -> Run:
        '''Perform inference on the model.'''
        
        return self.provider(prompt)

@dataclass
class AgentEntry:
    '''Kernel's representation of an agent.'''
    
    id: int
    '''ID of the agent.'''
    
    name: str
    '''Name of the agent.'''
    
    agent: Agent
    '''Agent object.'''
    
    state_hash: int
    '''Hash of the most recent state.'''

class Kernel(PausableMixin):
    '''
    Brings all the components together into a single coherent system.
    '''
    
    config: dict
    '''Configuration for the kernel.'''
    
    context: AsyncExitStack
    '''Async context manager for providers.'''
    exit_signal: asyncio.Event
    '''Signal to exit the kernel.'''
    tg: asyncio.TaskGroup
    '''Active task group.'''
    
    agents: dict[int|str, AgentEntry]
    '''Agent map by id and name.'''
    providers: dict[str, Provider]
    '''Map of all active providers.'''
    models: dict[str, Model]
    '''Map of names to models.'''
    commands: dict[str, Callable[[str], Awaitable[str]]]
    '''Map of commands available to run.'''
    subs: defaultdict[str, set[Agent]]
    '''Subscription map.'''
    
    matrix: Matrix
    
    #channels: dict[str, db.Channel]
    '''Map of channels by name.'''
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.exit_signal = asyncio.Event()
        self.config = config
        self.context = AsyncExitStack()
        
        self.agents = {}
        self.providers = {}
        self.models = {}
        self.commands = {}
        self.subs = defaultdict(set)
        
        self.tg = asyncio.TaskGroup()
        
        self.memory = Memory(config['memory'])
        #self.channels = self.db.channels()
        self.matrix = Matrix(config['matrix'])
        
        @self.register_command
        async def help(msg: str) -> str:
            '''List all available commands.'''
            return "Available commands:\n  " + "\n  ".join(
                f"{name}: {cmd.__doc__}" for name, cmd in self.commands.items()
            )
        
        @self.register_command
        async def quit(msg: str) -> str:
            '''Exit the kernel.'''
            self.exit()
            return "Exiting..."
        
        @self.register_command
        async def pause(msg: str) -> str:
            '''Pause the kernel.'''
            
            if msg == "*":
                for ae in self.agents.values():
                    ae.agent.pause()
                return "Paused all agents."
            elif msg:
                self.agents[msg].agent.pause()
                return f"Paused agent {msg}."
            else:
                self.pause()
                return "Paused kernel."
        
        @self.register_command
        async def resume(msg: str) -> str:
            '''Resume the kernel.'''
            
            if msg == "*":
                for ae in self.agents.values():
                    ae.agent.resume()
                return "Resumed all agents."
            elif msg:
                self.agents[msg].agent.resume()
                return f"Resumed agent {msg}."
            else:
                self.resume()
                return "Resumed kernel."
        
        @self.register_command
        async def state(msg: str) -> str:
            '''Print the state of the given agent.'''
            return str(self.agents[msg].agent.state_dict())
        
        @self.register_command
        async def agents(msg: str) -> str:
            '''List all agents.'''
            return "Agents:\n  " + "\n  ".join(
                f"{name}: {typename(ae.agent)}" for name, ae in self.agents.items()
            )
        
        @self.register_command
        async def poke(msg: str) -> str:
            '''Poke the given agent prompting it to create a chat completion.'''
            self.agents[msg].agent.poke()
            return f"Poked agent {msg}."
    
    def subscribe(self, chan: str, agent: Agent):
        '''Subscribe an agent to a topic.'''
        self.subs[chan].add(agent)
    
    def unsubscribe(self, chan: str, agent: Agent):
        '''Unsubscribe an agent from a topic.'''
        self.subs[chan].discard(agent)
    
    def publish(self, chan: str, agent: Agent, msg: MessageMemory):
        '''Publish a message to a topic.'''
        
        rows = []
        for agent in self.subs[chan]:
            agent.push(msg)
            rows.append(PushMemory(msg.id, agent.id))
        
        self.memory.add(*rows)
    
    def register_command(self, func: Callable[[str], Awaitable[str]]):
        '''Register a command.'''
        self.commands[func.__name__] = func
        return func
    
    def unregister_command(self, func: str):
        '''Unregister a command.'''
        del self.commands[func]
    
    async def command(self, cmd: str) -> str:
        '''Run a command.'''
        name, rest = cmd.split(" ", 1)
        return await self.commands[name](rest)
    
    def foreign_agent(self, name: str) -> Agent:
        if ag := self.agents.get(name):
            return ag.agent
        
        
    
    def agent(self, ref: int|str) -> Agent:
        '''Return the agent with the given reference.'''
        return self.agents[ref].agent
    
    def model(self, name: str) -> Model:
        '''Return the model with the given name.'''
        return self.models[name]
    
    def add_agents(self, **agents: str|Agent):
        '''Add an agent to the kernel.'''
        
        raise NotImplementedError()
        for name, agent in agents.items():
            if not isinstance(agent, str):
                self.agents[name] = AgentEntry(
                    0, name, agent, hash(agent.state_dict())
                )
                agent.name = name
                continue
            
            AgentClass, config = BaseAgent.from_uri(agent)
            
            if dbagent := self.memory.select(AgentMemory).where(AgentMemory.name == name).fetchone():
                AgentClass, config = BaseAgent.from_uri(dbagent.kind)
                agent = AgentClass({**dbagent.config, **config})
            else:
                agent = AgentClass(parse_qs(u.query))
                dbagent = self.memory.add(AgentMemory(name, AgentClass.__name__, now()))
                
                self.agents[name] = AgentEntry(
                    dbagent.id, name, agent, hash(None)#dbagent.state)
                )
            return agent
    
    async def add_models(self, **models: str|Model):
        '''
        Add a model to the kernel, mapping a name to a model initialized
        from a URI. May initialize a new provider if necessary.
        
        Example:
        >>> kernel.add_model(
        ...     "chatbot", 'openai:gpt-4-turbo'
        ... )
        '''
        
        for name, model in models.items():
            if not isinstance(model, str):
                self.models[name] = model
                continue
            
            uri = model
            
            # Return the model if it's already initialized
            if model := self.models.get(name):
                return model
            
            u = urlparse(uri)
            scheme = u.scheme
            
            config = self.config.get(scheme, {})
            
            # Parse the model config from the URI
            model_config = filter_dict(
                unalias_dict({
                    **config, **parse_qs(u.query)
                }, {
                    "T": "temperature",
                    "p": "top_p",
                    "max_token": "max_tokens"
                }), {
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "presence_penalty",
                    "frequency_penalty",
                    "stop"
                }
            )
            
            provider = self.providers.get(scheme)
            
            # Ensure the provider is initialized
            if provider is None:
                context = Provider.get(scheme).connect({
                    **config,
                    "api_key": config["api_key"],
                    "host": u.hostname,
                    "port": u.port,
                    "model": u.path
                })
                provider = await self.context.enter_async_context(context)
                self.providers[scheme] = provider
            
            self.models[name] = Model(provider, model_config)
    
    def exit(self):
        '''Exit the kernel.'''
        self.exit_signal.set()
    
    async def run(self):
        '''Start the kernel and run indefinitely.'''
        
        logger.info("Starting kernel")
        
        await self.memory.setup(self)
        await self.context.enter_async_context(self.tg)
        
        async with self.context:
            await self.add_models(**self.config.get('models', {}))
            #self.add_agents(**self.config.get('agents', {}))
            
            for ae in self.agents.values():
                self.tg.create_task(ae.agent.run(self))
            
            self.tg.create_task(self.matrix.run(self, self.memory.state("matrix")))
            
            # Wait for the exit signal
            await self.exit_signal.wait()
            
            logger.info("Stopping kernel")
            
            # Exit the agents
            # NOTE: Use of private API until TaskGroup.cancel is implemented
            for task in self.tg._tasks:
                task.cancel()