'''
SQLAlchemy uses too much magic for Pyright to understand, so we need to
manually define the types for the database classes.
'''

import logging
from typing import Optional

import sqlalchemy
from sqlalchemy.orm import DeclarativeMeta

from old3.src.ChatDev.camel import messages

from .kernel import Kernel

class Base(metaclass=DeclarativeMeta):
    def __init__(self, **kwargs): ...

class NoId(Base):
    __abstract__ = True

class HasId(Base):
    __abstract__ = True

    id: int

class Agent(HasId):
    __tablename__ = 'agents'
    
    name: str
    kind: str
    created_at: int
    destroyed_at: Optional[int]
    subs: list["Subscription"]
    messages: list["Message"]
    states: list["State"]
    threads: list["Thread"]

    @property
    def state(self) -> dict: ...

class State(HasId):
    __tablename__ = 'states'
    
    def __init__(self, *, agent_id: Optional[int]=None, state: dict) -> None: ...

    agent_id: int
    state: dict
    agent: Agent

class Channel(HasId):
    __tablename__ = 'channels'
    
    def __init__(self, *, name: str, created_at: int) -> None: ...

    name: str
    created_at: int
    threads: list["Thread"]
    messages: list["Message"]

class Subscription(NoId):
    '''Maps agents to the set of channels they are subscribed to.'''
    __tablename__ = 'subs'
    
    def __init__(self, *, agent_id: Optional[int]=None, channel_id: Optional[int]=None) -> None: ...
    
    agent_id: int
    channel_id: int
    agent: Agent
    channel: Channel

class Thread(HasId):
    __tablename__ = 'threads'
    
    def __init__(self, *, channel_id: Optional[int]=None, agent_id: Optional[int]=None, created_at: int, destroyed_at: Optional[int]=None) -> None: ...
    
    channel_id: int
    agent_id: int
    created_at: int
    destroyed_at: Optional[int]
    channel: Channel
    agent: Agent
    pushes: list["Push"]
    messages: list["Message"]

class Message(HasId):
    __tablename__ = 'messages'
    
    def __init__(self, *, agent_id: Optional[int]=None, created_at: int) -> None: ...
    
    agent_id: int
    created_at: int
    agent: Agent
    steps: list["Step"]
    pushes: list["Push"]
    threads: list[Thread]

class Step(HasId):
    __tablename__ = 'steps'
    
    def __init__(self, *, message_id: Optional[int]=None, kind: str, content: str) -> None: ...
    
    message_id: int
    kind: str
    content: str
    message: Message

class Dependency(HasId):
    '''Maps messages to the set of messages used to create them.'''
    __tablename__ = 'depends'
    
    def __init__(self, *, message_id: Optional[int]=None, dependency_id: Optional[int]=None) -> None: ...
    
    message_id: int
    dependency_id: int
    message: Message
    dependency: Message

class Push(NoId):
    __tablename__ = 'pushes'
    
    def __init__(self, *, thread_id: Optional[int]=None, message_id: Optional[int]=None) -> None: ...

    thread_id: int
    message_id: int
    thread: Thread
    message: Message

class Summary(HasId):
    __tablename__ = 'summaries'
    
    def __init__(self, *, agent_id: Optional[int]=None, created_at: int, content: str) -> None: ...

    checkpoint_id: int
    created_at: int
    content: str
    checkpoint: Message

class Log(HasId):
    __tablename__ = 'logs'
    
    def __init__(self, *, level: int, created_at: int, content: str) -> None: ...

    level: int
    created_at: int
    content: str

class DBLogHandler(logging.Handler):
    def __init__(self, memory: "Memory") -> None: ...
    def emit(self, record: logging.LogRecord) -> None: ...

class Memory:
    config: dict
    engine: sqlalchemy.engine.Engine
    
    def __init__(self, config: dict) -> None: ...
    
    def link_kernel(self, kernel: Kernel) -> None: ...
    def add(self, *obj: Base): ...
    def agents(self) -> list[Agent]: ...
    def channels(self) -> dict[str, Channel]: ...
    def get_thread_buffer(self, thread_id: int, checkpoint_id: int) -> list[Message]: ...
    def message_steps(self, msg: Message) -> list[Step]: ...
