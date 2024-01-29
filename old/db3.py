'''
All memory related functions and classes. Implementation details of memories
SHALL NOT be exposed. SQLAlchemy is too insane and unpythonic to pollute the
rest of the codebase. To compensate, we have to be really aggressive about how
tightly we control the abstraction from leaking.
'''

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Literal, Optional, cast, override

from prettytable import PrettyTable

from .util import now, logger
#from .orm import NoId, HasId, Column, Relationship, Database, PrimaryKey, ForeignKey

from sqlalchemy import Column, ForeignKey, PrimaryKeyConstraint, UniqueConstraint, Integer, String, DateTime, Boolean, JSON, Table, MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker, Session, declarative_base


if TYPE_CHECKING:
    from .kernel import Kernel

class Base(DeclarativeBase):
    pass

class NoId(Base):

class PushMemory(NoId):
    '''Messages pushed to agent threads.'''
    __tablename__ = "pushes"
    
    message_id = Column(int)
    agent_id = Column(int)
    
    def __init__(self, message_id: int, agent_id: int): ...
    
    __table_args__ = (
        PrimaryKey(message_id, agent_id),
        ForeignKey(message_id, references=lambda: MessageMemory.id),
        ForeignKey(agent_id, references=lambda: AgentMemory.id)
    )

class AgentMemory(HasId):
    '''Information required to reconstruct agents.'''
    __tablename__ = "agents"
    
    name = Column(str)
    kind = Column(str)
    created_at = Column(int)
    destroyed_at: Column[Optional[int]] = Column(cast(type, Optional[int]), default=None)
    
    def __init__(self, name: str, kind: str, created_at: int, destroyed_at: Optional[int]=None): ...
    
    messages = Relationship(list["MessageMemory"], backref=lambda: MessageMemory.agent_id)

class MessageMemory(HasId):
    '''Messages created by agents.'''
    __tablename__ = "messages"
    
    event_id = Column(str)
    agent_id = Column(int)
    channel = Column(str)
    created_at = Column(int)
    
    def __init__(self, agent_id: int, channel: str, created_at: int): ...
    
    agent = Relationship(AgentMemory, foreign=agent_id)
    steps = Relationship(list['StepMemory'], backref=lambda: StepMemory.message_id)
    agents = Relationship(set[AgentMemory], foreign=PushMemory.agent_id, backref=PushMemory.message_id)
    
    __table_args__ = (
        ForeignKey(agent_id, references=AgentMemory.id),
    )

class StepMemory(HasId):
    '''Steps which make up messages.'''
    __tablename__ = "steps"
    
    message_id = Column(int)
    kind: Column[Literal['text', 'action']] = Column(Literal['text', 'action'])
    content = Column(str)
    
    #def __init__(self, message_id: int, kind: Literal['text', 'action'], content: str): ...
    
    message = Relationship(MessageMemory, foreign=message_id)
    
    __table_args__ = (
        ForeignKey(message_id, references=MessageMemory.id),
    )

class LogMemory(HasId):
    '''System logs.'''
    __tablename__ = "logs"
    
    level = Column(int)
    content = Column(str)
    args = Column(str)
    created_at = Column(int)
    
    def __init__(self, level: int, content: str, args: str, created_at: int): ...

class StateMemory(NoId):
    '''Global memory table.'''
    __tablename__ = "states"
    
    owner = Column(str)
    value = Column(str)
    
    def __init__(self, owner: str, value: str): ...
    
    __table_args__ = (
        PrimaryKey(owner),
    )

class DBLogHandler(logging.Handler):
    '''Log handler for emitting SQL.'''
    
    def __init__(self, memory: 'Memory'):
        super().__init__()
        self.memory = memory
    
    @override
    def emit(self, record: logging.LogRecord):
        self.memory.add(LogMemory(
            level=record.levelno,
            created_at=now(),
            args=json.dumps(record.args),
            content=record.message
        ))

class Memory(Database):
    '''Memory interface.'''
    
    tables = [AgentMemory, MessageMemory, StepMemory, PushMemory, LogMemory, StateMemory]
    
    def __init__(self, config: dict):
        super().__init__(config['database'])
    
    async def setup(self, kernel: "Kernel"):
        logger.addHandler(DBLogHandler(self))
        
        @kernel.register_command
        async def sql(query):
            '''Execute raw SQL.'''
            
            t = time.time()
            cur = self.execute(query)
            result = cur.fetchall()
            dt = time.time() - t
            
            if cur.rowcount == -1:
                if len(result) == 0:
                    content = "empty set"
                else:
                    table = PrettyTable(result[0].keys())
                    table.add_rows(result)
                    content = f"{table}\n\n{len(result)} rows in set"
            else:
                content = f"{cur.rowcount} affected"
            
            return f"{content} ({dt:.2f} sec)"
        
        @kernel.register_command
        async def select(query):
            '''Execute raw SQL using SELECT.'''
            return await sql(f"SELECT {query}")
    
    def state(self, name: str) -> StateMemory:
        '''Get a global state.'''
        if row := self.select(StateMemory).where(StateMemory.owner == name).fetchone():
            return row
        else:
            self.insert(StateMemory).values(
                owner=name,
                value=json.dumps(None)
            ).execute()
            return self.state(name)
    
    def message(self, event_id: str, sender_id: int, timestamp: int, content: str) -> MessageMemory:
        '''Fetch a message with the event id or create a new one.'''
        
        if row := self.select(MessageMemory).where(MessageMemory.event_id == event_id).fetchone():
            return row
        
        msg = self.add(MessageMemory(event_id, sender_id, timestamp))
        self.add(StepMemory(msg.id, "text", content))
        return msg
    
    def get_thread_buffer(self, channel: str, checkpoint_id: int=0) -> list[MessageMemory]:
        '''
        Get the buffer of a thread.
        
        Parameters:
        thread_id: int
            The thread to retrieve the buffer of.
        checkpoint_id: int
            The id of the oldest message to retrieve.
        '''
        
        return self.select(MessageMemory).where(
            MessageMemory.channel == channel,
            MessageMemory.id > checkpoint_id
        ).fetch()