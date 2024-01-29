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
from .orm import Base, NoId, HasId, PrimaryKey, column, one_to_many, many_to_many, many_to_one, Database, select, insert, update, delete, func

if TYPE_CHECKING:
    from .kernel import Kernel

class PushMemory(NoId):
    '''Messages pushed to agent threads.'''
    __tablename__ = "pushes"
    
    message_id = column(int, primary_key=True)
    agent_id = column(int, foreign_key=lambda: AgentMemory.id)
    
    def __init__(self, message_id: int, agent_id: int):
        super().__init__(message_id=message_id, agent_id=agent_id)
    
    __table_args__ = (
        PrimaryKey(message_id, agent_id),
    )

class AgentMemory(HasId):
    '''Information required to reconstruct agents.'''
    __tablename__ = "agents"
    
    name = column(str)
    kind = column(str)
    created_at = column(int)
    destroyed_at = column(Optional[int], default=None)
    
    def __init__(self, name: str, kind: str, created_at: int):
        super().__init__(name=name, kind=kind, created_at=created_at)
    
    messages = one_to_many(lambda: MessageMemory, backref="agent")

class MessageMemory(HasId):
    '''Messages created by agents.'''
    __tablename__ = "messages"
    
    event_id = column(Optional[str], unique=True)
    agent_id = column(int, foreign_key=lambda: AgentMemory.id)
    channel = column(str)
    created_at = column(int)
    
    def __init__(self, event_id: Optional[str], agent_id: int, channel: str, created_at: int):
        super().__init__(event_id=event_id, agent_id=agent_id, channel=channel, created_at=created_at)
    
    agent = many_to_one(lambda: AgentMemory, backref="messages")
    steps = one_to_many(lambda: StepMemory, backref="message")
    agents = many_to_many(lambda: AgentMemory, secondary=lambda: PushMemory, backref="messages")

class StepMemory(HasId):
    '''Steps which make up messages.'''
    __tablename__ = "steps"
    
    message_id = column(int, foreign_key=lambda: MessageMemory.id)
    kind = column(str)
    content = column(str)
    
    def __init__(self, message_id: int, kind: str, content: str):
        super().__init__(message_id=message_id, kind=kind, content=content)
    
    message = many_to_one(lambda: MessageMemory, backref="steps")

class LogMemory(HasId):
    '''System logs.'''
    __tablename__ = "logs"
    
    level = column(int)
    content = column(str)
    args = column(str)
    created_at = column(int)
    
    def __init__(self, level: int, content: str, args: str, created_at: int):
        super().__init__(level=level, content=content, args=args, created_at=created_at)

class StateMemory(NoId):
    '''Global memory table.'''
    __tablename__ = "states"
    
    owner = column(str, primary_key=True)
    value = column(str)
    
    def __init__(self, owner: str, value: str):
        super().__init__(owner=owner, value=value)

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
    
    def __init__(self, config: dict):
        super().__init__(config['database'])
    
    async def setup(self, kernel: "Kernel"):
        logger.addHandler(DBLogHandler(self))
        
        @kernel.register_command
        async def sql(query):
            t = time.time()
            with self.engine.connect() as conn:
                result = conn.execute(query)
                dt = time.time() - t
                cols = result.keys()
                rows = result.fetchall()
                
                # Determine number of rows affected or in set
                if result.returns_rows:
                    if len(rows) == 0:
                        content = "empty set"
                    else:
                        table = PrettyTable(cols)
                        table.add_rows(rows)
                        content = f"{table}\n\n{len(rows)} rows in set"
                else:
                    content = f"{result.rowcount} rows affected"
            
            return f"{content} ({dt:.2f} sec)"
        
        @kernel.register_command
        async def select(query):
            '''Execute raw SQL using SELECT.'''
            return await sql(f"SELECT {query}")
    
    def state(self, name: str) -> StateMemory:
        '''Get a global state.'''
        
        with self.session() as conn:
            row = conn.execute(select(StateMemory).where(StateMemory.owner == name)).one()
            if row is not None:
                return row[0]
            
            row = StateMemory(
                owner=name,
                value=json.dumps(None)
            )
            self.session.add(row)
            self.session.commit()
            
            if row := self.execute(select(StateMemory).where(StateMemory.owner == name)).fetchone():
                return row[0]
            
            return self.execute(insert(StateMemory).values(
                owner=name,
                value=json.dumps(None)
            )).one()[0]
    
    def message(self, event_id: str, sender_id: int, timestamp: int, channel: str, content: str) -> MessageMemory:
        '''Fetch a message with the event id or create a new one.'''
        
        if row := self.execute(select(MessageMemory).where(MessageMemory.event_id == event_id)).fetchone():
            return row[0]
        
        msg = MessageMemory(event_id, sender_id, channel, timestamp)
        msg.steps.append(StepMemory(msg.id, "text", content))
        msg = self.add(msg)
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
        
        return [
            row[0] for row in self.execute(select(MessageMemory).where(
                MessageMemory.channel == channel,
                MessageMemory.id > checkpoint_id
            )).fetchall()
        ]