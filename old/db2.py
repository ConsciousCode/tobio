'''
Database management code. Implements a simple ORM for the database based on
SQLAlchemy, just without the magic. It proved too arcane to use effectively.
'''

import time
import logging
from typing import TYPE_CHECKING, Literal, Optional, override

from .orm import *
from .util import logger
import sqlite3

DATABASE_FILE = "private/database.db"

if TYPE_CHECKING:
    from .kernel import Kernel

class Agent(HasId):
    '''Autonomous agents within the system.'''
    __tablename__ = 'agents'
    
    name: str
    kind: str
    created_at: timestamp
    destroyed_at: Optional[timestamp]=None
    
    __table_args__ = [Unique('name')]
    
    sent: list['Message'] = relationship('agent_id')
    '''Messages sent by this agent.'''
    
    received: list['Message'] = relationship('message_id', 'agent_id', secondary='Message', order_by='rowid')
    '''Messages received by this agent.'''

class Message(HasId):
    '''Messages sent between agents.'''
    __tablename__ = 'messages'
    
    agent_id: 'Agent.primary_key'
    channel: str
    created_at: timestamp
    
    agent: 'Agent' = relationship('agent_id')
    '''The agent which sent this message.'''
    steps: list['Step'] = relationship('message_id')
    '''The steps of this message.'''
    threads: list["Agent"] = relationship('agent_id', 'thread_id', secondary='Push', order_by='rowid')
    '''The agents which received this message.'''

class Step(HasId):
    '''Steps used to construct messages.'''
    __tablename__ = 'steps'
    
    message_id: 'Message.primary_key'
    kind: Literal['text', 'action']
    content: str

class Push(NoId):
    '''Pushes of messages to threads/agents.'''
    __tablename__ = 'pushes'
    
    message_id: 'Message.primary_key'
    agent_id: 'Agent.primary_key'
    
    __table_args__ = [PrimaryKey("message_id", "agent_id")]

class Log(HasId):
    '''System event logs.'''
    __tablename__ = 'logs'
    
    level: int
    content: str
    created_at: timestamp

class DBLogHandler(logging.Handler):
    '''Log handler for emitting SQL.'''
    
    def __init__(self, memory: 'Memory'):
        super().__init__()
        self.memory = memory
    
    @override
    def emit(self, record: logging.LogRecord):
        self.memory.add(Log(
            level=record.levelno,
            created_at=now(),
            content=record.getMessage()
        ))

def format_result(cursor: sqlite3.Cursor, rows: list[sqlite3.Row]):
    '''Format the results of a SQL query.'''

    if len(rows) == 0:
        return "empty set"

    if cursor.description is not None:
        # Column headers
        headers = [desc[0] for desc in cursor.description]

        # Find the maximum length for each column
        lengths = [max(len(str(cell)) for cell in col) for col in zip(*rows)]
        lengths = [max(len(header), colmax) + 2 for header, colmax in zip(headers, lengths)]

        # Create a format string with dynamic padding
        tpl = '|'.join(f"{{:^{length}}}" for length in lengths)
        tpl = f"| {tpl} |"
        
        return '\n'.join([
            tpl.format(*headers),
            tpl.replace("|", "+").format(*('-'*length for length in lengths)),
            *(tpl.format(*map(str, row)) for row in rows)
        ])

    if cursor.rowcount >= 0:
        return f"{cursor.rowcount} rows affected"

    if len(rows) > 0:
        return f"{len(rows)} rows in set"
    
    return "empty set (end of `format_result`)"

class Memory(Database):
    '''Holds logic for database persistence.'''
    
    config: dict
    '''Database configuration.'''
    
    sql: sqlite3.Connection
    '''Connection to the database.'''
    
    tables = [Agent, Message, Step, Push, Log]
    
    def __init__(self, config: dict):
        super().__init__(config.get('database', DATABASE_FILE))
        logger.addHandler(DBLogHandler(self))
        
        self.config = config
    
    def setup(self, kernel: "Kernel"):
        @kernel.register_command
        async def sql(query):
            '''Execute raw SQL.'''
            
            t = time.time()
            with self.session() as session:
                cur = session.execute(query)
                values = cur.fetchall()
                dt = time.time() - t
                
                return f"{format_result(cur, values)} ({dt:.2f} sec)"
        
        @kernel.register_command
        async def select(query):
            '''Execute raw SQL using SELECT.'''
            return await sql(f"SELECT {query}")
    
    def agents(self) -> list[Agent]:
        '''Get all agents.'''
        return self.select(Agent).fetch()
    
    def get_thread_buffer(self, thread_id: Agent.primary_key, checkpoint_id: Message.primary_key=0) -> list[Message]:
        '''
        Get the buffer of a thread.
        
        Parameters:
        thread_id: int
            The thread to retrieve the buffer of.
        checkpoint_id: int
            The id of the oldest message to retrieve.
        '''
        
        return self.select(Push).where(
            "rowid == ?", "message_id >= ?", values=(thread_id, checkpoint_id)
        ).fetch()