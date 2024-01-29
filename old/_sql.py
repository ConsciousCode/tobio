'''
SQLAlchemy is absurdly unpythonic, and writing a custom ORM is more trouble
than it's worth. So KISS and just use raw SQL. I DON'T FUCKING CARE IF YOU
HAVE TO WRITE THE SCHEMA BY HAND.

This file starts with a _ because it should not be imported by anything other
than the memory module.
'''

from dataclasses import dataclass
import logging
import json
from typing import TYPE_CHECKING, Any, Literal, Optional, dataclass_transform, override

from sqlalchemy import Column, Connection, ForeignKey, Integer, PrimaryKeyConstraint, Table, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, declarative_base, relationship, Mapped, mapped_column, Session

from sqlalchemy.sql import text
from sqlalchemy.types import JSON

from .util import logger, now

DATABASE_FILE = "private/database.db"

if TYPE_CHECKING:
    from .kernel import Kernel
    class Base(DeclarativeBase): pass
else:
    Base = declarative_base()

# NOTE: Dataclass transform duplicated across the two base classes so the id
#  is not listed as one of the required fields in __init__.

@dataclass_transform(
    eq_default=False,
    kw_only_default=True,
    #field_specifiers=(mapped_column, relationship)
)
class NoId(Base):
    __abstract__ = True

@dataclass_transform(
    eq_default=False,
    kw_only_default=True,
    #field_specifiers=(mapped_column, relationship)
)
class HasId(Base):
    __abstract__ = True
    id: Mapped[int] = mapped_column(primary_key=True)

Push_table = Table(
    'pushes',
    Base.metadata,
    Column('message_id', ForeignKey('messages.id')),
    Column('agent_id', ForeignKey('agents.id')),
    PrimaryKeyConstraint('message_id', 'agent_id'),
    comment='Mapping of messages to agents.'
)

class AgentRow(HasId):
    '''Autonomous agents within the system.'''
    __tablename__ = 'agents'
    
    name: Mapped[str] = mapped_column()
    kind: Mapped[str] = mapped_column()
    created_at: Mapped[int] = mapped_column()
    destroyed_at: Mapped[Optional[int]] = mapped_column(nullable=True, default=None)
    
    # Pyright is "clever" and thinks destroyed_at's default parameter means no
    #  more non-default fields can be added.
    messages: Mapped[list['MessageRow']] = relationship(secondary=Push_table, back_populates='owner')

class MessageRow(HasId):
    '''Messages sent between agents.'''
    __tablename__ = 'messages'
    
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"))
    channel: Mapped[str] = mapped_column()
    created_at: Mapped[int] = mapped_column()
    
    owner: Mapped[AgentRow] = relationship(back_populates='messages')
    steps: Mapped[list['StepRow']] = relationship(order_by='StepRow.id', back_populates='message')
    agents: Mapped[set[AgentRow]] = relationship(secondary=Push_table, back_populates='messages')

class StepRow(HasId):
    '''Steps used to construct messages.'''
    __tablename__ = 'steps'
    
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))
    kind: Mapped[Literal['text', 'action']] = mapped_column()
    content: Mapped[str] = mapped_column()
    
    message: Mapped[MessageRow] = relationship(back_populates='steps')

class LogRow(HasId):
    '''System event logs.'''
    __tablename__ = 'logs'
    
    level = mapped_column(Integer)
    content: Mapped[str] = mapped_column()
    args: Mapped[Any] = mapped_column(JSON)
    created_at: Mapped[int] = mapped_column()

@dataclass
class WrappedCursor:
    columns: list[str]
    rows: list[tuple]

class DBLogHandler(logging.Handler):
    '''Log handler for emitting SQL.'''
    
    def __init__(self, db: 'Database'):
        super().__init__()
        self.db = db
    
    @override
    def emit(self, record: logging.LogRecord):
        self.db.add(LogRow(
            level=record.levelno,
            created_at=now(),
            content=record.getMessage(),
            args=record.args
        ))

class Database:
    '''Holds logic for database persistence.'''
    
    config: dict
    '''Database configuration.'''
    
    engine: Engine
    '''Connection to the database.'''
    
    sql: Connection
    '''Connection to the database.'''
    
    def __init__(self, config: dict):
        self.config = config
        self.engine = create_engine(config.get('database', DATABASE_FILE), echo=True)
        self.sql = self.engine.connect()
        logger.addHandler(DBLogHandler(self))
        
        Base.metadata.create_all(self.engine)
    
    def session(self):
        '''Get a session.'''
        return Session(self.engine)
    
    def raw_query(self, query: str):
        with self.session() as session:
            cur = session.execute(text(query))
            return WrappedCursor(list(cur.keys()), list(cur.scalars().all()))
    
    def add(self, ob: Base) -> Base:
        '''Add an object to the database.'''
        with self.session() as session:
            session.add(ob)
            return ob
    
    def add_all(self, *objects: Base):
        '''Add objects to the database.'''
        with self.session() as session:
            session.add_all(objects)
    
    def agents(self) -> list[AgentRow]:
        '''Get all agents.'''
        return list(self.sql.execute(select(AgentRow).order_by(AgentRow.id.asc())).scalars())
    
    def get_thread_buffer(self, channel: str, checkpoint_id: int=0) -> list[MessageRow]:
        '''
        Get the buffer of a thread.
        
        Parameters:
        thread_id: int
            The thread to retrieve the buffer of.
        checkpoint_id: int
            The id of the oldest message to retrieve.
        '''
        
        with self.session() as session:
            return list((
                session.execute(select(MessageRow)
                    .where(MessageRow.id > checkpoint_id)
                    .where(MessageRow.channel == channel)
                    .order_by(MessageRow.id.asc())
                )
            ).scalars())