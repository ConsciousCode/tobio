import time
import logging
from typing import TYPE_CHECKING, overload, override
import sqlalchemy
from sqlalchemy import ForeignKey, UniqueConstraint, create_engine, select
from sqlalchemy.schema import Column
from sqlalchemy.types import JSON, Integer, String
from sqlalchemy.orm import DeclarativeMeta, Session, declarative_base, relationship

from .util import logger

class Base(metaclass=DeclarativeMeta):
    '''Base class for all database models.'''
    
    # Dummy __init__ accepting any kwargs
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

if TYPE_CHECKING:
    from .kernel import Kernel
else:
    # Pyright has no idea how to handle this dynamic "variable" type and
    #  thinks NoIdBase and IdBase are somehow compatible types, so we have
    #  to use a fake base class during type checking.
    Base = declarative_base(cls=Base)

DATABASE_FILE = 'sqlite://private/memory.db'

class NoId(Base):
    '''Base for all database models without an id.'''
    __abstract__ = True

class HasId(Base):
    '''Base for all database models with an id.'''
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    '''Primary key id for the row.'''

class Agent(HasId):
    '''Agents participating in threads.'''
    __tablename__ = 'agents'
    
    name = Column(String)
    '''Name of the agent.'''
    kind = Column(String)
    '''Class name of the agent for deserialization.'''
    created_at = Column(Integer)
    '''Timestamp of agent creation.'''
    destroyed_at = Column(Integer, nullable=True)
    '''Timestamp of agent destruction.'''
    
    subs = relationship("Subscription", back_populates="agent")
    '''Subscriptions of the agent.'''
    messages = relationship("Message", back_populates="agent", order_by="Message.id")
    '''Messages created by the agent.'''
    states = relationship("State", back_populates="agent", order_by="State.id")
    '''States of the agent.'''
    threads = relationship("Thread", back_populates="agent", order_by="Thread.id")
    '''Threads the agent created.'''
    
    @property
    def state(self) -> dict:
        '''Get the latest state of the agent.'''
        return self.states.order_by(State.id.desc()).first().state

class State(HasId):
    '''States of the agents.'''
    __tablename__ = 'states'
    
    # Columns

    agent_id = Column(Integer, ForeignKey("agents.id"))
    '''Id of the agent this state belongs to.'''
    state = Column(JSON)
    '''State dictionary stored as JSON.'''
    
    agent = relationship("Agent", back_populates="states")
    '''Agent this state belongs to.'''

class Channel(HasId):
    '''Agent intercommunication.'''
    __tablename__ = 'channels'
    
    name = Column(String, unique=True)
    '''Name of the channel.'''
    created_at = Column(Integer)
    '''Timestamp of channel creation.'''
    
    threads = relationship("Thread", back_populates="channel", order_by="Thread.id")
    '''Threads in the channel.'''
    messages = relationship("Message", secondary="pushes", viewonly=True, back_populates="channel", order_by="Message.id")
    '''Messages in the channel.'''

class Subscription(NoId):
    '''Maps agents to the set of channels they are subscribed to.'''
    __tablename__ = 'subs'
    
    agent_id = Column(Integer, ForeignKey("agents.id"), primary_key=True)
    '''Id of the agent.'''
    channel_id = Column(Integer, ForeignKey("channels.id"), primary_key=True)
    '''Id of the channel.'''
    
    agent = relationship("Agent", back_populates="subs")
    '''Agent this subscription belongs to.'''
    channel = relationship("Channel", back_populates="subs")
    '''Channel this subscription belongs to.'''

class Thread(HasId):
    '''An agent's subjective view of a channel.'''
    __tablename__ = 'threads'
    
    channel_id = Column(Integer, ForeignKey("channels.id"))
    '''Id of the channel this thread is associated with.'''
    agent_id = Column(Integer, ForeignKey("agents.id"))
    '''Id of the agent this thread belongs to.'''
    created_at = Column(Integer)
    '''Timestamp of thread creation (aka subscription time).'''
    destroyed_at = Column(Integer, nullable=True)
    '''Timestamp of thread destruction.'''
    
    __table_args__ = (UniqueConstraint('channel_id', 'agent_id'),)
    
    channel = relationship("Channel", back_populates="threads")
    '''Channel this thread is associated with.'''
    agent = relationship("Agent", back_populates="threads")
    '''Agent this thread belongs to.'''
    pushes = relationship("Push", back_populates="thread", order_by="Push.id")
    '''Pushes to this thread.'''
    messages = relationship("Message", secondary="pushes", viewonly=True, back_populates="thread", order_by="Message.id")
    '''Messages in this thread.'''

class Message(HasId):
    '''All messsages in all threads.'''
    __tablename__ = 'messages'
    
    # Invariant: All threads.channel are the same
    
    agent_id = Column(Integer, ForeignKey("agents.id"))
    '''Id of the agent which created this message.'''
    created_at = Column(Integer)
    '''Timestamp of message creation.'''
    
    agent = relationship("Agent", back_populates="messages")
    '''Agent which created this message.'''
    steps = relationship("Step", back_populates="message")
    '''Steps in the creation of this message.'''
    pushes = relationship("Push", back_populates="message", order_by="Push.id")
    '''Pushes of this message.'''
    threads = relationship("Thread", secondary="pushes", viewonly=True, back_populates="messages", order_by="Thread.id")
    '''Threads this message was pushed to.'''
    dependencies = relationship("Dependency", back_populates="dependency", foreign_keys="Dependency.dependency_id", order_by="Dependency.dependency_id")
    '''Messages this message depends on.'''
    dependents = relationship("Dependency", back_populates="message", foreign_keys="Dependency.message_id", order_by="Dependency.dependency_id")
    '''Messages which depend on this message.'''

class Step(HasId):
    '''Steps in the creation of a message.'''
    __tablename__ = 'steps'
    
    message_id = Column(Integer, ForeignKey("messages.id"))
    '''Id of the message this step belongs to.'''
    kind = Column(String)
    '''Kind of step.'''
    content = Column(String)
    '''Content of the message.'''
    
    message = relationship("Message", back_populates="steps")
    '''Message this step belongs to.'''

class Dependency(NoId):
    '''Maps messages to the set of messages used to create them.'''
    __tablename__ = 'depends'
    
    message_id = Column(Integer, ForeignKey("messages.id"), primary_key=True)
    '''Id of the message this dependency belongs to.'''
    dependency_id = Column(Integer, ForeignKey("messages.id"), primary_key=True)
    '''Id of the message this message depends on.'''
    
    message = relationship("Message", back_populates="depends")
    '''Message this dependency belongs to.'''
    dependency = relationship("Message", back_populates="dependents")
    '''Message this message depends on.'''

class Push(NoId):
    '''A message pushed to a particular agent.'''
    __tablename__ = "pushes"
    
    thread_id = Column(Integer, ForeignKey("threads.id"), primary_key=True)
    '''Id of the thread the message was pushed to.'''
    message_id = Column(Integer, ForeignKey("messages.id"), primary_key=True)
    '''Id of he message pushed to the agent.'''
    
    __table_args__ = (UniqueConstraint('thread_id', 'message_id'),)
    
    thread = relationship("Thread", back_populates="pushes")
    '''Thread the message was pushed to.'''
    message = relationship("Message", back_populates="pushes")
    '''Message pushed to the agent.'''

class Summary(HasId):
    '''Summaries of the conversation to compress the context.'''
    __tablename__ = 'summaries'
    
    checkpoint_id = Column(Integer, ForeignKey("messages.id"))
    '''Checkpoint at which the summary was created.'''
    created_at = Column(Integer)
    '''Timestamp of summary creation.'''
    content = Column(String)
    '''Content of the summary.'''
    
    checkpoint = relationship("Message")
    '''Checkpoint at which the summary was created.'''

class User(HasId):
    '''User profile and authentication information.'''
    
    agent_id = Column(Integer, ForeignKey("agents.id"))
    '''Id of the agent the user is associated with.'''
    created_at = Column(Integer)
    '''Timestamp of user creation.'''
    destroyed_at = Column(Integer, nullable=True)
    '''Timestamp of user destruction.'''
    
    agent = relationship("Agent", back_populates="user")
    '''Agent the user is associated with.'''

class Log(HasId):
    '''Log messages.'''
    __tablename__ = 'logs'
    
    level = Column(Integer)
    '''Log level.'''
    created_at = Column(Integer)
    '''Timestamp of log creation.'''
    content = Column(String)
    '''Log message content.'''

class DBLogHandler(logging.Handler):
    '''Log handler for emitting SQL.'''
    
    def __init__(self, memory: 'Memory'):
        super().__init__()
        self.memory = memory
    
    @override
    def emit(self, record: logging.LogRecord):
        self.memory.add(Log(
            level=record.levelno,
            created_at=time.time(),
            content=record.getMessage()
        ))

class Memory:
    '''SQLAlchemy database-backed memory.'''
    
    config: dict
    '''Database configuration.'''
    
    engine: sqlalchemy.engine.Engine
    '''SQLAlchemy engine.'''
    
    def __init__(self, config):
        super().__init__()
        
        logger.addHandler(DBLogHandler(self))
        
        self.config = config
        self.engine = create_engine(
            config.get('database', DATABASE_FILE), echo=True
        )
        
        # Initialize the database
        if not sqlalchemy.inspect(self.engine).has_table('agents'):
            Base.metadata.create_all(self.engine)
            with Session(self.engine) as session:
                session.add_all([
                    Thread(name='main')
                ])
                session.commit()
    
    def setup(self, kernel: "Kernel"):
        @kernel.register_command
        async def sql(query):
            '''Execute raw SQL.'''
            with Session(self.engine) as session:
                # TODO
                session.execute(query)
                return "{table}\n\n{x} rows updated in {dt}s"
        
        @kernel.register_command
        async def select(query):
            '''Execute raw SQL using SELECT.'''
            with Session(self.engine) as session:
                # TODO
                session.execute(query)
                return "{table}\n\n{x} rows updated in {dt}s"
    
    def add(self, *obj: Base):
        '''Add an object to the database.'''
        with Session(self.engine) as session:
            session.add_all(obj)
            session.commit()
    
    def agents(self) -> list[Agent]:
        '''Get all agents.'''
        with Session(self.engine) as session:
            return list(session.execute(select(Agent)).scalars())
    
    def channels(self) -> dict[str, Channel]:
        '''Get all channels.'''
        with Session(self.engine) as session:
            return {str(chan.name): chan for chan in session.execute(select(Channel)).scalars()}
    
    def get_thread_buffer(self, thread_id: int, checkpoint_id: int) -> list[Message]:
        '''
        Get the buffer of a thread.
        
        Parameters:
        thread_id: int
            The thread to get the buffer of.
        checkpoint_id: int
            The checkpoint to get the buffer from.
        '''
        
        with Session(self.engine) as session:
            pushes = session.execute(
                select(Push)
                    .where(Push.thread_id == thread_id)
                    .where(Push.message_id > checkpoint_id)
            ).scalars()
            
            # GPT-4 insists using in_ is optimized, comparable to sqlite's executemany
            messages = session.execute(
                select(Message)
                    .where(Message.id.in_(p.message_id for p in pushes))
            ).scalars()
            
            return list(messages)
    
    def message_steps(self, msg: Message) -> list[Step]:
        '''Get the steps of a message.'''
        
        with Session(self.engine) as session:
            return list(session.execute(
                select(Step).where(Step.message_id == msg.id)
            ).scalars())