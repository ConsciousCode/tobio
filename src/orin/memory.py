from typing import Any, Literal, Optional, override
import uuid
import time
from datetime import datetime

import chainlit as cl
from chainlit import TrueStepType
import chainlit.types as cl_types
from chainlit.element import ElementDict, ElementDisplay, ElementType
from chainlit.step import StepDict
import chainlit.data as cl_data
from chainlit.user import UserDict

from literalai import PageInfo, PaginatedResponse
from peewee import IntegrityError

from .orm import Base, Column, Database, NoId, HasId, PrimaryKey, insert_with_on_conflict, many_to_one, one_to_many, many_to_many, select, update, insert, delete, func, UUID

class Tag(HasId):
    __tablename__ = "tags"
    
    id = Column[int](primary_key=True)
    name = Column[str](unique=True)
    
    def __init__(self, name: str): ...

class StepTag(NoId):
    __tablename__ = "step_tags"
    
    step_guid = Column[UUID](foreign_key=lambda: Step.guid)
    tag_rowid = Column[int](foreign_key=lambda: Tag.id)
    
    def __init__(self, step_guid: UUID, tag_rowid: int): ...
    
    __table_args__ = (
        PrimaryKey(step_guid, tag_rowid),
    )

class ThreadTag(NoId):
    __tablename__ = "thread_tags"
    
    thread_guid = Column[UUID](foreign_key=lambda: Thread.guid)
    tag_rowid = Column[int](foreign_key=lambda: Tag.id)
    
    def __init__(self, thread_guid: UUID, tag_rowid: int): ...
    
    __table_args__ = (
        PrimaryKey(thread_guid, tag_rowid),
    )

class User(HasId):
    __tablename__ = "users"
    
    guid = Column[UUID](primary_key=True)
    name = Column[str](unique=True)
    created_at = Column[float]()
    deleted_at = Column[Optional[float]](default=None)
    metadata_ = Column[Optional[dict]]("metadata")
    
    def __init__(self, guid: UUID, name: str, created_at: float, metadata: Optional[Any]=None): ...
    
    threads = one_to_many(lambda: Thread, backref=lambda: Thread.user)
    sessions = one_to_many(lambda: UserSession, backref=lambda: UserSession.user)
    
    def to_dict(self) -> UserDict:
        return {
            "id": str(self.guid),
            "identifier": self.name,
            "metadata": self.metadata_ or {}
        }

class UserSession(HasId):
    __tablename__ = "user_sessions"
    
    guid = Column[UUID](primary_key=True)
    created_at = Column[float]()
    deleted_at = Column[Optional[float]](default=None)
    anon_user_id = Column[str]()
    user_guid = Column[Optional[UUID]](foreign_key=lambda: User.guid)
    is_interactive = Column[bool]()
    
    def __init__(self, guid: UUID, started_at: float, anon_user_id: str, user_guid: Optional[UUID]=None, is_interactive=False): ...
    
    user = many_to_one(lambda: User, backref=lambda: User.sessions)
    
    def to_dict(self) -> dict:
        return {
            "id": str(self.guid),
            "startedAt": datetime.utcfromtimestamp(self.created_at).isoformat(),
            "anonUserId": self.anon_user_id,
            "userId": str(self.user_guid) if self.user_guid else None
        }

class Thread(HasId):
    __tablename__ = "threads"
    
    guid = Column[UUID](primary_key=True)
    user_guid = Column[Optional[UUID]](foreign_key=lambda: User.guid)
    title = Column[str]()
    created_at = Column[float]()
    updated_at = Column[float]()
    deleted_at = Column[Optional[float]](default=None)
    summary = Column[Optional[str]]()
    metadata_ = Column[Optional[Any]]("metadata")
    
    def __init__(self, guid: UUID, user_guid: Optional[UUID], title: str, created_at: float, updated_at: float, summary: Optional[str]=None, metadata: Optional[Any]=None): ...
    
    user = many_to_one(lambda: User, backref=lambda: User.threads)
    tags = many_to_many(lambda: Tag, secondary=lambda: ThreadTag)
    steps = one_to_many(lambda: Step, backref=lambda: Step.thread)
    elements = one_to_many(lambda: Element, backref=lambda: Element.thread)

class Step(HasId):
    __tablename__ = "steps"
    
    guid = Column[UUID](primary_key=True)
    name = Column[str]()
    type = Column[TrueStepType]()
    metadata_ = Column[Optional[Any]]("metadata")
    parent_guid = Column[Optional[UUID]](foreign_key=lambda: Step.guid)
    thread_guid = Column[UUID](foreign_key=lambda: Thread.guid)
    created_at = Column[float]()
    finished_at = Column[Optional[float]]()
    deleted_at = Column[Optional[float]](default=None)
    
    def __init__(self, guid: UUID, name: str, type: TrueStepType, metadata: Optional[Any], parent_guid: Optional[UUID], thread_guid: Optional[UUID], created_at: float, finished_at: Optional[float]=None): ...
    
    thread = many_to_one(lambda: Thread, backref=lambda: Thread.steps)
    parent = many_to_one(lambda: Step, backref=lambda: Step.children)
    children = one_to_many(lambda: Step, backref=lambda: Step.parent)
    tags = many_to_many(lambda: Tag, secondary=lambda: StepTag)
    
    def to_dict(self) -> StepDict:
        metadata = self.metadata_ or {}
        return {
            "name": self.name,
            "type": self.type,
            "id": str(self.guid),
            "threadId": str(self.thread_guid),
            "parentId": str(self.parent_guid) if self.parent_guid else None,
            "disableFeedback": metadata.get("disableFeedback", False),
            "streaming": metadata.get("streaming", False),
            "waitForAnswer": metadata.get("waitForAnswer"),
            "isError": metadata.get("isError"),
            "metadata": metadata,
            "input": metadata.get("input", ""),
            "output": metadata.get("output", ""),
            "createdAt": datetime.utcfromtimestamp(self.created_at).isoformat(),
            "start": None,
            "end": None,
            "generation": None,
            "showInput": metadata.get("showInput"),
            "language": metadata.get("language"),
            "indent": metadata.get("indent"),
            "feedback": None
        }

class Element(HasId):
    __tablename__ = "elements"
    
    guid = Column[UUID](primary_key=True)
    type = Column[ElementType]()
    metadata_ = Column[Optional[Any]]("metadata")
    thread_guid = Column[UUID](foreign_key=lambda: Thread.guid)
    display = Column[ElementDisplay]()
    created_at = Column[float]()
    deleted_at = Column[Optional[float]](default=None)
    
    def __init__(self, guid: UUID, type: ElementType, metadata: Optional[Any], thread_guid: UUID, display: ElementDisplay, created_at: float): ...
    
    thread = many_to_one(lambda: Thread, backref=lambda: Thread.elements)
    
    def to_dict(self) -> ElementDict:
        metadata = self.metadata_ or {}
        return {
            "id": str(self.guid),
            "threadId": str(self.thread_guid),
            "type": self.type,
            "chainlitKey": metadata.get("chainlitKey"),
            "url": metadata.get("url"),
            "objectKey": metadata.get("objectKey"),
            "name": metadata.get("name", ""),
            "display": metadata.get("display", "inline"),
            "size": metadata.get("size"),
            "language": metadata.get("language"),
            "page": metadata.get("page"),
            "forId": metadata.get("forId"),
            "mime": metadata.get("mime")
        }

class Feedback(HasId):
    __tablename__ = "feedback"
    
    guid = Column[UUID](primary_key=True)
    step_guid = Column[UUID](foreign_key=lambda: Step.guid)
    value = Column[Literal[-1, 0, 1]]()
    strategy = Column[cl_types.FeedbackStrategy]()
    created_at = Column[float]()
    comment = Column[Optional[str]]()
    
    def __init__(self, guid: UUID, step_guid: UUID, value: Literal[-1, 0, 1], strategy: cl_types.FeedbackStrategy, created_at: float, comment: Optional[str]): ...

class DataLayer(cl_data.BaseDataLayer):
    def __init__(self, uri):
        self.db = Database(uri)
    
    @override
    async def get_user(self, identifier: str):
        user = self.db.execute(
            select(User).where(User.name == identifier)
        ).one()[0]
        
        return cl.PersistedUser(
            id=user.guid,
            createdAt=user.created_at,
            identifier=user.name,
            metadata=user.metadata_
        )
    
    @override
    async def create_user(self, user: cl.User):
        guid = uuid.uuid4()
        ct = time.time()
        self.db.add(User(guid, user.identifier, ct, None))
        return cl.PersistedUser(
            id=str(guid),
            createdAt=datetime.utcfromtimestamp(ct).isoformat(),
            identifier=user.identifier
        )
    
    @override
    async def update_user_session(
            self, id: str, is_interactive: bool, ended_at: Optional[str]
        ) -> dict:
        session: UserSession = self.db.execute(
            select(UserSession).where(UserSession.guid == id)
        ).one()[0]
        
        session.is_interactive = is_interactive
        if ended_at:
            session.deleted_at = datetime.fromisoformat(ended_at).timestamp()
        
        self.db.commit()
        return session.to_dict()
    
    @override
    async def create_user_session(
            self,
            id: str,
            started_at: str,
            anon_user_id: str,
            user_id: Optional[str],
        ) -> dict:
        return self.db.add(UserSession(
            guid=uuid.UUID(id),
            started_at=datetime.fromisoformat(started_at).timestamp(),
            anon_user_id=anon_user_id,
            user_guid=uuid.UUID(user_id) if user_id else None
        )).to_dict()
    
    @override
    async def delete_user_session(self, id: str) -> bool:
        self.db.execute(
            update(UserSession).where(UserSession.guid == id).values(
                deleted_at=time.time()
            )
        )
        self.db.commit()
        return True
    
    @override
    @cl_data.queue_until_user_message()
    async def create_step(self, step_dict: StepDict):
        # TODO: StepDict.id exists
        guid = uuid.uuid4()
        thread_id = step_dict.get("threadId")
        thread = None if thread_id is None else uuid.UUID(thread_id)
        self.db.add(Step(
            guid,
            name=step_dict.get("name", "undefined"),
            type=step_dict.get('type'),
            metadata=step_dict.get("metadata"),
            parent_guid=step_dict.get("parent"),
            thread_guid=thread,
            created_at=time.time()
        ))
    
    @override
    @cl_data.queue_until_user_message()
    async def update_step(self, step_dict: StepDict):
        self.db.execute(
            update(Step).where(Step.guid == step_dict.get('id')).values(
                name=step_dict.get("name", "undefined"),
                type=step_dict.get('type'),
                metadata=step_dict.get("metadata"),
                parent_guid=step_dict.get("parent"),
                finished_at=step_dict.get("end")
            )
        )
        self.db.commit()
    
    @override
    @cl_data.queue_until_user_message()
    async def delete_step(self, step_id: str):
        self.db.execute(
            update(Step).where(Step.guid == step_id).values(
                deleted_at=time.time()
            )
        )
        self.db.commit()
    
    @override
    async def get_thread_author(self, thread_id: str) -> str:
        thread: Thread = self.db.execute(
            select(Thread).where(Thread.guid == thread_id)
        ).one()[0]
        return thread.user.name
    
    @override
    async def list_threads(
            self,
            pagination: cl_types.Pagination,
            filters: cl_types.ThreadFilter
        ) -> PaginatedResponse[cl_types.ThreadDict]:
        # TODO: pagination
        threads = self.db.execute(
            select(Thread).where(Thread.deleted_at == None)
        ).fetchall()
        
        data: list[cl_types.ThreadDict] = []
        for tr in threads:
            thread: Thread = tr[0]
            data.append({
                "id": str(thread.guid),
                "createdAt": datetime.utcfromtimestamp(thread.created_at).isoformat(),
                "user": thread.user.to_dict(),
                "tags": [tag.name for tag in thread.tags],
                "metadata": thread.metadata_,
                "steps": [step.to_dict() for step in thread.steps],
                "elements": None
            })
        
        return PaginatedResponse(
            PageInfo(hasNextPage=False, endCursor=None), data
        )
    
    @override
    async def update_thread(
            self,
            thread_id: str,
            user_id: Optional[str] = None,
            metadata: Optional[dict] = None,
            tags: Optional[list[str]] = None,
        ):
        '''Undocumented, but this is an upsert not an update.'''
        
        row = self.db.execute(
            select(Thread).where(Thread.guid == thread_id)
        ).fetchone()
        # Insert
        if row is None:
            md = metadata or {}
            now = time.time()
            thread = self.db.add(Thread(
                uuid.UUID(thread_id),
                user_guid=uuid.UUID(user_id) if user_id else None,
                title=md.get("title", "undefined"),
                created_at=now,
                updated_at=now,
                metadata=metadata
            ))
            return
        
        # Update
        thread: Thread = row[0]
        if user_id:
            thread.user_guid = uuid.UUID(user_id)
        thread.metadata_ = metadata
        if tags:
            # Ensure tags exist, then update the thread's tags
            with self.db.session() as session:
                thread.tags.update(
                    session.merge(Tag(name=tag)) for tag in tags
                )
                session.commit()
    
    @override
    async def get_thread(self, thread_id: str):
        return self.db.execute(
            select(Thread).where(Thread.guid == thread_id)
        ).one()[0]
    
    @override
    async def delete_thread(self, thread_id: str):
        self.db.execute(
            delete(Thread).where(Thread.guid == thread_id)
        )
        self.db.commit()
    
    @override
    async def upsert_feedback(self, feedback: cl_types.Feedback) -> str:
        # Update
        if feedback.id:
            self.db.execute(
                update(Feedback).where(Feedback.id == feedback.id).values(
                    value=feedback.value,
                    strategy=feedback.strategy,
                    comment=feedback.comment
                )
            )
            self.db.commit()
            return feedback.id
        
        # Insert
        guid = uuid.uuid4()
        self.db.add(Feedback(
            guid,
            uuid.UUID(feedback.forId),
            feedback.value,
            feedback.strategy,
            time.time(),
            feedback.comment
        ))
        return str(guid)
    
    @override
    @cl_data.queue_until_user_message()
    async def create_element(self, element_dict: ElementDict):
        guid = uuid.uuid4()
        thread_id = uuid.UUID(element_dict.get("threadId", ""))
        self.db.add(Element(
            guid,
            type=element_dict.get("type"),
            metadata=element_dict.get("metadata"),
            thread_guid=thread_id,
            display=element_dict.get("display"),
            created_at=time.time()
        ))
    
    @override
    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional[ElementDict]:
        element = self.db.execute(
            select(Element).where(Element.guid == element_id)
        ).fetchone()
        return None if element is None else element[0].to_dict()
    
    @override
    @cl_data.queue_until_user_message()
    async def delete_element(self, element_id: str):
        self.db.execute(
            update(Element).where(Element.guid == element_id).values(
                deleted_at=time.time()
            )
        )
        self.db.commit()