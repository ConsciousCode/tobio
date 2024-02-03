from typing import Any, Literal, Optional, override
import uuid
import time
from datetime import datetime

import chainlit as cl
import chainlit.types as cl_types
from chainlit.element import ElementDict, ElementDisplay, ElementType
from chainlit.step import StepDict
import chainlit.data as cl_data
from chainlit.user import UserDict

from literalai import PageInfo, PaginatedResponse
from literalai.step import StepType

from .orm import Base, Column, Database, PrimaryKey, many_to_one, one_to_many, many_to_many, one_to_one, select, update, insert, delete, func, UUID

def optional_uuid(s: Optional[str]) -> Optional[UUID]:
    return None if s is None else uuid.UUID(s)

def optional_str(s: Optional[Any]) -> Optional[str]:
    return None if s is None else str(s)

def to_iso(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).isoformat()

def from_iso(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()

class Tag(Base):
    __tablename__ = "tags"
    
    id = Column[int](primary_key=True)
    name = Column[str](unique=True)
    
    def __init__(self, name: str): ...

class StepTag(Base):
    __tablename__ = "step_tags"
    
    step_guid = Column[UUID](foreign="steps.guid")
    tag_rowid = Column[int](foreign="tags.id")
    
    def __init__(self, step_guid: UUID, tag_rowid: int): ...
    
    __table_args__ = (
        PrimaryKey(step_guid, tag_rowid),
    )

class ThreadTag(Base):
    __tablename__ = "thread_tags"
    
    thread_guid = Column[UUID](foreign="threads.guid")
    tag_rowid = Column[int](foreign="tags.id")
    
    def __init__(self, thread_guid: UUID, tag_rowid: int): ...
    
    __table_args__ = (
        PrimaryKey(thread_guid, tag_rowid),
    )

class User(Base):
    __tablename__ = "users"
    
    guid = Column[UUID](primary_key=True)
    name = Column[str](unique=True)
    created_at = Column[float]()
    deleted_at = Column[Optional[float]](default=None)
    metadata_ = Column[Optional[dict]]("metadata")
    
    def __init__(self, guid: UUID, name: str, created_at: float, metadata_: Optional[Any]=None): ...
    
    threads = one_to_many(lambda: Thread, backref=lambda: Thread.user)
    sessions = one_to_many(lambda: UserSession, backref=lambda: UserSession.user)
    
    def to_dict(self) -> UserDict:
        return {
            "id": str(self.guid),
            "identifier": self.name,
            "metadata": self.metadata_ or {}
        }

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    guid = Column[UUID](primary_key=True)
    created_at = Column[float]()
    deleted_at = Column[Optional[float]](default=None)
    anon_user_id = Column[str]()
    user_guid = Column[Optional[UUID]](foreign="users.guid")
    is_interactive = Column[bool]()
    
    def __init__(self, guid: UUID, created_at: float, anon_user_id: str, user_guid: Optional[UUID]=None, is_interactive=False): ...
    
    user = many_to_one(lambda: User, backref=lambda: User.sessions)
    
    def to_dict(self) -> dict:
        return {
            "id": str(self.guid),
            "startedAt": datetime.utcfromtimestamp(self.created_at).isoformat(),
            "anonUserId": self.anon_user_id,
            "userId": str(self.user_guid) if self.user_guid else None
        }

class Thread(Base):
    __tablename__ = "threads"
    
    guid = Column[UUID](primary_key=True)
    user_guid = Column[Optional[UUID]](foreign="users.guid")
    title = Column[str]()
    created_at = Column[float]()
    updated_at = Column[float]()
    deleted_at = Column[Optional[float]](default=None)
    summary = Column[Optional[str]]()
    metadata_ = Column[Optional[Any]]("metadata")
    
    def __init__(self, guid: UUID, user_guid: Optional[UUID], title: str, created_at: float, updated_at: float, summary: Optional[str]=None, metadata_: Optional[Any]=None): ...
    
    user = many_to_one(lambda: User, backref=lambda: User.threads)
    tags = many_to_many(lambda: Tag, secondary=lambda: ThreadTag)
    steps = one_to_many(lambda: Step, backref=lambda: Step.thread)
    elements = one_to_many(lambda: Element, backref=lambda: Element.thread)

class Step(Base):
    __tablename__ = "steps"
    
    guid = Column[UUID](primary_key=True)
    name = Column[str]()
    type = Column[StepType]()
    metadata_ = Column[Optional[Any]]("metadata")
    parent_guid = Column[Optional[UUID]](foreign="steps.guid")
    thread_guid = Column[UUID](foreign="threads.guid")
    created_at = Column[float]()
    finished_at = Column[Optional[float]]()
    deleted_at = Column[Optional[float]](default=None)
    
    def __init__(self, guid: UUID, name: str, type: StepType, metadata_: Optional[Any], parent_guid: Optional[UUID], thread_guid: Optional[UUID], created_at: float, finished_at: Optional[float]=None): ...
    
    thread = many_to_one(lambda: Thread, backref=lambda: Thread.steps)
    parent = many_to_one(lambda: Step, backref=lambda: Step.children)
    children = one_to_many(lambda: Step, backref=lambda: Step.parent)
    tags = many_to_many(lambda: Tag, secondary=lambda: StepTag)
    feedback = one_to_one(lambda: Feedback, backref=lambda: Feedback.step)
    
    def to_dict(self) -> StepDict:
        metadata = self.metadata_ or {}
        return {
            "name": self.name,
            "type": self.type,
            "id": str(self.guid),
            "threadId": str(self.thread_guid),
            "parentId": optional_str(self.parent_guid),
            "disableFeedback": metadata.get("disableFeedback", False),
            "streaming": metadata.get("streaming", False),
            "waitForAnswer": metadata.get("waitForAnswer"),
            "isError": metadata.get("isError"),
            "metadata": metadata,
            "input": metadata.get("input", ""),
            "output": metadata.get("output", ""),
            "createdAt": to_iso(self.created_at),
            "start": None,
            "end": None,
            "generation": None,
            "showInput": metadata.get("showInput"),
            "language": metadata.get("language"),
            "indent": metadata.get("indent"),
            "feedback": None
        }

class Element(Base):
    __tablename__ = "elements"
    
    guid = Column[UUID](primary_key=True)
    type = Column[ElementType]()
    metadata_ = Column[Optional[Any]]("metadata")
    thread_guid = Column[Optional[UUID]](foreign="threads.guid")
    display = Column[ElementDisplay]()
    created_at = Column[float]()
    deleted_at = Column[Optional[float]](default=None)
    
    def __init__(self, guid: UUID, type: ElementType, metadata_: Optional[Any], thread_guid: Optional[UUID], display: ElementDisplay, created_at: float): ...
    
    thread = many_to_one(lambda: Thread, backref=lambda: Thread.elements)
    
    def to_dict(self) -> ElementDict:
        metadata = self.metadata_ or {}
        return {
            "id": str(self.guid),
            "threadId": optional_str(self.thread_guid),
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

class Feedback(Base):
    __tablename__ = "feedback"
    
    guid = Column[UUID](primary_key=True)
    step_guid = Column[UUID](foreign="steps.guid")
    value = Column[Literal[-1, 0, 1]]()
    strategy = Column[cl_types.FeedbackStrategy]()
    created_at = Column[float]()
    comment = Column[Optional[str]]()
    
    def __init__(self, guid: UUID, step_guid: UUID, value: Literal[-1, 0, 1], strategy: cl_types.FeedbackStrategy, created_at: float, comment: Optional[str]): ...
    
    step = one_to_one(lambda: Step, backref=lambda: Step.feedback)

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
            createdAt=to_iso(ct),
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
            session.deleted_at = from_iso(ended_at)
        
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
            created_at=from_iso(started_at),
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
        self.db.add(Step(
            uuid.uuid4(),
            name=step_dict.get("name", "undefined"),
            type=step_dict.get('type', "undefined"),
            metadata_=step_dict.get("metadata"),
            parent_guid=step_dict.get("parent"),
            thread_guid=optional_uuid(step_dict.get("threadId")),
            created_at=time.time()
        ))
    
    @override
    @cl_data.queue_until_user_message()
    async def update_step(self, step_dict: StepDict):
        step: Step = self.db.execute(
            select(Step).where(Step.guid == step_dict.get('id'))
        ).one()[0]
        
        if name := step_dict.get("name"):
            step.name = name
        if type := step_dict.get("type"):
            step.type = type
        if thread := step_dict.get("threadId"):
            step.thread_guid = uuid.UUID(thread)
        
        step.parent_guid = optional_uuid(step_dict.get("parentId"))
        
        # start is considered immutable
        if end := step_dict.get("end"):
            step.finished_at = from_iso(end)
        
        if new_md := step_dict.get("metadata"):
            step.metadata_ = {
                **(step.metadata_ or {}),
                **new_md,
                "disableFeedback": step_dict.get("disableFeedback", False),
                "streaming": step_dict.get("streaming", False),
                "waitForAnswer": step_dict.get("waitForAnswer"),
                "isError": step_dict.get("isError"),
                "input": step_dict.get("input", ""),
                "output": step_dict.get("output", ""),
                "createdAt": step_dict.get("createdAt"),
                "generation": step_dict.get("generation"),
                "showInput": step_dict.get("showInput"),
                "language": step_dict.get("language"),
                "indent": step_dict.get("indent")
            }
        else:
            step.metadata_ = None
        
        if feedback := step_dict.get("feedback"):
            if step.feedback is None:
                step.feedback = self.db.add(Feedback(
                    uuid.uuid4(),
                    step.guid,
                    feedback.get("value"),
                    feedback.get("strategy"),
                    time.time(),
                    feedback.get("comment")
                ))
        
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
        
        after_row = self.db.execute(
            select(Thread).where(Thread.guid == pagination.cursor)
        ).one_or_none()
        after: float = after_row[0].created_at if after_row else 0
        
        query = (select(Thread)
            .where(Thread.deleted_at == None)
            .where(Thread.created_at > after)
        )
        if filters.userIdentifier:
            query = query.where(
                Thread.user_guid == uuid.UUID(filters.userIdentifier)
            )
        
        if filters.search:
            query = query.where(
                Thread.title.ilike(f"%{filters.search}%")
            )
        
        if filters.feedback:
            query = query.join(Step).join(Feedback).where(
                Feedback.value == filters.feedback
            )
        
        threads = self.db.execute(
            query
                .order_by(Thread.created_at.desc())
                .limit(pagination.first + 1)
        ).fetchall()
        hasNextPage = (len(threads) > pagination.first)
        
        data: list[cl_types.ThreadDict] = []
        for tr in threads:
            thread: Thread = tr[0]
            data.append({
                "id": str(thread.guid),
                "createdAt": to_iso(thread.created_at),
                "user": thread.user.to_dict(),
                "tags": [tag.name for tag in thread.tags],
                "metadata": thread.metadata_,
                "steps": [step.to_dict() for step in thread.steps],
                "elements": None
            })
        
        return PaginatedResponse(
            PageInfo(
                hasNextPage=hasNextPage,
                endCursor=threads[-1][0].id if hasNextPage else None
            ), data
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
        
        thread_guid = uuid.UUID(thread_id)
        
        row = self.db.execute(
            select(Thread).where(Thread.guid == thread_guid)
        ).fetchone()
        # Insert
        if row is None:
            md = metadata or {}
            now = time.time()
            thread = self.db.add(Thread(
                thread_guid,
                user_guid=optional_uuid(user_id),
                title=md.get("title", "undefined"),
                created_at=now,
                updated_at=now,
                metadata_=metadata
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
                update(Feedback).where(Feedback.guid == feedback.id).values(
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
        thread_id = optional_uuid(element_dict.get("threadId"))
        self.db.add(Element(
            guid,
            type=element_dict.get("type"),
            metadata_=element_dict.get("metadata"),
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