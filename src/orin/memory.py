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

from .orm import Base, Column, Database, PrimaryKey, many_to_one, one_to_many, many_to_many, one_to_one, select, update, delete, UUID, NoResultFound, Mapped
from .util import logger

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
    
    id: Mapped[int] = Column[int](primary_key=True)
    name: Mapped[str] = Column[str](unique=True)

class StepTag(Base):
    __tablename__ = "step_tags"
    
    step_guid: Mapped[UUID] = Column[UUID](foreign="steps.guid")
    tag_rowid: Mapped[int] = Column[int](foreign="tags.id")
    
    __table_args__ = (
        PrimaryKey(step_guid, tag_rowid),
    )

class ThreadTag(Base):
    __tablename__ = "thread_tags"
    
    thread_guid: Mapped[UUID] = Column[UUID](foreign="threads.guid")
    tag_rowid: Mapped[int] = Column[int](foreign="tags.id")
    
    __table_args__ = (
        PrimaryKey(thread_guid, tag_rowid),
    )

class User(Base):
    __tablename__ = "users"
    
    guid: Mapped[UUID] = Column[UUID](primary_key=True)
    name: Mapped[str] = Column[str](unique=True)
    created_at: Mapped[float] = Column[float]()
    deleted_at: Mapped[Optional[float]] = Column[Optional[float]](default=None)
    metadata_: Mapped[Optional[dict]] = Column[Optional[dict]]("metadata")
    
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
    
    guid: Mapped[UUID] = Column[UUID](primary_key=True)
    created_at: Mapped[float] = Column[float]()
    deleted_at: Mapped[Optional[float]] = Column[Optional[float]](default=None)
    anon_user_id: Mapped[str] = Column[str]()
    user_guid: Mapped[Optional[UUID]] = Column[Optional[UUID]](foreign="users.guid")
    is_interactive: Mapped[bool] = Column[bool]()
    
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
    
    guid: Mapped[UUID] = Column[UUID](primary_key=True)
    user_guid: Mapped[Optional[UUID]] = Column[Optional[UUID]](foreign="users.guid")
    title: Mapped[str] = Column[str]()
    created_at: Mapped[float] = Column[float]()
    updated_at: Mapped[float] = Column[float]()
    deleted_at: Mapped[Optional[float]] = Column[Optional[float]](default=None)
    summary: Mapped[Optional[str]] = Column[Optional[str]]()
    metadata_: Mapped[Optional[Any]] = Column[Optional[Any]]("metadata")
    
    user = many_to_one(lambda: User, backref=lambda: User.threads)
    tags = many_to_many(lambda: Tag, secondary=lambda: ThreadTag)
    steps = one_to_many(lambda: Step, backref=lambda: Step.thread)
    elements = one_to_many(lambda: Element, backref=lambda: Element.thread)

class Step(Base):
    __tablename__ = "steps"
    
    guid: Mapped[UUID] = Column[UUID](primary_key=True)
    name: Mapped[str] = Column[str]()
    type: Mapped[StepType] = Column[StepType]()
    metadata_: Mapped[Optional[Any]] = Column[Optional[Any]]("metadata")
    parent_guid: Mapped[Optional[UUID]] = Column[Optional[UUID]](foreign="steps.guid")
    thread_guid: Mapped[UUID] = Column[UUID](foreign="threads.guid")
    created_at: Mapped[float] = Column[float]()
    finished_at: Mapped[Optional[float]] = Column[Optional[float]]()
    deleted_at: Mapped[Optional[float]] = Column[Optional[float]](default=None)
    
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
    
    @staticmethod
    def metadata_from_dict(d: StepDict) -> dict:
        '''
        StepDict has a lot of data we don't store in the table explicitly -
        this function takes a StepDict and returns a dict that can be stored
        in the metadata_ column.
        '''
        return {
            **d.get("metadata", {}),
            "disableFeedback": d.get("disableFeedback", False),
            "streaming": d.get("streaming", False),
            "waitForAnswer": d.get("waitForAnswer"),
            "isError": d.get("isError"),
            "input": d.get("input", ""),
            "output": d.get("output", ""),
            "createdAt": d.get("createdAt"),
            "generation": d.get("generation"),
            "showInput": d.get("showInput"),
            "language": d.get("language"),
            "indent": d.get("indent")
        }

class Element(Base):
    __tablename__ = "elements"
    
    guid: Mapped[UUID] = Column[UUID](primary_key=True)
    type: Mapped[ElementType] = Column[ElementType]()
    metadata_: Mapped[Optional[Any]] = Column[Optional[Any]]("metadata")
    thread_guid: Mapped[Optional[UUID]] = Column[Optional[UUID]](foreign="threads.guid")
    display: Mapped[ElementDisplay] = Column[ElementDisplay]()
    created_at: Mapped[float] = Column[float]()
    deleted_at: Mapped[Optional[float]] = Column[Optional[float]](default=None)
    
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
    
    guid: Mapped[UUID] = Column[UUID](primary_key=True)
    step_guid: Mapped[UUID] = Column[UUID](foreign="steps.guid")
    value: Mapped[Literal[-1, 0, 1]] = Column[Literal[-1, 0, 1]]()
    strategy: Mapped[cl_types.FeedbackStrategy] = Column[cl_types.FeedbackStrategy]()
    created_at: Mapped[float] = Column[float]()
    comment: Mapped[Optional[str]] = Column[Optional[str]]()
    
    step: Mapped[Step] = one_to_one(lambda: Step, backref=lambda: Step.feedback)

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
        logger.debug(f"Creating user %s", user.identifier)
        guid = uuid.uuid4()
        ct = time.time()
        with self.db.transaction() as session:
            session.add(User(
                guid=guid,
                name=user.identifier,
                created_at=ct,
                metadata_=None
            ))
        
        return cl.PersistedUser(
            id=str(guid),
            createdAt=to_iso(ct),
            identifier=user.identifier
        )
    
    @override
    async def update_user_session(
            self, id: str, is_interactive: bool, ended_at: Optional[str]
        ) -> dict:
        logger.debug(f"Updating session %s", id)
        
        with self.db.transaction() as session:
            try:
                us: UserSession = session.execute(
                    select(UserSession).where(UserSession.guid == uuid.UUID(id))
                ).one()[0]
                
                us.is_interactive = is_interactive
                if ended_at:
                    us.deleted_at = from_iso(ended_at)
                
                return us.to_dict()
            except NoResultFound:
                logger.warn("Stale session %s", id)
                # The return is unused by the caller, but return something invalid
                #  so if they ever do use it we get a clear error
                return None # type: ignore
    
    @override
    async def create_user_session(
            self,
            id: str,
            started_at: str,
            anon_user_id: str,
            user_id: Optional[str],
        ) -> dict:
        '''Undocumented: Select or insert, not insert.'''
        logger.debug(f"create_user_session %s", id)
        with self.db.transaction() as session:
            cur = session.execute(
                select(UserSession).where(UserSession.guid == uuid.UUID(id))
            ).one_or_none()
            if cur:
                logger.debug(f"Session %s already exists, recovering", id)
                return cur[0].to_dict()
            us = UserSession(
                guid=uuid.UUID(id),
                created_at=from_iso(started_at),
                anon_user_id=anon_user_id,
                user_guid=optional_uuid(user_id),
                is_interactive=False
            )
            session.add(us)
            return us.to_dict()
    
    @override
    async def delete_user_session(self, id: str) -> bool:
        #logger.debug(f"Deleting session %s", id)
        with self.db.transaction() as session:
            session.execute(
                update(UserSession).where(UserSession.guid == uuid.UUID(id)).values(
                    deleted_at=time.time()
                )
            )
            return True
    
    @override
    @cl_data.queue_until_user_message()
    async def create_step(self, step_dict: StepDict):
        #logger.debug(f"Creating step %s", step_dict.get("id"))
        with self.db.transaction() as session:
            step = Step(
                guid=uuid.UUID(step_dict.get("id")),
                name=step_dict.get("name", "undefined"),
                type=step_dict.get('type', "undefined"),
                metadata_=Step.metadata_from_dict(step_dict),
                parent_guid=step_dict.get("parent"),
                thread_guid=uuid.UUID(step_dict["threadId"]),
                created_at=time.time(),
                finished_at=None
            )
            session.add(step)
    
    @override
    @cl_data.queue_until_user_message()
    async def update_step(self, step_dict: StepDict):
        #logger.debug(f"Updating step %s", step_dict.get("id"))
        with self.db.transaction() as session:
            try:
                step: Step = session.execute(
                    select(Step).where(Step.guid == optional_uuid(step_dict.get('id')))
                ).one()[0]
            except NoResultFound:
                logger.warn("Failed to find step %s", step_dict.get('id'))
                return
            
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
            
            # Just unconditionally set it, there's too much to check
            step.metadata_ = {
                **(step.metadata_ or {}),
                **Step.metadata_from_dict(step_dict)
            }
            
            if feedback := step_dict.get("feedback"):
                if step.feedback is None:
                    step.feedback = Feedback(
                        guid=uuid.uuid4(),
                        step_guid=step.guid,
                        value=feedback.get("value"),
                        strategy=feedback.get("strategy"),
                        created_at=time.time(),
                        comment=feedback.get("comment")
                    )
    
    @override
    @cl_data.queue_until_user_message()
    async def delete_step(self, step_id: str):
        #logger.debug(f"Deleting step %s", step_id)
        
        with self.db.transaction() as session:
            session.execute(
                update(Step).where(Step.guid == uuid.UUID(step_id)).values(
                    deleted_at=time.time()
                )
            )
    
    @override
    async def get_thread_author(self, thread_id: str) -> str:
        thread: Thread = self.db.execute(
            select(Thread).where(Thread.guid == uuid.UUID(thread_id))
        ).one()[0]
        return thread.user.name
    
    @override
    async def list_threads(
            self,
            pagination: cl_types.Pagination,
            filters: cl_types.ThreadFilter
        ) -> PaginatedResponse[cl_types.ThreadDict]:
        after_row = self.db.execute(
            select(Thread).where(Thread.guid == optional_uuid(pagination.cursor))
        ).one_or_none()
        after: float = after_row[0].created_at if after_row else 0
        
        query = (select(Thread)
            .where(Thread.deleted_at == None)
            .where(Thread.created_at > after)
        )
        # Add the filters
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
        
        if hasNextPage:
            # Extra assignment step for type checking, SQLAlchemy doesn't type
            #  rows properly so indexing is otherwise Any
            lt: Thread = threads[-1][0]
            last_thread = str(lt.guid)
            threads = threads[:-1]
        else:
            last_thread = None
        
        # Convert to ThreadDict
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
        
        return PaginatedResponse(PageInfo(hasNextPage, last_thread), data)
    
    @override
    async def update_thread(
            self,
            thread_id: str,
            user_id: Optional[str] = None,
            metadata: Optional[dict] = None,
            tags: Optional[list[str]] = None,
        ):
        '''Undocumented, but this is an upsert not an update.'''
        
        #logger.debug(f"Upserting thread %s", thread_id)
        with self.db.transaction() as session:
            thread_guid = uuid.UUID(thread_id)
            
            row = session.execute(
                select(Thread).where(Thread.guid == thread_guid)
            ).fetchone()
            if row is None:
                # Insert
                md = metadata or {}
                now = time.time()
                thread = Thread(
                    guid=thread_guid,
                    user_guid=optional_uuid(user_id),
                    title=md.get("title", "undefined"),
                    created_at=now,
                    updated_at=now,
                    summary=md.get("summary"),
                    metadata_=metadata
                )
                session.add(thread)
            else:
                # Update
                thread: Thread = row[0]
                if user_id:
                    thread.user_guid = uuid.UUID(user_id)
                thread.metadata_ = metadata
            
            if tags:
                # Ensure tags exist, then update the thread's tags
                thread.tags.update(
                    session.merge(Tag(
                        id=None, # type: ignore
                        name=tag
                    )) for tag in tags
                )
    
    @override
    async def get_thread(self, thread_id: str):
        with self.db.transaction() as session:
            return session.execute(
                select(Thread).where(Thread.guid == thread_id)
            ).one()[0]
    
    @override
    async def delete_thread(self, thread_id: str):
        #logger.debug(f"Deleting thread %s", thread_id)
        with self.db.transaction() as session:
            session.execute(
                delete(Thread).where(Thread.guid == thread_id)
            )
    
    @override
    async def upsert_feedback(self, feedback: cl_types.Feedback) -> str:
        #logger.debug(f"Upserting feedback for step %s", feedback.forId)
        with self.db.transaction() as session:
            if feedback.id:
                # Update
                session.execute(
                    update(Feedback).where(Feedback.guid == feedback.id).values(
                        value=feedback.value,
                        strategy=feedback.strategy,
                        comment=feedback.comment
                    )
                )
                fb_id = feedback.id
            else:
                # Insert
                guid = uuid.uuid4()
                session.add(Feedback(
                    guid=guid,
                    step_guid=uuid.UUID(feedback.forId),
                    value=feedback.value,
                    strategy=feedback.strategy,
                    created_at=time.time(),
                    comment=feedback.comment
                ))
                fb_id = str(guid)
            
            return fb_id
    
    @override
    @cl_data.queue_until_user_message()
    async def create_element(self, element_dict: ElementDict):
        #logger.debug(f"Creating element %s", element_dict.get("chainlitKey"))
        with self.db.transaction() as session:
            guid = uuid.uuid4()
            thread_id = optional_uuid(element_dict.get("threadId"))
            session.add(Element(
                guid=guid,
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
        with self.db.transaction() as session:
            element = session.execute(
                select(Element).where(Element.guid == element_id)
            ).fetchone()
            return None if element is None else element[0].to_dict()
    
    @override
    @cl_data.queue_until_user_message()
    async def delete_element(self, element_id: str):
        #logger.debug(f"Deleting element %s", element_id)
        with self.db.transaction() as session:
            session.execute(
                update(Element).where(Element.guid == element_id).values(
                    deleted_at=time.time()
                )
            )