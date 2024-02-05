from typing import Any, Literal, Optional, override
import uuid
import time

from .orm import Base, Column, Database, PrimaryKey, many_to_one, one_to_many, many_to_many, one_to_one, select, update, insert, delete, UUID, JSON, NoResultFound, Mapped
from .util import logger

class Persistent(Base):
    __tablename__ = "persistent"
    
    key: Mapped[str] = Column[str](primary_key=True)
    value: Mapped[JSON] = Column[JSON]()

class Author(Base):
    __tablename__ = "authors"
    
    guid: Mapped[UUID] = Column[UUID](primary_key=True)
    name: Mapped[str] = Column[str](unique=True)
    created_at: Mapped[float] = Column[float]()
    deleted_at: Mapped[Optional[float]] = Column[Optional[float]](default=None)

class Message(Base):
    __tablename__ = "messages"
    
    guid: Mapped[UUID] = Column[UUID](primary_key=True)
    name: Mapped[str] = Column[str]()
    parent_guid: Mapped[Optional[UUID]] = Column[Optional[UUID]](foreign="messages.guid")
    created_at: Mapped[float] = Column[float]()
    finished_at: Mapped[Optional[float]] = Column[Optional[float]]()
    deleted_at: Mapped[Optional[float]] = Column[Optional[float]](default=None)
    
    parent = many_to_one(lambda: Message, backref=lambda: Message.children)
    children = one_to_many(lambda: Message, backref=lambda: Message.parent)

class Memory:
    def __init__(self, uri):
        self.db = Database(uri)
        self.persistent_cache = {}
    
    def __getitem__(self, name):
        if name in self.persistent_cache:
            return self.persistent_cache[name]
        
        try:
            value = self.db.execute(
                select(Persistent).where(Persistent.key == name)
            ).one()[0].value
            self.persistent_cache[name] = value
            return value
        except NoResultFound:
            raise KeyError(name) from None
    
    def __setitem__(self, name, value):
        self.persistent_cache[name] = value
        self.db.execute(
            update(Persistent).where(Persistent.key == name).values(value=value)
        )
    
    def author(self, name):
        author = self.db.execute(
            select(Author).where(Author.name == name)
        ).one_or_none()
        if author:
            return author[0]
        
        guid = uuid.uuid4()
        ct = time.time()
        self.db.execute(
            insert(Author).values(
                guid=guid,
                name=name,
                created_at=ct
            )
        )
        return guid
    
    def get_author(self, guid: uuid.UUID) -> Author:
        return self.db.execute(
            select(Author).where(Author.guid == guid)
        ).one()[0]
    
    def get_messages(self, limit: int):
         self.db.execute(
            select(Message).limit(limit)
        ).fetchall()
    
    @override
    async def get_user(self, identifier: str):
        user = self.db.execute(
            select(Author).where(Author.name == identifier)
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
            session.add(Author(
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
            step = Message(
                guid=uuid.UUID(step_dict.get("id")),
                name=step_dict.get("name", "undefined"),
                type=step_dict.get('type', "undefined"),
                metadata_=Message.metadata_from_dict(step_dict),
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
                step: Message = session.execute(
                    select(Message).where(Message.guid == optional_uuid(step_dict.get('id')))
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
                **Message.metadata_from_dict(step_dict)
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
                update(Message).where(Message.guid == uuid.UUID(step_id)).values(
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
            query = query.join(Message).join(Feedback).where(
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