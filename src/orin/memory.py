from typing import Optional

from chainlit import TrueStepType

from .orm import Base, Column, NoId, HasId, PrimaryKey, many_to_one, one_to_many, many_to_many

class Thread(HasId):
    __tablename__ = "threads"
    
    guid = Column[str](primary_key=True)
    created_at = Column[float]()
    updated_at = Column[float]()
    deleted_at = Column[Optional[float]]()
    title = Column[str]()
    summary = Column[Optional[str]]()

class Tag(HasId):
    __tablename__ = "tags"
    
    id = Column[int](primary_key=True)
    name = Column[str](unique=True)

class StepTag(NoId):
    __tablename__ = "step_tags"
    
    step_guid = Column[str](foreign_key=lambda: Step.guid)
    tag_rowid = Column[int](foreign_key=lambda: Tag.id)
    
    __table_args__ = (
        PrimaryKey(step_guid, tag_rowid),
    )

class Step(HasId):
    __tablename__ = "steps"
    
    guid = Column[str](primary_key=True)
    name = Column[str]()
    type = Column[TrueStepType]()
    metadata_ = Column[Optional[str]]("metadata")
    parent_guid = Column[Optional[str]](foreign_key=lambda: Step.guid)
    thread_guid = Column[str](foreign_key=lambda: Thread.guid)
    created_at = Column[float]()
    finished_at = Column[Optional[float]]()
    
    parent = many_to_one(lambda: Step, backref=lambda: Step.children)
    children = one_to_many(lambda: Step, backref=lambda: Step.parent)
    tags = many_to_many(lambda: Tag, secondary=lambda: StepTag)