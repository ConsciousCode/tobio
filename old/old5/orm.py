'''
Thin wrapper around SQLAlchemy's ORM to narrow it to a restricted subset and
provide helpers.
'''

from types import UnionType
from typing import Any, Mapping, Optional, Sequence, Union, dataclass_transform, get_args, get_origin, overload
from sqlalchemy import  CursorResult, Executable, PrimaryKeyConstraint, Result, UpdateBase, create_engine
from sqlalchemy.sql import select, insert, update, delete, func
from sqlalchemy.sql.selectable import TypedReturnsRows
from sqlalchemy.types import Integer, String
from sqlalchemy.orm import DeclarativeBase, InstrumentedAttribute, Mapped, mapped_column, relationship, scoped_session, sessionmaker

from .util import Thunk

__all__ = [
    # Pass-through exports
    "select",
    "insert",
    "update",
    "delete",
    "func",
    
    # Alias exports
    "PrimaryKey",
    
    # New exports
    "Base",
    "NoId",
    "HasId",
    'column',
    "Database"
]

PrimaryKey = PrimaryKeyConstraint

class Base(DeclarativeBase):
    pass

type DeferColumn[T] = Thunk[InstrumentedAttribute[T]]

@overload
def column[T](t: UnionType, /, *,
    primary_key=False,
    foreign_key: Optional[DeferColumn]=None,
    unique=False,
    default: Optional[Any]=None
) -> Mapped[Optional[T]]: ...
@overload
def column[T](t: type[T], /, *,
    primary_key=False,
    foreign_key: Optional[DeferColumn]=None,
    unique=False,
    default: Optional[Any]=None
) -> Mapped[T]: ...

def column[T](t, /, *,
        primary_key=False,
        foreign_key: Optional[DeferColumn]=None,
        unique=False,
        default: Optional[Any]=None
    ) -> Any:
    '''Tighter type hinting.'''
    
    ct = {
        int: Integer,
        str: String
    }.get(t) # type: ignore
    
    nullable = False
    
    if ct is None:
        origin, args = get_origin(t), get_args(t)
        
        if origin is Union:
            if len(args) != 2 or type(None) not in args:
                raise TypeError(f"Unsupported Union type: {t}")
            ct = next(a for a in args if a is not type(None))
            nullable = True
        else:
            raise TypeError(f"Unsupported type: {t}")
    
    return mapped_column(ct,
        primary_key=primary_key,
        foreign_key=foreign_key,
        unique=unique,
        nullable=nullable,
        default=default
    )

def one_to_many[T: Base](table: Thunk[type[T]], /, *, backref: Optional[str]=None) -> Mapped[list[T]]:
    return relationship(table, back_populates=backref)

def many_to_one[T: Base](table: Thunk[type[T]], /, *, backref: Optional[str]=None) -> Mapped[T]:
    return relationship(table, backref=backref)

def many_to_many[T: Base](table: Thunk[type[T]], /, *, secondary: Thunk[type[Base]], backref: Optional[str]=None) -> Mapped[set[T]]:
    return relationship(table, secondary=lambda: secondary().__table__, backref=backref)

class NoId(Base):
    '''Base class for tables without an id.'''
    __abstract__ = True

class HasId(Base):
    '''Base class for tables with an id.'''
    __abstract__ = True
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

type SingleExecParams = Mapping[str, Any]
type MultiExecParams = Sequence[SingleExecParams]
type ExecParams = SingleExecParams | MultiExecParams

class Database:
    def __init__(self, url):
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.session = scoped_session(sessionmaker(bind=self.engine))
    
    def add[T: Base](self, row: T) -> T:
        with self.session() as session:
            session.add(row)
        return row
    
    def add_all(self, *args: Base) -> tuple[Base, ...]:
        with self.session() as session:
            session.add_all(args)
        return args
    
    def commit(self):
        self.session.commit()
    
    @overload
    def execute[T: tuple](self, statement: TypedReturnsRows[T], params: Optional[ExecParams]=None) -> Result[T]: ...
    @overload
    def execute(self, statement: UpdateBase, params: Optional[ExecParams]=None) -> CursorResult[Any]: ...
    @overload
    def execute(self, statement: Executable, params: Optional[ExecParams]=None) -> Result[Any]: ...
    
    def execute(self, statement, params=None):
        return self.session.execute(statement, params)
    