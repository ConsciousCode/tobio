'''
Thin wrapper around SQLAlchemy's ORM to narrow it to a restricted subset and
provide helpers.
'''

from contextlib import contextmanager
from functools import cached_property
import inspect
from types import GenericAlias
from typing import Any, Callable, Generic, Literal, Mapping, Optional, Self, Sequence, TypeAlias, TypeVar, Union, cast, dataclass_transform, get_args, get_origin, overload, override
from urllib.parse import urlparse
import uuid
from nio import dataclass

from sqlalchemy import Boolean, CursorResult, Executable, Float, ForeignKey, PrimaryKeyConstraint, Result, UpdateBase, Uuid, create_engine, JSON
from sqlalchemy.exc import NoResultFound
from sqlalchemy.sql import select, insert, update, delete, func
from sqlalchemy.sql.selectable import TypedReturnsRows
from sqlalchemy.types import Integer, String, TypeEngine
from sqlalchemy.orm import DeclarativeBase, InstrumentedAttribute, Mapped, MappedColumn, Session, declared_attr, mapped_column, relationship, scoped_session, sessionmaker

from .util import Thunk, typename

__all__ = [
    # Pass-through exports
    "select",
    "insert",
    "update",
    "delete",
    "func",
    
    'UUID',
    'JSON',
    
    "NoResultFound",
    
    # Alias exports
    "PrimaryKey",
    
    # New exports
    "Base",
    'Column',
    "Database",
    
    "one_to_one",
    "one_to_many",
    "many_to_one",
    "many_to_many",
]

UUID = uuid.UUID
PrimaryKey = PrimaryKeyConstraint

type DeferColumn[T] = Thunk[InstrumentedAttribute[T]]

class ColumnGeneric[T](GenericAlias):
    '''Hook into generic protocol so we can get the subscript.'''
    
    def __call__(self,
            name: Optional[str]=None, /, *,
            primary_key=False,
            foreign: Optional[str]=None,
            unique=False,
            default: Optional[Any]=None,
            nullable=False
        ) -> Mapped[T]:
        
        t = self.__args__[0]
        if isinstance(t, TypeEngine):
            ct = t
        else:
            CT_MAP: dict[Any, type[TypeEngine]] = {
                int: Integer,
                float: Float,
                str: String,
                bool: Boolean,
                uuid.UUID: Uuid,
                Any: JSON,
                dict: JSON
            }
            ct = CT_MAP.get(t)
            
            if ct is None:
                origin, args = get_origin(t), get_args(t)
                
                if origin is dict:
                    ct = JSON
                elif origin is Literal:
                    lt: type = type(args[0])
                    if not all(isinstance(a, lt) for a in args):
                        raise TypeError(f"Mixed type literals are not supported: Expected {typename(lt)} but got {args}")
                    ct = CT_MAP.get(lt)
                    if ct is None:
                        raise TypeError(f"Unsupported Literal type: {t}")
                elif origin is Union:
                    if type(None) in args:
                        args = tuple(a for a in args if a is not type(None))
                        nullable = True
                    
                    # Check for literal unions
                    lt = args[0]
                    if get_origin(lt) is Literal:
                        lt = type(get_args(lt)[0])
                        if not all(sa for a in args for sa in get_args(a)):
                            raise TypeError(f"Mixed type literals are not supported: Expected {typename(lt)} but got {args}")
                    elif len(args) != 2 and not nullable:
                        raise TypeError(f"Unsupported Union type: {t}")
                    ct = CT_MAP.get(lt)
                    if ct is None:
                        raise TypeError(f"Unsupported Union type: {t}")
                else:
                    raise TypeError(f"Unsupported type: {t}")
        
        if foreign is not None:
            ct = ForeignKey(foreign)
        
        if name is None:
            return mapped_column(
                ct,
                primary_key=primary_key,
                unique=unique,
                nullable=nullable,
                default=default
            )
        else:
            return mapped_column(
                name, ct,
                primary_key=primary_key,
                unique=unique,
                nullable=nullable,
                default=default
            )

class Column[T](Mapped[T]):
    def __class_getitem__(cls, item):
        return ColumnGeneric[T](cls, item)
    
    def __new__(cls,
            name: Optional[str]=None, /, *,
            primary_key=False,
            foreign: Optional[str]=None,
            unique=False,
            default: Optional[Any]=None,
            nullable=False
        ) -> Mapped[T]:
            raise NotImplementedError("Column must have a type")
    
    def __init__(self,
            name: Optional[str]=None, /, *,
            primary_key=False,
            foreign: Optional[str]=None,
            unique=False,
            default: Optional[Any]=None,
            nullable=False
        ):
            raise NotImplementedError("Column must have a type")

class Relationship[T, M](Mapped[M]):
    def __init__(self,
            table: Thunk[type[T]],
            backref: Optional[DeferColumn]=None,
            secondary: Optional[Thunk[type['Base']]]=None
        ):
        self.table = table
        self.backref = backref
        self.secondary = secondary
    
    def __set_name__(self, owner, name):
        self.name = name
    
    @overload
    def __get__(self, instance: None, owner) -> InstrumentedAttribute[M]: ...
    @overload
    def __get__(self, instance: object, owner) -> M: ...
    
    @override
    def __get__(self, instance, owner) -> InstrumentedAttribute[M]|M:
        backref = self.backref and self.backref().key
        secondary = self.secondary and (lambda: self.secondary().__table__) # type: ignore
        
        rel = relationship(
            self.table,
            back_populates=backref,
            secondary=secondary
        )
        setattr(owner, self.name, rel)
        return cast(InstrumentedAttribute[M]|M, rel.__get__(instance, owner))

def one_to_one[T: 'Base'](table: Thunk[type[T]], /, *, backref: Optional[DeferColumn]=None) -> Mapped[T]:
    '''This one row referring to one other row.'''
    return Relationship[T, T](table, backref)

def one_to_many[T: 'Base'](table: Thunk[type[T]], /, *, backref: Optional[DeferColumn]=None) -> Mapped[list[T]]:
    '''This one row referring to many rows.'''
    return Relationship[T, list[T]](table, backref)

def many_to_one[T: 'Base'](table: Thunk[type[T]], /, *, backref: Optional[DeferColumn]=None) -> Mapped[T]:
    '''Many rows referring to one row.'''
    return Relationship[T, T](table, backref)

def many_to_many[T: 'Base'](table: Thunk[type[T]], /, *, secondary: Thunk[type['Base']], backref: Optional[DeferColumn]=None) -> Mapped[set[T]]:
    '''Many rows referring to some subset of many other rows.'''
    return Relationship[T, set[T]](table, backref, secondary)

@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(Column,)
)
class Base(DeclarativeBase):
    pass

type SingleExecParams = Mapping[str, Any]
type MultiExecParams = Sequence[SingleExecParams]
type ExecParams = SingleExecParams | MultiExecParams

class Database:
    def __init__(self, url: str):
        u = urlparse(url)
        if u.netloc == '':
            # SQLAlchemy doesn't do proper URI parsing
            url = f"{u.scheme}:///{u._replace(scheme='').geturl()}"
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.session = scoped_session(sessionmaker(bind=self.engine))
    
    @contextmanager
    def transaction(self):
        session = self.session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
    
    @overload
    def execute[T: tuple](self, statement: TypedReturnsRows[T], params: Optional[ExecParams]=None) -> Result[T]: ...
    @overload
    def execute(self, statement: UpdateBase, params: Optional[ExecParams]=None) -> CursorResult[Any]: ...
    @overload
    def execute(self, statement: Executable, params: Optional[ExecParams]=None) -> Result[Any]: ...
    
    def execute(self, statement, params=None):
        return self.session.execute(statement, params)