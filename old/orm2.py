from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
import traceback
from typing import Any, Callable, ClassVar, Generator, Iterable, Literal, NewType, Optional, Self, Sequence, Sized, TypeAliasType, Union, assert_type, cast, dataclass_transform, get_args, get_origin, overload, override
import sqlite3
from contextlib import contextmanager
from urllib.parse import urlparse

from .util import NOT_GIVEN, NotGiven, OnePlus, indent, typecheck, typename, Defer, deferred_property

class Constraint(ABC):
    columns: tuple['Column', ...]
    
    def __init__(self, *columns: 'Column'):
        self.columns = columns
    
    def _colnames(self):
        return ', '.join(col.__name__ for col in self.columns)
    
    @abstractmethod
    def schema(self) -> str: ...

class Unique(Constraint):
    def schema(self) -> str:
        return f"UNIQUE ({self._colnames()})"

class PrimaryKey(Constraint):
    def schema(self) -> str:
        return f"PRIMARY KEY ({self._colnames()})"

class ForeignKey(Constraint):
    references = deferred_property[tuple['Column', ...]]()
    
    def __init__(self, *columns: 'Column', references: Defer[OnePlus["Column"]]):
        super().__init__(*columns)
        
        refs: Defer[tuple['Column', ...]]
        
        if isinstance(references, Callable):
            @wraps(references)
            def thunk_wrapper() -> tuple['Column', ...]:
                result = references()
                return tuple(result) if isinstance(result, Sequence) else (result,)
            
            refs = thunk_wrapper
        elif isinstance(references, Sequence):
            refs = tuple(references)
        else:
            refs = (references,)
        
        type(self).references.defer(self, refs)
    
    def schema(self) -> str:
        keys = ', '.join(col.__name__ for col in self.columns)
        table = self.references[0].table.__tablename__
        return f"FOREIGN KEY ({keys}) REFERENCES {table}({self._colnames()})"

class Operand(ABC):
    '''Mixin for implementing all the expression operators.'''
    
    def __add__(self, other): return Expression("+", self, other)
    def __sub__(self, other): return Expression("-", self, other)
    def __mul__(self, other): return Expression("*", self, other)
    def __truediv__(self, other): return Expression("/", self, other)
    def __mod__(self, other): return Expression("%", self, other)
    
    def __eq__(self, other): return Expression("=", self, other) # type: ignore
    def __ne__(self, other): return Expression("!=", self, other) # type: ignore
    def __lt__(self, other): return Expression("<", self, other)
    def __le__(self, other): return Expression("<=", self, other)
    def __gt__(self, other): return Expression(">", self, other)
    def __ge__(self, other): return Expression(">=", self, other)
    
    def __and__(self, other): return Expression("AND", self, other)
    def __or__(self, other): return Expression("OR", self, other)
    def __invert__(self): return Expression("NOT", self, None)
    
    def __radd__(self, other): return Expression("+", other, self)
    def __rsub__(self, other): return Expression("-", other, self)
    def __rmul__(self, other): return Expression("*", other, self)
    def __rtruediv__(self, other): return Expression("/", other, self)
    def __rmod__(self, other): return Expression("%", other, self)
    def __rpow__(self, other): return Expression("^", other, self)
    
    def __req__(self, other): return Expression("=", other, self)
    def __rne__(self, other): return Expression("!=", other, self)
    def __rlt__(self, other): return Expression("<", other, self)
    def __rle__(self, other): return Expression("<=", other, self)
    def __rgt__(self, other): return Expression(">", other, self)
    def __rge__(self, other): return Expression(">=", other, self)
    
    def __rand__(self, other): return Expression("AND", other, self)
    def __ror__(self, other): return Expression("OR", other, self)
    
    def __neg__(self): return Expression("-", self, None)
    def __pos__(self): return Expression("+", self, None)
    
    @abstractmethod
    def expr(self) -> tuple[str, tuple]: ...

@dataclass
class Expression(Operand):
    '''Simple expression constructed using operators.'''
    
    op: str
    left: Operand|Any
    right: Optional[Operand|Any]
    
    def __str__(self):
        return f"({self.left} {self.op} {self.right})"
    
    @override
    def expr(self) -> tuple[str, tuple]:
        if isinstance(self.left, Operand):
            lx, lv = self.left.expr()
            if self.right is None:
                return f"{self.op}{lx}", lv
            elif isinstance(self.right, Operand):
                rx, rv = self.right.expr()
                return f"({lx} {self.op} {rx})", lv + rv
            else:
                return f"({lx} {self.op} {self.right!r})", lv
        else:
            assert isinstance(self.right, Operand)
            rx, rv = self.right.expr()
            return f"({self.left!r} {self.op} {rx})", rv

class Descriptor[T]:
    __name__: str
    
    def __set_name__(self, owner, name):
        self.__name__ = name
    
    @overload
    def __get__(self, instance: None, owner) -> Self: ...
    @overload
    def __get__(self, instance, owner) -> T: ...
    
    def __get__(self, instance, owner) -> T|Self:
        if instance is None:
            return self
        return instance.__dict__[self.__name__]

class Column[T](Descriptor[T], Operand):
    '''Column descriptor.'''
    
    name: str
    '''Name of the column in the database, defaults to the attribute name.'''
    table: type['Row']
    '''The table the column belongs to.'''
    supertype: type
    '''The more general type which can be mapped to a database type.'''
    subtype: Any
    '''The more specific type hint for the application.'''
    nullable: bool
    default: T
    hidden: bool
    
    def __init__(self, kind: type[T], name: Optional[str]=None,default: T|NotGiven=NOT_GIVEN, hidden=False):
        self.subtype = kind
        if name is not None:
            self.name = name
        if not isinstance(default, NotGiven):
            self.default = default
        self.nullable = False # Set to True if Optional
        self.hidden = hidden
        
        if isinstance(kind, TypeAliasType):
            supertype = kind.__value__
        elif isinstance(kind, NewType):
            supertype = kind.__supertype__
        else:
            origin, args = get_origin(kind), get_args(kind)
            if origin is None:
                supertype = kind
            elif origin is Literal:
                LT = type(args[0])
                if not all(type(arg) is LT for arg in args[1:]):
                    raise TypeError(f'Only homogeneous literal types supported, got {kind}.')
                supertype = type(args[0])
            elif origin is not None and origin is Union:
                if len(args) != 2 or type(None) not in args:
                    raise TypeError('Only Optional unions supported.')
                
                supertype = [arg for arg in args if arg is not type(None)][0]
                self.nullable = True
            else:
                raise NotImplementedError(f'Unsupported type {kind}')
            
        self.supertype = supertype
    
    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        if not hasattr(self, "name"):
            self.name = name
        self.table = owner
    
    def __set__(self, instance: 'Row', value):
        assert_type(value, self.subtype)
        if instance.is_bound():
            instance._db.update(self.table).set(
                self.name, value
            ).where(**{
                col.__name__: getattr(instance, col.__name__)
                for col in type(instance).primary_keys
            }).execute()
        
        instance.__dict__[self.name] = value
    
    def has_default(self):
        return hasattr(self, "default")
    
    def type_schema(self) -> str:
        base = {
            int: "INTEGER",
            float: "REAL",
            str: "TEXT",
            bytes: "BLOB",
            bool: "INTEGER",
        }[self.supertype]
        
        if not self.nullable:
            base += " NOT NULL"
        if self.has_default():
            base += " DEFAULT "
            base += "NULL" if self.default is None else repr(self.default)
        return base
    
    def schema(self) -> str:
        return f"{self.name} {self.type_schema()}"
    
    @override
    def expr(self) -> tuple[str, tuple]:
        return f"{self.table.__tablename__}.{self.name}", ()

class RowidColumn(Column[int]):
    '''Special column for the rowid.'''
    
    def __init__(self):
        super().__init__(int, "rowid", hidden=True)
    
    # Have to respecify the overloads every fucking time apparently
    @overload
    def __get__(self, instance: None, owner) -> Self: ...
    @overload
    def __get__(self, instance, owner) -> int: ...
    
    def __get__(self, instance, owner) -> Self|int:
        if instance is None:
            return self
        if not instance.is_bound():
            raise TypeError('Cannot get the rowid of an unbound row.')
        return instance.__dict__[self.__name__]
    
    @override
    def __set__(self, instance, value):
        if self.__name__ not in instance.__dict__:
            instance.__dict__[self.__name__] = value
        else:
            raise RuntimeError("Cannot assign id twice.")

class Relationship[T](Descriptor[T]):
    '''Virtual relationship between two tables.'''
    
    table: type['Row']
    '''The table the relationship belongs to.'''
    kind: type[T]
    '''The raw type of the relationship.'''
    target: type['Row']
    '''The target table of the relationship.'''
    secondary: deferred_property[Optional[type['Row']]] = deferred_property()
    foreign = deferred_property[Optional[Column]]()
    backref = deferred_property[Optional[Column]]()
    
    def __init__(self,
            kind: type[T], *,
            foreign: Optional[Defer[Column]]=None,
            backref: Optional[Defer[Column]]=None
        ):
        if foreign is None and backref is None:
            raise TypeError('Must specify at least one of foreign or backref.')
        
        def resolve_secondary() -> Optional[Table]:
            fc = self.foreign
            bc = self.backref
            
            if fc is None or bc is None:
                return None
            
            if fc.table is not bc.table:
                raise TypeError('Cannot have a secondary table with foreign and backref on different tables.')
            
            # Secondary table is implied when foreign and backref are on the same table
            return fc.table
        
        type(self).secondary.defer(self, resolve_secondary)
        type(self).foreign.defer(self, foreign)
        type(self).backref.defer(self, backref)
        
        self.kind = kind
        
        origin, args = get_origin(kind), get_args(kind)
        if origin is None:
            # For god only knows why, TypeAliasType doesn't work with isinstance or issubclass
            if not issubclass(kind, (NoId, HasId)):
                raise TypeError(f'Unsupported type {kind}')
            self.target = kind
        elif issubclass(origin, Iterable):
            self.target = args[0]
        else:
            raise NotImplementedError(f'Unsupported type {kind}')
    
    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        self.table = owner
        if self.__dict__.get('backref') is not None and isinstance(owner, NoId):
            raise TypeError('Cannot have a backref on a table with no id.')
    
    @overload
    def __get__(self, instance: None, owner: type['Row']) -> Self: ...
    @overload
    def __get__(self, instance: 'Row', owner: type['Row']) -> T: ...
    
    def __get__(self, instance: Optional['Row'], owner: type['Row']) -> T|Self:
        if instance is None:
            return self
        
        # {one, many} to one will have T <: Row and only requires foreign
        # One to many will have T <: Sequence[Row] and only requires backref
        # Many to many is T <: Sequence[Row] and requires both
        
        query = instance._db.select(self.target)
        
        # singular => T <: Row and only requires foreign
        if self.kind is self.target:
            if self.foreign is None:
                raise TypeError('Cannot get a singular relationship without a foreign key.')
            
            if not isinstance(self.target, HasId):
                raise TypeError('Cannot use a foreign key for a table with no id.')
            
            return self.kind(query.where(self.target.id == self.foreign).fetchone())
        
        # one to many => T <: Sequence[Row] and only requires backref
        if self.secondary is None:
            if self.backref is None:
                raise TypeError('Cannot get a one to many relationship without a backref.')
            
            if not isinstance(instance, HasId):
                raise TypeError('Cannot use a backref for a table with no id.')
            
            query = query.where(self.backref == instance.id)
        
        # many to many => T <: Sequence[Row] and requires both
        else:
            if self.foreign is None or self.backref is None:
                raise TypeError('Cannot get a many to many relationship without both a foreign key and a backref.')
            
            if not isinstance(self.target, HasId):
                raise TypeError('Cannot use a foreign key for a table with no id.')
            
            if not isinstance(instance, HasId):
                raise TypeError('Cannot use a backref for a table with no id.')
            
            query = query.join(self.secondary).on(self.target.id == self.foreign).where(self.backref == instance.id)
        
        return self.kind(query.fetch())

class Table(type):
    __abstract__: bool
    __doc__: Optional[str]
    __tablename__: str
    __table_args__: tuple[Constraint, ...]
    _columns: dict[str, Column]
    _relationships: dict[str, Relationship]
    _defaults: dict[str, Any]
    
    def __new__(mcls, name, bases, ns):
        if not ns.get("__abstract__"):
            if "__tablename__" not in ns:
                raise TypeError('Table model missing __tablename__.')
            
            if "__table_args__" not in ns:
                ns["__table_args__"] = ()
            elif isinstance(ns["__table_args__"], list):
                ns["__table_args__"] = tuple(ns["__table_args__"])
            
            cols = {}
            rels = {}
            defs = {}
            
            for n, v in ns.items():
                if isinstance(v, Column):
                    if v.hidden:
                        continue
                    df = getattr(v, "default", NOT_GIVEN)
                    if df is not NOT_GIVEN:
                        defs[n] = df
                    cols[n] = v
                elif isinstance(v, Relationship):
                    rels[n] = v
            
            ns["_columns"] = cols
            ns["_relationships"] = rels
            ns["_defaults"] = defs
            
            # __init__ is just to make the type checker STFU, so remove it.
            ns.pop("__init__")
        
        return super().__new__(mcls, name, bases, ns)
    
    @property
    def primary_keys(cls) -> tuple["Column", ...]:
        if cls.is_abstract():
            raise TypeError('Cannot get primary key of abstract table.')
        
        if hasattr(cls, "id"):
            return (cls.id,) # type: ignore [attr-defined]
        
        for col in cls.__table_args__:
            if isinstance(col, PrimaryKey):
                return col.columns
        else:
            raise TypeError('No primary key defined.')
    
    def is_abstract(cls):
        return cls.__dict__.get("__abstract__") # Only check own properties
    
    def column_schemata(cls):
        return ',\n'.join(column.schema() for column in cls._columns.values())
    
    def constraint_schemata(cls):
        return ',\n'.join(constraint.schema() for constraint in cls.__table_args__)
    
    def schema(cls):
        if cls.is_abstract():
            raise TypeError('Cannot create schema for abstract table.')
        comment = f"/* {cls.__doc__} */\n" if cls.__doc__ else ""
        header = f"{comment}CREATE TABLE IF NOT EXISTS {cls.__tablename__}"
        body = indent(',\n\n'.join(filter(None, [cls.column_schemata(), cls.constraint_schemata()])))
        return f"{header} (\n{body}\n)"

class _BaseRow(metaclass=Table):
    __abstract__ = True
    _all_columns = "*"
    _db: 'Database'
    
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        
        return super().__new__(cls)
    
    def __init__(self, *args, **kwargs):
        '''Initialize the unbound row excluding any rowid.'''
        
        fields = type(self)._defaults.copy()
        df_count = len(fields)
        columns = type(self)._columns
        
        def assign(name, value):
            kind = columns[name].subtype
            if not typecheck(value, kind):
                raise TypeError(f"{typename(self)}.{name} must be {typename(kind)}, got {value!r}")
            
            fields[name] = value
        
        if len(args) > len(columns):
            raise TypeError(f"{typename(self)} expected {len(columns)} arguments, got {len(args)}")
        
        for name, arg in zip(columns.keys(), args):
            assign(name, arg)
        
        for name, arg in kwargs.items():
            if name not in columns:
                raise TypeError(f"{typename(self)} got an unexpected keyword argument {name!r}")
            
            assign(name, arg)
        
        if len(fields) != len(columns):
            raise TypeError(f"{typename(self)} expected {len(columns) - df_count} arguments, got {len(fields)}")
        
        self.__dict__.update(fields)
    
    def is_bound(self):
        return hasattr(self, "_db")
    
    def _bind(self, db: "Database", id: int):
        self._db = db
    
    @classmethod
    def from_row(cls, db: 'Database', row: Iterable):
        ob = cls(*row)
        ob._db = db
        return ob
    
    def __iter__(self):
        for name in self._columns:
            yield getattr(self, name)

# NoId and HasId have dataclass_transform separately to avoid id being
#  considered a field

@dataclass_transform(
    eq_default=False
)
class NoId(_BaseRow):
    '''Row with no id column.'''
    __abstract__ = True
    
    @classmethod
    def schema(cls):
        return f"{Table.schema(cls)} WITHOUT ROWID"

@dataclass_transform(
    eq_default=False
)
class HasId(_BaseRow):
    '''Row with an id column.'''
    __abstract__ = True
    _all_column_select = "rowid, *"
    id: Column[int]
    
    def __init_subclass__(cls):
        cls.id = RowidColumn() # type: ignore [assignment]
        cls.id.__set_name__(cls, "id")
    
    @override
    def _bind(self, db: "Database", id: int):
        self._db = db
        self.id = id
    
    @override
    @classmethod
    def from_row(cls, db: 'Database', row: Iterable):
        id, *cols = row
        ob = super().from_row(db, cols)
        ob.id = id
        return ob
    
    @classmethod
    def column_schemata(cls):
        return f"/* rowid */\n{Table.column_schemata(cls)}"

type Row = NoId|HasId
'''Row with or without an id. Defined as a union to emulate closed subclassing.'''

class Cursor[T](sqlite3.Cursor):
    '''Typed wrapper for sqlite3 Cursor.'''
    
    def __init__(self, db: 'Database', factory: Optional[type[T]|Callable[["Cursor", sqlite3.Row], T]]=None):
        super().__init__(db.connection)
        
        if factory is not None:
            if isinstance(factory, type):
                if issubclass(factory, (NoId, HasId)):
                    self.row_factory = lambda cur, row: factory.from_row(db, row)
                else:
                    self.row_factory = lambda cur, row: factory(*row)
            else:
                self.row_factory = lambda cur, row: factory(Cursor(db, cur.row_factory), row)
    
    def fetch(self, size: Optional[int]=None) -> list[T]:
        return self.fetchall() if size is None else self.fetchmany(size)
    
    def __iter__(self): return super().__iter__()
    def fetchmany(self, size: Optional[int]=None) -> list[T]: return super().fetchmany(size)
    def fetchall(self) -> list[T]: return super().fetchall()
    def fetchone(self) -> Optional[T]: return super().fetchone()
    
    def one(self) -> T:
        '''Get the ONLY row from the cursor, or raise an error.'''
        results = self.fetchmany(2)
        if len(results) == 1:
            return results[0]
        elif len(results) == 0:
            raise ValueError('No rows returned.')
        else:
            raise ValueError('More than one row returned.')

class Query[T]:
    '''
    Builds a statement within the context of a database for execution using
    a fluent interface.
    '''
    
    db: 'Database'
    clauses: list[str]
    binds: list
    factory: Optional[type[T]|Callable[["Cursor", sqlite3.Row], T]]
    
    def __init__(self, db: 'Database', clause: str="", factory: Optional[type[T]|Callable[["Cursor", sqlite3.Row], T]]=None):
        self.db = db
        self.clauses = [clause] if clause else []
        self.binds = []
        self.factory = factory
    
    def __str__(self): return ' '.join(self.clauses)
    def __repr__(self): return f"Query({str(self)!r})"
    
    @overload
    @contextmanager
    def _cursor(self, cur: None=None) -> Generator[Cursor[T], Any, Any]: ...
    @overload
    @contextmanager
    def _cursor[U](self, cur: Cursor[U]) -> Generator[Cursor[U], Any, Any]: ...
    
    @contextmanager
    def _cursor(self, cur: Optional[Cursor[T]]=None):
        if cur is None:
            with self.db.cursor(self.factory) as cur:
                yield cur
        else:
            yield cur
    
    def copy(self):
        '''Get a copy of the query.'''
        copy = Query.__new__(Query)
        copy.db = self.db
        copy.clauses = self.clauses[:]
        copy.binds = self.binds[:]
        copy.factory = self.factory
        return copy
    
    def raw(self, clause: str, values: Iterable=()) -> Self:
        '''Add a raw clause to the statement.'''
        self.clauses.append(clause)
        self.binds.extend(values)
        return self
    
    @overload
    def where(self, first: str|Expression, /, *rest: str|Expression, values: Iterable[Any]=()) -> Self: ...
    @overload
    def where(self, /, **kwargs: Any) -> Self:
        '''Shorthand for WHERE clauses consisting entirely of equality.'''
    
    def where(self, first: Optional[str|Expression]=None, /, *rest: str|Expression, values: Optional[Iterable[Any]]=None, **kwargs) -> Self:
        if first:
            values = tuple(values or ())
            clauses = []
            for clause in [first, *rest]:
                if isinstance(clause, Expression):
                    clause, binds = clause.expr()
                    values += binds
                clauses.append(clause)
        else:
            # Just in case values is a proper column
            if values is not None:
                kwargs['values'] = values
            values = tuple(kwargs.values())
            clauses = [f"{k} = ?" for k in kwargs]
        
        return self.raw(f"WHERE {' AND '.join(clauses)}", values=values)
    
    def order_by(self, *columns: str): return self.raw(f"ORDER BY {', '.join(columns)}")
    def limit(self, limit: int): return self.raw(f"LIMIT {limit}")
    def offset(self, offset: int): return self.raw(f"OFFSET {offset}")
    def join(self, other: type[Row]): return self.raw(f"JOIN {other.__tablename__}")
    
    @overload
    def set(self, name: str, value: Any, /) -> Self: ...
    @overload
    def set(self, **values: Any) -> Self: ...
    
    def set(self, name=None, value=None, **values: Any):
        if name is None and value is None:
            return self.raw(f"SET {', '.join(f'{k} = ?' for k in values)}", values.values())
        elif name is None or value is None:
            raise TypeError('Must specify both name and value or neither.')
        else:
            return self.raw(f"SET {name} = ?", (value,))
    
    @overload
    def values(self, *args: Any) -> Self: ...
    @overload
    def values(self, **kwargs: Any) -> Self: ...
    
    def values(self, *args: Any, **kwargs: Any):
        if args and kwargs:
            raise TypeError('Cannot mix positional and keyword arguments.')
        
        if args:
            return self.raw(f"VALUES ({', '.join('?'*len(args))})", args)
        else:
            return self.raw(
                f"({', '.join(kwargs.keys())}) VALUES({', '.join('?'*len(kwargs))})",
                kwargs.values()
            )
    
    def on(self, on: str|Expression, values: tuple=()) -> Self:
        if isinstance(on, Expression):
            on, binds = on.expr()
            values += binds
        return self.raw(f"ON {on}", values)
    
    @overload
    def execute(self) -> Cursor[T]: ...
    @overload
    def execute[U](self, cursor: Cursor[U]) -> Cursor[U]: ...
    
    def execute[U](self, cursor: Optional[Cursor[U]]=None) -> Cursor[T]|Cursor[U]:
        '''Execute the statement.'''
        with self._cursor(cursor) as cur:
            return cur.execute(str(self), tuple(self.binds))
    
    @overload
    def executemany(self, cursor: None=None) -> Cursor[T]: ...
    @overload
    def executemany[U](self, cursor: Cursor[U]) -> Cursor[U]: ...
    
    def executemany[U](self, cursor: Optional[Cursor[U]]=None) -> Cursor[T]|Cursor[U]:
        # Check that all binds are iterables of the same length
        L = None
        for col in self.binds:
            if not isinstance(col, Sized):
                raise TypeError("executemany requires iterables of rows for all bound values.")
            
            if L is None:
                L = len(col)
            elif len(col) != L:
                raise ValueError("Not all rows are the same length.")
        
        with self._cursor(cursor) as cur:
            return cur.executemany(str(self), zip(*self.binds))
    
    def fetch(self, size: Optional[int]=None) -> list[T]:
        '''Utility to execute and then fetch a number of rows.'''
        return self.execute().fetch(size)
    
    def fetchone(self) -> Optional[T]:
        '''Utility to execute and then fetch a single row.'''
        return self.execute().fetchone()
    
    def one(self) -> T:
        '''Utility to execute and then fetch a single row, or raise an error.'''
        return self.execute().one()

class Database:
    tables: ClassVar[list[Table]]
    connection: sqlite3.Connection
    
    def __init_subclass__(cls):
        if not hasattr(cls, "tables"):
            raise TypeError('Database subclass missing tables.')
    
    def __init__(self, filename: str):
        uri = urlparse(filename)
        if uri.scheme not in {"", "file", "sqlite"}:
            raise NotImplementedError(f"Database URI scheme {uri.scheme!r}")
        self.connection = sqlite3.connect(uri.path)
        self.connection.executescript(self.schema())
    
    @contextmanager
    def cursor[T](self, factory: Optional[Callable[[Cursor, sqlite3.Row], T]]=None):
        '''Context manager for a logical session.'''
        try:
            yield Cursor(self, factory)
        except:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()
    
    def schema(self):
        return ';\n\n'.join(table.schema() for table in self.tables)
    
    def execute[T](self, query: str, values: tuple=(), factory: Optional[Callable[[Cursor, sqlite3.Row], T]]=None) -> Cursor[T]:
        '''Execute a raw SQL query.'''
        with self.cursor(factory) as cur:
            return cur.execute(query, values)
    
    # Need to specify overloads for a bound type vs an unbound type because otherwise Pyright
    #  will try to downcast subtypes of Row to NoId|HasId which loses the type information
    @overload
    def query[T: Row](self, query: str, factory: Optional[type[T]]=None) -> Query[T]: ...
    @overload
    def query[T](self, query: str, factory: Optional[Callable[[Cursor, sqlite3.Row], T]]=None) -> Query[T]: ...
    
    def query[T](self, query: str, factory: Optional[type[T]|Callable[[Cursor, sqlite3.Row], T]]=None):
        return Query(self, query, factory)
    
    def select[T: Row](self, table: type[T]):
        return self.query(f"SELECT {table._all_columns} FROM {table.__tablename__}", table)
    def count(self, table: type[Row]):
        return self.query(f"SELECT COUNT(*) FROM {table.__tablename__}", lambda cur, row: int(row[0]))
    def insert[T: Row](self, table: type[T]):
        return self.query(f"INSERT INTO {table.__tablename__}", table)
    def update[T: Row](self, table: type[T]):
        return self.query(f"UPDATE {table.__tablename__}", table)
    def delete[T: Row](self, table: type[T]):
        return self.query(f"DELETE FROM {table.__tablename__}", table)
    
    def add[T: Row](self, row: T) -> T:
        cur = self.insert(type(row)).values(*row).execute()
        if isinstance(row, HasId):
            row._bind(self, cast(int, cur.lastrowid))
        return row
        
    def add_all(self, *rows: Row) -> tuple[Row, ...]:
        with self.cursor() as cur:
            for row in rows:
                self.insert(type(row)).values(*row).execute(cur)
                if isinstance(row, HasId):
                    row._bind(self, cast(int, cur.lastrowid))

        return rows