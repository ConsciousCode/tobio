'''
Custom ORM implementation for sqlite3 based on SQLAlchemy. I tried to use it,
but had these complaints:
- Too arcane to use effectively, lots of "magical" nonlocal actions.
- Created before PEP484, so it interacts poorly with type checkers.
- "Explicit is better than implicit" and "In the face of ambiguity, refuse the
   temptation to guess", SQLAlchemy will automatically and implicitly link
   relationships by attributes marked as foreign keys. This doesn't work in
   the general case because a table might have multiple foreign keys to the
   same table, meaning there's 2 ways to do the same thing and one might guess
   incorrectly.
- "There should be one-- and preferably only one --obvious way to do it",
   SQLAlchemy has many ways to do the same thing, probably because of
   accumulated technical debt and backwards compatibility.

Foreign keys are defined by the presence of a field of type 'HasId.primary_key',
using the name of the class the primary_key was retrieved from as a ForwardRef
to link to the correct table. primary_key is a type alias for int representing
the rowid of the table.

Ordinarily I might consider writing a schema.sql file and loading it to be run
on startup, but this presents problems with consistency between the schema and
the code. This way, the schema is generated from the code so it's always up to
date.
'''

from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from functools import cached_property
import sqlite3
import sys
import time
from typing import Any, ClassVar, Concatenate, ForwardRef, Literal, Optional, Self, Sequence, Union, dataclass_transform, get_args, get_origin, get_type_hints, overload, override
import inspect
from typing_extensions import deprecated

from sqlalchemy import over

from .util import NOT_GIVEN, indent, resolve_forward_ref, typename, typecheck

__all__ = [
    'timestamp',
    'attr_name',
    'now',
    'Constraint',
    'Unique',
    'PrimaryKey',
    'ForeignKey',
    'Descriptor',
    'Column',
    'Relationship',
    'relationship',
    'Table',
    'Row',
    'NoId',
    'HasId',
    'AnyId',
    'Database'
]

def as_primary_key(note) -> Optional[str]:
    '''Get the primary key table for a foreign key, or None if it's not a foreign key.'''
    PK = ".primary_key"
    if isinstance(note, str) and note.endswith(PK):
        return note[:-len(PK)]

type timestamp = int
'''Type for unix epoch timestamps in seconds.'''

type attr_name = str
'''For clarity, an alias explicitly denoting an attribute name.'''

def now() -> timestamp:
    '''Get the current timestamp.'''
    return int(time.time())

class Constraint:
    '''Base class for constraints.'''
    
    @abstractmethod
    def schema(self) -> str: '''Get the SQL representation of the constraint.'''
    def __str__(self): return self.schema()

class Unique(Constraint):
    '''Unique constraint.'''
    
    columns: tuple[str, ...]
    
    def __init__(self, *columns: str):
        super().__init__()
        self.columns = columns
    
    def schema(self): return f"UNIQUE({', '.join(self.columns)})"
    def __repr__(self): return f"Unique({', '.join(self.columns)})"

class PrimaryKey(Constraint):
    '''Primary key constraint.'''
    
    columns: tuple[str, ...]
    
    def __init__(self, *columns: str):
        super().__init__()
        self.columns = columns
    
    def schema(self): return f"PRIMARY KEY({', '.join(self.columns)})"
    def __repr__(self): return f"PrimaryKey({', '.join(self.columns)})"

class ForeignKey(Constraint):
    '''Foreign key constraint.'''
    
    columns: tuple[str, ...]
    foreign_table: str
    references: list[str]
    on_delete: str
    on_update: str
    
    def __init__(self,
        *columns: str,
        references: str|list[str],
        on_delete: str="",
        on_update: str=""
    ):
        super().__init__()
        
        self.columns = columns
        self.references = [references] if isinstance(references, str) else references
        self.on_delete = on_delete
        self.on_update = on_update
    
    def schema(self):
        return (
            f"FOREIGN KEY({', '.join(self.columns)}) REFERENCES {self.foreign_table}({', '.join(self.references)})" +
                (self.on_delete and f" ON DELETE {self.on_delete}") +
                (self.on_update and f" ON UPDATE {self.on_update}")
        )
    
    def __repr__(self):
        return f"ForeignKey({', '.join([
            *self.columns,
            f"references={', '.join(f"{self.foreign_table}.{r}" for r in self.references)}",
            f"on_delete={self.on_delete}",
            f"on_update={self.on_update}"
        ])})"

class Descriptor:
    '''Base class for descriptors.'''
    __name__: str
    '''Name of the descriptor in the class.'''
    __owner__: 'Table'
    '''Class the descriptor belongs to.'''
    annotation: Any
    '''Type annotation given to the descriptor.'''
    
    def __set_name__(self, owner: 'Table', name: str):
        self.__name__ = name
        self.__owner__ = owner
    
    def schema(self):
        '''Get the SQL representation of the descriptor.'''
        return self.__name__
    
    def __str__(self): return self.schema()

class Column(Descriptor):
    '''Base class for columns.'''
    
    default: Any
    '''Default value for the column.'''
    
    def __init__(self, annotation: Any, default: Any=NOT_GIVEN):
        super().__init__()
        
        self.annotation = annotation
        if default is not NOT_GIVEN:
            self.default = default
    
    def has_default(self):
        '''Check if the column has a default value.'''
        return hasattr(self, 'default')
    
    @cached_property
    def clause(self) -> str:
        '''Convert the annotationa to a SQL column type.'''
        
        types = {
            int: "INTEGER",
            str: "TEXT",
            bytes: "BLOB",
            float: "REAL",
            bool: "INTEGER",
            timestamp: "INTEGER",
            HasId.primary_key: "INTEGER"
        }
        
        note = self.annotation
        orig, args = get_origin(note), get_args(note)
        if orig is Optional:
            return types[args[0]]
        
        if orig is Union:
            if len(args) == 2 and args[1] is type(None):
                return types[args[0]]
            raise TypeError(f"Unsupported type {note}")
        
        if orig is Literal:
            # Assumes all literals are the same type
            note = type(args[0])
        
        if schema := types.get(note):
            return f"{schema} NOT NULL"
        
        raise TypeError(f"Unsupported type {note}")
    
    def schema(self):
        schema = f"{self.__name__} {self.clause}"
        if not self.has_default():
            return schema
        
        match self.default:
            case None: return f"{schema} DEFAULT NULL"
            case int(d)|float(d): return f"{schema} DEFAULT {d}"
            case str(d)|bytes(d): return f"{schema} DEFAULT {d!r}"
            case bool(d): return f"{schema} DEFAULT {int(d)}"
            
            # NOTE: Future feature maybe, support for HasId default
            case d:
                raise TypeError(f"Unsupported default type {typename(d)}")
    
    def __repr__(self):
        if self.has_default():
            return f"Column({self.annotation}, {self.default!r})"
        return f"Column({self.annotation})"
    
    def __expr__(self): return self.__name__, ()

# Need a helper function which explicitly returns Any so it can be used in
#  type annotations.
def column(annotation: Any, default: Any=NOT_GIVEN) -> Any:
    '''Descriptor for a column.'''
    return Column(annotation, default)

class Relationship(Descriptor):
    '''More explicit SQLAlchemy-style relationship descriptor.'''
    
    annotation: Any
    '''Type annotation for the relationship.'''
    
    foreign_key: attr_name
    '''Name of the attribute which holds the reference to a foreign table.'''
    
    local_key: attr_name
    '''Name of the attribute which holds the reference to this row.'''
    
    secondary: Union[str, ForwardRef, 'Table']
    '''Secondary table for many-to-many relationships.'''
    
    order_by: str
    '''Ordering clause.'''
    
    def __init__(self,
        foreign_key: attr_name,
        local_key: attr_name="",
        secondary: Union[str, ForwardRef, 'Table']="",
        order_by: str=""
    ):
        super().__init__()
        if (local_key is None) != (secondary is None):
            raise TypeError("self_ref and secondary must be specified together.")
        
        self.foreign_key = foreign_key
        self.local_key = local_key
        self.secondary = secondary
        self.order_by = order_by
    
    def __get__(self, instance: 'HasId', owner: 'Table'):
        if instance is None:
            return ForwardRef(self.__name__)
        
        def attr(): return f"{typename(instance)}.{self.__name__}"
        
        if instance.db is None:
            raise RuntimeError(f"Cannot access relationship {attr()} on unbound row.")
        
        hint = get_type_hints(instance)[self.__name__]
        origin, args = get_origin(hint), get_args(hint)
        
        if self.secondary is None:
            cur = (instance.db.select(hint)
                .where(f"{self.foreign_key} = ?", values=(instance.rowid,))
                .order_by(self.order_by)
            ).execute()
        else:
            if origin is None:
                raise TypeError(f"Relationship {attr()} is not a valid type (Got {typename(hint)})")
            
            # Make sure we have a secondary table reference and not a ForwardRef
            secondary = resolve_forward_ref(
                self.secondary,
                globalns=vars(sys.modules[owner.__module__]),
                localns=dict(vars(owner))
            )
            cur = (instance.db.select(args[0])
                .where(
                    f"rowid IN (SELECT {self.foreign_key} FROM {secondary.__name__} WHERE {self.local_key} = ?)",
                    values=(instance.rowid,)
                )
                .order_by(self.order_by)
            ).execute()
        
        if origin is None:
            return hint._bind_new(instance.db, cur.fetch(1))
        
        if origin in {list, set, tuple}:
            return origin(hint._bind_new(instance.db, row) for row in cur.fetch())
        
        raise TypeError(f"Relationship {attr()} is not a valid type (Got {typename(hint)})")
    
    def _validate(self, owner: type['Row'], name: str, note: Any):
        '''
        Validate the annotation for a relationship. Must be called after
        forward references have been resolved.
        '''
        
        def attr(): return f"{typename(owner)}.{name}"
        
        origin = get_origin(note)
        if origin in {list, set, tuple}:
            args = get_args(note)
            if origin in {list, set}:
                if len(args) != 1:
                    raise TypeError(f"{typename(origin)} relationship {attr()} must have only one type.")
            else: # tuple
                if len(args) != 2 or args[1] is not ...:
                    raise TypeError(f"tuple relationship {attr()} must be tuple[T: HasId, ...].")
            
            note = args[0]
        
        if not isinstance(note, type):
            raise TypeError(f"Relationship {attr()} annotation must be a type (Got {note!r}).")
        
        if not issubclass(note, HasId):
            raise TypeError(f"Relationship {attr()} must refer to a subclass of HasId.")
    
    @override
    def schema(self):
        raise TypeError("Relationships cannot be used in table schemas.")
    
    @override
    def __str__(self):
        return f"Relationship({', '.join(filter(None, [
            self.foreign_key,
            self.local_key,
            self.secondary and f"secondary={self.secondary}",
            self.order_by and f"order_by={self.order_by}"
        ]))})"

# https://github.com/microsoft/pyright/issues/5102#issuecomment-1545152637
# "Type checkers assume that the functional form of a field specifier will
#  return the type of the field or Any, not a Field object". Even though the
#  documentation of dataclasses.field says it returns a Field object, and
#  the documentation of dataclasses.dataclass_transform says field_specifiers
#  supports classes, which can't be typed as returning Any.
def relationship(
    foreign_key: attr_name, local_key: attr_name="",
    secondary: Union[str, ForwardRef, 'Table']="", order_by: str=""
) -> Any:
    '''More explicit SQLAlchemy-style relationship descriptor.'''
    return Relationship(foreign_key, local_key, secondary, order_by)

class Table(type):
    '''Metaclass for tables.'''
    __tablename__: str
    '''Name of the table.'''
    __table_args__: Sequence[Constraint]
    '''Table arguments (eg constraints).'''
    __table_fields__: dict[str, Descriptor]
    '''Table fields.'''
    
    def __init__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]):
        #self = super().__new__(cls, name, bases, attrs)
        
        # Don't process abstract tables further
        if attrs.get("__abstract__", False):
            return
        
        if not hasattr(cls, '__tablename__'):
            raise TypeError(f"Table model {name} missing __tablename__")
        
        fields: dict[str, Descriptor] = {}
        table_args: list[Constraint] = list(getattr(cls, "__table_args__", []))
        
        # Set of foreign key attributes seen so far
        foreign: set[attr_name] = set(
            col for fk in table_args if isinstance(fk, ForeignKey) for col in fk.columns
        )
        
        for name, note in inspect.get_annotations(cls).items():
            # For reference:
            #   name: note = attr
            #   agent_id: int = some_default
            #   agent: Agent = relationship('agent_id')
            if ref := as_primary_key(note):
                # If an explicit ForeignKey is present, don't add an implicit one
                if ref not in foreign:
                    foreign.add(ref)
                    table_args.append(ForeignKey(name, references=ref))
            
            if name in attrs:
                attr = attrs[name]
                if isinstance(attr, Descriptor):
                    attr.annotation = note
                else:
                    attr = Column(note, attr)
                    attr.__set_name__(cls, name)
            else:
                attr = Column(note)
                attr.__set_name__(cls, name)
            
            fields[name] = attr
        
        cls.__table_fields__ = fields
        cls.__table_args__ = tuple(table_args)
    
    def fields(cls):
        '''Get all fields for the table.'''
        return cls.__table_fields__
    
    def columns(cls):
        '''Get the columns for the table.'''
        cols: dict[str, Column] = {}
        for name, note in cls.fields().items():
            if isinstance(note, Column):
                cols[name] = note
        return cols
    
    def relationships(cls):
        '''Get the relationships for the table.'''
        cols: dict[str, Relationship] = {}
        for name, note in cls.fields().items():
            if isinstance(note, Relationship):
                cols[name] = note
        return cols
    
    def defaults(cls):
        '''Get the default values for the table.'''
        defs: dict[str, Any] = {}
        for name, col in cls.columns().items():
            if hasattr(col, 'default'):
                defs[name] = col.default
        return defs
    
    def constraints(cls):
        '''Get the constraints for the table.'''
        return cls.__table_args__
    
    def foreign_keys(cls):
        '''Get the foreign keys for the table.'''
        fks: list[ForeignKey] = []
        for fk in cls.constraints():
            if isinstance(fk, ForeignKey):
                fks.append(fk)
        return fks
    
    def unique(cls):
        '''Get the unique constraints for the table.'''
        uniques: list[Unique] = []
        for unique in cls.constraints():
            if isinstance(unique, Unique):
                uniques.append(unique)
        return uniques
    
    def primary(cls):
        '''Get the primary key constraint for the table.'''
        for pk in cls.constraints():
            if isinstance(pk, PrimaryKey):
                return pk
        return PrimaryKey("rowid")
    
    def schema(cls) -> str:
        '''
        Get the schema for the table. This is only valid for tables which are
        not abstract, and only after all type hint forward references have been
        resolved.
        '''
        
        if cls.is_abstract():
            raise TypeError(f"Cannot get schema for abstract table {cls.__name__}")
        
        notes = inspect.get_annotations(cls)
        hints = get_type_hints(cls)
        for name in notes.keys():
            if field := cls.fields().get(name):
                # NOTE: Maybe add a hook for assigning the annotation to the field?
                hint = hints[name]
                
                field.annotation = hint
                if isinstance(field, Relationship):
                    field._validate(cls, name, hint)
                
                # Resolve the implicit foreign keys to their equivalent
                #  forward references. eg Message.primary_key -> messages.rowid
                note = notes[name]
                if ref := as_primary_key(note):
                    for fk in cls.foreign_keys():
                        try:
                            # If the column name matches, replace the reference
                            # eg FOREIGN KEY (agent_id) REFERENCES agents
                            idx = fk.columns.index(name)
                        except ValueError:
                            continue
                        
                        table: Table = resolve_forward_ref(
                            ref,
                            globalns=vars(sys.modules[cls.__module__]),
                            localns=dict(vars(cls))
                        )
                        
                        fk.foreign_table = table.__tablename__
                        fk.references[idx] = pk = table.primary().columns[idx]
        
        cols_clauses = "{}"
        if cons := cls.constraints():
            cols_clauses += ",\n\n{}"
        
        return f"CREATE TABLE IF NOT EXISTS {cls.__tablename__} (\n{cols_clauses}\n)".format(
            indent('\n'.join([
                "/* rowid */",
                ',\n'.join(col.schema() for col in cls.columns().values()),
            ])),
            indent(',\n'.join(con.schema() for con in cons))
        )
    
    def is_abstract(cls) -> bool:
        '''
        Check if a table is abstract. That is, if it has a non-inherited
        __abstract__ property set to something truthy.
        '''
        return bool(getattr(cls.__dict__, '__abstract__', False))

class Row(metaclass=Table):
    '''Base class for database rows.'''
    __abstract__ = True
    
    db: Optional['Database']
    '''Database the row belongs to.'''
    
    def __init__(self, *args, **kwargs):
        '''Initialize the unbound row excluding any rowid.'''
        
        fields = type(self).defaults().copy()
        df_count = len(fields)
        columns = type(self).columns()
        table_fields = type(self).fields()
        
        if len(args) > len(columns):
            raise TypeError(f"{typename(self)} expected {len(columns)} arguments, got {len(args)}")
        
        # Positional arguments
        for name, arg in zip(columns, args):
            note = table_fields[name].annotation
            if as_primary_key(note):
                note = int
            
            # Special case for Literal for a better error message
            if get_origin(note) is Literal:
                if arg not in get_args(note):
                    raise TypeError(f"{typename(self)}.{name} must be Literal[{', '.join(map(repr, get_args(note)))}], got {arg!r}")
            
            if not typecheck(arg, note):
                raise TypeError(f"{typename(self)}.{name} must be {typename(note)}, got {typename(arg)}")
            fields[name] = arg
        
        # Named arguments
        for name, arg in kwargs.items():
            if name not in columns:
                raise TypeError(f"{typename(self)} got an unexpected keyword argument '{name}'")
            
            note = table_fields[name].annotation
            if as_primary_key(note):
                note = int
            
            # Special case for Literal for a better error message
            if get_origin(note) is Literal:
                if arg not in get_args(note):
                    raise TypeError(f"{typename(self)}.{name} must be Literal[{', '.join(map(repr, get_args(note)))}], got {arg!r}")
            
            if not typecheck(arg, note):
                raise TypeError(f"{typename(self)}.{name} must be {typename(note)}, got {typename(arg)}")
            fields[name] = arg
        
        if len(fields) != len(columns):
            raise TypeError(f"{typename(self)} expected {len(columns) - df_count} arguments, got {len(fields)}")
        
        self.__dict__.update(fields)
    
    @classmethod
    def _bind_new(cls, db: 'Database', row: sqlite3.Row):
        '''Create an ORM row from a database row.'''
        
        self = cls(**{
            name: row[name] for name in row.keys()
        })
        self.db = db
        return self
    
    def is_bound(self):
        '''Check if the row is bound to a database.'''
        return self.db is not None
    
    def __len__(self):
        return len(self.__table_fields__)
    
    def __iter__(self):
        for name in self.__table_fields__:
            yield getattr(self, name)
    
    def __repr__(self):
        return f"{typename(self)}({
            ', '.join(f'{name}={getattr(self, name)!r}' for name in self.__table_fields__)
        })"

# Note: dataclass_transform is duplicated here so db and rowinfo are not
#  considered part of the __init__ parameters. They must be set by ._bind_new.

@dataclass_transform(
    eq_default=False,
    order_default=False,
    kw_only_default=False,
    field_specifiers=(column,)
)
class NoId(Row):
    '''Base class for objects without an ID.'''
    __abstract__ = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'rowid' in kwargs:
            raise TypeError(f"{typename(self)} got an unexpected keyword argument 'rowid'")
    
    @classmethod
    def schema(cls: type[Row]) -> str:
        schema  = Table.schema(cls)
        schema = schema.replace('    /* rowid */\n', '')
        return f"{schema} WITHOUT ROWID"

@dataclass_transform(
    eq_default=False,
    order_default=False,
    kw_only_default=False,
    field_specifiers=(column,)
)
class HasId(Row):
    '''Base class for objects with an ID.'''
    __abstract__ = True
    
    type primary_key = int
    '''
    Type alias for the primary key allowing relationships to be linked using
    valid type hints.
    '''
    rowid: primary_key = column(primary_key) # type: ignore
    '''Primary key for the row.'''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'rowid' in kwargs:
            raise TypeError(f"{typename(self)} got an unexpected keyword argument 'rowid'")
    
    @classmethod
    def _bind_new(cls, db: 'Database', row: sqlite3.Row):
        self = super()._bind_new(db, row)
        self.rowid = row['rowid']
        return self

type AnyId = NoId|HasId
'''Type for objects with or without an ID.'''

class Cursor[T: Row]:
    '''Wrapper for sqlite3 Cursor to provide a more consistent interface.'''
    
    cursor: sqlite3.Cursor
    '''Wrapped cursor.'''
    
    def __init__(self, db: 'Database', table: type[T], cursor: sqlite3.Cursor):
        cursor.row_factory = lambda conn, row: table._bind_new(db, row)
        self.cursor = cursor
    
    def __iter__(self): return iter(self.cursor)
    
    def fetch(self, size: Optional[int]=None):
        '''Fetch a number of rows.'''
        match size:
            case None: return self.cursor.fetchall()
            case size: return self.cursor.fetchmany(size)
    
    def fetchone(self) -> T:
        '''Fetch a single row.'''
        return self.cursor.fetchone()
    
    @property
    def rowcount(self):
        '''Number of rows returned by the last query.'''
        return self.cursor.rowcount
    
    @property
    def lastrowid(self):
        '''Last row ID inserted by the last query.'''
        return self.cursor.lastrowid
    
    @property
    def description(self):
        '''Description of the columns returned by the last query.'''
        return self.cursor.description

class Query[T: Row]:
    '''
    Builds a statement within the context of a database for execution using
    a fluent interface.
    '''
    
    db: 'Database'
    '''Database the query belongs to.'''
    table: type[T]
    '''Table the query is for.'''
    clauses: list[str]
    '''Clauses in the statement being built.'''
    binds: list[Any]
    '''Values to be bound to the statement.'''
    
    def __init__(self, db: 'Database', table: type[T], clause: str=""):
        self.db = db
        self.table = table
        self.clauses = [clause] if clause else []
        self.binds = []
    
    def __str__(self): return ' '.join(self.clauses)
    def __repr__(self): return f"Query({str(self)!r})"
    
    def copy(self):
        '''Get a copy of the query.'''
        copy = Query(self.db, self.table)
        copy.clauses = self.clauses.copy()
        copy.binds = self.binds.copy()
        return copy
    
    def raw(self, clause: str, values: tuple=()) -> Self:
        '''Add a raw clause to the statement.'''
        self.clauses.append(clause)
        self.binds.extend(values)
        return self
    
    def values(self, *values: Any) -> Self:
        '''Add a VALUES clause to the statement.'''
        return self.raw(f"VALUES ({','.join('?'*len(values))})", values)
    
    def where(self, first: str, *rest: str, values: tuple=()) -> Self:
        '''Add a WHERE clause to the statement.'''
        return self.raw(f"WHERE {' AND '.join([first, *rest])}", values=values)
    
    def set(self, **values: Any) -> Self:
        '''Add a SET clause to the statement.'''
        self.binds.extend(values.values())
        return self.raw(f"SET {', '.join(f'{k} = ?' for k in values)}")
    
    def order_by(self, *columns: str) -> Self:
        '''Add an ORDER BY clause to the statement.'''
        return self.raw(f"ORDER BY {', '.join(columns)}")
    
    def limit(self, limit: int) -> Self:
        '''Add a LIMIT clause to the statement.'''
        return self.raw(f"LIMIT {limit}")
    
    def offset(self, offset: int) -> Self:
        '''Add an OFFSET clause to the statement.'''
        return self.raw(f"OFFSET {offset}")
    
    def execute(self, session: Optional[Any]=None) -> Cursor:
        '''Execute the statement.'''
        with (session or nullcontext()) as sess:
            return Cursor(self.db, self.table, sess.execute(str(self), tuple(self.binds)))
    
    def fetch(self, size: Optional[int]=None):
        '''Utility to execute and then fetch a number of rows.'''
        return self.execute().fetch(size)
    
    def fetchone(self) -> Row:
        '''Utility to execute and then fetch a single row.'''
        return self.execute().fetchone()

class Database:
    '''Base class for databases, provides some nonspecific utility methods.'''
    
    tables: ClassVar[Sequence[Table]]
    '''Recognized tables in the database.'''
    path: str
    '''Path to the database file.'''
    sql: sqlite3.Connection
    '''Connection to the database.'''
    
    def __init__(self, path: str):
        self.path = path
        self.sql = sqlite3.connect(path)
        self.sql.row_factory = sqlite3.Row
        self.sql.executescript(self.schema())
    
    def __repr__(self):
        return f"{typename(self)}({self.path!r}, [{', '.join(t.__name__ for t in self.tables)}])"
    
    @classmethod
    def schema(cls) -> str:
        '''Get the schema for the database.'''
        doc = inspect.getdoc(cls) or ''
        return (
            doc and f"/**\n * {'\n * '.join(doc.splitlines())}\n**/\n\n"
            "PRAGMA foreign_keys = ON;\n\n" +
            ';\n\n'.join(table.schema() for table in cls.tables)
        )
    
    ## Raw access methods ##
    
    @overload
    def execute[T: Row](self, query: Query[T], /) -> Cursor[T]: ...
    @overload
    def execute(self, q1: Query, q2: Query, /, *queries: Query) -> tuple[Cursor, ...]: ...
    
    def execute(self, first: Query, *rest: Query) -> Cursor|tuple[Cursor, ...]:
        '''Execute multiple queries in a single transaction.'''
        with self.session() as session:
            result1 = first.execute(session)
            return (result1, *(query.execute(session) for query in rest)) if rest else result1
    
    ## Utilities ##
    
    @contextmanager
    def session(self):
        '''Context to commit any pending transactions.'''
        
        cursor = self.sql.cursor()
        try:
            yield cursor
            self.sql.commit()
        except:
            self.sql.rollback()
            raise
        finally:
            cursor.close()
    
    @deprecated("Use ORM methods instead.")
    def cast_execute(self, schema: type, query: str, values: tuple=()):
        '''Execute a SQL query and cast the results to a schema.'''
        cursor = self.sql.cursor()
        cursor.row_factory = lambda conn, row: schema(*row)
        return cursor.execute(query, values)
    
    ## Logical access methods ##
    
    def select[T: Row](self, table: type[T]):
        '''Start a SELECT query.'''
        cols = "*" if issubclass(table, NoId) else "rowid, *"
        return Query(self,table, f"SELECT {cols} FROM {table.__tablename__}")
    
    def insert[T: Row](self, table: type[T]):
        '''Start an INSERT query.'''
        return Query(self, table, f"INSERT INTO {table.__tablename__}")
    
    def update[T: Row](self, table: type[T]):
        '''Start an UPDATE query.'''
        return Query(self, table, f"UPDATE {table.__tablename__}")
    
    def delete[T: Row](self, table: type[T]):
        '''Delete rows from a table.'''
        return Query(self, table, f"DELETE FROM {table.__tablename__}")
    
    ## high-level access methods ##
    
    @overload
    def add[T: Row](self, row: T, /) -> Cursor[T]: ...
    @overload
    def add(self, r1: Row, r2: Row, /, *rows: Row) -> tuple[Row, ...]: ...
    
    def add(self, *rows: Row): #type: ignore
        '''Add one or more objects to the database.'''
        return self.execute(*(self.insert(type(row)).values(*row) for row in rows))