'''
Database modeling and related types.
'''

import sqlite3
from typing import Literal, Optional, cast
from urllib.parse import urlparse
import time

from prettytable import PrettyTable
from pydantic import BaseModel

from .util import logger

type StepKind = Literal['text', 'tool', 'action']
type Role = Literal['user', 'assistant', 'system', 'tool']

class Author(BaseModel):
    id: int
    role: Role
    name: Optional[str]

class Step(BaseModel):
    id: int
    author: Author
    created_at: float
    updated_at: float
    kind: StepKind
    content: str

class Database:
    _config: dict
    _db: sqlite3.Connection
    
    _authors: dict[int|tuple[Role, Optional[str]], Author]
    '''Local author cache.'''
    
    def __init__(self, config):
        self._config = config
        
        u = urlparse(config['database'])
        if u.scheme not in {'', 'sqlite'}:
            raise ValueError('Only sqlite databases are supported')
        logger.info('Connecting to database %s', u.path)
        
        self._db = sqlite3.connect(u.path)
        self._db.row_factory = sqlite3.Row
        
        with open(config['schema']) as f:
            self._db.executescript(f.read())
        
        self._db.commit()
        
        self._authors = {}
    
    def raw(self, query: str, *args):
        '''Raw SQL query.'''
        return self._db.execute(query, args)
    
    def raw_format(self, query: str):
        '''Formatted raw SQL query.'''
        
        t = time.time()
        cur = self._db.execute(query)
        result = cur.fetchall()
        dt = time.time() - t
        
        if cur.rowcount == -1:
            if len(result) == 0:
                content = "empty set"
            else:
                table = PrettyTable(result[0].keys())
                table.add_rows(result)
                content = f"{table}\n\n{len(result)} rows in set"
        else:
            content = f"{cur.rowcount} affected"
        
        return f"{content} ({dt:.2f} sec)"
    
    def get_author(self, id: int) -> Author:
        '''Get author by ID.'''
        
        if id not in self._authors:
            if row := self._db.execute(
                'SELECT * FROM authors WHERE rowid = ?', (id,)
            ).fetchone():
                self._authors[id] = Author(**row)
            else:
                raise KeyError(id)
        return self._authors[id]
    
    def put_author(self, role: Role, name: Optional[str]) -> Author:
        '''Get or create author by role and name.'''
        
        if author := self._authors.get((role, name)):
            return author
        
        cur = self._db.execute(
            'SELECT * FROM authors WHERE role = ? AND name = ?',
            (role, name)
        )
        if row := cur.fetchone():
            author = Author(**row)
            self._authors[(role, name)] = author
            return author
        
        cur = self._db.execute(
            'INSERT INTO authors (role, name) VALUES (?, ?)',
            (role, name)
        )
        self._db.commit()
        author = Author(
            id=cur.lastrowid, # type: ignore
            role=role,
            name=name
        )
        self._authors[(role, name)] = author
        return author
    
    def get_history(self, limit: Optional[int]=None):
        '''Get history of steps.'''
        
        rows = self._db.execute(f'''
            SELECT * FROM steps ORDER BY created_at DESC LIMIT
            {limit or self._config['history_limit']}
        ''').fetchall()
        
        return [
            Step(
                id=row['id'],
                author=self.get_author(row['author_id']),
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                kind=row['kind'],
                content=row['content']
            ) for row in rows
        ]
    
    def add_step(self, author: Author, kind: StepKind, content: str):
        '''Append step to history.'''
        
        now = time.time()
        cur = self._db.execute('''
            INSERT INTO steps (author_id, created_at, updated_at, kind, content)
            VALUES (?, ?, ?, ?, ?)
        ''', (author.id, now, now, kind, content)
        )
        self._db.commit()
        step_id = cast(int, cur.lastrowid)
        return Step(
            id=step_id,
            author=author,
            created_at=now,
            updated_at=now,
            kind=kind,
            content=content
        )
    
    def stream_step(self, step: Step, content: str):
        '''Update step content.'''
        
        now = time.time()
        self._db.execute('''
            UPDATE steps SET updated_at = ?, content = content || ? WHERE rowid = ?
        ''', (now, content, step.id))
        self._db.commit()