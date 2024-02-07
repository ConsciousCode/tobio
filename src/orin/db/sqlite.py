'''
Sqlite implementation of the database.
'''

import sqlite3
from typing import Optional, cast
from urllib.parse import urlparse
import time
import os

from prettytable import PrettyTable

from ..util import logger
from .base import Author, Step, Row, Unbound

class Database:
    _config: dict
    _db: sqlite3.Connection
    
    _authors: dict[int|tuple[Author.Role, Optional[str]], Author]
    '''Local author cache.'''
    
    def __init__(self, config):
        self._config = config
        self._authors = {}
        
        u = urlparse(config['database'])
        if u.scheme not in {'', 'sqlite'}:
            raise ValueError('Only sqlite databases are supported')
        logger.info('Connecting to database %s', u.path)
        
        self._db = sqlite3.connect(
            os.path.join(
                os.path.dirname(__file__),
                "schema.sql"
            )
        )
        self._db.row_factory = sqlite3.Row
        
        with open(config['schema']) as f:
            self._db.executescript(f.read())
        
        self._db.commit()
    
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
    
    def add_author(self, author: Author.Unbound) -> Author:
        '''Add author to the database.'''
        
        cur = self._db.execute(
            'INSERT INTO authors (role, name) VALUES (?, ?)',
            (author.role, author.name)
        )
        self._db.commit()
        return Author(
            id=cur.lastrowid, # type: ignore
            role=author.role,
            name=author.name
        )
    
    def get_author(self, id: int) -> Author:
        '''Get author by ID.'''
        
        if author := self._authors.get(id):
            return author
        
        if row := self._db.execute(
            'SELECT * FROM authors WHERE rowid = ?', (id,)
        ).fetchone():
            author = Author(**row)
            self._authors[id] = author
            return author
        
        raise KeyError(id)
    
    def put_author(self, role: Author.Role, name: Optional[str]) -> Author:
        '''Get or create author by role and name.'''
        
        if author := self._authors.get((role, name)):
            return author
        
        cur = self._db.execute(
            'SELECT * FROM authors WHERE role = ? AND name = ?',
            (role, name)
        )
        if row := cur.fetchone():
            rowid = row['id']
            if author := self.get_author(rowid):
                return author
            author = Author(**row)
            self._authors[rowid] = author
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
                status=row['status'],
                content=row['content']
            ) for row in rows
        ]
    
    def add_step(self, step: Step.Unbound) -> Step:
        '''Append step to history.'''
        
        if step.kind == "text":
            assert step.text is not None
            content: str = step.text
        else:
            assert step.data is not None
            content = step.data.model_dump_json()
        
        if step.status is None:
            raise ValueError('Step status must be set before insertion.')
        
        created_at = step.created_at or time.time()
        cur = self._db.execute('''
            INSERT INTO steps (author_id, created_at, updated_at, kind, content)
            VALUES (?, ?, ?, ?, ?)
        ''', (step.author.id, created_at, created_at, step.kind, content)
        )
        self._db.commit()
        return Step(
            id=cur.lastrowid, # type: ignore
            author=step.author,
            created_at=created_at,
            updated_at=created_at,
            kind=step.kind,
            status=step.status,
            content=content
        )
    
    def set_step_content(self, step: Step, content: str):
        '''Update step content.'''
        
        if step.status != "stream":
            raise ValueError('Only stream steps can be updated')
        
        now = time.time()
        self._db.execute('''
            UPDATE steps SET updated_at = ?, content = ? WHERE rowid = ?
        ''', (now, content, step.id))
        self._db.commit()
    
    def finalize_step(self, step: Step):
        '''Finalize a streaming step.'''
        
        if step.status != "stream":
            raise ValueError('Only stream steps can be finalized')
        
        now = time.time()
        self._db.execute('''
            UPDATE steps SET updated_at = ?, status = "done" WHERE rowid = ?
        ''', (now, step.id))
        self._db.commit()