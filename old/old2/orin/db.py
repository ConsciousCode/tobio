from contextlib import contextmanager
import json
import time
import sqlite3
import dill

import dataclasses
from dataclasses import dataclass

from .util import read_file
from .typing import json_value, Iterable, Iterator, Optional

VERSION = 0
'''Version counter for database consistency.'''

SCHEMA = read_file("schema.sql")

@dataclass
class Row:
    def __iter__(self):
        return iter(dataclasses.astuple(self))

@dataclass
class AgentRow(Row):
    type primary_key = int
    rowid: primary_key
    
    type: str
    created_at: int
    deleted_at: Optional[int]
    name: str
    config: str

@dataclass
class MessageRow(Row):
    type primary_key = int
    rowid: primary_key
    
    agent_id: AgentRow.primary_key
    content: str
    created_at: int

@dataclass
class PushRow(Row):
    agent_id: AgentRow.primary_key
    message_id: MessageRow.primary_key

@dataclass
class StateRow(Row):
    type primary_key = int
    rowid: primary_key
    
    created_at: int
    agent_id: AgentRow.primary_key
    data: bytes

class Database:
    '''Holds logic for database persistence.'''
    
    sql: sqlite3.Connection
    '''Connection to the database.'''
    
    def __init__(self, sql: sqlite3.Connection):
        self.sql = sql
        self.sql.row_factory = sqlite3.Row
        self.sql.executescript(SCHEMA)
    
    @contextmanager
    @staticmethod
    def connect(path: str):
        with sqlite3.connect(path) as sql:
            yield Database(sql)
    
    @contextmanager
    def transaction(self):
        '''Wrap a logical transaction to commit any pending transactions.'''
        
        try:
            cursor = self.sql.cursor()
            yield cursor
            self.sql.commit()
        except:
            self.sql.rollback()
            raise
        finally:
            cursor.close()
    
    def cast_execute(self, schema: type, query: str, values: tuple=()):
        cursor = self.sql.cursor()
        cursor.row_factory = lambda conn, row: schema(*row)
        return cursor.execute(query, values)
    
    def create_agent(self, type: str, name: str, config: json_value) -> AgentRow.primary_key:
        with self.transaction() as cursor:
            return cursor.execute('''
                INSERT INTO agent
                    (created_at, type, name, config) VALUES (?, ?, ?, ?)
                ''', (
                    int(time.time()), type, name, json.dumps(config)
                )
            ).lastrowid or 0
    
    def destroy_agent(self, agent: AgentRow.primary_key):
        with self.transaction() as cursor:
            cursor.execute('''
                UPDATE agent SET deleted_at=? WHERE rowid=?
            ''', (int(time.time()), agent,))
    
    def set_config(self, agent: AgentRow.primary_key, config: json_value):
        with self.transaction() as cursor:
            cursor.execute('''
                UPDATE agent SET config=? WHERE rowid=?
            ''', (json.dumps(config), agent))
    
    def push(self,
        agent: AgentRow.primary_key,
        message: MessageRow.primary_key
    ):
        with self.transaction() as cursor:
            cursor.execute('''
                INSERT INTO push (agent_id, message_id) VALUES (?, ?)
            ''', (agent, message))
    
    def push_many(self, rows: Iterable[tuple[int, int]]):
        with self.transaction() as cursor:
            cursor.executemany('''
                INSERT OR IGNORE INTO push (agent_id, message_id) VALUES (?, ?)
            ''', rows)
    
    def add_state(self, agent_id: AgentRow.primary_key, env: object):
        res = self.sql.execute('''
            SELECT data FROM state WHERE agent_id=? ORDER BY rowid DESC LIMIT 1
        ''', (agent_id,)).fetchall()
        
        # Don't persist state if it hasn't changed
        pickle = dill.dumps(env)
        if res and res[0]['data'] == pickle:
            return
        
        with self.transaction() as cursor:
            cursor.execute('''
                INSERT INTO state (created_at, agent_id, data) VALUES (?, ?, ?)
            ''', (int(time.time()), agent_id, pickle,))
    
    def message(self,
        agent: AgentRow.primary_key,
        content: str,
        created_at: int
    ) -> MessageRow:
        with self.transaction() as cursor:
            msg_id = cursor.execute('''
                INSERT INTO message
                    (agent_id, content, created_at) VALUES (?, ?, ?)
            ''', (agent, content, created_at)).lastrowid or 0
            
            return MessageRow(msg_id, agent, content, created_at)
    
    def load_agents(self) -> Iterator[AgentRow]:
        return self.cast_execute(AgentRow,
            "SELECT rowid, * FROM agent WHERE deleted_at IS NULL"
        )
    
    def load_state(self, agent_id: AgentRow.primary_key) -> Optional[object]:
        res = self.sql.execute(
            "SELECT * FROM state WHERE agent_id=? ORDER BY rowid DESC LIMIT 1",
            (agent_id,)
        ).fetchall()
        if res:
            if data := res[0]['data']:
                return dill.loads(data)
        return None