import sqlite3
from typing import Literal, Optional
from urllib.parse import urlparse

from util import logger

class Row:
    id: int

class TextStep(Row):
    content: str

class ToolStep(Row):
    name: str
    args: dict

type Step = TextStep | ToolStep

class Message(Row):
    role: Literal['user', 'assistant', 'system', 'tool']
    name: str
    created_at: str
    steps: list[Step]

class Database:
    config: dict
    db: sqlite3.Connection
    
    def __init__(self, config):
        self.config = config
        
        u = urlparse(config['database'])
        if u.scheme not in {'', 'sqlite'}:
            raise ValueError('Only sqlite databases are supported')
        logger.info('Connecting to database %s', u.path)
        
        self.db = sqlite3.connect(u.path)
        self.db.row_factory = sqlite3.Row
        
        with open(config['schema']) as f:
            self.db.executescript(f.read())
        
        self.db.commit()
    
    def history(self, limit: Optional[int]=None):
        rows = self.db.execute(f'''
            SELECT
            messages.rowid AS message_id,
            steps.rowid AS step_id,
            messages.*, steps.* FROM (
                SELECT rowid FROM messages
                ORDER BY created_at DESC LIMIT
                {limit or self.config['history_limit']}
            ) AS lastmessages
            JOIN messages ON messages.rowid = lastmessages.rowid
            LEFT JOIN steps ON messages.rowid = steps.message_id
            ORDER BY messages.created_at DESC, steps.step
        ''').fetchall()
        
        messages = {}
        for row in rows:
            message_id = row['message_id']
            if message_id not in messages:
                messages[message_id] = Message(**row)
                messages[message_id].steps = []
            if row['step_id']:
                messages[message_id].steps.append(Step(**row))
        