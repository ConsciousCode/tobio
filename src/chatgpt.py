'''
Utility for consolidating ChatGPT exports into a single database.
'''

import re
from typing import Generator, NotRequired, Optional, TypedDict, cast
import uuid
import logging
import os
import sys
import json
import sqlite3

from pydantic import TypeAdapter

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
logger.addHandler(logging.StreamHandler())

# Type models to make sure the schema of exports hasn't changed.

class Author(TypedDict, total=True):
    role: str
    name: NotRequired[Optional[str]]
    metadata: NotRequired[Optional[dict]]

class Message(TypedDict, total=True):
    id: str
    author: Author
    create_time: NotRequired[Optional[float]]
    update_time: NotRequired[Optional[float]]
    content: dict
    status: str
    end_turn: NotRequired[Optional[bool]]
    weight: float
    metadata: Optional[dict]
    recipient: str

class MessageMapping(TypedDict, total=True):
    id: str
    message: NotRequired[Optional[Message]]
    parent: NotRequired[Optional[str]]
    children: NotRequired[list[str]]

class Conversation(TypedDict, total=True):
    id: str
    title: str
    create_time: NotRequired[Optional[float]]
    update_time: NotRequired[Optional[float]]
    current_node: NotRequired[Optional[str]]
    conversation_template_id: NotRequired[Optional[str]]
    gizmo_id: NotRequired[Optional[str]]
    moderation_results: NotRequired[list]
    mapping: dict[str, MessageMapping]

Conversation_typecheck = TypeAdapter(Conversation)

def optional_uuid(s: Optional[str]) -> Optional[str]:
    return None if s is None else uuid.UUID(s).hex

def breadth(M: dict[str, MessageMapping]) -> Generator[MessageMapping, None, None]:
    seen: set[str] = set()
    todo: list[str] = []

    # Find and process root nodes (nodes without parents)
    for k, v in M.items():
        if v.get('parent') is None:
            yield v
            seen.add(k)
            todo.extend(v.get("children", []))

    # Process remaining nodes
    while todo:
        current_key = todo.pop(0)
        if current_key in seen:
            continue

        current_node = M[current_key]
        # Ensure the parent has been processed
        if current_node.get('parent') in seen:
            yield current_node
            seen.add(current_key)
            todo.extend(current_node.get("children", []))
        else:
            # Requeue the current node to be processed after its parent
            todo.append(current_key)

class Database:
    '''Collect database operations into a single class with shared state.'''
    
    conn: sqlite3.Connection
    
    def __init__(self, db: str):
        self.conn = sqlite3.connect(db)
        self.conn.executescript('''
        PRAGMA foreign_keys = ON;
        
        CREATE TABLE IF NOT EXISTS authors (
            id INTEGER PRIMARY KEY,
            role TEXT NOT NULL,
            name TEXT,
            metadata TEXT,
            
            UNIQUE(role, name)
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            create_time REAL,
            update_time REAL,
            current_node TEXT,
            conversation_template_id TEXT,
            gizmo_id TEXT
        );
        
        CREATE TABLE IF NOT EXISTS mappings (
            id BLOG PRIMARY KEY,
            conversation TEXT,
            parent TEXT,
            message TEXT,
            
            FOREIGN KEY(conversation) REFERENCES conversations(id),
            FOREIGN KEY(parent) REFERENCES mappings(id)
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            author_id INTEGER NOT NULL,
            create_time REAL,
            update_time REAL,
            content TEXT NOT NULL,
            status TEXT NOT NULL,
            end_turn BOOLEAN,
            weight REAL,
            metadata TEXT,
            recipient TEXT,
            
            FOREIGN KEY(author_id) REFERENCES authors(id)
        );
        ''')
    
    def load_author(self, author: Author) -> int:
        name = author.get("name")
        cur = self.conn.execute(
            "SELECT id FROM authors WHERE role = ? AND (name = ? OR name is NULL);",
            (author["role"], name)
        ).fetchone()
        if cur:
            return cur[0]
        
        logger.info("Author %s:%s does not exist, creating", author['role'], name)
        md = author.get("metadata")
        cur = self.conn.execute(
            "INSERT INTO authors (role, name, metadata) VALUES (?, ?, ?)",
            (
                author["role"],
                name,
                json.dumps(md) if md else None
            )
        )
        return cast(int, cur.lastrowid)
    
    def insert_message(self, message: Message):
        '''Raw insert of a message, without validation.'''
        md = message.get("metadata")
        self.conn.execute("""
            INSERT INTO messages (id, author_id, create_time, update_time, content, status, end_turn, weight, metadata, recipient) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            uuid.UUID(message["id"]).hex,
            self.load_author(message["author"]),
            message.get("create_time"),
            message.get("update_time"),
            json.dumps(message["content"]),
            message["status"],
            message.get("end_turn"),
            message["weight"],
            json.dumps(md) if md else None,
            message["recipient"]
        ))
        
        return message['id']
    
    def load_message(self, message: Optional[Message]) -> Optional[str]:
        if message is None:
            return
        
        message_id = uuid.UUID(message['id']).hex
        cur = self.conn.execute("SELECT COUNT(*) FROM messages WHERE id = ?;", (message_id,))
        if cur.fetchone()[0] != 0:
            logger.info("Message %s already exists", message_id)
        else:
            self.insert_message(message)
        return message['id']
    
    def insert_mapping(self, conversation: str, mapping: MessageMapping):
        self.conn.execute(
            "INSERT INTO mappings (id, conversation, message, parent) VALUES (?, ?, ?, ?);",
            (
                uuid.UUID(mapping["id"]).hex,
                conversation,
                self.load_message(mapping.get("message")),
                optional_uuid(mapping.get("parent"))
            )
        )
    
    def load_mapping(self, conversation: str, mapping: MessageMapping) -> str:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM mappings WHERE id = ?;",
            (uuid.UUID(mapping["id"]).hex,)
        )
        if cur.fetchone()[0] != 0:
            logger.info("Mapping %s already exists", mapping["id"])
        else:
            self.insert_mapping(conversation, mapping)
        
        return mapping["id"]
    
    def insert_conversation(self, convo: Conversation):
        '''Raw insert of a conversation, without validation.'''
        self.conn.execute("""
            INSERT INTO conversations (id, title, create_time, update_time, current_node, conversation_template_id, gizmo_id) VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            uuid.UUID(convo['id']).hex,
            convo["title"],
            convo.get("create_time"),
            convo.get("update_time"),
            optional_uuid(convo.get("current_node")),
            convo.get("conversation_template_id"),
            convo.get("gizmo_id")
        ))
        
        return convo["id"]
    
    def load_conversation(self, convo: Conversation) -> str:
        # Validation
        
        # Want to be very picky so I don't accidentally discard any new
        #  fields, openai is sure to add more in the future
        Conversation_typecheck.validate_python(convo)
        
        if convo.get("moderation_results"):
            raise ValueError(f"Moderation results are not empty, revisit this code. {convo.get('moderation_results')}")
        
        # Check if conversation already exists
        if convo.get("id") != convo.get("conversation_id"):
            raise ValueError("conversations.id != conversations.conversation_id")
        
        convo_id = uuid.UUID(convo["id"]).hex
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE id = ?;",
            (convo_id,)
        )
        if cur.fetchone()[0] != 0:
            logger.info("Conversation %s already exists", convo['id'])
        else:
            self.insert_conversation(convo)
        
        for msg in breadth(convo['mapping']):
            self.load_mapping(convo_id, msg)
            
        self.conn.commit()
        return convo["id"]
    
    def insert_sparse_message(self, role, content):
        mid = uuid.uuid4().hex
        self.insert_message({
            "id": mid,
            "author": {"role": role},
            "content": {
                "content_type": "text",
                "parts": [content]
            },
            "status": "finished_successfully",
            "end_turn": True,
            "weight": 0.0,
            "metadata": {
                "_loaded_format": "markdown"
            },
            "recipient": "all"
        })
        
        return mid
    
    def load_markdown(self, fn: str, md: str):
        title = os.path.splitext(os.path.basename(fn))[0]
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE title = ?;",
            (title,)
        )
        if cur.fetchone()[0] != 0:
            logger.info("Conversation %s already exists", title)
            return
        
        convo_id = self.insert_conversation({
            "id": uuid.uuid4().hex,
            "title": title,
            "mapping": {}
        })
        
        messages: list[str] = []
        last = 0
        last_role = ""
        for mid in re.finditer("^## (Prompt|Response):$", md, re.M):
            next_role = {
                "Prompt": "user",
                "Response": "assistant"
            }[mid[1]]
            
            if last != 0:
                messages.append(self.insert_sparse_message(
                    last_role, md[last:mid.start()].strip()
                ))
            last = mid.end()
            last_role = next_role
        
        # Append the last message
        if last != 0:
            messages.append(self.insert_sparse_message(
                last_role, md[last:].strip()
            ))
        
        # Now link everything in the mappings table
        parent = None
        for mid in messages:
            self.insert_mapping(
                uuid.UUID(convo_id).hex,
                {
                    "id": mid,
                    "parent": parent
                }
            )
            parent = mid
        
        self.conn.commit()

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Consolidate ChatGPT exports into a single database.")
    parser.add_argument("-db", "--database", type=str, help="The database file to use.")
    parser.add_argument("export", type=str, nargs="+", help="The export to load.")
    
    args = parser.parse_args()
    
    db = Database(args.database or "chatgpt.db")
    
    from zipfile import ZipFile
    
    for export in args.export:
        logger.info("Loading export %s", export)
        if os.path.isdir(export):
            with open(f"{export}/conversations.json", "r") as f:
                conversations = json.load(f)
        else:
            match os.path.splitext(export)[1]:
                case ".json":
                    with open(export, "r") as f:
                        conversations = json.load(f)
                
                case ".zip":
                    with ZipFile(export) as z:
                        with z.open("conversations.json", "r") as f:
                            conversations = json.load(f)
                
                case ".md":
                    with open(export, "r") as f:
                        raw = f.read()
                    db.load_markdown(export, raw)
                    continue
                
                case _:
                    logger.error("Unknown file type: %s", export)
                    continue
        
        for convo in conversations:
            db.load_conversation(convo)
    
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))