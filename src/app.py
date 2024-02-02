'''
BOTTOM. UP. ONLY BOTTOM UP. DO NOTHING TOP-DOWN. Every change MUST result in a
working program.
'''

from typing import Literal, Optional, assert_never, cast

import chainlit as cl
import chainlit.data as cl_data

import openai as oai
from openai.types.chat import ChatCompletionMessageParam
import httpx

from orin import load_config
from orin.util import logger

type Role = Literal["user", "assistant", "system", "tool"]
type Message = ChatCompletionMessageParam

"""
class DataLayer(cl_data.BaseDataLayer):
    def __init__(self, uri):
        u = urlparse(uri)
        if u.scheme not in {"", "file", "sqlite"}:
            raise ValueError("Only SQLite is supported")
        
        self.conn = sqlite3.connect(u.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript('''
            PRAGMA foreign_keys = ON;
            
            CREATE TABLE IF NOT EXISTS threads (
                guid TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                deleted_at REAL,
                title TEXT NOT NULL,
                summary TEXT
            );
            
            CREATE TABLE IF NOT EXISTS tags (
                name TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS step_tags (
                step_guid TEXT NOT NULL,
                tag_rowid INTEGER NOT NULL,
                
                PRIMARY KEY(step_id, tag_id)
                FOREIGN KEY(step_guid) REFERENCES steps(guid)
                FOREIGN KEY(tag_rowid) REFERENCES tags(rowid)
            ) WITHOUT ROWID;
            
            CREATE TABLE IF NOT EXISTS steps (
                guid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                metadata TEXT,
                parent_guid TEXT,
                thread_guid TEXT NOT NULL,
                created_at REAL NOT NULL,
                finished_at REAL,
                
                FOREIGN KEY(parent_guid) REFERENCES steps(guid),
                FOREIGN KEY(thread_guid) REFERENCES threads(guid)
            );
        ''')
    
    async def get_user(self, identifier: str):
        return cl.PersistedUser(id="test", createdAt=now, identifier=identifier)

    async def create_user(self, user: cl.User):
        return cl.PersistedUser(id="test", createdAt=now, identifier=user.identifier)

    @cl_data.queue_until_user_message()
    async def create_step(self, step_dict: StepDict):
        global create_step_counter
        create_step_counter += 1

    async def get_thread_author(self, thread_id: str):
        return "admin"

    async def list_threads(
        self, pagination: cl_data.Pagination, filter: cl_data.ThreadFilter
    ) -> cl_data.PaginatedResponse[cl_data.ThreadDict]:
        return cl_data.PaginatedResponse(
            data=[t for t in thread_history if t["id"] not in deleted_thread_ids],
            pageInfo=cl_data.PageInfo(hasNextPage=False, endCursor=None),
        )

    async def get_thread(self, thread_id: str):
        return next((t for t in thread_history if t["id"] == thread_id), None)

    async def delete_thread(self, thread_id: str):
        deleted_thread_ids.append(thread_id)

cl_data._data_layer = DataLayer()
"""

class Context:
    http: httpx.AsyncClient
    openai: oai.AsyncOpenAI
    history: list[Message]
    
    chat_prompt: Message
    summary_prompt: Message
    
    def __init__(self, config):
        self.config = config
        with open(config['persona'], "r") as f:
            self.chat_prompt = {
                "role": "system",
                "name": "prompt",
                "content": f.read()
            }
        self.summary_prompt = {
            "role": "system",
            "name": "prompt",
            "content": "You are the summarizing agent of Orin. Summarize the conversation so far, including information contained in previous summaries, and provide an informative summary."
        }
        
        if config['history_limit'] < 2:
            raise ValueError("History limit must be at least 2.")
        """
        if config['history_limit'] % 2 != 0:
            logger.warn("History limit should be even to ensure the most recent message is from the assistant. Otherwise, the (not very smart) summarizer can get confused and try to answer the user's question.")
            config['history_limit'] += 1
        """
        self.history = [
            {"role": "system", "name": "summary", "content": "[SUMMARY] First boot."}
        ]
        self.oldest_summary = 0
    
    async def __aenter__(self):
        self.http = await httpx.AsyncClient().__aenter__()
        self.openai = await oai.AsyncOpenAI(
            api_key=self.config['openai']['api_key'],
            http_client=self.http
        ).__aenter__()
        return self
    
    async def __aexit__(self, exc_type=None, exc_value=None, traceback=None):
        await self.openai.__aexit__(exc_type, exc_value, traceback)
        await self.http.__aexit__(exc_type, exc_value, traceback)
    
    async def add_summary(self, oldest: Message):
        result = await self.openai.chat.completions.create(
            messages=[
                self.chat_prompt,
                oldest,
                *self.history,
                self.summary_prompt
            ],
            **self.config['models']['summarize']
        )
        content = result.choices[0].message.content
        if content is None:
            raise ValueError("No content in completion")
        
        self.history.append({
            "role": "system",
            "content": f"[SUMMARY] {content}",
            "name": "summary"
        })
        return content
    
    async def add(self, role: Role, content: str, name: Optional[str]=None):
        message = {"role": role, "content": content}
        if name is not None:
            message["name"] = name
        self.history.append(message) # type: ignore
        
        # Only truncate if the assistant is speaking
        #if role != "assistant":
        #    return
        
        if len(self.history) > self.config['history_limit']:
            oldest = self.history.pop(0)
            if oldest.get('name') == "summary":
                summary = await self.add_summary(oldest)
                print("Summary", summary)
    
    def stream(self, delta: str):
        '''Stream deltas to the last message'''
        
        last = self.history[-1]
        content = last.get('content')
        if content is None:
            last["content"] = delta
        elif isinstance(content, str):
            last['content'] = content + delta
        elif isinstance(content, list):
            step = content[-1]
            if step['type'] == "text":
                step['text'] += delta
            else:
                content.append({"type": "text", "text": delta})
        else:
            assert_never(content)
    
    def chatlog(self):
        return [self.chat_prompt, *self.history]

@cl.on_chat_start
async def on_chat_start():
    config = load_config("private/config.toml")
    context = await Context(config).__aenter__()
    cl.user_session.set("context", context)
    
    #cl_data._data_layer = DataLayer(config['memory']['database'])

@cl.on_chat_end
async def on_chat_end():
    context = cast(Context, cl.user_session.get("context"))
    await context.__aexit__(None, None, None)

@cl.step(type="run", name="command")
async def command(context: Context, cmd: str, rest: str):
    match cmd:
        case "history":
            return {"history": context.history}
        
        case _:
            return f"Unknown command {cmd}"

@cl.step(type="llm", name="orin")
async def call_llm(context: Context, msg: cl.Message):
    settings = context.config['models']['chat']
    chatlog = context.chatlog()
    
    step = cl.context.current_step
    if step is not None:
        step.generation = cl.ChatGeneration(
            provider="openai-chat",
            messages=[
                cl.GenerationMessage(
                    role=m['role'],
                    formatted=m['content'], # type: ignore
                    name=m.get('name')
                ) for m in chatlog
            ],
            settings=settings
        )
        step.input = msg.content
        
        msg = cl.Message(author="Orin", content="")
        await msg.send()
        
        stream = await context.openai.chat.completions.create(
            messages=context.chatlog(),
            stream=True,
            **settings,
        )
        
        async for part in stream:
            delta = part.choices[0].delta
            if delta.content:
                # Stream the output of the step
                await msg.stream_token(delta.content)
                
                context.stream(delta.content)
        
        return msg.content

@cl.on_message
async def on_message(message: cl.Message):
    context = cast(Context, cl.user_session.get("context"))
    
    content = message.content
    if content.startswith("/") and not content.startswith("//"):
        cmd, *rest = content[1:].split(None, 1)
        rest = rest[0] if rest else ""
        await command(context, cmd, rest)
    else:
        await context.add("user", message.content)
        await call_llm(context, message)