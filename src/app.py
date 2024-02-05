'''
BOTTOM. UP. ONLY BOTTOM UP. DO NOTHING TOP-DOWN. Every change MUST result in a
working program.
'''

from functools import cache
import json
from typing import Literal, Optional, assert_never, cast
import uuid

import streamlit as st

import openai as oai
from openai.types.chat import ChatCompletionMessageParam
import httpx

from orin import load_config
from orin.util import logger

type Role = Literal["user", "assistant", "system", "tool"]
type Message = ChatCompletionMessageParam

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

if 'config' not in st.session_state:
    st.session_state.config = load_config("private/config.toml")
config = st.session_state.config

conn = st.connection("sql", url=config['memory']['database'])
conn.connect()

@cache
def author(name):
    author = conn.query(
        "SELECT id FROM authors WHERE name = ?",
        params=(name,)
    )
    if len(author):
        return author[0]
    
    guid = uuid.uuid4()
    conn.query(
        "INSERT INTO authors (guid, name) VALUES (?, ?)",
        params=(guid, name)
    )
    return guid

def persistent(name, value):
    if name in st.session_state:
        return st.session_state[name]
    st.session_state[name] = value
    return value

if 'session_id' not in st.session_state:
    session_id = uuid.uuid4()
    conn.query(
        "INSERT INTO sessions (id) VALUES (?)",
        params=(str(session_id),)
    )
    st.session_state.session_id = uuid.uuid4()



st.title("Hello world")

if 'history' not in st.session_state:
    st.session_state.history = []

print(st.session_state)
print(st.session_state.history)
for message in st.session_state.history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("What is up?"):
    st.session_state.history.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)