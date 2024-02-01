#!/usr/bin/env python3

import asyncio
import os
import sys
import json
import traceback
import uvicorn
import tomllib
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from tobio.agent import UserAgent
from tobio.util import typename, logger
from tobio.kernel import Kernel
from tobio.memory import StepMemory

CONFIG_FILE = "private/config.toml"

with open(CONFIG_FILE, 'rb') as f:
    config = tomllib.load(f)
print(config)
kernel = Kernel(config)

## FastAPI ##

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(kernel.run())
    yield

app = FastAPI(debug=True, lifespan=lifespan)

static = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

@app.get('/')
async def root():
    return FileResponse(os.path.join(static, 'index.html'))

async def user_input(user: UserAgent, ws: WebSocket):
    '''Handle input from the user.'''
    while True:
        msg = await ws.receive_json()
        out = await user.receive(msg)
        if out is not None:
            await ws.send_json({"type": "output", "content": out})

def unroll_steps(steps: list[StepMemory]):
    for step in steps:
        match step.kind:
            case "text": yield step.content
            case "action": yield json.loads(step.content)
            case _: raise NotImplementedError(step.kind)

async def user_output(user: UserAgent, ws: WebSocket):
    '''Handle outputs pushed to the user.'''
    while True:
        # Pull messages pushed to the user's message queue
        poke = True
        async for msg in user.pull():
            poke = False
            await ws.send_json({
                "type": "message",
                "name": msg.agent.name,
                "created_at": msg.created_at,
                "steps": list(unroll_steps(msg.steps))
            })
        
        # No messages were pushed but pull yielded, so another agent poked the user?
        if poke:
            await ws.send_json({"type": "poke"})

@app.websocket('/socket')
async def socket(ws: WebSocket):
    await ws.accept()
    try:
        name = await ws.receive_text()
        logger.info("Logging in as %s", name)
        user = kernel.agent(name)
        if not isinstance(user, UserAgent):
            raise TypeError(f"Can't log in as {typename(user)} {user!r}")
        
        async with asyncio.TaskGroup() as tg:
            tg.create_task(user_input(user, ws))
            tg.create_task(user_output(user, ws))
    except BaseException as e:
        content = ''.join(traceback.format_exception(e))
        await ws.send_json({"type": "error", "content": content})
        raise

# Must be last to prevent conflicts with other routes
app.mount("/", StaticFiles(directory=static), name="static")

async def main(argv):
    #sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main(sys.argv))