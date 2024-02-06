#!/usr/bin/env python3

import asyncio
from contextlib import asynccontextmanager
import sys
import traceback
import uvicorn
import tomllib
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketState

from orin import Orin, FunctionTool
from util import logger

CONFIG_FILE = "private/config.toml"

with open(CONFIG_FILE, 'rb') as f:
    config = tomllib.load(f)

@FunctionTool
async def bash(cmd: str) -> str:
    '''Run a bash command and return stdout and stderr together.'''
    print("Bash", cmd)
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return stdout.decode()

orin = Orin(config)
orin.add_tools(bash)

## FastAPI ##

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with orin:
        yield

app = FastAPI(debug=True, lifespan=lifespan)

@app.websocket('/chat')
async def socket(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Keepalive
            while True:
                if prompt := await ws.receive_text():
                    break
            
            # Stream
            async for delta in orin.chat(prompt):
                await ws.send_text(delta)
            
            # Halt
            await ws.send_text("[DONE]")
    except BaseException as e:
        content = ''.join(traceback.format_exception(e))
        logger.error(content)
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_text(content)
        raise

async def main(argv):
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main(sys.argv))