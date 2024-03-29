#!/usr/bin/env python3

import asyncio
from contextlib import asynccontextmanager
import os
import signal
import sys
import traceback
import uvicorn
import tomllib
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketState

from orin import Orin, FunctionTool, logger

CONFIG_FILE = "private/config.toml"

with open(CONFIG_FILE, 'rb') as f:
    config = tomllib.load(f)

@FunctionTool
async def bash(cmd: str) -> str:
    '''Run a bash command and return stdout and stderr together. Call using a valid JSON object, as in {"cmd": "the command to execute"}'''
    print("Bash", cmd)
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return stdout.decode()

orin = Orin(config)
orin.toolbox.add(bash)

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
            if prompt.startswith("/") and not prompt.startswith("//"):
                cmd, *args = prompt[1:].split(" ", 1)
                
                match cmd:
                    case "q"|"quit":
                        await ws.close()
                        # Apparently neither FastAPI nor uvicorn have a native
                        #  way to gracefully initiate a shutdown??
                        os.kill(os.getpid(), signal.SIGINT)
                        break
                
                    case _:
                        stream = orin.cmd(cmd, args[0] if args else "")
            else:
                stream = orin.chat(prompt)
            
            async for delta in stream:
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