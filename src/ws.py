#!/usr/bin/env python3

import asyncio
from contextlib import asynccontextmanager
import sys
import json
import traceback
import uvicorn
import tomllib
from fastapi import FastAPI, WebSocket

from orin import Orin

CONFIG_FILE = "private/config.toml"

with open(CONFIG_FILE, 'rb') as f:
    config = tomllib.load(f)

orin = Orin(config)

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
        prompt = await ws.receive_text()
        async for delta in orin.chat(prompt):
            await ws.send_text(delta)
    except BaseException as e:
        content = ''.join(traceback.format_exception(e))
        await ws.send_text(content)
        raise

async def main(argv):
    #sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main(sys.argv))