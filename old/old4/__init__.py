#!/usr/bin/env python3

import asyncio
import os
import sys
import uvicorn
import tomllib
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .kernel import Model, Kernel

__all__ = [
    'Model',
    'Kernel',
    'main',
    'app',
    'kernel'
]

CONFIG_FILE = "private/config.toml"

with open(CONFIG_FILE, 'rb') as f:
    config = tomllib.load(f)

kernel = Kernel(config)

## FastAPI ##

@asynccontextmanager
async def lifespan(app: FastAPI):
    await kernel.run()
    yield

app = FastAPI(lifespan=lifespan)

@app.get('/')
async def root():
    return FileResponse('static/index.html')

class Message(BaseModel):
    user: str
    message: str

@app.post('/sendMessage')
async def sendMessage(message: Message):
    user = kernel.user(message.user)
    response = await user.sendMessage(message.message)
    return {'response': response}

app.mount("/", StaticFiles(directory="static"), name="static")

async def main(argv):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == '__main__':
    asyncio.run(main(sys.argv))