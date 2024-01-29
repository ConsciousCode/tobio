'''
Manage a Matrix client as the primary chat interface for a tobio instance.
'''

import json
import time
from typing import TypedDict, cast
from nio import TYPE_CHECKING, AsyncClient, ErrorResponse, Event, LoginResponse, MatrixRoom, Response, RoomCreateResponse, RoomSendResponse, RoomMessageText, SyncError, SyncResponse, Event

from .memory import StateMemory
from .util import logger

if TYPE_CHECKING:
    from .kernel import Kernel

ROOM_ID = "!sIyitIciGAyzCGQNvi:consciouscode.cloud"

def unwrap[T: Response, E: ErrorResponse](response: T|E) -> T:
    '''Raise an error if the response is not a success.'''
    if isinstance(response, ErrorResponse):
        raise RuntimeError(f"Matrix failure: {response}")
        
    return response

class Gateway:
    pass

class MatrixState(TypedDict):
    sync: str
    device: str
    access: str

class Matrix(Gateway):
    def __init__(self, config):
        homeserver = config['homeserver']
        username = config['username']
        self.client = AsyncClient(homeserver, username)
        self.username = username
        self.password = config['password']
        self.last_time = int(time.time()*1000)

    async def create_room(self, is_direct, invite):
        response = unwrap(await self.client.room_create(
            is_direct=is_direct,
            invite=invite
        ))
        return response.room_id

    async def send_message(self, room_id, message_type, content):
        return unwrap(await self.client.room_send(
            room_id=room_id,
            message_type=message_type,
            content=content
        ))

    async def close(self):
        await self.client.close()

    async def send_text(self, text):
        await self.send_message(
            room_id=ROOM_ID,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": text
            }
        )
    
    async def run(self, kernel: "Kernel", state: StateMemory):
        def on_message(room: MatrixRoom, event: Event):
            ev: RoomMessageText = cast(RoomMessageText, event)
            if ev.sender == self.username:
                return
            
            msg = kernel.memory.message(
                ev.event_id,
                kernel.foreign_agent(ev.sender).id,
                ev.server_timestamp,
                room.room_id,
                ev.body
            )
            
            if ev.body.startswith("!"):
                return
            
            logger.info(f"Received message: {ev.body}")
            kernel.push_message(msg)
        
        self.client.add_event_callback(on_message, RoomMessageText)
        
        state_value: MatrixState = json.loads(state.value)
        try:
            # Login or restore the login
            if state_value is None:
                logger.info("No matrix state found, logging in")
                login = unwrap(await self.client.login(self.password))
                state.value = json.dumps({
                    "device": login.device_id,
                    "access": login.access_token
                })
                sync_token = None
            else:
                logger.info("Restoring matrix login")
                self.client.restore_login(
                    self.username,
                    state_value['device'],
                    state_value['access']
                )
                sync_token = state_value.get('sync')
            
            # Sync forever
            while True:
                res = await self.client.sync(
                    since=sync_token
                )
                sync_token = unwrap(res).next_batch
                state.value = json.dumps({
                    "sync": sync_token,
                    "device": self.client.device_id,
                    "access": self.client.access_token
                })
        
        finally:
            await self.client.close()