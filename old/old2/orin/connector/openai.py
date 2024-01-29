from contextlib import asynccontextmanager
from typing import reveal_type
import httpx
import json
from openai import AsyncClient
from dataclasses import dataclass, field

from ..util import logger
from ..typing import override

from .base import Connector, MessageHandle, MessageSchema, ProcessorHandle, Role, Stopped, TextContent, ImageContent, ActionRequired, Step, Content, Stopped, Waiting, Failed, RunHandle, ThreadHandle, FileHandle, ToolCodeInterpreter, ToolFunction, ToolRetrieval, ToolSchema, ExpiredError, APIError

from ..typing import Optional, NotGiven, NOT_GIVEN, Literal, override, Any

type APIResourceId = str

type AssistantId = APIResourceId
type ThreadId = APIResourceId
type MessageId = APIResourceId
type RunId = APIResourceId
type StepId = APIResourceId
type FileId = APIResourceId

def check_id(kind: type[APIResourceId], id: str):
    if not isinstance(id, str):
        return False
    
    return id.startswith({
        AssistantId: "asst_",
        ThreadId: "thread_",
        MessageId: "msg_",
        RunId: "run_",
        StepId: "step_",
        FileId: "file_"
    }[kind])

@dataclass
class OpenAIMessageHandle(MessageHandle):
    '''OpenAI message handle.'''
    
    conn: 'OpenAIConnector'
    
    thread_id: ThreadId
    message_id: MessageId
    
    def __post_init__(self):
        assert check_id(ThreadId, self.thread_id)
        assert check_id(MessageId, self.message_id)
    
    @override
    def handle(self) -> str:
        return f"{self.message_id},{self.thread_id}"
    
    @override
    async def retrieve(self) -> MessageSchema:
        res = await self.conn.openai.beta.threads.messages.retrieve(
            thread_id=self.thread_id,
            message_id=self.message_id
        )
        content = []
        for msg in res.content:
            match msg.type:
                case "text":
                    content.append(TextContent(msg.text.value)) # type: ignore
                case "image":
                    content.append(ImageContent(
                        self.conn.file_handle(msg.image_file.file_id) # type: ignore
                    ))
                case _:
                    raise NotImplementedError(f"Unknown message type {msg.type!r}")
        
        return MessageSchema(
            role=res.role,
            content=content
        )

@dataclass
class OpenAIFileHandle(FileHandle):
    '''File API resource.'''
    
    conn: 'OpenAIConnector'
    
    file_id: FileId
    
    def __post_init__(self):
        assert check_id(FileId, self.file_id)
    
    @override
    def handle(self):
        return self.file_id
    
    @override
    async def retrieve(self) -> FileHandle.Object:
        res = await self.conn.openai.files.retrieve(
            file_id=self.file_id
        )
        return FileHandle.Object(res.filename)
    
    @override
    async def content(self) -> bytes:
        res = await self.conn.openai.files.content(
            file_id=self.file_id
        )
        return res.content
    
    @override
    async def delete(self) -> bool:
        res = await self.conn.openai.files.delete(
            file_id=self.file_id
        )
        return res.deleted

@dataclass
class OpenAIRunHandle(RunHandle):
    '''An in-progress completion in a thread.'''
    
    conn: 'OpenAIConnector'
    
    thread_id: ThreadId
    run_id: RunId
    
    def __post_init__(self):
        assert check_id(ThreadId, self.thread_id)
        assert check_id(RunId, self.run_id)
    
    @override
    def handle(self):
        return f"{self.run_id},{self.thread_id}"
    
    @override
    async def __aenter__(self):
        res = await self.conn.openai.beta.threads.runs.retrieve(
            run_id=self.run_id,
            thread_id=self.thread_id
        )
        
        while True:
            match res.status:
                # If it's cancelling, assume it will be cancelled and exit early
                case "cancelling"|"cancelled": yield Stopped("cancelled")
                case "completed": yield Stopped("completed")
                case "expired": yield Stopped("expired")
                case "failed":
                    assert res.last_error is not None
                    if res.last_error.code == "server_error":
                        yield Failed("server_error")
                    elif res.last_error.code == "rate_limit_exceeded":
                        yield Failed("rate_limit")
                    else:
                        raise NotImplementedError(f"Unknown error code {res.last_error.code!r}")
                
                case "in_progress"|"queued":
                    res = await self.conn.openai.beta.threads.runs.retrieve(
                        run_id=self.run_id,
                        thread_id=self.thread_id
                    )
                    yield Waiting()
                
                case "requires_action":
                    assert res.required_action is not None
                    
                    # Yield to the caller for each action required
                    # This lets us keep track of tool ids internally
                    tool_outputs = []
                    for tool_call in res.required_action.submit_tool_outputs.tool_calls:
                        tool_id = tool_call.id
                        func = tool_call.function
                        
                        args = await self.conn.parse_json(func.arguments)
                        assert isinstance(args, dict)
                        
                        output = yield ActionRequired(func.name, args)
                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "output": json.dumps(output)
                        })
                    
                    self.res = await self.conn.openai.beta.threads.runs.submit_tool_outputs(
                        run_id=self.run_id,
                        thread_id=self.thread_id,
                        tool_outputs=tool_outputs
                    )
                
                case status:
                    raise NotImplementedError(f"Unknown run status {status!r}")
    
    @override
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cancel()
    
    @override
    async def cancel(self):
        await self.conn.openai.beta.threads.runs.cancel(
            run_id=self.run_id,
            thread_id=self.thread_id
        )

@dataclass
class OpenAIProcessorHandle(ProcessorHandle):
    '''Abstract handle to assistant API resource.'''
    
    conn: 'OpenAIConnector'
    
    assistant_id: AssistantId
    
    def __post_init__(self):
        assert check_id(AssistantId, self.assistant_id)
    
    @override
    def handle(self):
        return self.assistant_id
    
    @override
    async def file_assign(self, file: OpenAIFileHandle):
        await self.conn.openai.beta.assistants.files.create(
            assistant_id=self.assistant_id,
            file_id=file.file_id
        )
    
    @override
    async def file_list(self):
        res = await self.conn.openai.beta.assistants.files.list(
            assistant_id=self.assistant_id
        )
        async for file in res:
            yield OpenAIFileHandle(self.conn, file.id)
            
    @override
    async def file_delete(self, file: OpenAIFileHandle):
        res = await self.conn.openai.beta.assistants.files.delete(
            file_id=file.file_id,
            assistant_id=self.assistant_id
        )
        return res.deleted
    
    @override
    async def retrieve(self):
        res = await self.conn.openai.beta.assistants.retrieve(
            assistant_id=self.assistant_id
        )
        tools: list[ToolSchema] = []
        for tool in res.tools:
            match tool.type:
                case 'code-interpreter':
                    tools.append(ToolCodeInterpreter())
                case 'retrieval':
                    tools.append(ToolRetrieval())
                case 'function':
                    tools.append(ToolFunction(
                        name=tool.function.name, # type: ignore
                        parameters=tool.function.parameters, # type: ignore
                        description=tool.function.description # type: ignore
                    ))
                
        return ProcessorHandle.Object(
            res.name,
            res.created_at,
            res.description,
            res.instructions, {
                "model": res.model
            },
            tools
        )
    
    @override
    async def update(self, *,
        description: Optional[str]|NotGiven=NOT_GIVEN,
        file_ids: list[str]|NotGiven=NOT_GIVEN,
        instructions: Optional[str]|NotGiven=NOT_GIVEN,
        model: str|NotGiven=NOT_GIVEN,
        tools: list[dict]|NotGiven=NOT_GIVEN
    ):
        await self.conn.openai.beta.assistants.update(
            assistant_id=self.assistant_id,
            **NotGiven.params(
                description=description,
                file_ids=file_ids,
                instructions=instructions,
                model=model,
                tools=tools
            )
        )
    
    @override
    async def delete(self):
        res = await self.conn.openai.beta.assistants.delete(
            assistant_id=self.assistant_id
        )
        return res.deleted

@dataclass
class OpenAIThreadHandle(ThreadHandle):
    '''An abstract conversation thread.'''
    
    conn: 'OpenAIConnector'
    
    thread_id: ThreadId
    
    def __post_init__(self):
        assert check_id(ThreadId, self.thread_id)
    
    @override
    def handle(self):
        return self.thread_id
    
    @override
    async def add(self, role: Role, content: str):
        if role != "user":
            raise NotImplementedError("Only user messages are supported")
        
        res = await self.conn.openai.beta.threads.messages.create(
            thread_id=self.thread_id,
            content=content,
            role=role
        )
        return OpenAIMessageHandle(self.conn, self.thread_id, res.id)
    
    @override
    async def run(self, processor: OpenAIProcessorHandle):
        res = await self.conn.openai.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=processor.assistant_id
        )
        return OpenAIRunHandle(
            conn=self.conn,
            thread_id=self.thread_id,
            run_id=res.id
        )
    
    @override
    async def list(self, *, order='asc'):
        res = await self.conn.openai.beta.threads.messages.list(
            thread_id=self.thread_id,
            order=order
        )
        async for msg in res:
            yield OpenAIMessageHandle(self.conn, self.thread_id, msg.id)
    
    @override
    async def delete(self):
        res = await self.conn.openai.beta.threads.delete(
            thread_id=self.thread_id
        )
        return res.deleted

class OpenAIConnector(Connector):
    '''OpenAI API connector.'''
    
    name = "openai"
    
    openai: AsyncClient
    
    def __init__(self, client: AsyncClient):
        self.openai = client
    
    def __repr__(self):
        return "<openai Connector>"
    
    @override
    @classmethod
    @asynccontextmanager
    async def connect(cls, config: dict):
        api_key = config['api_key']
        async with httpx.AsyncClient() as http:
            async with AsyncClient(api_key=api_key, http_client=http) as client:
                yield OpenAIConnector(client)
    
    @override
    def message_handle(self, config: str):
        message, thread = config.split(',')
        return OpenAIMessageHandle(self, thread, message)
    
    @override
    def run_handle(self, config: str):
        run, thread = config.split(',')
        return OpenAIRunHandle(self, thread, run)
    
    @override
    def processor_handle(self, config: str):
        return OpenAIProcessorHandle(self, config)
    
    @override
    async def processor_create(self, name: str, description: str, instructions: str, config: dict):
        res = await self.openai.beta.assistants.create(
            name=name,
            description=description,
            model=config['model'],
            instructions=instructions
        )
        return OpenAIProcessorHandle(self, res.id)
    
    @override
    def thread_handle(self, config: str) -> ThreadHandle:
        return OpenAIThreadHandle(self, config)
    
    @override
    async def thread_create(self, messages: list[MessageSchema]|NotGiven=NOT_GIVEN) -> ThreadHandle:
        res = await self.openai.beta.threads.create(
            **NotGiven.params(messages=messages)
        )
        return OpenAIThreadHandle(self, res.id)
    
    @override
    def file_handle(self, config: str) -> FileHandle:
        return OpenAIFileHandle(self, config)
    
    @override
    async def file_create(self, filename: str, purpose: Literal['assistants']) -> FileHandle:
        res = await self.openai.files.create(file=open(filename, 'rb'), purpose=purpose)
        return OpenAIFileHandle(self, res.id)
    
    async def parse_json(self, doc: str):
        '''Attempt to parse the JSON using an LLM as backup for typos.'''
        
        for i in range(3):
            try:
                return json.loads(doc)
            except json.JSONDecodeError as e:
                logger.debug(f"Argument parse {e!r}")
                raise NotImplementedError("Todo: json parse backup")