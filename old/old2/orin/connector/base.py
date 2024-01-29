'''
Abstract base classes which must be implemented by LLM providers.
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..util import Registrant, typename
from ..typing import Optional, NotGiven, NOT_GIVEN, Literal, AsyncIterator, AsyncGenerator, AsyncContextManager, TypedDict

class APIError(RuntimeError): pass
class ExpiredError(RuntimeError): pass

@dataclass
class ToolCodeInterpreter: pass

@dataclass
class ToolRetrieval: pass

@dataclass
class ToolFunction:
    name: str
    parameters: dict[str, object]
    description: Optional[str]

type ToolSchema = ToolCodeInterpreter|ToolRetrieval|ToolFunction

type Role = Literal['user', 'assistant', 'system']

@dataclass
class TextContent:
    '''Text message.'''
    text: str

@dataclass
class ImageContent:
    '''Generated image.'''
    image: 'FileHandle'

@dataclass
class ActionRequired:
    '''Function call/action is required.'''
    function: str
    arguments: dict[str, object]

@dataclass
class Waiting:
    '''Waiting for an action to complete.'''
    pass

@dataclass
class Stopped:
    '''Run stopped for some reason.'''
    reason: Literal["completed", "cancelled", "expired"]

@dataclass
class Failed:
    '''Miscellaneous failures.'''
    error: Literal["server_error", "rate_limit"]

type Content = TextContent|ImageContent
type Step = Content|ActionRequired|Waiting|Stopped

class MessageSchema(TypedDict):
    role: Role
    content: list[Content]

class APIHandle(ABC):
    '''Handle to an abstract API resource.'''
    
    @abstractmethod
    def handle(self) -> str:
        '''Retrieve API resource handle for persistence.'''
    
    def __repr__(self):
        return f"{typename(self)}({self.handle()!r})"

class MessageHandle(APIHandle):
    '''Handle to a message.'''
    
    @abstractmethod
    async def retrieve(self) -> MessageSchema:
        '''Retrieve message specification.'''

class FileHandle(APIHandle):
    '''File API resource.'''
    
    @dataclass
    class Object:
        filename: str
    
    @abstractmethod
    def handle(self) -> str:
        '''Retrieve API resource handle for persistence.'''
    
    @abstractmethod
    async def retrieve(self) -> Object:
        '''Retrieve API resource specification.'''
    
    @abstractmethod
    async def content(self) -> bytes:
        '''Retrieve file contents.'''
    
    @abstractmethod
    async def delete(self) -> bool:
        '''Delete the file.'''

class RunHandle(APIHandle):
    '''
    In-progress completion, allowing for bidirectional data flow between user
    code and the LLM provider, eg tool use
    '''
    
    @abstractmethod
    async def __aenter__(self) -> AsyncGenerator[Step, Optional[str]]:
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback):
        pass
    
    @abstractmethod
    async def cancel(self):
        '''Cancel the run.'''

class ProcessorHandle(APIHandle):
    '''Handle to a provider's processor implementation.'''
    
    @dataclass
    class Object:
        name: Optional[str]
        created_at: int
        description: Optional[str]
        instructions: Optional[str]
        config: object
        tools: list[ToolSchema]
    
    @abstractmethod
    async def file_assign(self, file: FileHandle):
        '''Assign a file to the processor.'''
    
    @abstractmethod
    async def file_list(self) -> AsyncIterator[FileHandle]:
        '''List files in the processor.'''
    
    @abstractmethod
    async def file_delete(self, file: FileHandle) -> bool:
        '''Delete a file from the processor.'''
    
    @abstractmethod
    async def update(self, *,
        name: str|NotGiven=NOT_GIVEN,
        description: str|NotGiven=NOT_GIVEN,
        instructions: str|NotGiven=NOT_GIVEN,
        config: object|NotGiven=NOT_GIVEN,
        tools: list[dict]|NotGiven=NOT_GIVEN
    ):
        '''Update processor configuration.'''
    
    @abstractmethod
    async def retrieve(self) -> Object:
        '''Retrieve processor configuration.'''
    
    @abstractmethod
    async def delete(self) -> bool:
        '''Delete the processor.'''

class ThreadHandle(APIHandle):
    '''
    Provider thread abstraction.
    '''
    
    @abstractmethod
    def handle(self) -> str:
        '''Retrieve API resource handle for persistence.'''
    
    @abstractmethod
    async def add(self, role: Role, content: str):
        '''Add a message to the thread.'''
    
    @abstractmethod
    async def run(self, processor: ProcessorHandle) -> RunHandle:
        '''Create a new run in the thread.'''
    
    @abstractmethod
    async def delete(self) -> bool:
        '''Delete the thread.'''
    
    @abstractmethod
    def list(self, *, order: Literal['asc', 'desc']='asc') -> AsyncIterator[MessageHandle]:
        '''List messages in the thread.'''

class Connector(Registrant):
    '''Connection to an external LLM provider.'''
    
    @classmethod
    @abstractmethod
    def connect[T](cls: type[T], config) -> AsyncContextManager[T]:
        '''Connect to the LLM provider.'''
    
    @abstractmethod
    def message_handle(self, config: str) -> MessageHandle:
        '''Load a message from a configuration.'''
    
    @abstractmethod
    def run_handle(self, config: str) -> RunHandle:
        '''Load a run from a configuration.'''
    
    @abstractmethod
    def processor_handle(self, config: str) -> ProcessorHandle:
        '''Load a processor from a configuration.'''
    
    @abstractmethod
    async def processor_create(self, name: str, description: str, instructions: str, config: object) -> ProcessorHandle:
        '''Create a new processor.'''
    
    @abstractmethod
    async def processor_list(self) -> AsyncIterator[ProcessorHandle]:
        '''List processors.'''
    
    @abstractmethod
    def thread_handle(self, config: str) -> ThreadHandle:
        '''Load a thread from a configuration.'''
    
    @abstractmethod
    async def thread_create(self, messages: list[MessageSchema]|NotGiven=NOT_GIVEN) -> ThreadHandle:
        '''Create a new thread.'''
    
    @abstractmethod
    def file_handle(self, config: str) -> FileHandle:
        '''Load a file from a configuration.'''
    
    @abstractmethod
    async def file_create(self, filename: str, purpose: Literal['assistants']) -> FileHandle:
        '''Create a new file.'''
    
    @abstractmethod
    async def file_list(self) -> AsyncIterator[FileHandle]:
        '''List files uploaded to the API.'''