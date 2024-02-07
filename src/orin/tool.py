'''
Tool-use abstractions.
'''

from abc import ABC, abstractmethod
from typing import Any, Optional, override

from openai.types.chat import ChatCompletionToolParam
from pydantic import TypeAdapter

class Tool(ABC):
    __name__: str
    __doc__: Optional[str]
    
    @abstractmethod
    async def __call__(self, **kwargs) -> Any: ...
    
    @abstractmethod
    def render(self) -> ChatCompletionToolParam: ...

class FunctionTool(Tool):
    def __init__(self, func):
        self.__func__ = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
    
    async def __call__(self, **kwargs) -> Any:
        print("Function call", self.__name__, kwargs)
        return await self.__func__(**kwargs)
    
    @override
    def render(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": str(self.__doc__),
                "parameters": TypeAdapter(self.__func__).json_schema()
            }
        }

class ToolBox:
    '''Collection of instantiated tools.'''
    
    tools: dict[str, Tool]
    
    def __init__(self, *tools: Tool):
        self.tools = {tool.__name__: tool for tool in tools}
    
    def __getitem__(self, name: str) -> Tool:
        return self.tools[name]
    
    def __iter__(self):
        return iter(self.tools.values())
    
    def __contains__(self, name: str) -> bool:
        return name in self.tools
    
    def render(self) -> list[ChatCompletionToolParam]:
        return [tool.render() for tool in self.tools.values()]
    
    def add(self, *tools: Tool):
        self.tools.update({tool.__name__: tool for tool in tools})