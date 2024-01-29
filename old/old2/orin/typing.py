'''
Consolidate typing in one file.
'''

from typing import Optional, override, overload, Any, Iterable, Iterator, Literal, Optional, Union, TypeVar, Generic, Callable, Awaitable, AsyncIterator, AsyncGenerator, Mapping, get_args, get_origin, cast, TYPE_CHECKING, TypeGuard, Protocol, IO, reveal_type, Self, AsyncContextManager, TypedDict
from os import PathLike

type json_value = None|bool|int|float|str|list[json_value]|dict[str, json_value]

class NotGiven:
    '''Placeholder for value which isn't given.'''
    
    def __init__(self): raise NotImplementedError()
    def __bool__(self): return False
    def __repr__(self): return "NOT_GIVEN"
    
    @staticmethod
    def params(**kwargs):
        '''Return a dict of the given parameters which are not NOT_GIVEN.'''
        return {k: v for k, v in kwargs.items() if not isinstance(v, NotGiven)}

NOT_GIVEN = object.__new__(NotGiven)
