from abc import ABCMeta
import asyncio
from functools import cache
import sys
import time
import importlib.util
import logging
import os
from types import GenericAlias
from typing import Annotated, Any, AnyStr, Callable, ClassVar, ForwardRef, Literal, LiteralString, Optional, Self, Sequence, TypeAliasType, cast, get_args, get_origin, get_type_hints, overload
from collections.abc import Iterable
from weakref import WeakKeyDictionary

from prompt_toolkit import print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText

class ColorLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        fmt ='[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'
        self.formatter = logging.Formatter(fmt)

    def emit(self, record):
        try:
            with patch_stdout():
                msg = self.format(record)
                color = self.get_color(record.levelno)
                formatted_text = FormattedText([(color, msg)])
                print_formatted_text(formatted_text)
        except Exception:
            self.handleError(record)

    def get_color(self, levelno):
        if levelno >= logging.ERROR:
            return 'ansired'
        elif levelno >= logging.WARNING:
            return 'ansiyellow'
        elif levelno >= logging.INFO:
            return 'ansigreen'
        else:  # DEBUG and NOTSET
            return 'ansiblue'

logger = logging.getLogger("orin")
LOG_LEVEL = "INFO"
if LOG_LEVEL := os.getenv("LOG_LEVEL", LOG_LEVEL).upper():
    #logger.addHandler(ColorLogHandler())
    logger.setLevel(LOG_LEVEL)
    logger.info(f"Set log level to {LOG_LEVEL}")
elif os.getenv("LOGLEVEL"):
    logger.warning("Use LOG_LEVEL, not LOGLEVEL")

if LOG_LEVEL == "DEBUG":
    asyncio.get_event_loop().set_debug(True)

class NotGiven:
    '''Placeholder for value which isn't given.'''
    
    def __init__(self): raise NotImplementedError()
    def __bool__(self): return False
    def __repr__(self): return "NOT_GIVEN"
    
    @staticmethod
    def params(**kwargs):
        '''Return a dict of the given parameters which are not NOT_GIVEN.'''
        return {k: v for k, v in kwargs.items() if not isinstance(v, NotGiven)}

# Using __new__ to implement singleton pattern
NOT_GIVEN = object.__new__(NotGiven)
'''Placeholder for value which isn't given.'''

#LiteralType: TypeAlias = type(Literal[0]) # type: ignore

# Pyright is too stupid to follow TypeVarType when passing type[T] to a function,
#  so this probably won't work.
type timestamp = int

type Thunk[T] = Callable[[], T]
type OnePlus[T] = T|Sequence[T]
type Defer[T] = T|Thunk[T]
type TypeRef = str|ForwardRef|GenericAlias|TypeAliasType|Annotated
type DeferTypeRef = Defer[type]|TypeRef
'''A typelike reference which can be wrapped to be resolved later.'''

def now() -> timestamp:
    return int(time.time())

def filter_dict(d: dict, keys: Iterable):
    '''Filter a dict by keys.'''
    return {k: v for k, v in d.items() if k in keys}

def unalias_dict(d: dict, aliases: dict):
    '''Unalias a dict.'''
    return {aliases.get(k, k): v for k, v in d.items()}

def read_file(filename):
    '''Open and read a text file.'''
    with open(filename, "rt") as f:
        return f.read()

def nop(x): return x

def typename(t: TypeRef) -> str:
    '''Return the name of a type, or the name of a value's type.'''
    
    if get_origin(t) is None:
        if not isinstance(t, type):
            t = type(t)
        return t.__name__ # type: ignore
    return str(t)

def typecheck(value: Any, t: TypeRef) -> bool:
    '''
    More featureful type checking. Supports isinstance, but also the zoo of
    typing types which are not supported by isinstance.
    '''
    
    try:
        # type, Optional, Union, @runtime_checkable
        return isinstance(value, t) # type: ignore
    except TypeError:
        pass
    
    if t is Any: return True
    if t in {None, type(None)}: return value is None
    if t in {AnyStr, LiteralString}: return isinstance(value, (str, bytes))
    
    # Generic types
    
    origin, args = get_origin(t), get_args(t)
    
    if origin is Literal:
        return value in args
    
    if origin is Annotated:
        return typecheck(value, args[0])
    
    if isinstance(t, TypeAliasType):
        return typecheck(value, t.__value__) # type: ignore [attr-defined]
    
    return False

def load_module(file_name: str, module_name: Optional[str]=None):
    '''Load a module from a file.'''
    
    if module_name is None:
        module_name = os.path.splitext(os.path.basename(file_name))[0]
    
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module {module_name} from {file_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Alternatively, could use ForwardRef._evaluate but that's private. This is at least public and legal.3
def resolve_forward_ref(
    obj: TypeRef,
    globalns: Optional[dict[str, Any]]=None,
    localns: Optional[dict[str, Any]]=None
):
    '''Resolve a singular forward reference.'''
    def dummy(x: obj): pass # type: ignore
    return get_type_hints(dummy, globalns, localns)['x']

class deferred_property[T]:
    '''A property which can be resolved later with minimal friction.'''
    
    deferral: WeakKeyDictionary[type, Thunk[T]]
    
    def __init__(self):
        self.deferral = WeakKeyDictionary()
    
    def __set_name__(self, owner, name: str):
        self.__name__ = name
    
    @overload
    def __get__(self, instance: None, owner) -> Self: ...
    @overload
    def __get__(self, instance, owner) -> T: ...
    
    def __get__(self, instance, owner) -> Self|T:
        if instance is None:
            return self
        
        value = instance.__dict__.get(self.__name__, NOT_GIVEN)
        if value is not NOT_GIVEN:
            return value
        
        try:
            value = self.deferral.pop(instance)()
            setattr(instance, self.__name__, value)
            return value
        except KeyError:
            raise AttributeError(f"{typename(owner)}.{self.__name__} has no deferral") from None
    
    def __set__(self, instance, value: T):
        instance.__dict__[self.__name__] = value
        return value
    
    def defer(self, instance, deferral: Defer[T]):
        '''Explicitly defer a value.'''
        if callable(deferral):
            self.deferral[instance] = deferral
        else:
            setattr(instance, self.__name__, deferral)
        return self

def lazy_type(t: DeferTypeRef, cls: Optional[type]=None) -> Defer[type]:
    '''Return a lazy type which can be resolved later.'''
    
    if isinstance(t, (type, Callable)): return cast(type|Callable, t)
    if cls is None: raise ValueError("cls must be given if t is unbound")
    
    @cache
    def factory():
        globalns = load_module(cls.__module__).__dict__
        return resolve_forward_ref(t, globalns, cls.__dict)
    return factory

def indent(text: str, level: int=1, indent: str="    "):
    '''Indent a block of text.'''
    return "\n".join(indent + line for line in text.splitlines())

class Registry(ABCMeta):
    '''Metaclass for registering subclasses.'''
    
    def __init__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]):
        super().__init__(name, bases, attrs)
        # Don't register resource base classes
        if name != "Registrant":
            # Register only Registrant subclass subclasses
            if Registrant in cls.__bases__:
                cls.registry = {}
                cls.lazy_registry = {}
            else:
                cls.register(cls) # type: ignore

class Registrant(metaclass=Registry):
    '''Self-registering resource.'''
    
    name: ClassVar[str]
    '''Name to register the resource under.'''
    
    registry: ClassVar[dict[str, type[Self]]]
    '''Registry of registered resources.'''
    
    lazy_registry: ClassVar[dict[str, Thunk[type[Self]]]]
    '''Registry of lazy registrations.'''
    
    def __init_subclass__(cls):
        # Only register subclasses of Resource subclasses
        if Registrant not in cls.__bases__:
            cls.register(cls)
    
    @classmethod
    def keyof(cls) -> str:
        '''Return the key of the resource.'''
        return getattr(cls, "name", None) or cls.__name__
    
    @classmethod
    def normalize_key(cls, key: str) -> str:
        '''Normalize a key.'''
        return key.lower()
    
    @classmethod
    def register(cls, alt: type[Self]):
        '''Directly register a subclass.'''
        
        name = cls.normalize_key(cls.keyof())
        # Pop any lazy registration
        cls.lazy_registry.pop(name, None)
        cls.registry[name] = alt
    
    @classmethod
    def lazy_register(cls, load: Thunk[type[Self]]):
        '''Register a hook for lazy initialization.'''
        if load.__name__ not in cls.registry:
            cls.lazy_registry[load.__name__] = load
    
    @classmethod
    def get(cls, name: str) -> type[Self]:
        '''Retrieve a registered subclass.'''
        
        name = cls.normalize_key(name)
        if factory := cls.lazy_registry.get(name):
            type = factory()
            del cls.lazy_registry[name]
            cls.registry[name] = type
            return type
        
        return cls.registry[name]

class PausableMixin:
    '''Mixin for pausing components of the system.'''
    
    pause_signal: asyncio.Event
    '''Signal used to coordinate pausing the mixin.'''
    
    def __init__(self, default=False):
        self.pause_signal = asyncio.Event()
        if default:
            self.resume()
    
    def until_unpaused(self):
        '''Wait until the agent is unpaused, if it's paused.'''
        return self.pause_signal.wait()
    
    def pause(self):
        '''Pause the agent.'''
        self.pause_signal.clear()
    
    def resume(self):
        '''Unpause the agent.'''
        self.pause_signal.set()
    
    def is_paused(self):
        '''Return whether the mixin is paused.'''
        return not self.pause_signal.is_set()