from abc import ABCMeta
import asyncio
import sys
import importlib.util
import logging
import os
from typing import Any, Callable, ClassVar, Optional, Self
from collections.abc import Iterable

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
if LOG_LEVEL := os.getenv("LOG_LEVEL", LOG_LEVEL):
    logger.addHandler(ColorLogHandler())
    logger.setLevel(LOG_LEVEL.upper())
    logger.info(f"Set log level to {LOG_LEVEL}")

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

def typename(value):
    '''Return the type name of the value.'''
    return type(value).__name__

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
    
    lazy_registry: ClassVar[dict[str, Callable[[], type[Self]]]]
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
    def lazy_register(cls, load: Callable[[], type[Self]]):
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