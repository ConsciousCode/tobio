import asyncio
import logging
import os
from prompt_toolkit import print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText

from .typing import Callable, Optional, overload, overload, Self

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

def read_file(filename):
    '''Open and read a text file.'''
    with open(filename, "rt") as f:
        return f.read()

def typename(value):
    '''Return the type name of the value.'''
    return type(value).__name__

@overload
def normalize_chan(chan: None) -> None: ...
@overload
def normalize_chan(chan: str) -> str: ...

def normalize_chan(chan: Optional[str]):
    '''Normalize a channel name.'''
    
    if chan is None:
        return
    
    if chan == "@":
        raise NotImplemented("Id channel with empty id")
    
    chan = chan.lower()
    # Logical id without 0 padding
    if chan.startswith("@"):
        chan = f"@{int(chan[1:], 16)}"
    return chan

class Registry(type):
    '''Metaclass for registering subclasses.'''
    
    def __init__(cls, name, bases, attrs):
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
    
    name: str
    registry: dict[str, type[Self]]
    lazy_registry: dict[str, Callable[[type[Self]], type[Self]]]
    
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
        cls.lazy_registry.pop(name, None)
        cls.registry[name] = alt
    
    @classmethod
    def lazy_register(cls, load: Callable[[type[Self]], type[Self]]):
        '''Register a hook for lazy initialization.'''
        cls.lazy_registry[load.__name__] = load
    
    @classmethod
    def get(cls, name: str) -> type[Self]:
        '''Retrieve a registered subclass.'''
        
        name = cls.normalize_key(name)
        
        if factory := cls.lazy_registry.get(name):
            type = factory(cls)
            del cls.lazy_registry[name]
            cls.registry[name] = type
            return type
        
        return cls.registry[name]

class PausableMixin:
    '''
    Mixin for pausing components of the system.
    '''
    
    pause_signal: asyncio.Event
    
    def __init__(self):
        self.pause_signal = asyncio.Event()
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
        return not self.pause_signal.is_set()