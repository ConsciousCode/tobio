'''
Module for connecting to abstract LLM providers.
'''

from base import ModelMessage, SystemMessage, AgentMessage, UserMessage, ActionStep, DeltaStep, Step, Run, Provider

__all__ = [
    'ModelMessage',
    'SystemMessage',
    'AgentMessage',
    'UserMessage',
    'ActionStep',
    'DeltaStep',
    'Step',
    'Run',
    'Provider'
]

@Provider.lazy_register
def openai():
    '''OpenAI API-based provider.'''
    
    from .openai import OpenAIProvider
    return OpenAIProvider

del openai