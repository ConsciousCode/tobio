import io
from contextlib import redirect_stdout, redirect_stderr
import traceback
import subprocess
import ast

from .typing import override
from .system import Tool, Kernel, Agent

class Python(Tool):
    '''
    Execute arbitrary python code in a persistent environment. Returns stdout.
    
    The agent swarm system kernel and agent are available as globals.
    '''
    
    parameters = {
        "code": {"type": "string", "description": "Python code to execute."}
    }
    
    def __init__(self, kernel: Kernel, agent: Agent):
        super().__init__(kernel, agent)
        self.globals = {
            "__name__": agent.name,
            "__package__": None,
            "__builtins__": __builtins__,
            "__annotations__": None,
            "kernel": kernel,
            "agent": agent
        }
        self.env = {}
    
    def state(self):
        return self.env
    
    @override
    def load_state(self, state: object):
        if state is not None:
            assert isinstance(state, dict)
            self.env = state
    
    async def __call__(self, code):
        print(f"{self.agent.name} executing:\n{code}")
        with redirect_stdout(io.StringIO()) as f:
            with redirect_stderr(f):
                ans = None
                try:
                    stmts = list(ast.iter_child_nodes(ast.parse(code)))
                    if len(stmts) == 0: return
                    
                    # LLMs like to use it like a REPL, so allow returning the last expression
                    if isinstance(stmts[-1], ast.Expr):
                        if len(stmts) > 1:
                            exec(compile(ast.Module(body=stmts[:-1]), filename="<ast>", mode="exec"), self.globals, self.env)
                        ans = eval(compile(ast.Expression(body=stmts[-1].value), filename="<ast>", mode="eval"), self.globals, self.env)
                    else:
                        # otherwise we just execute the entire code
                        ans = exec(code, self.globals, self.env)
                except Exception as e:
                    traceback.print_exception(e, file=f)
        
        # Persist environment state
        self.kernel.add_state(self.agent)
        
        stdout = f.getvalue().strip()
        ans = "" if ans is None else f"_ = {ans!r}"
        result = '\n'.join(filter(None, [stdout, ans])).strip()
        
        print(f"{self.agent.name} result:\n{result}")
        return result

class Shell(Tool):
    '''Execute shell code. Will timeout if it takes more than 3 seconds.'''
    
    # TODO: It may be worthwhile to have shell agents which use the Agent push system
    
    parameters = {
        "code": {"type": "string", "description": "Bash code to execute."}
    }
    
    async def __call__(self, code):
        return subprocess.run(["bash", "-c", code], capture_output=True, timeout=3)
