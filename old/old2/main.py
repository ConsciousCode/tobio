import asyncio
import sys
import tomllib as toml

# Automatically links names to agent types
from orin import Kernel

DBPATH = "private/system.db"

async def main(argv):
    try:
        with open("private/config.toml", "rb") as f:
            config = toml.load(f)
        
        async with Kernel.start(config) as system:
            if not system.agents:
                await system.create_agent("User")
                await system.create_agent("Hermes")
                await system.create_agent("Prometheus")
                await system.create_agent("Daedalus")
    
    finally:
        print(flush=True)

if __name__ == "__main__":
    asyncio.run(main(sys.argv))