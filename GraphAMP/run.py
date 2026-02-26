from .accumulate import accumulate
from .compute_session import compute_session
from .seed import seed_new
import asyncio


async def main():
    await seed_new()
    if await compute_session():
        accumulate()

if __name__ == "__main__":
    asyncio.run(main())