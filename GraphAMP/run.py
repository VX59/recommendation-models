from .accumulate import accumulate
from .compute_session import compute_session
from .seed import seed
import asyncio


async def main():
    await seed()
    await compute_session()
    accumulate()

if __name__ == "__main__":
    asyncio.run(main())