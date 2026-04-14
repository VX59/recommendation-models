from .accumulate import accumulate
from .compute_session import compute_session
from .seed import seed_new
import asyncio


async def main():
    G = await seed_new()
    if Gdt := await compute_session():
        await accumulate(G, Gdt)


if __name__ == "__main__":
    asyncio.run(main())
