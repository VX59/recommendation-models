from sqlalchemy.future import select
from sqlalchemy import func
from db import async_session
from models import MusiqlRepository
import numpy as np
import asyncio
from datetime import datetime
import os

async def GraphAMP_seed():
    stmt = select(func.count()).select_from(MusiqlRepository)

    async with async_session() as session:
        result = await session.execute(stmt)
        total = result.scalar_one()
    seed_matrix = np.random.randint(0,100, size=(total,total))
    seed_markov_chain = np.array(list(map(lambda row: row / row.sum(), seed_matrix)))

    return seed_markov_chain
    

async def main():
    seed_markov_chain = await GraphAMP_seed()
    dttm = datetime.now().isoformat()

    file_name = f"recommendation-models/GraphAMP/models/GraphAMP-{dttm}.npy"
    
    os.makedirs("recommendation-models/GraphAMP/models", exist_ok=True)

    with open(file_name, "wb") as f:
        np.save(f, seed_markov_chain)


if __name__ == "__main__":
    asyncio.run(main())