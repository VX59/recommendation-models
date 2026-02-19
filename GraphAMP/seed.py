from sqlalchemy.future import select
from sqlalchemy import func
from db import async_session
from models import MusiqlRepository
import numpy as np
import asyncio
from datetime import datetime
import os
import networkx as nx
from itertools import combinations

async def GraphAMP_seed():
    stmt = select(MusiqlRepository)

    async with async_session() as session:
        result = await session.execute(stmt)
        rows:list[MusiqlRepository] = result.scalars().all()

    G = nx.DiGraph()
    for record in rows:
        if record.uri not in G.nodes:
            G.add_node(record.uri)

    for i, j in combinations(G.nodes, 2):
        G.add_edge(i,j)

    return G

    seed_matrix = np.random.randint(0,100, size=(total,total))
    GraphAMP_seed = np.array(list(map(lambda row: row / row.sum(), seed_matrix)))

    return GraphAMP_seed
    

async def main():
    GraphAMP_seed = await GraphAMP_seed()
    dttm = datetime.now().isoformat()

    file_name = f"recommendation-models/GraphAMP/models/GraphAMP-{dttm}.npy"
    
    os.makedirs("recommendation-models/GraphAMP/models", exist_ok=True)

    with open(file_name, "wb") as f:
        np.save(f, GraphAMP_seed)


if __name__ == "__main__":
    asyncio.run(main())