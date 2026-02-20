from sqlalchemy.future import select
from sqlalchemy import func
from db import async_session
from models import MusiqlRepository
import numpy as np
import asyncio
import os
import networkx as nx
from itertools import product
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

async def GraphAMP_seed():
    stmt = select(MusiqlRepository)

    async with async_session() as session:
        result = await session.execute(stmt)
        rows:list[MusiqlRepository] = result.scalars().all()

    G = nx.DiGraph()
    for record in tqdm(rows):
        if record.uri not in G.nodes:
            G.add_node(record.uri)

    for i, j in tqdm(product(G.nodes, repeat=2)):
        G.add_edge(i,j, weight=np.random.rand())
        G.add_edge(j,i, weight=np.random.rand())

    for node in G.nodes:
        out_edges = G.out_edges(node, data=True)
        
        total_weight = sum(data["weight"] for _, _, data in out_edges)
        
        if total_weight > 0:
            for _, _, data in out_edges:
                data["weight"] /= total_weight
    
    return G

async def main():
    seed = await GraphAMP_seed()
    file_name = f"recommendation-models/GraphAMP/models/GraphAMP.graph"
    
    os.makedirs("recommendation-models/GraphAMP/models", exist_ok=True)
    joblib.dump(seed, file_name, compress=3)
    
    A = nx.to_numpy_array(seed, weight="weight", nodelist=sorted(seed.nodes()))
    plt.imshow(A, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Weight')
    plt.xticks(ticks=np.arange(len(seed.nodes())), labels=sorted(seed.nodes()), rotation=90)
    plt.yticks(ticks=np.arange(len(seed.nodes())), labels=sorted(seed.nodes()))
    
    plt.title("GraphAMP Seed Heatmap")
    plt.savefig("GraphAMP_seed_heatmap.png")

if __name__ == "__main__":
    asyncio.run(main())