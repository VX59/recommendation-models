from sqlalchemy.future import select
from db import async_session
from models import MusiqlRepository
import numpy as np
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
    
    return G

async def seed():
    seed = await GraphAMP_seed()
    file_name = f"recommendation-models/GraphAMP/models/GraphAMP.graph"
    
    os.makedirs("recommendation-models/GraphAMP/models", exist_ok=True)
    joblib.dump(seed, file_name, compress=3)
    
    A = nx.to_numpy_array(seed, weight="weight")
    plt.imshow(A, cmap='turbo', interpolation='nearest')
    plt.colorbar(label='Weight')
    
    plt.title("GraphAMP Seed Heatmap")
    plt.savefig("GraphAMP_seed_heatmap.png")
    plt.close()