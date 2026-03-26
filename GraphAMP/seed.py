from sqlalchemy.future import select
from musiql_api.db import async_session
from musiql_api.models import MusiqlRepository
import os
import networkx as nx
from itertools import product
import joblib

async def fetch_library() -> list[MusiqlRepository]:
    stmt = select(MusiqlRepository)

    async with async_session() as session:
        result = await session.execute(stmt)
        rows:list[MusiqlRepository] = result.scalars().all()

        return rows

async def GraphAMP_seed(file_name:str):
    if not os.path.exists(file_name):
        G = nx.DiGraph()
    else:
        G:nx.DiGraph = joblib.load(file_name)

    rows:list[MusiqlRepository] = await fetch_library()

    new_nodes, new_edges = [] , 0

    for record in rows:
        if record.uri not in G.nodes:
            G.add_node(record.uri)
            new_nodes.append(record.uri)

    for i, j in product(G.nodes, repeat=2):
        if i in new_nodes or j in new_nodes and not G.has_edge(i,j):
            G.add_edge(i,j, weight=1)
            new_edges += 1
            
    print(f"added {len(new_nodes)} new nodes, {new_edges} new edges")

    return G


async def seed_new():
    os.makedirs("recommendation-models/GraphAMP/models", exist_ok=True)
    file_name = f"recommendation-models/GraphAMP/models/GraphAMP.graph"

    seed:nx.DiGraph = await GraphAMP_seed(file_name)
    joblib.dump(seed, file_name, compress=3)