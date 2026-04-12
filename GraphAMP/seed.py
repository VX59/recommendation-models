from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from database.db import get_session
from database.models import MusiqlRepository
from s3_service import S3Service

import networkx as nx
from itertools import product
import pickle

async def fetch_library() -> list[MusiqlRepository]:
    stmt = select(MusiqlRepository)

    session_maker:sessionmaker = get_session()

    async with session_maker() as session:
        result = await session.execute(stmt)
        rows:list[MusiqlRepository] = result.scalars().all()

        return rows


async def GraphAMP_seed():
    obj_key = "recommendation_models/GraphAMP.model"

    s3_service = S3Service.get_s3_service()

    try:
        file_stream = s3_service.pull_obj_stream(obj_key)
        graph_data = file_stream.read()
        G = pickle.loads(graph_data)

    except KeyError:
        G = nx.DiGraph()

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


async def seed_new() -> nx.DiGraph:
    seed:nx.DiGraph = await GraphAMP_seed()
    return seed