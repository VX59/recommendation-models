from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from database.db import get_session
from database.models import MusiqlRepository, UserLirbary, Models
from s3_service import S3

import networkx as nx
import pickle


async def fetch_library(user_id:str) -> list[MusiqlRepository]:
    stmt = (
        select(MusiqlRepository)
        .select_from(UserLirbary)
        .join(MusiqlRepository, UserLirbary.record_id == MusiqlRepository.uri)
        .where(UserLirbary.user_id == user_id)
    ).order_by(MusiqlRepository.created.desc())

    session_maker: sessionmaker = get_session()

    async with session_maker() as session:
        result = await session.execute(stmt)
        rows: list[MusiqlRepository] = result.scalars().all()

        return rows


async def update_nodes(
    model: Models,
    s3_service: S3,
):
    obj_key = f"recommendation_models/GAMP/{model.uri}.gamp"
    try:
        file_stream = s3_service.pull_obj_stream(obj_key)
        graph_data = file_stream.read()
        G = pickle.loads(graph_data)

    except Exception:
        G = nx.DiGraph()

    rows: list[MusiqlRepository] = await fetch_library(model.user_id)
    db_uris = {record.uri for record in rows}

    # --- 1. Add new nodes ---
    new_nodes = set()

    for uri in db_uris:
        if uri not in G:
            G.add_node(uri)
            new_nodes.add(uri)


    # --- 2. Remove stale nodes ---
    removed_nodes = 0
    for node in list(G.nodes):
        if node not in db_uris:
            G.remove_node(node)
            removed_nodes += 1

    # --- 3. Initialize edges for new nodes (weak prior only) ---
    new_edges = 0
    nodes = list(G.nodes)

    for i in nodes:
        for j in nodes:

            if not G.has_edge(i, j):
                # only lightly connect new nodes (NOT full dense init)
                if i in new_nodes or j in new_nodes:
                    G.add_edge(i, j, weight=0.05)
                    new_edges += 1

    # --- 4. Light self-reinforcement for new nodes ---
    for uri in new_nodes:
        # helps prevent uniform collapse in normalization
        for neighbor in G.successors(uri):
            G[uri][neighbor]["weight"] *= 1.5

    print(f"removed {removed_nodes} nodes")
    print(f"added {len(new_nodes)} new nodes, {new_edges} new edges")

    return G


async def seed_new(
    model:Models,
    s3_service:S3
) -> nx.DiGraph:
    seed: nx.DiGraph = await update_nodes(model, s3_service)
    return seed
