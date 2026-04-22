import networkx as nx
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy.orm import sessionmaker

from s3_service import S3Service, get_s3_service
from database.db import get_session
from database.models import ModelUpdates


async def accumulate(library_graph: nx.DiGraph, session_graph: nx.DiGraph):

    lr = 1
    existence_threshold = 1e-4

    for u, v in session_graph.edges:
        session_weight = session_graph[u][v]["weight"]

        if not library_graph.has_edge(u, v):
            if lr * session_weight > existence_threshold:
                library_graph.add_edge(u, v, weight=lr * session_weight)
        else:
            new_weight = max(0, library_graph[u][v]["weight"] + lr * session_weight)
            library_graph[u][v]["weight"] = new_weight

            if new_weight < existence_threshold:
                library_graph.remove_edge(u, v)

    data = pickle.dumps(library_graph)
    obj_key = "recommendation_models/GraphAMP.model"

    s3_service:S3Service = get_s3_service()
    s3_service.put_object(data, obj_key)

    session_maker: sessionmaker = get_session()

    async with session_maker() as session, session.begin():
        session.add(ModelUpdates())

    A = nx.to_numpy_array(library_graph, weight="weight")

    plt.imshow(A, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Weight")

    plt.title(f"GraphAMP Heatmap, lr={lr}, {datetime.now().isoformat()}")
    plt.savefig("GraphAMP_heatmap.png")
    plt.close()
