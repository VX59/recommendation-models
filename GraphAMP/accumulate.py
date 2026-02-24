import networkx as nx
import joblib
import matplotlib.pyplot as plt
import numpy as np

def accumulate():
    library_file_name = f"recommendation-models/GraphAMP/models/GraphAMP.graph"
    session_file_name = f"recommendation-models/GraphAMP/models/GraphAMP_session.graph"

    library_graph:nx.DiGraph = joblib.load(library_file_name)
    session_graph:nx.DiGraph = joblib.load(session_file_name)

    lr = 1e-2

    library_edges = len([u for u,v in library_graph.edges if library_graph[u][v]["weight"] > 0])

    for u,v in session_graph.edges:
        session_weight = session_graph[u][v]["weight"]

        if not library_graph.has_edge(u, v):
            library_graph.add_edge(u, v, weight=session_weight)
            continue

        new_weight = (1 - lr) * library_graph[u][v]["weight"] + lr * session_weight

        if new_weight <= 1e-10:
            library_graph.remove_edge(u, v)
        else:
            library_graph[u][v]["weight"] = new_weight

    session_nodes = len(session_graph.nodes)
    library_nodes = len(library_graph.nodes)

    nodes_modified_amount = float(session_nodes)/float(library_nodes)

    session_edges = len([u for u,v in session_graph.edges if session_graph[u][v]["weight"] > 0])

    edges_modified_amount = float(session_edges)/float(library_edges)

    new_libary_edges = len([u for u,v in library_graph.edges if library_graph[u][v]["weight"] > 0])
    dropped_edges_amount = 1-(float(new_libary_edges)/float(library_edges))

    print(f"updating {session_nodes}/{library_nodes} - {nodes_modified_amount}% of songs in library")
    print(f"updating {session_edges}/{library_edges} - {edges_modified_amount}% of transitions in library")
    print(f"dropped {library_edges-new_libary_edges}/{library_edges} - {dropped_edges_amount}% of transitions in library")

    for node in session_graph.nodes:
        out_edges = library_graph.out_edges(node, data=True)
        
        total_weight = sum(data["weight"] for _, _, data in out_edges)
        
        if total_weight > 0:
            for _, _, data in out_edges:
                data["weight"] /= total_weight

    joblib.dump(library_graph, library_file_name, compress=3)

    A = nx.to_numpy_array(library_graph, weight="weight")

    plt.imshow(A, cmap='turbo', interpolation='nearest')
    plt.colorbar(label='Weight')

    plt.title(f"GraphAMP Heatmap Session Pass, learning rate ({lr})")
    plt.savefig("GraphAMP_heatmap.png")
    plt.close()