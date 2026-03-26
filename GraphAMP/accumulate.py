import networkx as nx
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timezone

def accumulate():
    library_file_name = f"recommendation-models/GraphAMP/models/GraphAMP.graph"
    session_file_name = f"recommendation-models/GraphAMP/models/GraphAMP_session.graph"

    library_graph:nx.DiGraph = joblib.load(library_file_name)
    session_graph:nx.DiGraph = joblib.load(session_file_name)

    lr = 1
    existence_threshold = 1e-4

    library_edges = len(library_graph.edges)

    for u,v in session_graph.edges:
        session_weight = session_graph[u][v]["weight"]

        if not library_graph.has_edge(u,v):
            if lr*session_weight > existence_threshold:
                library_graph.add_edge(u,v, weight=lr*session_weight)
        else:
            new_weight = max(0,library_graph[u][v]["weight"] + lr * session_weight)
            library_graph[u][v]["weight"] = new_weight

            if new_weight < existence_threshold:
                library_graph.remove_edge(u,v)


    session_nodes = len(session_graph.nodes)
    library_nodes = len(library_graph.nodes)

    nodes_modified_amount = float(session_nodes)/float(library_nodes)

    session_edges = len([u for u,v in session_graph.edges if session_graph[u][v]["weight"] != 0])

    edges_modified_amount = float(session_edges)/float(library_edges)

    new_libary_edges = len(library_graph.edges)
    dropped_edges_amount = 1-(float(new_libary_edges)/float(library_edges))

    print(f"updating {session_nodes}/{library_nodes} - {nodes_modified_amount}% of songs in library")
    print(f"updating {session_edges}/{library_edges} - {edges_modified_amount}% of transitions in library")
    print(f"dropped {library_edges-new_libary_edges}/{library_edges} - {dropped_edges_amount}% of transitions in library")

    joblib.dump(library_graph, library_file_name, compress=3)

    now = datetime.now().isoformat()

    try:
        with open("updates.log", "r") as reader:
            old_updates = reader.read()
    except FileNotFoundError:
        old_updates = ""

    with open("updates.log", "w") as writer:
        writer.write(now + "\n")
        writer.write(old_updates)

    A = nx.to_numpy_array(library_graph, weight="weight")

    plt.imshow(A, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Weight')


    plt.title(f"GraphAMP Heatmap, lr={lr}, {now}")
    plt.savefig("GraphAMP_heatmap.png")
    plt.close()