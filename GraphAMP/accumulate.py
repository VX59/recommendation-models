import networkx as nx
import joblib
import matplotlib.pyplot as plt
import numpy as np

def main():
    library_file_name = f"recommendation-models/GraphAMP/models/GraphAMP.graph"
    session_file_name = f"recommendation-models/GraphAMP/models/GraphAMP_session.graph"

    library_graph:nx.DiGraph = joblib.load(library_file_name)
    session_graph:nx.DiGraph = joblib.load(session_file_name)

    lr = 1e-2


    for u,v in session_graph.edges:
        lib_weight = library_graph[u][v]["weight"]
        session_weight = session_graph[u][v]["weight"]
        updated_weight = max((1-lr)*lib_weight + lr*session_weight,0)

        library_graph[u][v]["weight"] = updated_weight

    for node in library_graph.nodes:
        out_edges = library_graph.out_edges(node, data=True)
        
        total_weight = sum(data["weight"] for _, _, data in out_edges)
        
        if total_weight > 0:
            for _, _, data in out_edges:
                data["weight"] /= total_weight

    joblib.dump(library_graph, library_file_name, compress=3)

    A = nx.to_numpy_array(library_graph, weight="weight", nodelist=sorted(library_graph.nodes()))
    plt.imshow(A, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Weight')
    plt.xticks(ticks=np.arange(len(library_graph.nodes())), labels=sorted(library_graph.nodes()), rotation=90)
    plt.yticks(ticks=np.arange(len(library_graph.nodes())), labels=sorted(library_graph.nodes()))
    
    plt.title(f"GraphAMP Heatmap First Pass, learning rate ({lr})")
    plt.savefig("GraphAMP_heatmap.png")

if __name__ == "__main__":
    main()