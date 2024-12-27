# LLM Centrality Function: Eigenvector Centrality Adapted for DiGraph
def LLM_centrality(flight_network, airport, max_iter=100, tol=1e-6):
   
    N = len(flight_network) # Initialize the number of nodes in the graph
    centrality = {node: 1.0 / N for node in flight_network} # Initialize centrality scores for all nodes equally

    # Iterate to refine centrality scores
    for _ in range(max_iter):
        prev_centrality = centrality.copy()

        # Calculate centrality considering incoming edges for directed graph
        for node in flight_network:
            centrality[node] = sum(
                prev_centrality[neighbor] * flight_network[neighbor][node]['weight']
                for neighbor in flight_network.predecessors(node)
            )

        # Normalization step
        norm = max(sum(centrality.values()), 1e-10) # Avoid division by zero
        for node in centrality:
            centrality[node] /= norm

        # Convergence check
        if max(abs(centrality[node] - prev_centrality[node]) for node in flight_network) < tol:
            break

    return centrality[airport]


'''
This module introduces a new centrality metric adapted for directed graphs, based on eigenvector centrality. 
It computes the importance of an airport by considering the weighted incoming edges and normalizes the values 
to ensure consistency. This metric allows for a deeper analysis of the role of airports in the network, providing 
a basis for comparison with traditional centrality measures. The implementation is tailored to highlight critical 
nodes in a weighted directed graph, such as a flight network.
'''