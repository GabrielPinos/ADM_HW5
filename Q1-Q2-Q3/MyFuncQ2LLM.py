# LLM Centrality Function: Eigenvector Centrality Adapted for DiGraph
def LLM_centrality(flight_network, airport, max_iter=100, tol=1e-6):
   
    N = len(flight_network)
    centrality = {node: 1.0 / N for node in flight_network}

    for _ in range(max_iter):
        prev_centrality = centrality.copy()

        # Calculate centrality considering incoming edges for directed graph
        for node in flight_network:
            centrality[node] = sum(
                prev_centrality[neighbor] * flight_network[neighbor][node]['weight']
                for neighbor in flight_network.predecessors(node)
            )

        # Normalization step
        norm = max(sum(centrality.values()), 1e-10)
        for node in centrality:
            centrality[node] /= norm

        # Convergence check
        if max(abs(centrality[node] - prev_centrality[node]) for node in flight_network) < tol:
            break

    return centrality[airport]


'''Il modulo introduce una nuova metrica di centralità adattata per grafi diretti, basata sulla centralità autovettoriale. 
Calcola l’importanza di un aeroporto considerando i pesi degli archi entranti e normalizza i valori per evitare discrepanze.
Questa metrica permette di analizzare ulteriormente il ruolo degli aeroporti nella rete e confrontare i risultati con le centralità 
tradizionali.'''