from collections import defaultdict, deque
import heapq
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display

# Degree Centrality Function
def degree_centrality(flight_network, airport):
    degree = flight_network.out_degree(airport)
    total_nodes = len(flight_network) - 1
    return degree / total_nodes if total_nodes > 0 else 0  # Normalizzato

# Closeness Centrality Function (dall'ultima versione)
def closeness_centrality(flight_network, airport, wf_improved=True):
    # Usa il grafo invertito per simulare le distanze entranti
    reversed_graph = flight_network.reverse()
    distances = dijkstra_distances(reversed_graph, airport)
    reachable_nodes = [dist for dist in distances.values() if dist < float('inf')]

    if len(reachable_nodes) <= 1:
        return 0

    reachable_sum = sum(reachable_nodes)
    n = len(flight_network)  # Numero totale di nodi nel grafo
    reachable = len(reachable_nodes) - 1  # Numero di nodi raggiungibili (escludendo il nodo stesso)

    # Closeness Centrality Standard
    closeness = reachable / reachable_sum

    # Miglioramento di Wasserman e Faust
    if wf_improved:
        closeness *= (n - 1) / reachable

    return closeness

# Betweenness Centrality Function (dalla versione vecchia)
def betweenness_centrality(flight_network, airport):
    total_paths = 0
    passing_paths = 0

    # Usa Dijkstra per percorsi ponderati
    for src in flight_network:
        if src == airport:
            continue

        paths, parents = dijkstra_paths_and_parents(flight_network, src)

        for dest in flight_network:
            if dest == airport or dest == src or paths[dest] == 0:
                continue

            path_count = count_paths_through_node(dest, airport, parents)
            passing_paths += path_count
            total_paths += paths[dest]

    if total_paths == 0:
        return 0

    return passing_paths / total_paths

# PageRank Function
def page_rank(flight_network, airport, d=0.85, max_iter=100, tol=1e-6):
    N = len(flight_network)
    rank = {node: 1 / N for node in flight_network}

    for _ in range(max_iter):
        prev_rank = rank.copy()
        for node in flight_network:
            if flight_network.out_degree(node) == 0:
                rank[node] = (1 - d) / N
                continue

            weighted_sum = sum(
                prev_rank[neighbor] * flight_network[neighbor][node]['weight'] /
                max(sum(flight_network[neighbor][n]['weight'] for n in flight_network[neighbor]), 1e-10)
                for neighbor in flight_network.predecessors(node)
            )
            rank[node] = (1 - d) / N + d * weighted_sum

        if max(abs(rank[node] - prev_rank[node]) for node in flight_network) < tol:
            break

    total_rank = sum(rank.values())
    return rank[airport] / total_rank if total_rank > 0 else 0

# Compare Centrality Metrics Across Airports
def compare_centralities(flight_network, target_airports):
    print("Calculating Centralities...")
    centralities = [
        analyze_centrality(flight_network, airport)
        for airport in tqdm(target_airports, desc="Calculating Centrality")
    ]

    results_df = pd.DataFrame(centralities)

    for col in ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality", "PageRank"]:
        plt.figure(figsize=(12, 6))
        plt.bar(results_df["Airport"], results_df[col], color='skyblue', edgecolor='black')
        plt.title(f"{col} Distribution")
        plt.xlabel("Airport")
        plt.ylabel(col)
        plt.xticks(rotation=90)
        plt.show()

    print("\nTop 5 Airports by Centrality:")
    for col in ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality", "PageRank"]:
        top_5 = results_df.nlargest(5, col)[["Airport", col]]
        display(top_5)

    return results_df

# Analyze Centrality for a Given Airport
def analyze_centrality(flight_network, airport):
    return {
        "Airport": airport,
        "Degree Centrality": degree_centrality(flight_network, airport),
        "Closeness Centrality": closeness_centrality(flight_network, airport),
        "Betweenness Centrality": betweenness_centrality(flight_network, airport),
        "PageRank": page_rank(flight_network, airport),
    }

# Dijkstra for Weighted Distances
def dijkstra_distances(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # Priority queue

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue

        for neighbor, attr in graph[current_node].items():
            weight = attr.get('weight', 1)
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

# Dijkstra for Weighted Paths
def dijkstra_paths_and_parents(graph, start):
    paths = defaultdict(int)
    parents = defaultdict(list)
    distances = {node: float('inf') for node in graph}
    paths[start] = 1
    distances[start] = 0
    pq = [(0, start)]  # Priority queue

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue

        for neighbor, attr in graph[current_node].items():
            weight = attr.get('weight', 1)
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node]
                parents[neighbor] = [current_node]
                heapq.heappush(pq, (distance, neighbor))
            elif distance == distances[neighbor]:
                paths[neighbor] += paths[current_node]
                parents[neighbor].append(current_node)

    return paths, parents

# Count Paths Passing Through a Specific Node
def count_paths_through_node(dest, node, parents):
    stack = deque([dest])
    path_count = 0

    while stack:
        current = stack.pop()
        if current == node:
            path_count += 1
        else:
            stack.extend(parents[current])

    return path_count