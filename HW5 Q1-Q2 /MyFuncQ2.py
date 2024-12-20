from collections import defaultdict, deque
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Degree Centrality Function
def degree_centrality(flight_network, airport):
    return flight_network.out_degree(airport)

# Closeness Centrality Function
def closeness_centrality(flight_network, airport):
    distances = bfs_shortest_paths(flight_network, airport)
    reachable_nodes = [dist for dist in distances.values() if dist != float('inf')]
    
    if len(reachable_nodes) <= 1:
        return 0
    
    reachable = sum(reachable_nodes)
    return (len(reachable_nodes) - 1) / reachable

# Betweenness Centrality Function
def betweenness_centrality(flight_network, airport):
    total_paths = 0
    passing_paths = 0
    
    for src in flight_network:
        if src == airport:
            continue
        
        paths, parents = bfs_paths_and_parents(flight_network, src)
        
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
                prev_rank[neighbor] * flight_network[node][neighbor]['weight'] /
                max(sum(flight_network[neighbor][n]['weight'] for n in flight_network[neighbor]), 1e-10)
                for neighbor in flight_network[node]
            )
            rank[node] = (1 - d) / N + d * weighted_sum

        if max(abs(rank[node] - prev_rank[node]) for node in flight_network) < tol:
            break

    return rank[airport]


# Analyze Centrality for a Given Airport
def analyze_centrality(flight_network, airport):
    return {
        "Airport": airport,
        "Degree Centrality": degree_centrality(flight_network, airport),
        "Closeness Centrality": closeness_centrality(flight_network, airport),
        "Betweenness Centrality": betweenness_centrality(flight_network, airport),
        "PageRank": page_rank(flight_network, airport),
    }

# Compare Centrality Metrics Across Airports
def compare_centralities(flight_network, target_airports):
    from tqdm import tqdm
    from IPython.display import display

    print("Calculating Centralities...")
    centralities = [
        analyze_centrality(flight_network, airport) 
        for airport in tqdm(target_airports, desc="Calculating Centrality")
    ]

    results_df = pd.DataFrame(centralities)

    for col in ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality", "PageRank"]:
        plt.figure(figsize=(8, 5))
        plt.bar(results_df["Airport"], results_df[col], color='skyblue', edgecolor='black')
        plt.title(f"{col} Distribution")
        plt.xlabel("Airport")
        plt.ylabel(col)
        plt.show()

    print("\nTop 5 Airports by Centrality:")
    for col in ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality", "PageRank"]:
        top_5 = results_df.nlargest(5, col)[["Airport", col]]
        display(top_5)

    return results_df


# BFS Shortest Paths for Closeness Centrality
def bfs_shortest_paths(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


# BFS Paths and Parents for Betweenness Centrality
def bfs_paths_and_parents(graph, start):
    paths = defaultdict(int)
    parents = defaultdict(list)
    distances = {node: float('inf') for node in graph}
    paths[start] = 1
    distances[start] = 0

    queue = deque([start])

    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

            if distances[neighbor] == distances[current] + 1:
                paths[neighbor] += paths[current]
                parents[neighbor].append(current)

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