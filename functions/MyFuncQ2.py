from collections import defaultdict, deque
import heapq
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display

# Degree Centrality Function
def degree_centrality(flight_network, airport):
    degree = flight_network.out_degree(airport) 
    total_nodes = len(flight_network) - 1 # Total number of nodes in the graph excluding the current node
    return degree / total_nodes if total_nodes > 0 else 0  # Normalized

# Closeness Centrality Function 
def closeness_centrality(flight_network, airport, wf_improved=True):
    reversed_graph = flight_network.reverse() # Reverse the graph to simulate incoming paths
    distances = dijkstra_distances(reversed_graph, airport)
    reachable_nodes = [dist for dist in distances.values() if dist < float('inf')] # Filter out unreachable nodes

    # If the airport is isolated or has only one connection, return 0
    if len(reachable_nodes) <= 1:
        return 0

    reachable_sum = sum(reachable_nodes)
    n = len(flight_network)  
    reachable = len(reachable_nodes) - 1  # Number of reachable nodes excluding the airport itself

    # Closeness Centrality Standard
    closeness = reachable / reachable_sum

    #Wasserman and Faust improvement
    if wf_improved:
        closeness *= (n - 1) / reachable

    return closeness

# Betweenness Centrality Function 
def betweenness_centrality(flight_network, airport):
    # Initialize total paths and paths passing through the specified airpor
    total_paths = 0
    passing_paths = 0

     # Iterate through all nodes in the graph as potential sources   
    for src in flight_network:
        if src == airport:# Skip the specified airport
            continue

        # Compute shortest paths and parent nodes from the source
        paths, parents = dijkstra_paths_and_parents(flight_network, src)

        for dest in flight_network:
            if dest == airport or dest == src or paths[dest] == 0: # Skip cases where the destination is invalid or the same as the source
                continue

            path_count = count_paths_through_node(dest, airport, parents)
            # Update passing paths and total paths
            passing_paths += path_count
            total_paths += paths[dest]

    if total_paths == 0:
        return 0

    return passing_paths / total_paths #ratio of passing paths to total paths

# PageRank Function
def page_rank(flight_network, airport, d=0.85, max_iter=100, tol=1e-6):
    N = len(flight_network)
    rank = {node: 1 / N for node in flight_network} # Initialize rank values equally among all nodes

    for _ in range(max_iter):
        prev_rank = rank.copy()
        for node in flight_network:
            if flight_network.out_degree(node) == 0: # Handle nodes with no outgoing edges
                rank[node] = (1 - d) / N
                continue

            # Compute the weighted sum of incoming rank contributions    
            weighted_sum = sum(
                prev_rank[neighbor] * flight_network[neighbor][node]['weight'] /
                max(sum(flight_network[neighbor][n]['weight'] for n in flight_network[neighbor]), 1e-10)
                for neighbor in flight_network.predecessors(node)
            )
            rank[node] = (1 - d) / N + d * weighted_sum # Apply the PageRank formula

        # Check for convergence by comparing changes in rank values
        if max(abs(rank[node] - prev_rank[node]) for node in flight_network) < tol:
            break

    total_rank = sum(rank.values())
    return rank[airport] / total_rank if total_rank > 0 else 0 # Normalize the rank of the specified airport

# Compare Centrality Metrics Across Airports
def compare_centralities(flight_network, target_airports):
    
    centralities = [
        analyze_centrality(flight_network, airport)
        for airport in tqdm(target_airports, desc="Calculating Centrality")
    ]

    results_df = pd.DataFrame(centralities) # Store results in a DataFrame

    for col in ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality", "PageRank"]:
        plt.figure(figsize=(12, 6))
        plt.bar(results_df["Airport"], results_df[col], color='skyblue', edgecolor='black')
        plt.title(f"{col} Distribution")
        plt.xlabel("Airport")
        plt.ylabel(col)
        plt.xticks(rotation=90)
        plt.show()

    # Display top 5 airports for each centrality measure
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
    distances = {node: float('inf') for node in graph} # Initialize distances to infinity for all nodes
    distances[start] = 0
    pq = [(0, start)]  # Priority queue

    # Process nodes in the queue
    while pq:
        current_distance, current_node = heapq.heappop(pq) # Get the node with the smallest distance
        if current_distance > distances[current_node]: # Skip nodes if a shorter path was already found
            continue

        # Update distances for neighboring nodes
        for neighbor, attr in graph[current_node].items():
            weight = attr.get('weight', 1)
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

# Dijkstra for Weighted Paths
def dijkstra_paths_and_parents(graph, start):
    # Initialize dictionaries for paths, parents, and distances
    paths = defaultdict(int)
    parents = defaultdict(list)
    distances = {node: float('inf') for node in graph}

    # Initialize the start node
    paths[start] = 1
    distances[start] = 0

    pq = [(0, start)]  ## Initialize a priority queue with the start node
    # Process nodes in the queue

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
    # Initialize a stack for depth-first traversal
    stack = deque([dest])
    path_count = 0

    while stack:
        current = stack.pop()
        if current == node: # If the node is encountered, count the path
            path_count += 1
        else:
            stack.extend(parents[current]) # Add parent nodes to the stack

    return path_count

'''
This module contains functions to analyze and evaluate the flight network's centrality and structure. 
It offers tools to compute key centrality metrics such as Degree Centrality, Closeness Centrality, 
Betweenness Centrality, and PageRank. The module also provides advanced pathfinding algorithms based 
on Dijkstra's method to calculate shortest paths and distances. By comparing centrality measures and 
highlighting critical nodes, this module facilitates a deep understanding of the network's connectivity 
and the role of specific airports. The functions are designed to analyze and visualize the importance 
and influence of nodes within the network.
'''
