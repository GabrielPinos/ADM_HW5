import networkx as nx 
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import folium
import random




def find_minimum_edge_cut(G):
    """
    Find the minimum edge cut in a directed graph using Ford-Fulkerson algorithm.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Directed graph
        
    Returns:
    --------
    set: Set of edges that form the minimum cut
    """
    def bfs(graph, source, sink, parent):
        """Find augmenting path using BFS"""
        visited = {node: False for node in graph.nodes()}
        queue = [source]
        visited[source] = True
        
        while queue:
            u = queue.pop(0)
            for v in graph.successors(u):  # Use successors for directed graph
                if not visited[v] and graph[u][v]['capacity'] > graph[u][v]['flow']:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
        
        return visited[sink]

    def ford_fulkerson(graph, source, sink):
        """Implementation of Ford-Fulkerson algorithm"""
        for u, v in graph.edges():
            graph[u][v]['flow'] = 0
        
        parent = {node: None for node in graph.nodes()}
        max_flow = 0
        
        while bfs(graph, source, sink, parent):
            path_flow = float("inf")
            s = sink
            
            while s != source:
                path_flow = min(path_flow, 
                              graph[parent[s]][s]['capacity'] - graph[parent[s]][s]['flow'])
                s = parent[s]
            
            max_flow += path_flow
            
            v = sink
            while v != source:
                u = parent[v]
                graph[u][v]['flow'] += path_flow
                v = parent[v]
        
        return max_flow

    def find_reachable_nodes(graph, source):
        """Find all nodes reachable from source in residual graph"""
        visited = set()
        stack = [source]
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in graph.successors(node):
                    if graph[node][neighbor]['capacity'] > graph[node][neighbor]['flow']:
                        stack.append(neighbor)
        
        return visited

    # Create flow graph (keeping the direction of edges)
    flow_graph = nx.DiGraph()
    for u, v in G.edges():
        flow_graph.add_edge(u, v, capacity=1, flow=0)
    
    min_cut_size = float('inf')
    min_cut_edges = set()
    
    nodes = list(G.nodes())
    source = nodes[0]
    
    for sink in nodes[1:]:
        if nx.has_path(G, source, sink):  # Only consider reachable sinks
            current_graph = flow_graph.copy()
            
            max_flow = ford_fulkerson(current_graph, source, sink)
            
            if max_flow < min_cut_size:
                min_cut_size = max_flow
                
                reachable = find_reachable_nodes(current_graph, source)
                cut_edges = set()
                
                for u in reachable:
                    for v in G.successors(u):
                        if v not in reachable:
                            cut_edges.add((u, v))
                
                min_cut_edges = cut_edges
    
    return min_cut_edges




def find_connected_components(G):
    """
    Find weakly connected components in a directed graph.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Directed graph
        
    Returns:
    --------
    list: List of sets, where each set contains nodes in a connected component
    """
    def dfs(node, visited, component):
        """Depth-first search to find all nodes in a component"""
        visited.add(node)
        component.add(node)
        
        # Check outgoing edges
        for neighbor in G.successors(node):
            if neighbor not in visited:
                dfs(neighbor, visited, component)
        
        # Check incoming edges (for weakly connected components)
        for neighbor in G.predecessors(node):
            if neighbor not in visited:
                dfs(neighbor, visited, component)
    
    visited = set()
    components = []
    
    # Iterate through all nodes
    for node in G.nodes():
        if node not in visited:
            # Start a new component
            current_component = set()
            dfs(node, visited, current_component)
            components.append(current_component)
    
    return components





class FlightNetworkPartitionOriented:
    """
    A class to handle flight network partitioning and visualization.
    """

    def __init__(self):
        """Initialize the FlightNetworkPartition class"""
        self.G_original = None
        self.G = None
        self.cut_edges = None
        self.subgraphs = None
        self.airport_coords = {}
        self.node_colors = {}

    def partition_and_visualize(self, data_frame, output_file='flight_network.html'):
        """
        Create an interactive map visualization of partitioned flight networks using folium.
        
        Parameters:
        -----------
        data_frame : DataFrame
            DataFrame containing flight network data with required columns
        output_file : str, optional
            Name of the output HTML file
            
        Returns:
        --------
        dict: Dictionary containing:
            - removed_flights: list of flights removed to disconnect the graph
            - partition_sizes: sizes of resulting partitions
            - partition_nodes: nodes in each partition
        """
        # Remove rows with missing coordinates
        df_clean = data_frame.dropna(subset=['Org_airport_lat', 'Org_airport_long', 
                                           'Dest_airport_lat', 'Dest_airport_long'])
        
        self.G_original = nx.DiGraph()
        for _, row in df_clean.iterrows():
            self.G_original.add_edge(row['Origin_airport'], row['Destination_airport'])

        # Create a copy for modification
        self.G = self.G_original.copy()

        # Compute the minimum edge cut using our custom function
        self.cut_edges = list(find_minimum_edge_cut(self.G))

        # Remove the edges to partition the graph
        self.G.remove_edges_from(self.cut_edges)

        # For directed graphs, we need to use weakly connected components
        self.subgraphs = find_connected_components(self.G)          
        
        # Create color maps for partitions
        partition_colors = {
            0: '#00FF00',  # Green for partition 1
            1: '#0000FF'   # Blue for partition 2
        }
        
        # Create node color mapping
        self.node_colors = {}
        for idx, subgraph in enumerate(self.subgraphs):
            for node in subgraph:
                self.node_colors[node] = partition_colors[idx]

        # Create airport coordinates dictionary
        for _, row in df_clean.iterrows():
            if row['Origin_airport'] not in self.airport_coords:
                self.airport_coords[row['Origin_airport']] = {
                    'latitude': row['Org_airport_lat'],
                    'longitude': row['Org_airport_long'],
                    'city': row['Origin_city']
                }
            if row['Destination_airport'] not in self.airport_coords:
                self.airport_coords[row['Destination_airport']] = {
                    'latitude': row['Dest_airport_lat'],
                    'longitude': row['Dest_airport_long'],
                    'city': row['Destination_city']
                }

        # Create and save the map
        self._create_map(output_file)

        # Prepare return data
        result = {
            'removed_flights': self.cut_edges,
            'partition_sizes': [len(sg) for sg in self.subgraphs],
            'partition_nodes': [list(sg) for sg in self.subgraphs]
        }

        # Print analysis
        self._print_analysis(result)

        return result
    
    def _create_map(self, output_file):
        """Create and return the folium map."""
        # Calculate map center
        center_lat = np.mean([coords['latitude'] for coords in self.airport_coords.values()])
        center_lon = np.mean([coords['longitude'] for coords in self.airport_coords.values()])

        # Create the map
        m = folium.Map(location=[center_lat, center_lon], 
                    zoom_start=4,
                    tiles='cartodbpositron')

        # Add active flight paths
        for edge in self.G.edges():
            origin = edge[0]
            dest = edge[1]
            coords = [
                [self.airport_coords[origin]['latitude'], self.airport_coords[origin]['longitude']],
                [self.airport_coords[dest]['latitude'], self.airport_coords[dest]['longitude']]
            ]
            color = self.node_colors.get(origin, 'gray')
            folium.PolyLine(
                coords,
                weight=2,
                color=color,
                opacity=0.6
            ).add_to(m)

        # Add removed flight paths (dashed blue lines)
        for edge in self.cut_edges:
            origin = edge[0]
            dest = edge[1]
            coords = [
                [self.airport_coords[origin]['latitude'], self.airport_coords[origin]['longitude']],
                [self.airport_coords[dest]['latitude'], self.airport_coords[dest]['longitude']]
            ]
            folium.PolyLine(
                coords,
                weight=2,
                color='blue',
                opacity=0.8,
                dash_array='10'
            ).add_to(m)

        # Add airports as markers
        for airport, coords in self.airport_coords.items():
            # Create icon
            icon = folium.Icon(
                color='white' if self.node_colors.get(airport) == '#00FF00' else 'white',
                icon_color=self.node_colors.get(airport),
                icon='plane',
                prefix='fa'
            )
            
            # Add marker with custom icon
            folium.Marker(
                location=[coords['latitude'], coords['longitude']],
                icon=icon,
                popup=f"""
                    <div style='text-align:center'>
                        <b>{airport}</b><br>
                        {coords['city']}<br>
                        Partition: {'1' if self.node_colors.get(airport) == '#00FF00' else '2'}
                    </div>
                """
            ).add_to(m)

        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    border:2px solid grey; z-index:9999; 
                    background-color:white;
                    padding: 10px;
                    font-size: 14px;">
        <p><span style="color:#00FF00;">●</span> Partition 1</p>
        <p><span style="color:#0000FF;">●</span> Partition 2</p>
        <p><span style="color:#0000FF;"><b>- - -</b></span> Removed Flights</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save the map
        m.save(output_file)

        # Return the map for rendering in the notebook
        return m

            
    def _print_analysis(self, result):
        """Print the analysis results handling single airport partitions"""
        print("\nNetwork Partition Analysis:")
        print(f"Number of flights removed: {len(self.cut_edges)}")
        if len(self.subgraphs) > 0:
            print(f"Partition 1 size: {len(self.subgraphs[0])} airports")
            print("Partition 1 airports:", sorted(list(self.subgraphs[0])))
        if len(self.subgraphs) > 1:
            print(f"Partition 2 size: {len(self.subgraphs[1])} airports")
            print("Partition 2 airports:", sorted(list(self.subgraphs[1])))
        
        print("\nRemoved flights:")
        for edge in self.cut_edges:
            print(f"  {edge[0]} ↔ {edge[1]}")