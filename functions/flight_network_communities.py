import networkx as nx
import folium
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import random
from IPython.display import display

class FlightNetworkCommunitiesOriented:
    def __init__(self):
        self.G = None
        self.communities = {}
        self.partition = None
        self.airport_coords = {}
        self.community_colors = {}
        
    def _fast_betweenness(self, graph):
        """
        Betweenness centrality calculation exploring all nodes
        """
        betweenness = defaultdict(float)
        nodes = list(graph.nodes())
        
        for s in nodes:  # Explore all nodes as sources
            # BFS from source
            visited = {s}
            paths = {s: [[s]]}  # Paths to reach each node
            queue = deque([s])
            
            while queue:
                v = queue.popleft()
                for w in graph.neighbors(v):
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
                        paths[w] = [p + [w] for p in paths[v]]
                        
                        # Add to betweenness
                        for path in paths[w]:
                            for i in range(len(path) - 1):
                                edge = tuple(sorted([path[i], path[i + 1]]))
                                betweenness[edge] += 1
        
        return betweenness

    def _find_communities(self, G):
        """
        Optimized community detection with handling for small components
        """
        if len(G) <= 2:
            return {node: 0 for node in G.nodes()}
            
        # Work with a copy
        g = G.copy()
        communities = {node: 0 for node in g.nodes()}
        
        max_iterations = max(50, g.number_of_edges() // 2)
        
        for _ in range(max_iterations):
            if g.number_of_edges() == 0:
                break
                
            # Calculate betweenness
            betweenness = self._fast_betweenness(g)
            if not betweenness:
                break
                
            # Remove highest betweenness edge
            max_edge = max(betweenness.items(), key=lambda x: x[1])[0]
            g.remove_edge(*max_edge)
            
            # Update communities
            components = list(nx.connected_components(g))
            for i, component in enumerate(components):
                if len(component) < 3:  # Small community
                    # Merge with the largest neighboring community
                    for node in component:
                        neighbors = list(G.neighbors(node))
                        if neighbors:
                            closest_neighbor = neighbors[0]  # Simplistic choice
                            communities[node] = communities[closest_neighbor]
                else:
                    for node in component:
                        communities[node] = i
        
        return communities

    def detect_and_visualize(self, data_frame, c1=None, c2=None, output_file='flight_communities.html'):
        """
        Detect communities and visualize the network
        """
        # Remove rows with missing coordinates
        df_clean = data_frame.dropna(subset=['Org_airport_lat', 'Org_airport_long', 
                                           'Dest_airport_lat', 'Dest_airport_long'])
        
        # Create graph and store coordinates
        self.G = nx.DiGraph()
        for _, row in df_clean.iterrows():
            origin_city = row['Origin_city']
            destination_city = row['Destination_city']
            self.G.add_edge(origin_city, destination_city)
            
            # Store coordinates
            if origin_city not in self.airport_coords:
                self.airport_coords[origin_city] = {
                    'latitude': row['Org_airport_lat'],
                    'longitude': row['Org_airport_long']
                }
            if destination_city not in self.airport_coords:
                self.airport_coords[destination_city] = {
                    'latitude': row['Dest_airport_lat'],
                    'longitude': row['Dest_airport_long']
                }

        # Detect communities
        undirected_G = self.G.to_undirected()
        self.partition = self._find_communities(undirected_G)

        # Generate colors
        unique_communities = set(self.partition.values())
        colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(len(unique_communities))]
        self.community_colors = {comm_id: color for comm_id, color in zip(unique_communities, colors)}

        # Organize communities
        self.communities = defaultdict(list)
        for city, comm_id in self.partition.items():
            self.communities[comm_id].append(city)
        self.communities = dict(self.communities)

        # Create visualization
        map_object = self._create_map(output_file)

        # Check community membership
        same_community = None
        if c1 and c2:
            if c1 in self.partition and c2 in self.partition:
                same_community = self.partition[c1] == self.partition[c2]
            else:
                print(f"Warning: One or both cities ({c1}, {c2}) not found in the network.")

        return {
            'total_communities': len(self.communities),
            'communities': self.communities,
            'community_membership_check': {
                'city1': c1,
                'city2': c2,
                'in_same_community': same_community
            },
            'map_object': map_object  # Return map object for notebook display
        }
    
    def _create_map(self, output_file):
        """Create and return the visualization with custom plane icons."""
        # Calculate center
        center_lat = np.mean([coords['latitude'] for coords in self.airport_coords.values()])
        center_lon = np.mean([coords['longitude'] for coords in self.airport_coords.values()])

        # Create map
        m = folium.Map(location=[center_lat, center_lon], 
                    zoom_start=4,
                    tiles='cartodbpositron')

        # Add edges
        for edge in self.G.edges():
            origin, dest = edge
            if origin in self.airport_coords and dest in self.airport_coords:
                coords = [
                    [self.airport_coords[origin]['latitude'], self.airport_coords[origin]['longitude']],
                    [self.airport_coords[dest]['latitude'], self.airport_coords[dest]['longitude']]
                ]
                color = self.community_colors[self.partition[origin]]
                
                # Add line
                folium.PolyLine(
                    coords,
                    weight=2,
                    color=color,
                    opacity=0.6
                ).add_to(m)

        # Add airport markers with custom plane icons
        for airport, coords in self.airport_coords.items():
            # Create icon
            icon = folium.Icon(
                color='white',  # Background color
                icon_color=self.community_colors[self.partition[airport]],  # Icon color matches community
                icon='plane',  # Plane icon
                prefix='fa'  # Use FontAwesome icons
            )
            
            # Add marker with custom icon
            folium.Marker(
                location=[coords['latitude'], coords['longitude']],
                icon=icon,
                popup=f"""
                    <div style='text-align:center'>
                        <b>{airport}</b><br>
                        {coords.get('city', 'Unknown City')}<br>
                        Community: {self.partition[airport]}
                    </div>
                """
            ).add_to(m)

        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    border:2px solid grey; z-index:9999; 
                    background-color:white;
                    padding: 10px;
                    font-size: 14px;">
        <p><span style="color:#00FF00;">●</span> Community 1</p>
        <p><span style="color:#0000FF;">●</span> Community 2</p>
        <p><span style="color:#0000FF;"><b>- - -</b></span> Removed Flights</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save and return map object
        m.save(output_file)
        return m
