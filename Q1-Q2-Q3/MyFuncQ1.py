import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import folium
from IPython.display import display

# Function to Analyze the Directed Graph
def analyze_graph_features(flight_graph):
    """
    Analyze the directed graph.
    """
    num_nodes = flight_graph.number_of_nodes()
    num_edges = flight_graph.number_of_edges()

    print(f"Number of Airports (Nodes): {num_nodes}")
    print(f"Number of Routes (Edges): {num_edges}")

    # Calculate Density (Formula for Directed Graphs)
    density = num_edges / (num_nodes * (num_nodes - 1))
    print(f"Graph Density: {density:.4f}")

    # In-Degree and Out-Degree Distribution
    in_degrees = dict(flight_graph.in_degree())
    out_degrees = dict(flight_graph.out_degree())

    # In-Degree Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(in_degrees.values(), bins=30, color='skyblue', edgecolor='black')
    plt.title('In-Degree Distribution (Incoming Connections)')
    plt.xlabel('In-Degree')
    plt.ylabel('Frequency')
    plt.show()

    # Out-Degree Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(out_degrees.values(), bins=30, color='lightgreen', edgecolor='black')
    plt.title('Out-Degree Distribution (Outgoing Connections)')
    plt.xlabel('Out-Degree')
    plt.ylabel('Frequency')
    plt.show()

    # Identify Hubs (Sum of In-Degree and Out-Degree)
    # Calculate the total degree (sum of in-degree and out-degree) for each node.
    degree_sum = {airport: in_degrees[airport] + out_degrees[airport] for airport in flight_graph.nodes()} 
    threshold = pd.Series(list(degree_sum.values())).quantile(0.9)
    # Filter airports with a total degree greater than the 90th percentile (considered hubs).
    hubs = {airport: degree for airport, degree in degree_sum.items() if degree > threshold}

    print(f"\n90th Percentile of Total Degree: {threshold:.2f}")
    print(f"Identified Hubs: {len(hubs)}")

    # Create a table with the identified hubs and their total degrees.
    hub_table = pd.DataFrame(list(hubs.items()), columns=['Airport', 'Degree']).sort_values(by='Degree', ascending=False)
    hub_table.reset_index(drop=True, inplace=True)
    print("\nHub Table:")
    print(hub_table.to_string(index=False))

    # Graph Classification
    print("The graph is DENSE." if density > 0.5 else "The graph is SPARSE.")

# Summary Report Function
def summarize_graph_features(flight_graph):
    """
    Generate a report on the directed graph features.
    """
    num_nodes = flight_graph.number_of_nodes()
    num_edges = flight_graph.number_of_edges()

    # Graph Density Calculation (Directed Graphs)
    density = num_edges / (num_nodes * (num_nodes - 1))

    # Degree Distribution
    in_degrees = dict(flight_graph.in_degree())
    out_degrees = dict(flight_graph.out_degree())

    # Identify Hubs
    degree_sum = {airport: in_degrees[airport] + out_degrees[airport] for airport in flight_graph.nodes()}
    threshold = pd.Series(list(degree_sum.values())).quantile(0.9)
    hubs = {airport: degree for airport, degree in degree_sum.items() if degree > threshold}

    # Generate Report
    print("==== GRAPH FEATURE REPORT ====\n")
    print(f"Number of Airports (Nodes): {num_nodes}")
    print(f"Number of Routes (Edges): {num_edges}")
    print(f"Graph Density: {density:.4f}")
    print(f"\n90th Percentile of Total Degree: {threshold:.2f}")
    print(f"Identified Hubs: {len(hubs)}")

    # Hub Table in Horizontal Format
    hub_table = pd.DataFrame(list(hubs.items()), columns=['Airport', 'Degree']).sort_values(by='Degree', ascending=False)
    hub_table.reset_index(drop=True, inplace=True)
    hub_table_transposed = hub_table.T
    print("\nHub Table (Transposed):")
    display(hub_table_transposed)

    # Final Classification
    print("The graph is DENSE." if density > 0.5 else "The graph is SPARSE.")
    print("\n==== END OF REPORT ====")


# Function to Plot the Busiest Routes
def plot_busiest_routes(passenger_flow, top_n=10):
    """
    Display the busiest routes by passenger count.
    """
    # Extract the top `top_n` busiest routes based on total passengers
    top_routes = passenger_flow.head(top_n)

    plt.figure(figsize=(10, 6))
    # Create a horizontal bar chart
    plt.barh(
        top_routes.apply(lambda x: f"{x['Origin_airport']} → {x['Destination_airport']}", axis=1), # Format each route as "Origin → Destination"
        top_routes['Total_Passengers'],
        color='skyblue',
        edgecolor='black'
    )
    plt.title("Busiest Routes (by Passengers)", fontsize=14)
    plt.xlabel("Total Passengers", fontsize=12)
    plt.ylabel("Routes", fontsize=12)
    plt.gca().invert_yaxis()  # Invert y-axis
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# Import Required Libraries
import pandas as pd
import folium

def save_routes_to_csv(data, filename="Routes_with_Coordinates.csv", top_n=500):
    """
    Save routes with geographic coordinates to a CSV file.
    """
    # Sort and filter the top_n routes by Total_Passengers
    data = data.sort_values(by='Total_Passengers', ascending=False).head(top_n)
    
    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"CSV File Created: '{filename}' with {len(data)} rows")


def create_flight_map(data, top_n=500):
    """
    Create an interactive flight route map.
    """
    # Sort and filter the top_n routes by Total_Passengers
    data = data.sort_values(by='Total_Passengers', ascending=False).head(top_n)
    
    # Calculate Central Position
    avg_lat = data[['Org_airport_lat', 'Dest_airport_lat']].mean().mean()
    avg_long = data[['Org_airport_long', 'Dest_airport_long']].mean().mean()
    
    # Create a folium map centered at the average latitude and longitude
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=5)
    
    # Find the maximum number of passengers to normalize the line thickness
    max_passengers = data['Total_Passengers'].max()
    
    # Identify the Top 10 Busiest Routes
    top_10_routes = data.sort_values(by='Total_Passengers', ascending=False).head(10)
    
    # Add Lines to the Map
    for _, row in data.iterrows():
        # Set the line color to red for the top 10 busiest routes and blue for others
        color = "red" if row['Origin_airport'] in top_10_routes['Origin_airport'].values and \
                       row['Destination_airport'] in top_10_routes['Destination_airport'].values else "blue"
        # Set the line thickness proportional to the number of passengers on the route
        weight = max(1, (row['Total_Passengers'] / max_passengers) * 10)

        # Add the route as a polyline to the map
        folium.PolyLine(
            locations=[
                [row['Org_airport_lat'], row['Org_airport_long']], #Start
                [row['Dest_airport_lat'], row['Dest_airport_long']] # End
            ],
            color=color,
            weight=weight,
            opacity=0.7
        ).add_to(m)

    # Group the data by 'Origin_airport' and sum the total passengers for each airport
    airport_stats = (
        data.groupby("Origin_airport")["Total_Passengers"]
        .sum()
        .reset_index()
        .sort_values(by="Total_Passengers", ascending=False)
        .reset_index(drop=True)
    )
    airport_stats["Rank"] = airport_stats.index + 1

    # Add markers for each airport
    for _, row in airport_stats.iterrows():
        airport = row['Origin_airport']
        lat = data[data['Origin_airport'] == airport]['Org_airport_lat'].iloc[0]
        long = data[data['Origin_airport'] == airport]['Org_airport_long'].iloc[0]

        # Change color for top 10 airports
        color = "red" if row["Rank"] <= 10 else "green"

        # Add a marker to the map for the airport
        folium.Marker(
            location=[lat, long],
            popup=folium.Popup(  # Information displayed when clicking on the marker
                f"Airport: {airport}",
                max_width=300
            ),
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)
    
    return m
'''
This module contains functions to analyze and visualize the flight network. 
It includes tools for calculating graph properties like density and degree distribution, 
as well as identifying hubs. Additionally, it provides functionalities for creating bar 
charts of the busiest routes and interactive maps with geographic coordinates. 
These features help understand and visually communicate the network's characteristics.
'''