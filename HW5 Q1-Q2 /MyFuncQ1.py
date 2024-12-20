import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import folium

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
    degree_sum = {airport: in_degrees[airport] + out_degrees[airport] for airport in flight_graph.nodes()}
    threshold = pd.Series(list(degree_sum.values())).quantile(0.9)
    hubs = {airport: degree for airport, degree in degree_sum.items() if degree > threshold}

    print(f"\n90th Percentile of Total Degree: {threshold:.2f}")
    print(f"Identified Hubs: {len(hubs)}")

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
    top_routes = passenger_flow.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_routes.apply(lambda x: f"{x['Origin_airport']} → {x['Destination_airport']}", axis=1),
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

# Save Routes to CSV File
def save_routes_to_csv(data, filename="Routes_with_Coordinates.csv"):
    """
    Save routes with geographic coordinates to a CSV file.
    """
    data.to_csv(filename, index=False)
    print(f"CSV File Created: '{filename}'")


# Function to Create an Interactive Map
def create_flight_map(data, top_n=500):
    """
    Create an interactive flight route map.
    """
    # Consider only the top `top_n` routes
    data = data.sort_values(by='Total_Passengers', ascending=False).head(top_n)
    
    # Calculate Central Position
    avg_lat = data[['Org_airport_lat', 'Dest_airport_lat']].mean().mean()
    avg_long = data[['Org_airport_long', 'Dest_airport_long']].mean().mean()
    
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=5)
    
    # Proportional Weight Calculation
    max_passengers = data['Total_Passengers'].max()
    
    # Identify the Top 10 Busiest Routes
    top_10_routes = data.sort_values(by='Total_Passengers', ascending=False).head(10)
    
    # Add Lines to the Map
    for _, row in data.iterrows():
        # Determine Color for the Top 10 Routes
        color = "red" if row['Origin_airport'] in top_10_routes['Origin_airport'].values and \
                       row['Destination_airport'] in top_10_routes['Destination_airport'].values else "blue"
        weight = max(1, (row['Total_Passengers'] / max_passengers) * 10)
        folium.PolyLine(
            locations=[
                [row['Org_airport_lat'], row['Org_airport_long']],
                [row['Dest_airport_lat'], row['Dest_airport_long']]
            ],
            color=color,
            weight=weight,
            opacity=0.7
        ).add_to(m)

    # Add Markers for Airports
    airport_stats = (
        data.groupby("Origin_airport")["Total_Passengers"]
        .sum()
        .reset_index()
        .sort_values(by="Total_Passengers", ascending=False)
        .reset_index(drop=True)
    )
    airport_stats["Rank"] = airport_stats.index + 1

    for _, row in airport_stats.iterrows():
        airport = row['Origin_airport']
        lat = data[data['Origin_airport'] == airport]['Org_airport_lat'].iloc[0]
        long = data[data['Origin_airport'] == airport]['Org_airport_long'].iloc[0]
        folium.Marker(
            location=[lat, long],
            popup=folium.Popup(
                f"Airport: {airport}<br>Rank: {row['Rank']}<br>Total Passengers: {row['Total_Passengers']}",
                max_width=300
            ),
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m)
    
    return m