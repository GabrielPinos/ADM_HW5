import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def analyze_graph_features(flight_network):
    """
    Esegue un'analisi interattiva dettagliata delle caratteristiche del grafo.
    Include grafici e stampe interattive.
    """
    # 1. Numero di nodi e archi
    num_nodes = flight_network.number_of_nodes()
    num_edges = flight_network.number_of_edges()
    print(f"Numero di aeroporti (nodi): {num_nodes}")
    print(f"Numero di rotte (archi): {num_edges}")

    # 2. Calcolo della densità del grafo
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    print(f"Densità del grafo: {density:.4f}")

    # 3. Distribuzione dei gradi
    degrees = dict(flight_network.degree())
    degree_values = list(degrees.values())

    # Istogrammi delle distribuzioni
    plt.figure(figsize=(8, 5))
    plt.hist(degree_values, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribuzione del Grado (Connessioni per Aeroporto)')
    plt.xlabel('Grado (Numero di Connessioni)')
    plt.ylabel('Frequenza')
    plt.show()

    # 4. Identificazione degli hub
    threshold = pd.Series(degree_values).quantile(0.9)
    hubs = {airport: degree for airport, degree in degrees.items() if degree > threshold}
    print(f"\n90° percentile del grado: {threshold:.2f}")
    print(f"Aeroporti identificati come hub: {len(hubs)}")

    # Tabella degli hub
    hub_table = pd.DataFrame(list(hubs.items()), columns=['Aeroporto', 'Grado']).sort_values(by='Grado', ascending=False)
    hub_table.reset_index(drop=True, inplace=True)
    print("\nTabella degli Hub:")
    print(hub_table.to_string(index=False))

    # 5. Classificazione del grafo
    if density > 0.5:
        print("Il grafo è DENSO.")
    else:
        print("Il grafo è SPARSO.")


import pandas as pd
from IPython.display import display

import pandas as pd
from IPython.display import display

def summarize_graph_features(flight_network):
    """
    Crea un report delle caratteristiche principali del grafo senza grafici.
    La tabella degli hub è mostrata in formato orizzontale.
    """
    # 1. Numero di nodi e archi
    num_nodes = flight_network.number_of_nodes()
    num_edges = flight_network.number_of_edges()

    # 2. Calcolo della densità
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1))

    # 3. Distribuzione dei gradi
    degrees = dict(flight_network.degree())
    degree_values = list(degrees.values())

    # 4. Identificazione degli hub
    threshold = pd.Series(degree_values).quantile(0.9)
    hubs = {airport: degree for airport, degree in degrees.items() if degree > threshold}

    # Creazione del report formattato
    print("==== REPORT CARATTERISTICHE DEL GRAFO ====\n")
    print(f"Numero di aeroporti (nodi): {num_nodes}")
    print(f"Numero di rotte (archi): {num_edges}")
    print(f"Densità del grafo: {density:.4f}")
    print(f"\n90° percentile del grado: {threshold:.2f}")
    print(f"Aeroporti identificati come hub: {len(hubs)}")

    # Creazione della tabella degli hub in formato orizzontale
    hub_table = pd.DataFrame(list(hubs.items()), columns=['Aeroporto', 'Grado']).sort_values(by='Grado', ascending=False)
    hub_table.reset_index(drop=True, inplace=True)  # Rimuove l'indice predefinito
    hub_table_transposed = hub_table.T  # Trasponi la tabella
    print("\nTabella degli Hub (in formato orizzontale):")
    display(hub_table_transposed)  # Mostra la tabella trasposta con display()

    # Classificazione finale
    if density > 0.5:
        print("\nIl grafo è DENSO.")
    else:
        print("\nIl grafo è SPARSO.")
    print("\n==== FINE DEL REPORT ====")


import matplotlib.pyplot as plt

def plot_busiest_routes(passenger_flow, top_n=10):
    """
    Visualizza le rotte più trafficate per numero di passeggeri.
    
    Args:
        passenger_flow (pd.DataFrame): DataFrame contenente le colonne 'Origin_airport', 'Destination_airport', e 'Total_passengers'.
        top_n (int): Numero di rotte più trafficate da visualizzare. Default: 10.
    """
    # Seleziona le prime `top_n` rotte più trafficate
    top_routes = passenger_flow.head(top_n)
    
    # Crea il grafico a barre orizzontale
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_routes.apply(lambda x: f"{x['Origin_airport']} → {x['Destination_airport']}", axis=1),
        top_routes['Total_passengers'],
        color='skyblue',
        edgecolor='black'
    )
    plt.title("Le Rotte più Trafficate (per Passeggeri)", fontsize=14)
    plt.xlabel("Totale Passeggeri", fontsize=12)
    plt.ylabel("Rotte", fontsize=12)
    plt.gca().invert_yaxis()  # Inverti l'asse y per avere la rotta più trafficata in cima
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()