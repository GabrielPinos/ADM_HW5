# Algorithmic Methods for Data Mining - Homework 5 ✈️

This is a GitHub repository created to submit the fifth homework of the **Algorithmic Methods for Data Mining (ADM)** course for the MSc. in Data Science at the Sapienza University of Rome.

If you can't visualize the notebook click here!


[Link for the notebook](https://nbviewer.org/github/GabrielPinos/ADM_HW5/blob/main/Final.ipynb)



---

## Overview

This project is an in-depth analysis of the USA Airport Dataset, which details air travel networks connecting cities across the United States. Through this assignment, we aim to uncover insights into airport connectivity, flight routes, and passenger traffic patterns. By applying advanced techniques in data visualization, network analysis, and optimization, this analysis explores:

- Identification of the busiest airport hubs.
- Flight route efficiency and passenger flow.
- Centrality and importance of airports within the network.
- Optimal route determination between cities.
- Community detection and airline network partitioning.

This work leverages Python with libraries such as NetworkX, Matplotlib, and PySpark, where necessary, to handle large-scale data and implement custom algorithms.

---

## Objectives

1. **Flight Network Analysis**
   - Analyze the graph structure, including size, density, and degree distribution.
   - Identify hubs and evaluate the graph's sparsity/density.
   - Visualize key metrics and generate a summary report.
   - Explore passenger flow and the efficiency of flight routes.

2. **Centrality Measures**
   - Compute betweenness, closeness, degree centrality, and PageRank for all airports.
   - Compare centrality measures and interpret their significance.
   - Evaluate alternative centrality metrics and their insights.

3. **Optimal Flight Routes**
   - Develop an algorithm to find the most efficient flight routes between cities, considering multiple airport connections and flight availability on specific dates.

4. **Airline Network Partitioning**
   - Solve the graph disconnection problem to divide the network into two disconnected subgraphs with minimal edge removal.

5. **Community Detection**
   - Use graph-based algorithms to identify communities of interconnected cities.
   - Visualize and analyze the detected communities.

6. **Bonus: Connected Components on MapReduce**
   - Implement a custom MapReduce algorithm using PySpark to find connected components in the network.
   - Compare the results with those obtained using GraphFrames.

7. **Algorithmic Question**
   - Solve a constrained shortest path problem to find the cheapest route between cities within a specified number of stops.

---

## Repository Structure

```
.
|-- flight_network_maps_html/            # Folder with the html files of the map visualization
|   |-- flight_communities.html/         # ChatGPT flight communities detection
|   |-- flight_communites_oriented.html/ # Flight communities detection
|   |-- flight_network_oriented.html/    # Flight partition
|   |-- Top_500_Routes_Map.html/         # Top 500 routes
|   |-- washington_to_newyork.html/      # Best Routes between Washington and NewYork
|   |-- newyork_to_eugene.html/          # Best Routes between NewYork and Eugene
|-- functions/                           # Folder with the functions
|   |-- MyFuncQ1.py/                     # All the functions for question 1
|   |-- MyFuncQ2.py/                     # All the functions for question 2
|   |-- MyFuncQ2LLM.py/                  # Function created by LLM
|   |-- flight_network_partition.py/     # Graph disconnection algorithms
|   |-- flight_network_communities.py/   # Community detection methods
|-- main.ipynb/                          # Jupyter notebooks with the complete analysis
|-- algorithm.ipynb/                     # Jupyter notebooks with the solution to the question about the algorithm
|-- BonusQuestion.ipynb/                 # Jupyter notebooks with the answer to bonus question
|-- Routes_with_Coordinates.csv          # File with coordinates used in Question for map display
|-- README.md                            # This file
```

---

## Dataset

The dataset is publicly available on Kaggle and includes:

- Airport information
- Routes and connections
- Passenger traffic data

[Download Dataset](https://www.kaggle.com/datasets/flashgordon/usa-airport-dataset)

---

## Key Features

### Flight Network Analysis
- Compute graph metrics: number of nodes and edges, density, in/out degree distributions.
- Identify hubs using the 90th percentile threshold.
- Evaluate busiest and most efficient flight routes by passenger flow.
- Visualize the flight network with an interactive geographic map.

### Centrality Measures
- Analyze airport significance using betweenness, closeness, degree centrality, and PageRank.
- Compare centrality distributions and identify top airports.
- Explore alternative centrality measures suggested by LLMs.

### Optimal Flight Routes
- Implement a route optimization algorithm for finding the shortest path based on flight distance and availability.
- Handle complex city-airport relationships with multiple possible connections.

### Community Detection
- Detect and visualize communities in the flight network.
- Analyze inter-city relationships and their impact on airline operations.
- Compare community detection methods and outcomes.

### MapReduce Connected Components
- Identify connected components in the network using a custom MapReduce algorithm.
- Benchmark against GraphFrames for scalability and efficiency.

---

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**: NetworkX, Matplotlib, Plotly, PySpark, GraphFrames
- **Visualization**: Matplotlib, Plotly, Geopandas
- **Collaboration**: Jupyter Notebooks, GitHub

---


## Visualizations

The following HTML files demonstrate the map visualizations for the flight network:

- [Top_500_Routes_Map](flight_network_maps_html/Top_500_Routes_Map.html)  
  [Download](https://github.com/GabrielPinos/ADM_HW5/raw/main/flight_network_maps_htmlTop_500_Routes_Map.html)
- [washington_to_newyork](flight_network_maps_html/washington_to_newyork.html)  
  [Download](https://github.com/GabrielPinos/ADM_HW5/raw/main/flight_network_maps_html/washington_to_newyork.html)
- [newyork_to_eugene](flight_network_maps_html/newyork_to_eugene.html)  
  [Download](https://github.com/GabrielPinos/ADM_HW5/raw/main/flight_network_maps_html/newyork_to_eugene.html)
- [Flight Communities Detection](flight_network_maps_html/flight_communities.html)  
  [Download](https://github.com/GabrielPinos/ADM_HW5/raw/main/flight_network_maps_html/flight_communities.html)
- [Oriented Flight Communities Detection](flight_network_maps_html/flight_communites_oriented.html)  
  [Download](https://github.com/GabrielPinos/ADM_HW5/raw/main/flight_network_maps_html/flight_communites_oriented.html)
- [Flight Network Partition](flight_network_maps_html/flight_network_oriented.html)  
  [Download](https://github.com/GabrielPinos/ADM_HW5/raw/main/flight_network_maps_html/flight_network_oriented.html)

You can download these files and open them in a browser to explore the visualizations interactively.

---
## Interactive Visualization of Flight Networks

https://github.com/user-attachments/assets/51b32538-172e-432a-9955-8e2367cc182a


---
## References

1. [Kaggle - USA Airport Dataset](https://www.kaggle.com/datasets/flashgordon/usa-airport-dataset)
2. [Community Detection in Graphs](https://www.analyticsvidhya.com/blog/2020/04/community-detection-graphs-networks/)
3. [NetworkX Documentation](https://networkx.org/documentation/stable/)


**Author:** Gabriel Pinos, Fabrizio Ferrara 

**Email:** [pinos.1965035@studenti.uniroma1.it], [fabrizioferrara17@gmail.com]

*MSc. in Data Science, Sapienza University of Rome*

---

## Team

- **Gabriel Pinos - 1965035** - [pinos.1965035@studenti.uniroma1.it](mailto:pinos.1965035@studenti.uniroma1.it)
- **Fabrizio Ferrara -2207087** - [fabrizioferrara17@gmail.com](mailto:fabrizioferrara17@gmail.com)
