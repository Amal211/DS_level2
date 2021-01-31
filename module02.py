
# Software for Complex Networks - NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
# https://networkx.org/documentation/stable/index.html

# Visualize Graphs in Python
# https://www.geeksforgeeks.org/visualize-graphs-in-python/

# Using NetworkX to Plot Graphs
# https://dustinoprea.com/2015/07/25/using-networkx-to-plot-graphs/

import time
import os
os.system("cls")

import warnings                                  
warnings.filterwarnings('ignore')

import networkx as nx 
import matplotlib.pyplot as plt 
from module02_class import GraphVisualization

def get_program_running(start_time):
    end_time = time.clock()
    diff_time = end_time - start_time
    result = time.strftime("%H:%M:%S", time.gmtime(diff_time)) 
    print("program runtime: {}".format(result))

def graph_example1():
    graph = GraphVisualization() 
    graph.add_edge(0, 2) 
    graph.add_edge(1, 2) 
    graph.add_edge(1, 3) 
    graph.add_edge(5, 3) 
    graph.add_edge(3, 4) 
    graph.add_edge(1, 0) 
    graph.visualize_graph("Graph Title Example1", 700, 8) 

def graph_example2():
    g = nx.DiGraph() 
    g.add_edge(2, 3, weight=1)
    g.add_edge(3, 4, weight=5)
    g.add_edge(5, 1, weight=10)
    g.add_edge(1, 3, weight=15) 
    g.add_edge(2, 7, weight=1)
    g.add_edge(13, 6, weight=5)
    g.add_edge(12, 5, weight=10)
    g.add_edge(11, 4, weight=15) 
    g.add_edge(9, 2, weight=1)
    g.add_edge(10, 13, weight=5)
    g.add_edge(7, 5, weight=10)
    g.add_edge(9, 4, weight=15) 
    g.add_edge(10, 3, weight=1)
    g.add_edge(11, 2, weight=5)
    g.add_edge(9, 6, weight=10)
    g.add_edge(10, 5, weight=15) 
    pos = nx.circular_layout(g) 
    edge_labels = { (u,v): d['weight'] for u,v,d in g.edges(data=True) } 
    nx.draw_networkx_nodes(g,pos,node_size=700)
    nx.draw_networkx_edges(g,pos)
    nx.draw_networkx_labels(g,pos)
    nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels) 
    plt.title("Graph Title Example2")
    plt.axis('off') 
    plt.savefig('output.png')
    plt.show()

def graph_example3():
    G = nx.Graph()
    G.add_edge("a", "b", weight=0.6)
    G.add_edge("a", "c", weight=0.2)
    G.add_edge("c", "d", weight=0.1)
    G.add_edge("c", "e", weight=0.7)
    G.add_edge("c", "f", weight=0.9)
    G.add_edge("a", "d", weight=0.3)
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]
    pos = nx.spring_layout(G) 
    nx.draw_networkx_nodes(G, pos, node_size=700)   
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    plt.title("Graph Title Example3")
    plt.axis("off")
    plt.show()

def graph_example_ds630_ct2():
    graph = GraphVisualization() 
    graph.add_edge("Dammam", "Riyadh") 
    graph.add_edge("Riyadh", "Makkah") 
    graph.add_edge("Makkah", "Jeddah") 
    graph.add_edge("Jeddah", "Madinah") 
    graph.add_edge("Madinah", "Tabuk") 
    graph.add_edge("Makkah", "Al Bahah") 
    graph.add_edge("Al Bahah", "Abha") 
    graph.add_edge("Abha", "Najran") 
    graph.add_edge("Hayil", "Buraydah") 
    graph.add_edge("Buraydah", "Riyadh") 
    graph.visualize_graph("DS630 CT2 Example", 1300, 7) 

def main():
    graph_example1()
    graph_example2()
    graph_example3()
    graph_example_ds630_ct2()
    
if __name__ == '__main__':
    start_time = time.clock()
    main()
    get_program_running(start_time)

