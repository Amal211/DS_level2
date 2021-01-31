
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
    
    
    
    
#BFS

graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}

visited = [] # List to keep track of visited nodes.
queue = []     #Initialize a queue

def bfs(visited, graph, node):
  visited.append(node)
  queue.append(node)

  while queue:
    s = queue.pop(0) 
    print (s, end = " ") 

    for neighbour in graph[s]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

# Driver Code
bfs(visited, graph, 'A')



#DFS

# Using a Python dictionary to act as an adjacency list
graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}

visited = set() # Set to keep track of visited nodes.

def dfs(visited, graph, node):
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

# Driver Code
dfs(visited, graph, 'A')



#Dijkstra's algorithm

import sys

# Function to find out which of the unvisited node 
# needs to be visited next
def to_be_visited():
  global visited_and_distance
  v = -10
  # Choosing the vertex with the minimum distance
  for index in range(number_of_vertices):
    if visited_and_distance[index][0] == 0 \
      and (v < 0 or visited_and_distance[index][1] <= \
      visited_and_distance[v][1]):
        v = index
  return v

# Creating the graph as an adjacency matrix
vertices = [[0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]]
edges =  [[0, 3, 4, 0],
          [0, 0, 0.5, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0]]

number_of_vertices = len(vertices[0])

# The first element of the lists inside visited_and_distance 
# denotes if the vertex has been visited.
# The second element of the lists inside the visited_and_distance 
# denotes the distance from the source.
visited_and_distance = [[0, 0]]
for i in range(number_of_vertices-1):
  visited_and_distance.append([0, sys.maxsize])

for vertex in range(number_of_vertices):
  # Finding the next vertex to be visited.
  to_visit = to_be_visited()
  for neighbor_index in range(number_of_vertices):
    # Calculating the new distance for all unvisited neighbours
    # of the chosen vertex.
    if vertices[to_visit][neighbor_index] == 1 and \
     visited_and_distance[neighbor_index][0] == 0:
      new_distance = visited_and_distance[to_visit][1] \
      + edges[to_visit][neighbor_index]
      # Updating the distance of the neighbor if its current distance
      # is greater than the distance that has just been calculated
      if visited_and_distance[neighbor_index][1] > new_distance:
        visited_and_distance[neighbor_index][1] = new_distance
  # Visiting the vertex found earlier
  visited_and_distance[to_visit][0] = 1

i = 0 

# Printing out the shortest distance from the source to each vertex       
for distance in visited_and_distance:
  print("The shortest distance of ",chr(ord('a') + i),\
  " from the source vertex a is:",distance[1])
  i = i + 1
    
    
    
#A* Search Algorithm

from collections import deque

class Graph:
    # example of adjacency list (or rather map)
    # adjacency_list = {
    # 'A': [('B', 1), ('C', 3), ('D', 7)],
    # 'B': [('D', 5)],
    # 'C': [('D', 12)]
    # }

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function with equal values for all nodes
    def h(self, n):
        H = {
            'A': 1,
            'B': 1,
            'C': 1,
            'D': 1
        }

        return H[n]

    def a_star_algorithm(self, start_node, stop_node):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None
   

