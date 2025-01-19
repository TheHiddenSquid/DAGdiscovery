import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random



fig, ax = plt.subplots()
G = nx.Graph()


colors = ["red", "green", "blue"]
node_list = [1,2,3,4,5]
edge_list = [(1,2), (2,3), (1,3), (1,4),(4,5)]

for node in node_list:
    G.add_node(node)

for i,j in edge_list:
    G.add_edge(i,j)

color_map = random.choices(population=colors, k=5)

def update(frame):
    fig.clear()
    ax.clear()
    flip = random.choice(node_list)
    current = color_map[flip-1]
    new = current
    while new == current:
        new = random.choice(colors)
    color_map[flip-1] = new

    nx.draw_circular(G, node_color=color_map, with_labels=True)
    


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=50)
plt.show()
