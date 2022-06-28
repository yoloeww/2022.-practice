import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
G = nx.random_graphs.barabasi_albert_graph(n=6, m=3) #随机生成一个图
"""
edge_index = np.load("edge_index.npy")
edge_w = np.load("edge_w.npy")
"""
G = nx.Graph()
        # 添加带权边
G.add_edge('a', 'b', weight=0.6)
G.add_edge('a', 'c', weight=0.2)
G.add_edge('c', 'd', weight=0.1)
G.add_edge('c', 'f', weight=0.9)
G.add_edge('c', 'e', weight=0.7)
G.add_edge('a', 'd', weight=0.3)
pos = nx.random_layout(G)
weights = nx.get_edge_attributes(G, "weight")
nx.draw_networkx(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
plt.show()

plt.axis('off')
plt.savefig("weighted_graph.png")  # save as png
plt.show()  # display
