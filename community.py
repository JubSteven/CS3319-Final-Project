import torch
import torch_geometric.transforms as T
import networkx as nx
import community
import matplotlib.pyplot as plt
import random

# Load the data
data = torch.load("data/data.pt")
x, y, edge_index, val_mask, train_mask = data.x, data.y, data.edge_index, data.val_mask, data.train_mask

# Create a networkx graph from the edge_index
graph = nx.Graph()
graph.add_edges_from(edge_index.t().tolist())

# Perform community detection using the Louvain algorithm
partition = community.best_partition(graph)

# Choose the number of nodes to sample
num_nodes_to_sample = 500

# Randomly sample a starting node
start_node = random.choice(list(graph.nodes()))

# Perform a breadth-first search to obtain a connected subgraph
subgraph_nodes = set()
queue = [start_node]
while queue and len(subgraph_nodes) < num_nodes_to_sample:
    node = queue.pop(0)
    subgraph_nodes.add(node)
    neighbors = list(graph.neighbors(node))
    random.shuffle(neighbors)  # Randomize the order of neighbors
    for neighbor in neighbors:
        if neighbor not in subgraph_nodes:
            queue.append(neighbor)

# Create a subgraph with the sampled nodes and their edges
subgraph = graph.subgraph(subgraph_nodes)

# Create a dictionary to store the node colors based on the community assignment
node_colors_partition = [partition[node_id] for node_id in subgraph.nodes()]

# Create a dictionary to store the node colors based on the node labels
node_colors_labels = [y[node_id] for node_id in subgraph.nodes()]

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Draw the first subplot with node colors based on partition
pos = nx.spring_layout(subgraph)
ax1.set_title("Community Detection")
nx.draw_networkx(subgraph,
                 pos=pos,
                 with_labels=False,
                 node_color=node_colors_partition,
                 cmap="viridis",
                 ax=ax1,
                 node_size=25)

# Draw the second subplot with node colors based on labels
ax2.set_title("Node Labels")
nx.draw_networkx(subgraph,
                 pos=pos,
                 with_labels=False,
                 node_color=node_colors_labels,
                 cmap="viridis",
                 ax=ax2,
                 node_size=25)

plt.tight_layout()
plt.show()
