import torch
from utils import *
from dataset import GraphData
import community as cm
import matplotlib.pyplot as plt


def remove_duplicate_edges(edge_list):
    # Create a set to store unique edges
    unique_edges = set()

    # Iterate through the edge list
    for edge in edge_list:
        # Check if the edge or its reverse is already in the set
        if tuple(edge) not in unique_edges and tuple(reversed(edge)) not in unique_edges:
            # If not, add the edge to the set
            unique_edges.add(tuple(edge))

    # Convert the set back to a list and return it
    return np.array(list(unique_edges))


plt_communities_full = []
num_runs = 7
for i in range(1, num_runs + 1):
    graph_data = GraphData("data\data.pt")
    graph = nx.Graph(graph_data.adj_train)
    partition = cm.best_partition(graph, random_state=42)

    communities_louvain = list(partition.values())
    nb_communities_louvain = np.max(communities_louvain) + 1

    original_edges = graph_data.edge_index.numpy()
    # Transform the edges into a set
    original_edges = set([tuple(edge) for edge in original_edges.T])

    df = pd.read_csv('submission-{}.csv'.format(i))
    edge_arr = df.to_numpy()[:, 1:]
    edge_list = []
    for i in range(edge_arr.shape[0]):
        edge_list.append(edge_arr[i][edge_arr[i] != -1].reshape(2, -1))

    # Get the deleted edges
    deleted_edges = []
    for i in range(len(edge_list)):
        edges = set([tuple(edge) for edge in edge_list[i].T])
        targ_edges = list(original_edges - edges)
        deleted_edges.append(np.array([list(edge) for edge in targ_edges]))

    targ_edges = [remove_duplicate_edges(each) for each in deleted_edges]

    plt_edges = targ_edges[0]
    # Check the community of the edge nodes and count the number of communities
    plt_communities = []
    for edge in plt_edges:
        for node in edge:
            plt_communities.append(partition[node])

    deleted_partition = [0 for _ in range(nb_communities_louvain)]
    for i, community in enumerate(plt_communities):
        deleted_partition[community] += 1
    print(deleted_partition, sum(deleted_partition))

    plt_communities_full.append(plt_communities)

# Plot a histogram of the communities
plt.hist(plt_communities_full, bins=nb_communities_louvain)
plt.xlabel("Community Index")
plt.ylabel("Number of Nodes")
plt.title("Communities of nodes from 600 deleted edges")
plt.legend(["Acc 80.02", "Acc 79.97", "Acc 79.43", "Acc 78.98", "Test", "Acc 80.20", "Current SOTA"])
plt.show()
assert False

# Plot a histogram of the communities
plt.hist(plt_communities, bins=nb_communities_louvain)
plt.xlabel("Community Index")
plt.ylabel("Number of Nodes")
plt.title("Communities of nodes from 600 deleted edges")
plt.show()

# Number of nodes in each community
node_count = np.zeros(nb_communities_louvain)
for k, v in partition.items():
    node_count[v] += 1
print(node_count)

deleted_node_count = np.zeros(nb_communities_louvain)
for community in plt_communities:
    deleted_node_count[community] += 0.5
print(deleted_node_count)

deleted_node_percentage = np.zeros(nb_communities_louvain)
for i in range(node_count.shape[0]):
    deleted_node_percentage[i] = (deleted_node_count[i] / node_count[i]) * 100

# Plot the percentage of nodes deleted from each community
plt.bar(range(nb_communities_louvain), deleted_node_percentage)
plt.xlabel("Community Index")
plt.ylabel("Percentage of Nodes Deleted (%)")
plt.title("Percentage of nodes deleted from each community")
plt.show()