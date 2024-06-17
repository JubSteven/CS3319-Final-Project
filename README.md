# Leveraging Node Features for Edge-level Graph Augmentation

> Pengxiang Zhu, Yizhou Liu and Benhao Huang, Shanghai Jiao Tong University

This repository contains the code for the final project of CS3319 in Spring 2024. We have explored different methods that targets edge-level graph augmentation. 


## Problem Statement

Edge-level graph augmentation involves manipulating edges to improve the performance of a node classification model. After exploring a series of existing methods, including GAUG, TopoInf, and AdaEdge, we present two methods that leverage node properties. We explore the effect of node communities and pseudo labels and evaluate the effectiveness of our approaches on the Kaggle platform.

Edge rewiring is a widely adopted augmentation technique used to modify the graph structure while keeping the nodes $\mathcal{V}$ unchanged, thereby facilitating training. In this technique, we aim to add or delete specific edges in order to construct an augmented graph $\hat{\mathcal{G}}=(\mathcal{V},\hat{\mathcal{E}})$, where the updated adjacency matrix is defined as follows, with $\mathbf{R}_{ij}$ denoting the rewiring location indicator
$$
\hat{\mathbf{A}}=\mathbf{A}\circ (1-\mathbf{R})+(1-\mathbf{A})\circ \mathbf{R}
$$

For this task, we will only consider deleting the edges, as it is more commonly used.

## Code Structure

`vgae`: Contains the code for the Variational Graph Autoencoder (VGAE) model.

`supervised.py`: The main code for training the VGAE model and generating the augmented graph.

`utils.py`: Contains utility functions for loading the dataset and training.

`run.py`: The main script to run the code.