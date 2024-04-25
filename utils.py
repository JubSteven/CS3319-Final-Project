import numpy as np
import torch
import pandas as pd

def to_submission(path="data\data_augmented.pt"):
    data = torch.load(path)
    edge_index = data.edge_index
    edge_index = edge_index.numpy().reshape(1, -1)
    df = pd.DataFrame(edge_index)
    df.insert(0, 'ID', [0])
    df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    to_submission()