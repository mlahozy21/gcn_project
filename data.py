"""
Dataset loading utilities for Cora, Citeseer, and Pubmed.
Downloads the Planetoid datasets (Yang et al., 2016) used by Kipf & Welling.
"""

import os
import pickle
import urllib.request

import numpy as np
import scipy.sparse as sp
import torch


BASE_URLS = [
    "https://github.com/kimiyoung/planetoid/raw/master/data/",
    "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/",
]

DATASET_FILES = [
    "x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"
]


def download_dataset(dataset_name: str, data_dir: str = "data") -> str:
    """Download dataset files if they don't exist locally."""
    dataset_name = dataset_name.lower()
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for suffix in DATASET_FILES:
        filename = f"ind.{dataset_name}.{suffix}"
        filepath = os.path.join(dataset_dir, filename)

        if not os.path.exists(filepath):
            downloaded = False
            for base_url in BASE_URLS:
                try:
                    url = base_url + filename
                    print(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                    downloaded = True
                    break
                except Exception:
                    continue
            if not downloaded:
                raise RuntimeError(
                    f"Failed to download {filename}. "
                    f"Please download manually from the Planetoid repository."
                )

    return dataset_dir


def parse_index_file(filename: str) -> list:
    """Parse an index file and return list of integers."""
    indices = []
    with open(filename, "r") as f:
        for line in f:
            indices.append(int(line.strip()))
    return indices


def load_dataset(dataset_name: str, data_dir: str = "data"):
    """
    Load a Planetoid dataset.

    Returns:
        adj: scipy sparse adjacency matrix (N x N)
        features: torch.FloatTensor (N x F)
        labels: torch.LongTensor (N,)
        idx_train: torch.LongTensor - training node indices
        idx_val: torch.LongTensor - validation node indices
        idx_test: torch.LongTensor - test node indices
    """
    dataset_name = dataset_name.lower()
    dataset_dir = download_dataset(dataset_name, data_dir)

    # Load all pickle files
    objects = []
    for suffix in DATASET_FILES[:-1]:  # all except test.index
        filepath = os.path.join(dataset_dir, f"ind.{dataset_name}.{suffix}")
        with open(filepath, "rb") as f:
            objects.append(pickle.load(f, encoding="latin1"))

    x, y, tx, ty, allx, ally, graph = objects

    # Load test indices
    test_index_path = os.path.join(dataset_dir, f"ind.{dataset_name}.test.index")
    test_idx = parse_index_file(test_index_path)
    test_idx_sorted = np.sort(test_idx)

    # Handle Citeseer isolated nodes (some test indices are missing)
    if dataset_name == "citeseer":
        # Citeseer has some isolated test nodes not in the graph.
        # We fill them with zero features and assign a random label.
        test_idx_range_full = range(min(test_idx), max(test_idx) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_sorted - min(test_idx_sorted), :] = tx
        tx = tx_extended.tocsr()

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_sorted - min(test_idx_sorted), :] = ty
        ty = ty_extended

    # Stack features: allx (train+unlabeled) and tx (test)
    features = sp.vstack([allx, tx]).tolil()
    # Reorder test features to match original ordering
    features[test_idx, :] = features[test_idx_sorted, :]
    features = features.toarray()

    # Stack labels
    labels = np.vstack([ally, ty])
    labels[test_idx] = labels[test_idx_sorted]
    labels = np.argmax(labels, axis=1)

    # Build adjacency matrix from graph dict
    num_nodes = features.shape[0]
    adj = build_adjacency(graph, num_nodes)

    # Standard splits from Kipf & Welling:
    # Training: first labeled nodes (20 per class)
    # Validation: indices 200-699 (500 nodes)
    # Test: indices 1000-2707 for Cora (1000 nodes)
    idx_train = list(range(len(y)))  # labeled training nodes
    idx_val = list(range(len(y), len(y) + 500))
    idx_test = test_idx

    # Row-normalize features (as described in Section 5.2 of Kipf & Welling)
    features = row_normalize(features)

    # Convert to torch tensors
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print(f"Dataset: {dataset_name}")
    print(f"  Nodes: {num_nodes}, Features: {features.shape[1]}, "
          f"Classes: {labels.max().item() + 1}")
    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, "
          f"Test: {len(idx_test)}")
    print(f"  Edges: {adj.nnz // 2}")

    return adj, features, labels, idx_train, idx_val, idx_test


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Row-normalize a feature matrix: each row is divided by its L1 norm.
    This is the preprocessing step described in Section 5.2 of Kipf & Welling.
    """
    row_sum = np.array(matrix.sum(axis=1)).flatten()
    row_sum_inv = np.where(row_sum > 0, 1.0 / row_sum, 0.0)
    # Multiply each row by its inverse sum
    return matrix * row_sum_inv[:, np.newaxis]


def build_adjacency(graph_dict: dict, num_nodes: int) -> sp.csr_matrix:
    """Build a symmetric adjacency matrix from an adjacency list dict."""
    edges = []
    for src, neighbors in graph_dict.items():
        for dst in neighbors:
            edges.append((src, dst))

    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    data = np.ones(len(edges))

    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    # Make symmetric (maximum avoids double-counting existing symmetric edges)
    adj = adj.maximum(adj.T)
    return adj.tocsr()


def normalize_adjacency(adj: sp.csr_matrix) -> torch.sparse.FloatTensor:
    """
    Compute the normalized adjacency with self-loops:
        A_hat = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
    where A_tilde = A + I (renormalization trick from Kipf & Welling).

    Returns a torch sparse tensor.
    """
    # Add self-loops: A_tilde = A + I
    num_nodes = adj.shape[0]
    adj_tilde = adj + sp.eye(num_nodes)

    # Compute D_tilde^{-1/2}
    degree = np.array(adj_tilde.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    adj_normalized = D_inv_sqrt @ adj_tilde @ D_inv_sqrt
    adj_normalized = adj_normalized.tocoo()

    # Convert to torch sparse tensor
    indices = torch.LongTensor(
        np.vstack([adj_normalized.row, adj_normalized.col])
    )
    values = torch.FloatTensor(adj_normalized.data)
    shape = torch.Size(adj_normalized.shape)

    return torch.sparse_coo_tensor(indices, values, shape)
