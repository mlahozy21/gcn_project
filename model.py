"""
GCN model implementation from scratch.

Implements the Graph Convolutional Network from:
    "Semi-Supervised Classification with Graph Convolutional Networks"
    Kipf & Welling, ICLR 2017.

The key operation per layer is:
    H^{(l+1)} = sigma( D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H^{(l)} W^{(l)} )

where:
    A_tilde = A + I_N         (adjacency with self-loops)
    D_tilde = diag(sum(A_tilde))  (degree matrix of A_tilde)
    W^{(l)}                   (learnable weight matrix)
    sigma                     (activation function, e.g. ReLU)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Apply dropout to a sparse tensor by masking non-zero values."""
    if not training or p == 0:
        return x
    mask = torch.rand(x._values().size()) > p
    indices = x._indices()[:, mask]
    values = x._values()[mask] / (1 - p)  # scale to preserve expectation
    return torch.sparse_coo_tensor(indices, values, x.size()).coalesce()


class GraphConvolution(nn.Module):
    """
    Single graph convolutional layer.

    Computes: output = A_hat @ input @ W + b
    where A_hat is the normalized adjacency matrix (precomputed).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot uniform (Glorot & Bengio, 2010)."""
        fan_in = self.in_features
        fan_out = self.out_features
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        self.weight.data.uniform_(-limit, limit)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # XW first: reduces dimension before sparse matmul
        # Use sparse.mm if input is sparse (first layer with sparse features)
        if x.is_sparse:
            support = torch.sparse.mm(x, self.weight)
        else:
            support = torch.mm(x, self.weight)
        # A_hat @ XW: sparse-dense matmul
        output = torch.sparse.mm(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return (f"GraphConvolution({self.in_features} -> {self.out_features})")


class GCN(nn.Module):
    """
    Two-layer GCN model as described in Kipf & Welling (2017).

    Returns raw logits (no log_softmax). Use F.cross_entropy as loss.
    """

    def __init__(self, n_features: int, n_hidden: int, n_classes: int,
                 dropout: float = 0.5):
        super().__init__()
        self.gc1 = GraphConvolution(n_features, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            x = sparse_dropout(x, self.dropout, self.training)
        else:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x  # raw logits


class MLP(nn.Module):
    """
    Two-layer MLP baseline (no graph structure). Returns raw logits.
    """

    def __init__(self, n_features: int, n_hidden: int, n_classes: int,
                 dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)
        self.dropout = dropout

        # Use same Glorot initialization as GCN for fair comparison
        for layer in [self.fc1, self.fc2]:
            fan_in = layer.in_features
            fan_out = layer.out_features
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)
            layer.bias.data.zero_()

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        # MLP needs dense input for nn.Linear
        if x.is_sparse:
            x = x.to_dense()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x  # raw logits


class DeepGCN(nn.Module):
    """
    GCN with variable depth for over-smoothing experiments. Returns raw logits.
    """

    def __init__(self, n_features: int, n_hidden: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        assert n_layers >= 2, "DeepGCN requires at least 2 layers"
        self.dropout = dropout
        self.layers = nn.ModuleList()

        self.layers.append(GraphConvolution(n_features, n_hidden))
        for _ in range(n_layers - 2):
            self.layers.append(GraphConvolution(n_hidden, n_hidden))
        self.layers.append(GraphConvolution(n_hidden, n_classes))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            x = sparse_dropout(x, self.dropout, self.training)
        else:
            x = F.dropout(x, self.dropout, training=self.training)
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, adj)
        return x  # raw logits


class ResidualDeepGCN(nn.Module):
    """
    GCN with residual connections (Eq. 14, Appendix B). Returns raw logits.
    """

    def __init__(self, n_features: int, n_hidden: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        assert n_layers >= 2, "ResidualDeepGCN requires at least 2 layers"
        self.dropout = dropout
        self.layers = nn.ModuleList()

        self.layers.append(GraphConvolution(n_features, n_hidden))
        for _ in range(n_layers - 2):
            self.layers.append(GraphConvolution(n_hidden, n_hidden))
        self.layers.append(GraphConvolution(n_hidden, n_classes))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            x = sparse_dropout(x, self.dropout, self.training)
        else:
            x = F.dropout(x, self.dropout, training=self.training)

        # First layer (no residual: dimension change F -> H)
        x = self.layers[0](x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Intermediate layers with residual connections
        for layer in self.layers[1:-1]:
            residual = x
            x = layer(x, adj)
            x = F.relu(x)
            x = x + residual
            x = F.dropout(x, self.dropout, training=self.training)

        # Last layer (no residual: dimension change H -> C)
        x = self.layers[-1](x, adj)
        return x  # raw logits
