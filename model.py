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

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node feature matrix (N x in_features)
            adj: Normalized adjacency matrix (N x N, sparse)

        Returns:
            Output features (N x out_features)
        """
        # XW: (N x in_features) @ (in_features x out_features) = (N x out_features)
        support = torch.mm(x, self.weight)
        # A_hat @ XW: sparse mm
        output = torch.spmm(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return (f"GraphConvolution({self.in_features} -> {self.out_features})")


class GCN(nn.Module):
    """
    Two-layer GCN model as described in Kipf & Welling (2017).

    Architecture:
        Input -> GCN Layer 1 (ReLU + Dropout) -> GCN Layer 2 -> Log-Softmax

    Hyperparameters from the paper:
        - Hidden units: 16
        - Dropout: 0.5
        - Learning rate: 0.01
        - Weight decay (L2 on first layer): 5e-4
    """

    def __init__(self, n_features: int, n_hidden: int, n_classes: int,
                 dropout: float = 0.5):
        super().__init__()
        self.gc1 = GraphConvolution(n_features, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (N x n_features)
            adj: Normalized adjacency (N x N, sparse)

        Returns:
            Log-probabilities (N x n_classes)
        """
        # Dropout on input features (applied before each layer, as in the
        # original tkipf/gcn implementation and confirmed by Appendix B)
        x = F.dropout(x, self.dropout, training=self.training)

        # Layer 1: graph convolution + ReLU
        x = self.gc1(x, adj)
        x = F.relu(x)

        # Dropout before layer 2
        x = F.dropout(x, self.dropout, training=self.training)

        # Layer 2: graph convolution + log-softmax
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    """
    Two-layer Multi-Layer Perceptron baseline (no graph structure).

    This corresponds to the "Multi-layer perceptron" row in Table 3 of
    Kipf & Welling (2017), where the propagation model is simply XΘ
    (no adjacency matrix multiplication). This baseline isolates the
    contribution of the graph structure to classification performance.

    Paper results (Table 3): Citeseer 46.5%, Cora 55.1%, Pubmed 71.4%
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

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor = None) -> torch.Tensor:
        """
        Forward pass. adj is accepted but ignored (for API compatibility).
        """
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DeepGCN(nn.Module):
    """
    GCN with variable depth for over-smoothing experiments.

    All hidden layers have the same number of units.
    ReLU + dropout between each pair of layers.
    """

    def __init__(self, n_features: int, n_hidden: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        assert n_layers >= 2, "DeepGCN requires at least 2 layers"
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # First layer: input -> hidden
        self.layers.append(GraphConvolution(n_features, n_hidden))

        # Intermediate hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GraphConvolution(n_hidden, n_hidden))

        # Last layer: hidden -> output
        self.layers.append(GraphConvolution(n_hidden, n_classes))

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        # Dropout on input features
        x = F.dropout(x, self.dropout, training=self.training)

        # All layers except last: GCN + ReLU + Dropout
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # Last layer: GCN + log-softmax
        x = self.layers[-1](x, adj)
        return F.log_softmax(x, dim=1)
