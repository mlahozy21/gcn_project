"""
Training and evaluation pipeline for GCN.

Hyperparameters from Kipf & Welling (2017):
    - Learning rate: 0.01
    - Weight decay (L2 regularization on first layer weights): 5e-4
    - Dropout: 0.5
    - Hidden units: 16
    - Epochs: 200
    - Early stopping: patience of 10 epochs on validation loss
"""

import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import GCN, MLP, DeepGCN


def train_epoch(model, optimizer, features, adj, labels, idx_train):
    """Run one training epoch."""
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    loss = F.nll_loss(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    # Training accuracy
    preds = output[idx_train].argmax(dim=1)
    acc = (preds == labels[idx_train]).float().mean().item()

    return loss.item(), acc


@torch.no_grad()
def evaluate(model, features, adj, labels, idx):
    """Evaluate model on given indices."""
    model.eval()
    output = model(features, adj)
    loss = F.nll_loss(output[idx], labels[idx]).item()
    preds = output[idx].argmax(dim=1)
    acc = (preds == labels[idx]).float().mean().item()
    return loss, acc


def train_gcn(features, adj, labels, idx_train, idx_val, idx_test,
              n_hidden=16, dropout=0.5, lr=0.01, weight_decay=5e-4,
              epochs=200, patience=10, verbose=True):
    """
    Train a 2-layer GCN with the paper's hyperparameters.

    Args:
        features: Node features (N x F)
        adj: Normalized adjacency (sparse tensor)
        labels: Node labels (N,)
        idx_train, idx_val, idx_test: Split indices
        n_hidden: Hidden layer size (default: 16)
        dropout: Dropout rate (default: 0.5)
        lr: Learning rate (default: 0.01)
        weight_decay: L2 regularization (default: 5e-4)
        epochs: Max training epochs (default: 200)
        patience: Early stopping patience (default: 10)
        verbose: Print progress (default: True)

    Returns:
        dict with test_acc, val_acc, train_acc, train_losses, val_losses,
        training_time, epochs_trained
    """
    n_features = features.shape[1]
    n_classes = labels.max().item() + 1

    model = GCN(n_features, n_hidden, n_classes, dropout)

    # L2 regularization on first layer WEIGHTS only (Section 5.2, 6.1 of paper)
    # Not applied to biases or second layer parameters
    no_decay = [p for name, p in model.named_parameters()
                if not (name == "gc1.weight")]
    optimizer = optim.Adam([
        {"params": [model.gc1.weight], "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, optimizer, features, adj, labels, idx_train
        )
        val_loss, val_acc = evaluate(model, features, adj, labels, idx_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    training_time = time.time() - t_start

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _, test_acc = evaluate(model, features, adj, labels, idx_test)
    _, val_acc = evaluate(model, features, adj, labels, idx_val)
    _, train_acc = evaluate(model, features, adj, labels, idx_train)

    if verbose:
        print(f"\nTraining time: {training_time:.2f}s")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")

    return {
        "test_acc": test_acc,
        "val_acc": val_acc,
        "train_acc": train_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "training_time": training_time,
        "epochs_trained": len(train_losses),
    }


def train_deep_gcn(features, adj, labels, idx_train, idx_val, idx_test,
                   n_layers=2, n_hidden=16, dropout=0.5, lr=0.01,
                   weight_decay=5e-4, epochs=200, patience=10, verbose=False):
    """
    Train a GCN with variable depth for over-smoothing experiments.

    Same as train_gcn but uses DeepGCN with configurable number of layers.
    """
    n_features = features.shape[1]
    n_classes = labels.max().item() + 1

    model = DeepGCN(n_features, n_hidden, n_classes, n_layers, dropout)

    # L2 regularization on first layer WEIGHTS only
    no_decay = [p for name, p in model.named_parameters()
                if not (name == "layers.0.weight")]
    optimizer = optim.Adam([
        {"params": [model.layers[0].weight], "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, features, adj, labels, idx_train)
        val_loss, _ = evaluate(model, features, adj, labels, idx_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _, test_acc = evaluate(model, features, adj, labels, idx_test)

    if verbose:
        print(f"  {n_layers} layers -> Test Acc: {test_acc:.4f}")

    return test_acc


def train_mlp(features, adj, labels, idx_train, idx_val, idx_test,
              n_hidden=16, dropout=0.5, lr=0.01, weight_decay=5e-4,
              epochs=200, patience=10, verbose=False):
    """
    Train a 2-layer MLP baseline (no graph structure).

    Uses the same hyperparameters as the GCN for a fair comparison.
    Corresponds to the "Multi-layer perceptron" row in Table 3 of the paper.
    """
    n_features = features.shape[1]
    n_classes = labels.max().item() + 1

    model = MLP(n_features, n_hidden, n_classes, dropout)

    # Same weight decay strategy: first layer weights only
    no_decay = [p for name, p in model.named_parameters()
                if not (name == "fc1.weight")]
    optimizer = optim.Adam([
        {"params": [model.fc1.weight], "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, features, adj, labels, idx_train)
        val_loss, _ = evaluate(model, features, adj, labels, idx_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _, test_acc = evaluate(model, features, adj, labels, idx_test)

    if verbose:
        print(f"  MLP -> Test Acc: {test_acc:.4f}")

    return test_acc
