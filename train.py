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

from model import GCN, MLP, DeepGCN, ResidualDeepGCN
from data import DEVICE


def train_epoch(model, optimizer, features, adj, labels, idx_train):
    """Run one training epoch. Returns loss and accuracy."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = model(features, adj)
    loss = F.cross_entropy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    # Training accuracy
    preds = output[idx_train].argmax(dim=1)
    acc = (preds == labels[idx_train]).float().mean().item()

    return loss.item(), acc


def train_epoch_fast(model, optimizer, features, adj, labels, idx_train):
    """Run one training epoch. Returns loss only (no accuracy computation)."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = model(features, adj)
    loss = F.cross_entropy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, features, adj, labels, idx):
    """Evaluate model on given indices."""
    model.eval()
    output = model(features, adj)
    loss = F.cross_entropy(output[idx], labels[idx]).item()
    preds = output[idx].argmax(dim=1)
    acc = (preds == labels[idx]).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate_loss_only(model, features, adj, labels, idx):
    """Evaluate model, returning only loss (faster: skips accuracy)."""
    model.eval()
    output = model(features, adj)
    return F.cross_entropy(output[idx], labels[idx]).item()


def _setup_optimizer_first_layer(model, first_layer_weight_name, lr, weight_decay):
    """Create Adam optimizer with L2 regularization on first layer weights only."""
    first_weight = dict(model.named_parameters())[first_layer_weight_name]
    no_decay = [p for name, p in model.named_parameters()
                if name != first_layer_weight_name]
    return optim.Adam([
        {"params": [first_weight], "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)


def train_gcn(features, adj, labels, idx_train, idx_val, idx_test,
              n_hidden=16, dropout=0.5, lr=0.01, weight_decay=5e-4,
              epochs=200, patience=10, verbose=True):
    """
    Train a 2-layer GCN with the paper's hyperparameters.
    """
    n_features = features.shape[1]
    n_classes = labels.max().item() + 1

    model = GCN(n_features, n_hidden, n_classes, dropout).to(DEVICE)
    optimizer = _setup_optimizer_first_layer(model, "gc1.weight", lr, weight_decay)

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


def _train_deep_model(model_class, features, adj, labels,
                      idx_train, idx_val, idx_test,
                      n_layers=2, n_hidden=16, dropout=0.5, lr=0.01,
                      weight_decay=5e-4, epochs=200, patience=10,
                      verbose=False):
    """
    Generic training function for deep GCN models (standard or residual).

    Uses fast training path: skips accuracy computation during training,
    only computes validation loss for early stopping.
    """
    n_features = features.shape[1]
    n_classes = labels.max().item() + 1

    model = model_class(n_features, n_hidden, n_classes, n_layers, dropout).to(DEVICE)
    optimizer = _setup_optimizer_first_layer(model, "layers.0.weight", lr, weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Fast path: no accuracy computation
        train_epoch_fast(model, optimizer, features, adj, labels, idx_train)
        val_loss = evaluate_loss_only(model, features, adj, labels, idx_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            # Save best model
            best_model_state = {k: v.detach().clone()
                                for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_epoch > 0:
        model.load_state_dict(best_model_state)

    _, test_acc = evaluate(model, features, adj, labels, idx_test)

    if verbose:
        label = "residual" if model_class == ResidualDeepGCN else "standard"
        print(f"  {n_layers} layers ({label}) -> Test Acc: {test_acc:.4f} "
              f"(best epoch: {best_epoch})")

    return test_acc


def train_deep_gcn(features, adj, labels, idx_train, idx_val, idx_test,
                   n_layers=2, n_hidden=16, dropout=0.5, lr=0.01,
                   weight_decay=5e-4, epochs=200, patience=10, verbose=False):
    """Train a standard DeepGCN with variable depth."""
    return _train_deep_model(
        DeepGCN, features, adj, labels,
        idx_train, idx_val, idx_test,
        n_layers=n_layers, n_hidden=n_hidden, dropout=dropout,
        lr=lr, weight_decay=weight_decay, epochs=epochs,
        patience=patience, verbose=verbose,
    )


def train_residual_deep_gcn(features, adj, labels, idx_train, idx_val, idx_test,
                            n_layers=2, n_hidden=16, dropout=0.5, lr=0.01,
                            weight_decay=5e-4, epochs=200, patience=10,
                            verbose=False):
    """Train a ResidualDeepGCN with variable depth."""
    return _train_deep_model(
        ResidualDeepGCN, features, adj, labels,
        idx_train, idx_val, idx_test,
        n_layers=n_layers, n_hidden=n_hidden, dropout=dropout,
        lr=lr, weight_decay=weight_decay, epochs=epochs,
        patience=patience, verbose=verbose,
    )


def train_mlp(features, adj, labels, idx_train, idx_val, idx_test,
              n_hidden=16, dropout=0.5, lr=0.01, weight_decay=5e-4,
              epochs=200, patience=10, verbose=False):
    """Train a 2-layer MLP baseline (no graph structure)."""
    n_features = features.shape[1]
    n_classes = labels.max().item() + 1

    model = MLP(n_features, n_hidden, n_classes, dropout).to(DEVICE)
    optimizer = _setup_optimizer_first_layer(model, "fc1.weight", lr, weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        train_epoch_fast(model, optimizer, features, adj, labels, idx_train)
        val_loss = evaluate_loss_only(model, features, adj, labels, idx_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.detach().clone()
                                for k, v in model.state_dict().items()}
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
