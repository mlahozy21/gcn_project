"""
Over-smoothing experiments.

Studies how GCN performance degrades as the number of layers increases.
This is a well-known limitation: deeper GCNs cause node representations
to converge to similar values, losing discriminative power.

Reference:
    Li, Q., Han, Z., & Wu, X. M. (2018). Deeper insights into graph
    convolutional networks for semi-supervised learning. AAAI 2018.
"""

import matplotlib.pyplot as plt
import numpy as np

from data import load_dataset, normalize_adjacency
from train import train_deep_gcn


DATASETS = ["cora", "citeseer", "pubmed"]
LAYER_COUNTS = [2, 4, 8, 16, 32, 64]
N_RUNS = 5  # average over multiple runs for stability


def run_oversmoothing_experiment(data_dir="data"):
    """
    Train GCN models with varying depth on all datasets.
    Reports mean and std of test accuracy over N_RUNS.
    """
    results = {}

    for dataset_name in DATASETS:
        print(f"\n{'='*50}")
        print(f"Over-smoothing experiment: {dataset_name}")
        print(f"{'='*50}")

        adj, features, labels, idx_train, idx_val, idx_test = load_dataset(
            dataset_name, data_dir
        )
        adj_norm = normalize_adjacency(adj)

        dataset_results = {}

        for n_layers in LAYER_COUNTS:
            accs = []
            for run in range(N_RUNS):
                acc = train_deep_gcn(
                    features, adj_norm, labels,
                    idx_train, idx_val, idx_test,
                    n_layers=n_layers,
                    n_hidden=16,
                    dropout=0.5,
                    lr=0.01,
                    weight_decay=5e-4,
                    epochs=200,
                    patience=10,
                    verbose=False,
                )
                accs.append(acc)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            dataset_results[n_layers] = (mean_acc, std_acc)
            print(f"  {n_layers:2d} layers: {mean_acc:.4f} +/- {std_acc:.4f}")

        results[dataset_name] = dataset_results

    return results


def plot_oversmoothing(results, save_path="oversmoothing.png"):
    """
    Plot test accuracy vs number of layers for each dataset.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    markers = {"cora": "o", "citeseer": "s", "pubmed": "^"}
    colors = {"cora": "#1f77b4", "citeseer": "#ff7f0e", "pubmed": "#2ca02c"}

    for dataset_name in DATASETS:
        if dataset_name not in results:
            continue

        data = results[dataset_name]
        layers = sorted(data.keys())
        means = [data[l][0] for l in layers]
        stds = [data[l][1] for l in layers]

        ax.errorbar(
            layers, means, yerr=stds,
            marker=markers[dataset_name],
            color=colors[dataset_name],
            label=dataset_name.capitalize(),
            linewidth=2, markersize=8,
            capsize=4,
        )

    ax.set_xlabel("Number of GCN layers", fontsize=13)
    ax.set_ylabel("Test accuracy", fontsize=13)
    ax.set_title("Over-smoothing: Effect of GCN Depth on Performance", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LAYER_COUNTS)
    ax.set_xticklabels([str(l) for l in LAYER_COUNTS])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close()


def print_results_table(results):
    """Print a formatted table of results."""
    print(f"\n{'='*60}")
    print("Over-smoothing Results Summary")
    print(f"{'='*60}")
    print(f"{'Layers':<10}", end="")
    for ds in DATASETS:
        print(f"{ds.capitalize():<20}", end="")
    print()
    print("-" * 60)

    for n_layers in LAYER_COUNTS:
        print(f"{n_layers:<10}", end="")
        for ds in DATASETS:
            if ds in results and n_layers in results[ds]:
                mean, std = results[ds][n_layers]
                print(f"{mean:.4f} +/- {std:.4f}  ", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()


if __name__ == "__main__":
    results = run_oversmoothing_experiment()
    print_results_table(results)
    plot_oversmoothing(results)
