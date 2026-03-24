"""
Over-smoothing experiments.

Studies how GCN performance degrades as the number of layers increases,
and how residual connections (Eq. 14 of Appendix B, Kipf & Welling 2017)
can partially mitigate this effect.

Reference:
    Li, Q., Han, Z., & Wu, X. M. (2018). Deeper insights into graph
    convolutional networks for semi-supervised learning. AAAI 2018.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load_dataset, normalize_adjacency
from train import train_deep_gcn, train_residual_deep_gcn


DATASETS = ["cora", "citeseer", "pubmed"]
LAYER_COUNTS = [2, 4, 8, 16]
N_RUNS = 3  # average over multiple runs for stability


def run_oversmoothing_experiment(data_dir="data"):
    """
    Train GCN models (standard and residual) with varying depth on all datasets.
    Reports mean and std of test accuracy over N_RUNS.
    """
    results = {}
    results_residual = {}

    for dataset_name in DATASETS:
        print(f"\n{'='*50}")
        print(f"Over-smoothing experiment: {dataset_name}")
        print(f"{'='*50}")

        adj, features, labels, idx_train, idx_val, idx_test = load_dataset(
            dataset_name, data_dir
        )
        adj_norm = normalize_adjacency(adj)

        dataset_results = {}
        dataset_results_res = {}

        for n_layers in LAYER_COUNTS:
            # Standard GCN
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
            print(f"  {n_layers:2d} layers (standard):  {mean_acc:.4f} +/- {std_acc:.4f}")

            # Residual GCN
            accs_res = []
            for run in range(N_RUNS):
                acc = train_residual_deep_gcn(
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
                accs_res.append(acc)

            mean_res = np.mean(accs_res)
            std_res = np.std(accs_res)
            dataset_results_res[n_layers] = (mean_res, std_res)
            print(f"  {n_layers:2d} layers (residual):  {mean_res:.4f} +/- {std_res:.4f}")

        results[dataset_name] = dataset_results
        results_residual[dataset_name] = dataset_results_res

    return results, results_residual


def plot_oversmoothing(results, results_residual=None,
                       save_path="oversmoothing.png"):
    """
    Plot test accuracy vs number of layers for each dataset.
    If results_residual is provided, adds dashed lines for residual models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    colors = {"cora": "#1f77b4", "citeseer": "#ff7f0e", "pubmed": "#2ca02c"}

    for ax, dataset_name in zip(axes, DATASETS):
        if dataset_name not in results:
            continue

        # Standard GCN
        data = results[dataset_name]
        layers = sorted(data.keys())
        means = [data[l][0] for l in layers]
        stds = [data[l][1] for l in layers]

        ax.errorbar(
            layers, means, yerr=stds,
            marker="o",
            color=colors[dataset_name],
            label="Standard GCN",
            linewidth=2, markersize=7,
            capsize=3,
        )

        # Residual GCN
        if results_residual and dataset_name in results_residual:
            data_res = results_residual[dataset_name]
            means_res = [data_res[l][0] for l in layers]
            stds_res = [data_res[l][1] for l in layers]

            ax.errorbar(
                layers, means_res, yerr=stds_res,
                marker="s",
                color=colors[dataset_name],
                linestyle="--",
                label="Residual GCN",
                linewidth=2, markersize=7,
                capsize=3, alpha=0.8,
            )

        ax.set_xlabel("Number of layers", fontsize=11)
        ax.set_ylabel("Test accuracy", fontsize=11)
        ax.set_title(dataset_name.capitalize(), fontsize=13)
        ax.set_xscale("log", base=2)
        ax.set_xticks(LAYER_COUNTS)
        ax.set_xticklabels([str(l) for l in LAYER_COUNTS])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Over-smoothing: Effect of GCN Depth on Performance",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close()


def print_results_table(results, results_residual=None):
    """Print a formatted table of results."""
    print(f"\n{'='*80}")
    print("Over-smoothing Results Summary")
    print(f"{'='*80}")

    for ds in DATASETS:
        print(f"\n  {ds.capitalize()}")
        print(f"  {'Layers':<10} {'Standard':<22} {'Residual':<22}")
        print(f"  {'-'*54}")
        for n_layers in LAYER_COUNTS:
            line = f"  {n_layers:<10}"
            if ds in results and n_layers in results[ds]:
                m, s = results[ds][n_layers]
                line += f"{m:.4f} +/- {s:.4f}      "
            else:
                line += f"{'N/A':<22}"
            if results_residual and ds in results_residual and n_layers in results_residual[ds]:
                m, s = results_residual[ds][n_layers]
                line += f"{m:.4f} +/- {s:.4f}"
            else:
                line += "N/A"
            print(line)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results, results_residual = run_oversmoothing_experiment()
    print_results_table(results, results_residual)
    plot_oversmoothing(results, results_residual,
                       save_path="results/oversmoothing.png")
