"""
Main entry point: reproduce the results from Kipf & Welling (2017).

Expected results from the paper:
    Table 2 (GCN):  Cora 81.5%, Citeseer 70.3%, Pubmed 79.0%
    Table 3 (MLP):  Cora 55.1%, Citeseer 46.5%, Pubmed 71.4%

Usage:
    python main.py                    # Run on all datasets
    python main.py --dataset cora     # Run on a single dataset
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data import load_dataset, normalize_adjacency
from train import train_gcn, train_mlp


# Expected results from Kipf & Welling (2017)
PAPER_GCN = {"cora": 0.815, "citeseer": 0.703, "pubmed": 0.790}
PAPER_MLP = {"cora": 0.551, "citeseer": 0.465, "pubmed": 0.714}

N_RUNS = 10  # number of runs for mean/std


def run_single_dataset(dataset_name, data_dir="data"):
    """Run GCN and MLP experiments on a single dataset."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()}")
    print(f"{'='*60}")

    adj, features, labels, idx_train, idx_val, idx_test = load_dataset(
        dataset_name, data_dir
    )
    adj_norm = normalize_adjacency(adj)

    # --- GCN ---
    print(f"\n--- GCN ({N_RUNS} runs) ---")
    gcn_accs = []
    all_train_losses = []
    all_val_losses = []

    for run in range(N_RUNS):
        result = train_gcn(
            features, adj_norm, labels,
            idx_train, idx_val, idx_test,
            n_hidden=16,
            dropout=0.5,
            lr=0.01,
            weight_decay=5e-4,
            epochs=200,
            patience=10,
            verbose=(run == 0),
        )
        gcn_accs.append(result["test_acc"])
        all_train_losses.append(result["train_losses"])
        all_val_losses.append(result["val_losses"])

    gcn_mean = np.mean(gcn_accs)
    gcn_std = np.std(gcn_accs)

    # --- MLP baseline ---
    print(f"\n--- MLP baseline ({N_RUNS} runs) ---")
    mlp_accs = []

    for run in range(N_RUNS):
        acc = train_mlp(
            features, adj_norm, labels,
            idx_train, idx_val, idx_test,
            n_hidden=16,
            dropout=0.5,
            lr=0.01,
            weight_decay=5e-4,
            epochs=200,
            patience=10,
            verbose=(run == 0),
        )
        mlp_accs.append(acc)

    mlp_mean = np.mean(mlp_accs)
    mlp_std = np.std(mlp_accs)

    # --- Summary ---
    paper_gcn = PAPER_GCN.get(dataset_name)
    paper_mlp = PAPER_MLP.get(dataset_name)

    print(f"\n{'='*55}")
    print(f"Results for {dataset_name.upper()} ({N_RUNS} runs):")
    print(f"  {'Model':<10} {'Ours':<22} {'Paper':<10}")
    print(f"  {'-'*42}")
    print(f"  {'GCN':<10} {gcn_mean:.4f} +/- {gcn_std:.4f}   {paper_gcn:.4f}")
    print(f"  {'MLP':<10} {mlp_mean:.4f} +/- {mlp_std:.4f}   {paper_mlp:.4f}")
    print(f"  {'-'*42}")
    print(f"  Graph structure contributes: "
          f"{(gcn_mean - mlp_mean)*100:+.1f} percentage points")
    print(f"{'='*55}")

    # Plot training curves from first GCN run
    plot_training_curves(
        all_train_losses[0], all_val_losses[0],
        dataset_name, save_path=f"training_{dataset_name}.png"
    )

    return {
        "gcn": (gcn_mean, gcn_std),
        "mlp": (mlp_mean, mlp_std),
    }


def plot_training_curves(train_losses, val_losses, dataset_name, save_path):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (NLL)", fontsize=12)
    ax.set_title(f"Training curves - {dataset_name.capitalize()}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce GCN results from Kipf & Welling (2017)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["cora", "citeseer", "pubmed"],
        help="Dataset to run on. If not specified, runs on all three."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory to store/load datasets."
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["cora", "citeseer", "pubmed"]

    all_results = {}
    for ds in datasets:
        all_results[ds] = run_single_dataset(ds, args.data_dir)

    # Final summary table
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print("FINAL SUMMARY")
        print(f"{'='*65}")
        print(f"{'Dataset':<12} {'GCN (ours)':<20} {'GCN (paper)':<14} "
              f"{'MLP (ours)':<20} {'MLP (paper)':<12}")
        print("-" * 65)
        for ds, res in all_results.items():
            gcn_m, gcn_s = res["gcn"]
            mlp_m, mlp_s = res["mlp"]
            print(f"{ds.capitalize():<12} "
                  f"{gcn_m:.4f} +/- {gcn_s:.4f}  {PAPER_GCN[ds]:.4f}         "
                  f"{mlp_m:.4f} +/- {mlp_s:.4f}  {PAPER_MLP[ds]:.4f}")


if __name__ == "__main__":
    main()
