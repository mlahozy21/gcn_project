# GCN: Semi-Supervised Classification with Graph Convolutional Networks

Implementation from scratch of the GCN model from Kipf and Welling (ICLR 2017) for the GRMDIL course project.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
gcn_project/
├── data.py           # Dataset downloading and loading (Cora, Citeseer, Pubmed)
├── model.py          # GCN layer, GCN model, MLP baseline, DeepGCN
├── train.py          # Training and evaluation pipeline
├── experiments.py    # Over-smoothing experiments (varying depth)
├── main.py           # Reproduce paper results + MLP baseline comparison
├── requirements.txt
└── README.md
```

## Usage

### Reproduce paper results (GCN + MLP baseline on all datasets)
```bash
python main.py
```

### Run on a single dataset
```bash
python main.py --dataset cora
```

### Run over-smoothing experiments
```bash
python experiments.py
```

## Experiments

1. **Reproduction**: GCN on Cora, Citeseer, Pubmed — compare with Table 2 of the paper.
2. **MLP baseline**: Same architecture without graph structure — compare with Table 3. Shows the contribution of graph convolutions.
3. **Over-smoothing**: GCN with 2 to 64 layers — shows performance degradation with depth.

## Reference

Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017. https://arxiv.org/abs/1609.02907
