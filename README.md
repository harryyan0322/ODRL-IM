# MPGNN

Offline deep reinforcement learning for influence maximization under the
weighted-cascade independent-cascade model.

## Install

```bash
conda activate odrlim
pip install -r requirements.txt
pip install -r MPGNN/requirements-inference.txt
```

Install the PyTorch Geometric packages matching the PyTorch/CUDA version in
your environment.

## Run evaluation

The repository does not contain graphs, replay buffers, checkpoints, or
baseline code. Download a graph and generate an offline-data manifest with the
tools in [`data/`](data/), then run:

```bash
python MPGNN/main.py \
  --data-config /path/to/manifest.json \
  --graph-path /path/to/graph.txt \
  --dataset wiki_vote \
  --budget 10 \
  --paper-architecture true \
  --gpu-id 0
```

Use `--gpu-id -1` for CPU execution. The formal protocol uses five training
seeds (`11,22,33,44,55`) and 1,000 IC simulations for evaluation.

## Generate data

See [`data/README.md`](data/README.md) for public dataset links and the
protocol-compatible offline-data generator. Generated files stay local and
are ignored by Git.

## Tests

```bash
./run_tests.sh
```
