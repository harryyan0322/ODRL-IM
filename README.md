# MPGNN

Offline deep reinforcement learning for influence maximization under the
weighted-cascade independent-cascade model.

This repository contains the reproducible model and evaluation implementation
used by the paper. It intentionally excludes graph data, offline replay
buffers, checkpoints, baseline implementations, plotting scripts, collection
utilities, and experiment logs. Those assets are required only for the private
experimental pipeline and are not part of the code release.

## Protocol

The released implementation follows the locked experiment protocol:

- undirected input edge lists are expanded to both directed diffusion edges;
- edge probability is `1 / in_degree(target)` after expansion;
- node features are in-degree, out-degree, PageRank, and seed availability;
- seed budget is `max(1, floor(rate * |V|))`;
- influence coverage is the final activated-node fraction, including seeds;
- formal runs use five independent training seeds and 1,000 IC simulations.

The paper architecture is exposed by `PaperActor` and `PaperCritic` in
`MPGNN/Model.py`. `--paper-architecture true` selects the three-layer,
four-head GAT representation and two-layer 128-wide GCN heads used by the
formal configuration. The ATAC objective is implemented in
`MPGNN/atac/atac.py`.

## Environment

The experiments were run with the `zhihao` conda environment. Install the
dependencies listed in `MPGNN/requirements-inference.txt` and install a
compatible PyTorch Geometric build for the selected PyTorch/CUDA version.

## Running a supplied manifest

From the repository root, provide a graph and an offline-data manifest. Paths
inside a manifest may be absolute or relative to that manifest:

```bash
cd MPGNN
python main.py --data-config /path/to/manifest.json \
  --graph-path /path/to/graph.txt --dataset wiki_vote --budget 10 \
  --paper-architecture true --gpu-id 0
```

The evaluator is invoked separately with a checkpoint and the same graph. It
records the seed set, diffusion configuration, MC trials, coverage, confidence
interval, and device in its JSON output.

## Tests

Use the provided wrapper so imports resolve consistently from a clean checkout:

```bash
./run_tests.sh
```

The wrapper uses `zhihao` when it is available and otherwise falls back to the
current Python interpreter.

## Scope of the release

Only core model, replay-buffer, graph/evaluator, partition-inference, ATAC,
tests, and documentation files should be committed. The root `.gitignore`
contains the exclusion policy for private data and experiment assets.
