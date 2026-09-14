# Data tools

This directory contains public data instructions and a generator. It does not
contain downloaded graphs, replay buffers, checkpoints, or baseline source.
Those files are ignored by Git and must be generated locally.

## Public graphs

The six graphs used by the experiments and their public sources are listed in
[`datasets.json`](datasets.json). The SNAP and Network Repository downloads
use plain edge lists. KONECT provides the Twitter-follows archive and its
format description.

For the paper protocol, treat an input edge list as an undirected simple graph
unless the experiment explicitly documents a directed source conversion. The
loader removes self-loops, validates consecutive zero-based node IDs, and
expands every stored edge to two directed diffusion edges.

## Offline data

Run the self-contained generator from the repository root:

```bash
python data/generate_offline_data.py \
  --graph-path /data/wiki-Vote.txt \
  --output /data/offline/wiki_vote_random_degree.pkl \
  --dataset wiki_vote \
  --num-samples 10000 \
  --budget-percent 10 \
  --strategies random:0.20,degree:0.40,pagerank:0.40 \
  --seed 42
```

The default collection budget is 100 IC simulations per transition, matching
the manuscript protocol. Evaluation remains a separate 1,000-trial operation.
The strategy weights are the offline-data composition; increasing the
`random` weight directly implements the random-sample ablation. The generator
stores compact `mpgnn_compact_v1` shards and writes a sidecar JSON containing
the graph protocol, strategy counts, seed, budget, and MC count.

The generator's open policies are deliberately simple and reproducible:
uniform random, out-degree, and PageRank. To reproduce learned behavior
sources (DeepIM, GCOMB, or PIANO), obtain the original implementations from
their paper/project pages and export each policy's selected seed sets in the
same format; the private adapters used for the authors' experiments are not
included in this code release.

Useful public starting points for the learned sources are the project/paper
search pages for [DeepIM](https://github.com/search?q=DeepIM+influence+maximization&type=repositories),
[GCOMB](https://github.com/idea-iitd/GCOMB), and
[PIANO](https://github.com/search?q=PIANO+influence+maximization&type=repositories).
The adapter below accepts the seed-set JSON exported by any faithful original
implementation.

```bash
python data/build_offline_from_seed_sets.py \
  --graph-path /data/wiki-Vote.txt \
  --seed-sets /tmp/deepim_seed_sets.json \
  --output /data/offline/deepim.pkl \
  --dataset wiki_vote \
  --budget-percent 10 \
  --mc-trials 100
```

## Manifest

After producing one or more shards, create a manifest consumed by
`MPGNN/offline_data.py`:

```json
{
  "version": 1,
  "name": "wiki_vote_mix",
  "dataset": "wiki_vote",
  "graph_path": "/data/wiki-Vote.txt",
  "budget_percent": 10,
  "total_samples": 10000,
  "sampling_seed": 42,
  "sources": [
    {"name": "random", "path": "/data/offline/random.pkl", "weight": 0.2},
    {"name": "degree", "path": "/data/offline/degree.pkl", "weight": 0.4},
    {"name": "pagerank", "path": "/data/offline/pagerank.pkl", "weight": 0.4}
  ]
}
```

Keep manifests and generated files outside the repository or under an ignored
local data directory.
