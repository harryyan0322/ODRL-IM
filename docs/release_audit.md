# Release Audit

## Alignment confirmed

The current formal protocol and implementation agree on graph expansion,
weighted-cascade probabilities, four-dimensional node features, node-budget
rounding, Leiden-aware inference, five training seeds, and 1,000 external IC
simulations. The protocol gate currently reports `pass` for all 14,380 offline
collection tasks.

## Important scope note

The paper describes a GAT representation followed by independent GCN Actor and
Critic heads. In the current implementation this is represented by
`PaperActor` (GAT representation plus Actor GCN) and independent
`PaperCritic` modules (two-layer GCN over state/action features). The Critic
modules do not share a live parameter object with the Actor's GAT stack. This
is intentional for checkpoint compatibility and must not be described as
cross-network parameter sharing in a manuscript or README.

## Publication boundary

No graph data, replay buffers, checkpoints, baseline source trees, plotting or
data-collection scripts, private logs, estimated results, or reviewer files
belong in the public repository. The release contains executable core code and
tests only; an experiment manifest must be supplied by the user at runtime.

## Verification performed

- 39 unit tests pass in the `zhihao` environment.
- Python byte-compilation passes for core modules and ATAC modules.
- The C++ IC evaluator smoke test returns valid spreads.
- Formal protocol gate: `pass`, 14,380 assets checked, zero errors.
