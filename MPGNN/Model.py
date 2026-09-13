import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.distributions import Normal


# from nocd.utils import to_sparse_tensor


def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor],
                     device,
                     ) -> Union[torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor]:
    """Convert a scipy sparse matrix to a torch sparse tensor.

    Args:
        matrix: Sparse matrix to convert.
        cuda: Whether to move the resulting tensor to GPU.

    Returns:
        sparse_tensor: Resulting sparse tensor (on CPU or on GPU).

    """
    if sp.issparse(matrix):
        coo = matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    elif torch.is_tensor(matrix):
        row, col = matrix.nonzero().t()
        indices = torch.stack([row, col])
        values = matrix[row, col]
        shape = torch.Size(matrix.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    else:
        raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
    sparse_tensor = sparse_tensor.to(device)
    return sparse_tensor.coalesce()

def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor,
                      torch.cuda.sparse.FloatTensor)):  # 判断x是否是torch.sparse.FloatTensor型或torch.cuda.sparse.FloatTensor型
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.cuda.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)

# 问题1q网络和策略网络结构是否是一样的
# 问题2结构一样为何输入不一样，是否应该统一输入
# Q网络
def _graph_conv(backbone, input_dim, output_dim, gat_heads=1):
    backbone = str(backbone).lower()
    if backbone == 'gcn':
        return GCNConv(input_dim, output_dim)
    if backbone == 'gat':
        return GATConv(
            input_dim,
            output_dim,
            heads=int(gat_heads),
            concat=False,
        )
    if backbone == 'sage':
        return SAGEConv(input_dim, output_dim)
    raise ValueError('Unknown GNN backbone: {}'.format(backbone))


class _GraphStack(nn.Module):
    """Configurable message-passing stack shared by actor and critic."""

    def __init__(self, input_dim, hidden_dim, output_dim, depth=3,
                 backbone='gcn', gat_heads=1):
        super().__init__()
        depth = int(depth)
        if depth < 1:
            raise ValueError('GNN depth must be at least 1')
        dimensions = [int(input_dim)]
        if depth > 1:
            dimensions.extend([int(hidden_dim)] * (depth - 1))
        dimensions.append(int(output_dim))
        self.layers = nn.ModuleList([
            _graph_conv(
                backbone,
                dimensions[index],
                dimensions[index + 1],
                gat_heads=gat_heads,
            )
            for index in range(depth)
        ])
        self.depth = depth
        self.backbone = str(backbone).lower()
        self.gat_heads = int(gat_heads)

    def forward_with_hidden(self, x, edge_index):
        hidden_states = []
        value = x
        batch_shape = None
        if value.dim() == 3:
            batch_size, node_count, _ = value.shape
            batch_shape = (batch_size, node_count)
            value = value.reshape(batch_size * node_count, value.shape[-1])
        for index, layer in enumerate(self.layers):
            value = layer(value, edge_index)
            if index + 1 < len(self.layers):
                value = torch.relu(value)
                if batch_shape is None:
                    hidden_states.append(value)
                else:
                    hidden_states.append(value.reshape(
                        batch_shape[0], batch_shape[1], value.shape[-1]
                    ))
        if batch_shape is not None:
            value = value.reshape(
                batch_shape[0], batch_shape[1], value.shape[-1]
            )
        return value, hidden_states

    def forward(self, x, edge_index):
        output, _ = self.forward_with_hidden(x, edge_index)
        return output


class GNNQFunction(_GraphStack):
    """Graph neural network Q-function."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes,
                 depth=3, backbone='gcn', gat_heads=1):
        del num_nodes  # Kept in the signature for legacy checkpoints/configs.
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            depth=depth,
            backbone=backbone,
            gat_heads=gat_heads,
        )

# 策略网络
class GCN(_GraphStack):
    """Graph convolutional network."""

    def __init__(self, input_dim, hidden_dim, output_dim, depth=3,
                 backbone='gcn', gat_heads=1):
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            depth=depth,
            backbone=backbone,
            gat_heads=gat_heads,
        )


class NodeMLP(nn.Module):
    """Per-node MLP with the same call signature as the graph stacks.

    The encoder ablation removes message passing while preserving the number
    of layers and hidden width. ``edge_index`` is accepted and ignored so the
    training and checkpoint-evaluation pipelines remain identical.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, depth=3):
        super().__init__()
        depth = int(depth)
        if depth < 1:
            raise ValueError('MLP depth must be at least 1')
        dimensions = [int(input_dim)]
        if depth > 1:
            dimensions.extend([int(hidden_dim)] * (depth - 1))
        dimensions.append(int(output_dim))
        self.layers = nn.ModuleList([
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(depth)
        ])
        self.depth = depth
        self.backbone = 'mlp'
        self.gat_heads = 0

    def forward_with_hidden(self, x, edge_index=None):
        del edge_index
        hidden_states = []
        value = x
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 < len(self.layers):
                value = torch.relu(value)
                hidden_states.append(value)
        return value, hidden_states

    def forward(self, x, edge_index=None):
        output, _ = self.forward_with_hidden(x, edge_index)
        return output


class PaperActor(nn.Module):
    """PDF architecture: 3x4 GAT representation followed by 2-layer GCN."""

    output_range = "unit_interval"

    def __init__(self, input_dim=4, gat_hidden=64, actor_hidden=128,
                 gat_depth=3, gat_heads=4):
        super().__init__()
        # The paper specifies four heads *each producing 64 features* and
        # concatenates the heads.  ``_GraphStack`` uses concat=False for
        # legacy ablations, so a dedicated encoder is required here.
        gat_layers = []
        in_dim = int(input_dim)
        for _ in range(int(gat_depth)):
            gat_layers.append(
                GATConv(in_dim, int(gat_hidden), heads=int(gat_heads), concat=True)
            )
            in_dim = int(gat_hidden) * int(gat_heads)
        self.gat_layers = nn.ModuleList(gat_layers)
        self.gat_output_dim = in_dim
        self.actor_layers = nn.ModuleList([
            GCNConv(self.gat_output_dim, actor_hidden),
            GCNConv(actor_hidden, 1),
        ])
        self.depth = int(gat_depth) + 2
        self.backbone = "paper_gat3x4_gcn2x128"
        self.gat_heads = int(gat_heads)

    def forward_with_hidden(self, x, edge_index):
        value = x
        hidden = []
        batch_shape = None
        if value.dim() == 3:
            batch_shape = value.shape[:2]
            value = value.reshape(-1, value.shape[-1])
        for layer in self.gat_layers:
            value = torch.relu(layer(value, edge_index))
            hidden.append(value if batch_shape is None else value.reshape(batch_shape[0], batch_shape[1], -1))
        hidden_states = list(hidden)
        for index, layer in enumerate(self.actor_layers):
            value = layer(value, edge_index)
            if index + 1 < len(self.actor_layers):
                value = torch.relu(value)
                hidden_states.append(value if batch_shape is None else value.reshape(batch_shape[0], batch_shape[1], -1))
        value = torch.sigmoid(value)
        if batch_shape is not None:
            value = value.reshape(batch_shape[0], batch_shape[1], -1)
        return value, hidden_states

    def forward(self, x, edge_index):
        value, _ = self.forward_with_hidden(x, edge_index)
        return value


class PaperCritic(nn.Module):
    """Independent two-layer 128-wide GCN critic from the PDF."""

    output_range = "raw"
    expects_state = True

    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([
            GCNConv(int(input_dim), int(hidden_dim)),
            GCNConv(int(hidden_dim), 1),
        ])
        self.depth = 2
        self.backbone = "gcn"
        self.gat_heads = 0

    def forward_with_hidden(self, x, edge_index):
        batch_shape = None
        if x.dim() == 3:
            batch_shape = x.shape[:2]
            x = x.reshape(-1, x.shape[-1])
        hidden = []
        for index, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if index + 1 < len(self.layers):
                x = torch.relu(x)
                hidden.append(x if batch_shape is None else x.reshape(batch_shape[0], batch_shape[1], -1))
        if batch_shape is not None:
            x = x.reshape(batch_shape[0], batch_shape[1], -1)
        return x, hidden

    def forward(self, x, edge_index):
        value, _ = self.forward_with_hidden(x, edge_index)
        return value
