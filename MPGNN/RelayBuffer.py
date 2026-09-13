import numpy as np
import graph_utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import Normalizer
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

from offline_data import load_compact_transition_store


def generate_sublists(input_list):
    result = []
    for i in range(1, len(input_list) + 1):
        sub_list = input_list[:i]
        result.append(sub_list)
    return result
    # return {key: buf_arr[idx] for key, buf_arr in self.storage}


class Replay_buffer():
    def __init__(self):
        self.storage = []
        self.n_transitions_stored = 0

    def add(self, data):
        if self.n_transitions_stored == 0:
            self.storage = data
            self.n_transitions_stored += len(data["rewards"])

        else:
            for key in ("obs", "next_obs", "actions", "rewards"):
                self.storage[key] = list(self.storage[key]) + list(data[key])
            if "terminals" in self.storage or "terminals" in data:
                old_terminals = self.storage.get(
                    "terminals", np.zeros(self.n_transitions_stored, dtype=np.float32)
                )
                new_terminals = data.get(
                    "terminals", np.zeros(len(data["rewards"]), dtype=np.float32)
                )
                self.storage["terminals"] = np.concatenate((old_terminals, new_terminals))
            self.n_transitions_stored += len(data["rewards"])

    def sample_transitions(self, batch_size=1):
        # np.random.seed()
        # print("batch_size=", batch_size)
        # print("n_transitions_stored)=", self.n_transitions_stored)
        idx = np.random.randint(0, len(self.storage), size=batch_size)
        print("idx=", idx)
        obs_list = []
        next_obs_list = []
        action_list = []
        reward_list = []
        for i in idx:
            obs_list.append(self.storage["observations"][i])
            next_obs_list.append(self.storage["next_observations"][i])
            action_list.append(self.storage["actions"][i])
            reward_list.append(self.storage["rewards"][i])

        terminal_list = np.repeat(False, batch_size)

        sample_data = {"obs": np.array(obs_list),
                       "next_obs": np.array(next_obs_list),
                       "action": np.array(action_list),
                       "reward": np.array(reward_list),
                       "terminal": np.array(terminal_list),
                       "edge_index": self.storage["edge_index"],
                       "edge_weight": self.storage["edge_weight"],
                       # "embeddings": np.array(embeddings_list),
                       "terminals": np.array([False] * batch_size, dtype=np.float32)}
        return sample_data

    def sample_transitions_option(self, batch_size=1):
        idx = np.random.randint(0, self.n_transitions_stored, size=batch_size)
        obs_list = [self.storage["obs"][index] for index in idx]
        next_obs_list = [self.storage["next_obs"][index] for index in idx]
        action_list = [self.storage["actions"][index] for index in idx]
        reward_list = [self.storage["rewards"][index] for index in idx]
        if "terminals" in self.storage:
            terminal_list = np.asarray(
                self.storage["terminals"], dtype=np.float32
            )[idx]
        else:
            terminal_list = np.zeros(len(reward_list), dtype=np.float32)
        row, col = np.nonzero(self.storage["adj_matrix"])
        return {
            "obs": obs_list,
            "next_obs": next_obs_list,
            "action": action_list,
            "reward": reward_list,
            "terminal": np.asarray(terminal_list, dtype=np.float32),
            "edge_index": torch.tensor([row, col], dtype=torch.long),
        }


class HeterogeneousReplayBuffer:
    """Replay transitions from differently sized community subgraphs."""

    def __init__(self, manifest, seed=0):
        self.entries = []
        self.rng = np.random.default_rng(seed)
        community_entries = manifest["communities"]
        for entry in community_entries:
            mapping = []
            mapping_path = entry.get("mapping_path")
            if mapping_path:
                pairs = []
                with open(mapping_path, "r", encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if not line or line.startswith(("#", "%")):
                            continue
                        global_id, local_id = map(int, line.split()[:2])
                        pairs.append((local_id, global_id))
                pairs.sort()
                mapping = [global_id for _, global_id in pairs]
            for source_name, paths in entry["sources"].items():
                if isinstance(paths, str):
                    paths = [paths]
                for path in paths:
                    store = load_compact_transition_store(path)
                    edge_index, _ = from_scipy_sparse_matrix(store["adj_matrix"])
                    self.entries.append({
                        "community_id": entry["id"],
                        "source": source_name,
                        "store": store,
                        "edge_index": edge_index.long(),
                        "local_to_global": mapping,
                        "selected_indices": None,
                    })
        source_weights = manifest.get("source_weights")
        total_samples = manifest.get("total_samples")
        if source_weights and total_samples:
            self._select_fixed_composition(
                int(total_samples), source_weights, int(seed)
            )
        for entry in self.entries:
            if entry["selected_indices"] is None:
                entry["selected_indices"] = np.arange(
                    len(entry["store"]["rewards"]), dtype=np.int64
                )
        self._sizes = np.asarray([
            len(entry["selected_indices"]) for entry in self.entries
        ], dtype=np.int64)
        self.entries = [
            entry for entry, size in zip(self.entries, self._sizes) if size > 0
        ]
        self._sizes = self._sizes[self._sizes > 0]
        self._offsets = np.cumsum(self._sizes)
        self.n_transitions_stored = int(self._sizes.sum())
        if not self.n_transitions_stored:
            raise ValueError("Heterogeneous replay buffer contains no samples")

    def _select_fixed_composition(self, total_samples, source_weights, seed):
        names = [name for name, weight in source_weights.items() if weight > 0]
        weights = np.asarray([source_weights[name] for name in names], dtype=np.float64)
        weights /= weights.sum()
        raw = total_samples * weights
        source_counts = np.floor(raw).astype(np.int64)
        for index in np.argsort(-(raw - source_counts))[:total_samples - int(source_counts.sum())]:
            source_counts[index] += 1
        rng = np.random.default_rng(seed)
        for source, requested in zip(names, source_counts):
            matching = [entry for entry in self.entries if entry["source"] == source]
            available = np.asarray([
                len(entry["store"]["rewards"]) for entry in matching
            ], dtype=np.int64)
            if requested > int(available.sum()):
                raise ValueError(
                    f"Source {source} requests {requested} samples but only "
                    f"{int(available.sum())} are available"
                )
            raw_counts = requested * available / available.sum()
            counts = np.floor(raw_counts).astype(np.int64)
            remainder = requested - int(counts.sum())
            for index in np.argsort(-(raw_counts - counts))[:remainder]:
                counts[index] += 1
            for entry, count in zip(matching, counts):
                size = len(entry["store"]["rewards"])
                entry["selected_indices"] = rng.permutation(size)[:count]
        for entry in self.entries:
            if entry["source"] not in names:
                entry["selected_indices"] = np.empty(0, dtype=np.int64)

    def sample_transitions_option(self, batch_size=1):
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # A GNN minibatch must share one graph topology.  Choose the first
        # transition with the global sample distribution, then draw the rest
        # from that same community shard.  This preserves the paper's batch
        # size while supporting differently sized Leiden communities.
        flat_index = int(self.rng.integers(0, self.n_transitions_stored))
        entry_index = int(np.searchsorted(self._offsets, flat_index, side="right"))
        previous = 0 if entry_index == 0 else int(self._offsets[entry_index - 1])
        selected_position = flat_index - previous
        entry = self.entries[entry_index]
        store = entry["store"]
        selected = entry["selected_indices"]
        positions = [selected_position]
        if batch_size > 1:
            positions.extend(
                self.rng.integers(0, len(selected), size=batch_size - 1).tolist()
            )
        sample_indices = [int(selected[position]) for position in positions]
        obs_batch = []
        next_obs_batch = []
        action_batch = []
        rewards = []
        for sample_index in sample_indices:
            obs = store["initial_obs"].copy()
            obs[:, -1] = 0.0
            next_obs = obs.copy()
            seeds = store["seed_sets"][sample_index]
            next_obs[seeds, -1] = 1.0
            action = np.zeros((obs.shape[0], 1), dtype=np.float32)
            action[seeds, 0] = 1.0
            obs_batch.append(obs)
            next_obs_batch.append(next_obs)
            action_batch.append(action)
            rewards.append(float(store["rewards"][sample_index]))
        return {
            "obs": obs_batch,
            "next_obs": next_obs_batch,
            "action": action_batch,
            "reward": rewards,
            "terminal": np.ones(batch_size, dtype=np.float32),
            "edge_index": entry["edge_index"],
        }

# standard_scaler = StandardScaler()
# data = graph_utils.load_data("./data-b5-n1000.pkl")
# print(data.keys())

def create_sparse_matrix_with_self_loops(file_path):
    """
    从文件读取边列表，并生成包含节点自环的稀疏邻接矩阵。

    参数:
    file_path (str): 边列表文件的路径。

    返回:
    coo_matrix: 包含节点自环的稀疏邻接矩阵（COO格式）。
    """

    # 从文件读取边列表
    def read_edge_list(file_path):
        edges = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue  # 忽略注释行
                from_node, to_node = map(int, line.strip().split())
                edges.append((from_node, to_node))
        return np.array(edges, dtype=np.int64)

    # 读取边列表
    edges = read_edge_list(file_path)

    # 确定节点总数
    num_nodes = edges.max() + 1  # 假设节点编号从 0 开始

    # 创建自环边
    self_loops = np.array([(i, i) for i in range(num_nodes)], dtype=np.int64)

    # 将自环边与原始边合并
    all_edges = np.vstack((edges, self_loops))

    # 从边列表中提取行、列信息
    row = all_edges[:, 0]
    col = all_edges[:, 1]
    weight = np.ones(len(all_edges), dtype=np.int32)  # 边的权重，假设所有边的权重为 1
    adj = [row, col]
    adj = torch.tensor(adj, dtype=torch.long)
    # adj.append(weight)

    # 创建 COO 格式的稀疏矩阵
    # adj_matrix = coo_matrix((data, (row, col)))

    return adj, weight


def getNormalization(data):
    scaler = StandardScaler()
    # scaler = MinMaxScaler()

    # data["obs"] = np.array(data["obs"], dtype=np.float32)
    # print("data['obs'].shape=", data["next_obs"])
    # exit()

    obs = np.array(data["obs"], dtype=np.float32)

    obs_reshaped = obs.reshape(-1, obs.shape[-1])
    obs_standardized = scaler.fit_transform(obs_reshaped)
    obs = obs_standardized.reshape(data["obs"].shape)
    data["obs"] = obs

    data["next_obs"] = np.array(data["next_obs"], dtype=np.float32)
    next_obs = np.array(data["next_obs"])
    next_obs_reshaped = next_obs.reshape(-1, next_obs.shape[-1])
    next_obs_reshaped_standardized = scaler.fit_transform(next_obs_reshaped)
    next_obs = next_obs_reshaped_standardized.reshape(data["next_obs"].shape)
    data["next_obs"] = next_obs

    rewords = data["rewards"]
    # # data["rewards"] = standard_scaler.fit_transform(rewords.reshape(-1, 1)).reshape(-1)
    data["rewards"] = rewords * 100

    return data


from torch_geometric.data import InMemoryDataset, Data


class MyGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['Wiki-2.txt-b5-n1000-p0.3.pt']

    def download(self):
        pass

    def process(self):
        data_path = './Wiki-2.txt-b5-n1000-p0.3.pkl'
        graph_path = './Wiki-2.txt'
        re_list = []

        # 读取图
        graph = graph_utils.read_graph(path=graph_path)
        # print("graph",graph.num_nodes)
        # graph.generate_edge_index()

        # 读取离线数据
        data_list = graph_utils.load_data(data_path)
        # 归一化
        data_list = getNormalization(data_list)

        # 生成边索引
        adj, weight = create_sparse_matrix_with_self_loops(graph_path)

        adj = torch.tensor(adj, dtype=torch.long)
        weight = torch.tensor(weight, dtype=torch.float)

        # 创建多个 Data 对象
        for i in range(len(data_list["rewards"])):  # 假设我们有 10 个图

            data = Data(obs=torch.tensor(data_list["observations"][i], dtype=torch.float), edge_index=adj,
                        next_obs=torch.tensor(data_list["next_observations"][i], dtype=torch.float),
                        action=torch.tensor(data_list["actions"][i], dtype=torch.long),
                        reward=torch.tensor(data_list["rewards"][i], dtype=torch.float),
                        weight=torch.tensor(weight, dtype=torch.float),
                        terminal=torch.tensor(data_list["terminals"][i], dtype=torch.float))
            re_list.append(data)
            # print(data)

        data, slices = self.collate(re_list)
        torch.save((data, slices), self.processed_paths[0])
