import pickle
import random
import statistics
import time
from collections import deque
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch

from paper_protocol import bidirected_edges

class Graph:
    ''' graph class '''

    def __init__(self, nodes, edges, children, parents):
        self.nodes = nodes  # set()
        self.edges = edges  # dict{(src,dst): weight, }
        self.children = children  # dict{node: set(), }
        self.parents = parents  # dict{node: set(), }
        # transfer children and parents to dict{node: list, }
        for node in self.children:
            self.children[node] = sorted(self.children[node])
        for node in self.parents:
            self.parents[node] = sorted(self.parents[node])

        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

        self._adj = None
        self._from_to_edges = None
        self._from_to_edges_weight = None

    def generate_edge_index(graph):
        """
        根据 Graph 实例生成符合 GCNConv 输入要求的邻接矩阵格式。

        参数:
        graph (Graph): Graph 类实例

        返回:
        torch.Tensor: 形状为 [2, num_edges] 的邻接矩阵张量
        """
        edge_list = list(graph.edges.keys())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def get_adj_matrix(self):
        # Create a mapping from node to index
        node_index = {node: i for i, node in enumerate(self.nodes)}

        # Initialize an empty adjacency matrix
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))

        # Populate the adjacency matrix using edges
        for (src, dst), weight in self.edges.items():
            i, j = node_index[src], node_index[dst]
            adj_matrix[i, j] = 1
            # adj_matrix[j, i] = 1  # Assuming the graph is undirected

        # Convert the adjacency matrix to a sparse matrix format
        adj_matrix_sparse = sp.csr_matrix(adj_matrix)

        return adj_matrix_sparse
    def to_csr_matrix(self):
        # Create a mapping from node to index
        node_index = {node: idx for idx, node in enumerate(sorted(self.nodes))}
        row = []
        col = []
        data = []

        # Fill the row, col, and data arrays
        for (src, dst), weight in self.edges.items():
            row.append(node_index[src])
            col.append(node_index[dst])
            data.append(1)

        # Create CSR matrix
        csr_matrix = sp.csr_matrix((data, (row, col)), shape=(self.num_nodes, self.num_nodes))
        return csr_matrix

    def get_children(self, node):
        ''' outgoing nodes '''
        return self.children.get(node, [])

    def get_parents(self, node):
        ''' incoming nodes '''
        return self.parents.get(node, [])

    def get_prob(self, edge):
        return self.edges[edge]

    def get_adj(self):
        ''' return scipy sparse matrix '''
        if self._adj is None:
            self._adj = np.zeros((self.num_nodes, self.num_nodes))
            for edge in self.edges:
                self._adj[edge[0], edge[1]] = self.edges[edge]  # may contain weight
            self._adj = csr_matrix(self._adj)
        return self._adj

    def from_to_edges(self):
        ''' return a list of edge of (src,dst) '''
        if self._from_to_edges is None:
            self._from_to_edges_weight = list(self.edges.items())
            self._from_to_edges = [p[0] for p in self._from_to_edges_weight]
        return self._from_to_edges

    def from_to_edges_weight(self):
        ''' return a list of edge of (src, dst) with edge weight '''
        if self._from_to_edges_weight is None:
            self.from_to_edges()
        return self._from_to_edges_weight


def read_graph(path, ind=0, directed=False):
    ''' method to load edge as node pair graph '''
    parents = {}
    children = {}
    edges = {}
    nodes = set()

    if not directed:
        num_nodes, protocol_edges = bidirected_edges(path, ind)
        nodes.update(range(num_nodes))
        edge_rows = protocol_edges
    else:
        edge_rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not len(line) or line.startswith('#') or line.startswith('%'):
                    continue
                row = line.split()
                edge_rows.append((int(row[0]) - ind, int(row[1]) - ind))
                nodes.update(edge_rows[-1])
    for src, dst in edge_rows:
        children.setdefault(src, set()).add(dst)
        parents.setdefault(dst, set()).add(src)
        edges[(src, dst)] = 0.0

    # print("parents=", parents)
    # print("children=", children)
    # 入度
    indegree = {}
    for node in nodes:
        indegree[node] = len(parents.get(node, []))
    # 节点边数量
    num_links = {}
    for node in nodes:
        num_links[node] = len(children.get(node, [])) + len(parents.get(node, []))
    # 如果是无向图，则将得到度
    if not directed:
        for node in nodes:
            indegree[node] = int(len(parents.get(node, [])))
            num_links[node] = len(children.get(node, []))

    # print("num_links=", num_links)
    #
    # print("indegree=", indegree)
    # 改变边的权重，作为IC模型的传播概率
    for src, dst in edges:
        edges[(src, dst)] = 1.0 / max(indegree[dst], 1)
        # edges[(src, dst)] = 0.5
    return Graph(nodes, edges, children, parents)


def workerMC(x):
    ''' for multiprocessing '''
    return computeMC(x[0], x[1], x[2])

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
def computeMC(graph, S, R):
    ''' compute expected influence using MC under IC
        R: number of trials
    '''
    sources = set(S)
    inf = 0
    for _ in range(R):
        source_set = sources.copy()
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(child for child in graph.get_children(curr_node) \
                                       if
                                       not (child in source_set) and random.random() <= graph.edges[(curr_node, child)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        inf += len(source_set)

    return inf / R


if __name__ == '__main__':
    # path of the graph file
    path = "./Data/Wiki-2.txt"
    # path = "./1.txt"
    # number of parallel processes
    num_process = 5
    # number of trials
    num_trial = 100
    # load the graph
    graph = read_graph(path, ind=0, directed=True)
    # print(graph.edges[(5, 6)])
    # print(graph.num_edges)
    print("num_nodes=", graph.num_nodes)
    # exit()
    # print(graph.nodes)
    # for i in graph.edges:
    #     print(i, graph.edges[i])

    print('Generating seed sets:')
    list_S = [(2, 4), (2183, 6869,), (2183, 882, 6869, 1162, 12)]
    num_list_S = len(list_S)
    # for _ in range(num_list_S):
    #     list_S.append(random.sample(range(graph.num_nodes), k=random.randint(3, 10)))
    #     print(f'({str(list_S[-1])[1:-1]})')

    # cached single-process RR

    # multi-process MC
    print('Multi-process MC:')
    es_infs = []
    times = []
    time_1 = time.time()
    for S in list_S:
        time_start = time.time()
        with Pool(num_process) as p:
            es_infs.append(
                statistics.mean(p.map(workerMC, [[graph, S, num_trial // num_process] for _ in range(num_process)])))
        times.append(time.time() - time_start)
    time_2 = time.time()
    for i in range(num_list_S):
        print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')
