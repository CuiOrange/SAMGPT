import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold


def find_2hop_neighbors_sp(adj, node):
    # print(adj.getrow(node))
    # print(adj.getrow(node).todense().A)
    nodeadj = adj.getrow(node).todense().A[0]
    neighbors = []
    # print(type(adj))
    for i in range(len(nodeadj)):
        if len(neighbors) >= 4:
            break
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if nodeadj[i] != 0 and node != i:
            neighbors.append(i)
    neighbors_2hop = []
    for i in neighbors:
        cnt = 0
        nodeadj = adj.getrow(i).todense().A[0]
        for j in range(len(nodeadj)):
            if cnt >= 2:
                break
            if nodeadj[j] != 0 and j != i:
                neighbors_2hop.append(j)
                cnt += 1
    return neighbors, neighbors_2hop


def sp_adj(adj,node1,node2):
    begin = 0
    for i in range(adj.row.shape):
        if adj.row[i] == node1:
            begin = i
            break

#寻找当前节点的邻居节点和二阶邻居节点
def find_2hop_neighbors(adj, node):
    neighbors = []
    # print(type(adj))
    for i in range(len(adj[node])):
        if len(neighbors) >= 10:
            break
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if adj[node][i] != 0 and node != i:
            neighbors.append(i)
    neighbors_2hop = []
    for i in neighbors:
        cnt = 0
        for j in range(len(adj[i])):
            if cnt >= 4:
                break
            if adj[i][j] != 0 and j != i:
                neighbors_2hop.append(j)
                cnt += 1
    return neighbors, neighbors_2hop

def build_subgraph(adj, idx_train, sparse = True):
    neighborslist = [[] for x in range(idx_train.shape[0])]
    neighbors_2hoplist = [[] for x in range(idx_train.shape[0])]
    mainindex = [[] for x in range(idx_train.shape[0])]
    mainlist = [[] for x in range(idx_train.shape[0])]
    idx_train_list = idx_train.tolist()
    for x in range(idx_train.shape[0]):        
        if sparse:
            neighborslist[x], neighbors_2hoplist[x] = find_2hop_neighbors_sp(adj, idx_train[x])
        else:
            neighborslist[x], neighbors_2hoplist[x] = find_2hop_neighbors(adj, idx_train[x])
        mainlist[x] = [idx_train_list[x]] + neighborslist[x] + neighbors_2hoplist[x]
        mainindex[x] = [x] * len(mainlist[x])
    neighborslist = sum(neighborslist,[])
    neighbors_2hoplist = sum(neighbors_2hoplist,[])
    mainlist = sum(mainlist,[])
    mainindex = sum(mainindex,[])
    return {
        'idx':torch.tensor(mainlist),
        'batch':torch.tensor(mainindex),        
    }

def plotlabels(feature, Trure_labels, name):
 # maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # 设置散点颜色

    S_lowDWeights = visual(feature)
    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab','hotpink']
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels)) # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape) # [num, 3]
    for index in range(4): # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=20, marker='.', c=colors[index], edgecolors=colors[index])
        plt.xticks([]) # 去掉横坐标值
        plt.yticks([]) # 去掉纵坐标值
    plt.title(name, fontsize=32, fontweight='normal', pad=20)
    
    plt.savefig('plt_graph/exceptcomputers/{}.png'.format(name),dpi=500)
    plt.show()
    plt.clf()

def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    print(x_ts.shape) # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final

def combine_dataset(*args):
    # print(feature1.shape)
    # print(feature2.shape)
    for step,adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))
            
    adj = sp.csr_matrix(adj1)
    
    return adj

def combine_dataset_list(args):
    # print(feature1.shape)
    # print(feature2.shape)
    for step,adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))
            
    adj = sp.csr_matrix(adj1)
    
    return adj

def combine_dataset_list_sp(args):
    # 初始化一个空的块对角稀疏矩阵
    adj1 = None
    
    for step, adj in enumerate(args):
        if step == 0:
            adj1 = adj  # 第一个矩阵直接赋值给adj1
        else:
            # 构建零矩阵
            num_rows1, num_cols1 = adj1.shape
            num_rows2, num_cols2 = adj.shape
            zeroadj1 = sp.csr_matrix((num_rows1, num_cols2))  # adj1右侧的零矩阵
            zeroadj2 = sp.csr_matrix((num_rows2, num_cols1))  # adj1下方的零矩阵
            
            # 拼接矩阵
            top = sp.hstack([adj1, zeroadj1])
            bottom = sp.hstack([zeroadj2, adj])
            adj1 = sp.vstack([top, bottom])
    
    return adj1.tocsr()

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, class_num):
    # print("len",nb_graphs)
    ft_size = data.num_features

    num = range(class_num)

    labelnum=range(class_num,ft_size)

    features = data.x[:, num]

    rawlabels = data.x[:, labelnum]
    # masks[g, :sizes[g]] = 1.0
    e_ind = data.edge_index
    # print("e_ind",e_ind)
    coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                        shape=(features.shape[0], features.shape[0]))
    # print("coo",coo)
    adjacency = coo


    adj = sp.csr_matrix(adjacency)

    # graphlabels = labels

    return features, adj

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_random_edges(adj, percentage=0.2):
    adj_coo = adj.tocoo()
    num_nodes = adj.shape[0]
    
    existing_edges = set(zip(adj_coo.row, adj_coo.col))
    num_existing_edges = len(existing_edges)
    
    num_additional_edges = int(num_existing_edges * percentage)
    
    new_edges = set()
    while len(new_edges) < num_additional_edges:
        src = np.random.randint(0, num_nodes)
        tgt = np.random.randint(0, num_nodes)

        if src != tgt and (src, tgt) not in existing_edges and (src, tgt) not in new_edges:
            new_edges.add((src, tgt))
    
    new_edges = np.array(list(new_edges))
    all_row = np.hstack((adj_coo.row, new_edges[:, 0]))
    all_col = np.hstack((adj_coo.col, new_edges[:, 1]))
    all_data = np.hstack((adj_coo.data, np.ones(len(new_edges))))
    
    new_adj = sp.coo_matrix((all_data, (all_row, all_col)), shape=(num_nodes, num_nodes))
    
    return new_adj.tocsr()

def generate_k_hop_subgraphs(data, k_hop=4, unify_dim=None, dataset_id=None):
    from torch_geometric.utils import k_hop_subgraph, subgraph
    from torch_geometric.data import Data
    import torch
    from sklearn.decomposition import PCA
    import numpy as np

    def pca_compression(features, k):
        pca = PCA(n_components=k)
        compressed_features = pca.fit_transform(features)
        return compressed_features

    k_hop_subgraph_list = []
    total_nodes = data.x.size(0)

    if unify_dim is not None:
        original_features = data.x.cpu().numpy()
        compressed_features = pca_compression(original_features, k=unify_dim)
        data.x = torch.FloatTensor(compressed_features).to(data.x.device)

    for node_index in range(total_nodes):
        subset, _, _, _ = k_hop_subgraph(node_idx=node_index, num_hops=k_hop, 
                                         edge_index=data.edge_index, relabel_nodes=True)

        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        sub_x = data.x[subset]
        sub_label = data.y[node_index].item()

        k_hop_subgraph_ = Data(x=sub_x, edge_index=sub_edge_index, y=sub_label)
        k_hop_subgraph_.dataset_id = dataset_id  
        k_hop_subgraph_list.append(k_hop_subgraph_)

    print(f'generated {len(k_hop_subgraph_list)}/{total_nodes} {k_hop}-hop subgraphs')
    return k_hop_subgraph_list
