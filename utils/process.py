import random
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

import torch
import torch.nn.functional as F
from torch import optim as optim
from torch_geometric.utils import to_undirected
import random
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from sklearn.manifold import TSNE
from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def np_rand_mask(num_edge, num_mask, edge_t):
    index = np.arange(num_edge)
    np.random.shuffle(index)

    index_left = torch.from_numpy(index[:-num_mask])
    index_mask = torch.from_numpy(index[-num_mask:])

    edge_mask = edge_t[index_mask].t()
    edge_left = edge_t[index_left].t()
    return edge_mask, edge_left


def pt_rand_mask(num_edge, num_mask, edge_t):
    perm = torch.randperm(num_edge)

    index_left = perm[:-num_mask]
    index_mask = perm[-num_mask:]

    edge_mask = edge_t[index_mask].t()
    edge_left = edge_t[index_left].t()
    return edge_mask, edge_left


def edgemask_um(mask_ratio, edge_t, num_nodes):
    # 不做处理，保持单向
    if isinstance(edge_t, torch.Tensor):
        edge_train_pos_t = edge_t
    else:
        edge_train_pos_t = edge_t['train']['pos']

    num_edge = len(edge_train_pos_t)
    num_mask = int(num_edge * mask_ratio)
    edge_mask, edge_left = pt_rand_mask(num_edge, num_mask, edge_train_pos_t)

    # 为了GCN，需要无向图进行编码
    edge_left = to_undirected(edge_left)
    edge_left, _ = add_self_loops(edge_left, num_nodes=num_nodes)

    return edge_left, edge_mask


def edgemask_dm(mask_ratio, edge_t, num_nodes):
    # 将所有边恢复为双向，包括掩盖边
    if isinstance(edge_t, torch.Tensor):
        edge_train_pos_t = to_undirected(edge_t.t())
    else:
        edge_train_pos_t = torch.stack(
            [edge_t['train']['pos'][:, 1], edge_t['train']['pos'][:, 0]], dim=1)
        edge_train_pos_t = torch.cat(
            [edge_t['train']['pos'], edge_train_pos_t], dim=0)

    num_edge = len(edge_train_pos_t)
    num_mask = int(num_edge * mask_ratio)
    edge_mask, edge_left = pt_rand_mask(num_edge, num_mask, edge_train_pos_t)

    edge_left, _ = add_self_loops(edge_left, num_nodes=num_nodes)

    return edge_left, edge_mask


def cosine_similarity_matrix(features):
    """
    根据节点特征矩阵计算对应余弦相似度矩阵
    :param features: 节点特征矩阵，形状为 (num_nodes, num_features)
    :return: 余弦相似度矩阵，形状为 (num_nodes, num_nodes)
    """
    # 计算余弦相似度
    similarity = 1 - squareform(pdist(features, metric='cosine'))
    return similarity


def adjacency_matrix_from_similarity(similarity, threshold=0.3):
    """
    根据余弦相似度矩阵生成邻接矩阵
    :param similarity: 余弦相似度矩阵，形状为 (num_nodes, num_nodes)
    :param threshold: 阈值，只有相似度大于该值的节点对才会在邻接矩阵中连接
    :return: 邻接矩阵，形状为 (num_nodes, num_nodes)
    """
    adjacency_matrix = (similarity > threshold).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)  # 去掉自连接
    return adjacency_matrix


def filter_metapath_edges(edge_index_dict, target_node):
    """
    筛选出元路径得到的边。

    参数:
    edge_index_dict (dict): 包含所有边的字典。
    target_node (str): 目标节点类型。

    返回:
    dict: 仅包含元路径得到的边的字典。
    """
    metapath_edges = {}
    for edge_type, edge_index in edge_index_dict.items():
        if edge_type[0] == target_node and edge_type[2] == target_node:
            metapath_edges[edge_type] = edge_index
    return metapath_edges


def find_hop_nodes(edge_index_dict, target_node, max_hop=2):
    hop_dict = {0: [target_node]}
    visited = set([target_node])

    for hop in range(1, max_hop + 1):
        current_hop_nodes = hop_dict[hop - 1]
        next_hop_nodes = set()

        for node_type in current_hop_nodes:
            for (source, _, target) in edge_index_dict.keys():
                if source == node_type and target not in visited:
                    next_hop_nodes.add(target)
                    visited.add(target)

        if next_hop_nodes:
            hop_dict[hop] = list(next_hop_nodes)
        else:
            break

    return hop_dict


def edge_index2adj(edge_index, shape):
    adj = torch.zeros(shape, dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    return adj


def adj2edge_index(adj):
    row, col = torch.nonzero(adj, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def cosine_similarity_matrix(features):
    features = features / torch.norm(features, dim=-1, keepdim=True)
    similarity = torch.mm(features, features.T)

    return similarity


def cosine_similarity(x1, x2):
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    similarity_matrix = torch.matmul(x1_norm, x2_norm.T)
    return similarity_matrix


def adjacency_matrix_from_similarity(similarity, threshold):
    adjacency_matrix = (similarity > threshold).to(torch.float)
    adjacency_matrix.fill_diagonal_(0)

    return adjacency_matrix


def adjacency_matrix_top_similarity(similarity, k=50):
    similarity.fill_diagonal_(0)
    top_k = torch.topk(similarity, k, dim=1)

    adjacency_matrix = torch.zeros_like(similarity)
    row_indices = torch.arange(adjacency_matrix.size(0)).unsqueeze(1).expand(-1, top_k.indices.size(1))
    adjacency_matrix[row_indices, top_k.indices] = 1

    return adjacency_matrix


def aggregate_mean(adjacency_matrix, x):

    out = torch.matmul(adjacency_matrix, x)
    degrees = adjacency_matrix.sum(dim=1).unsqueeze(1)
    degrees[degrees == 0] = 1
    out = out.to('cpu') / degrees.to('cpu')

    out = out.to(x.device)
    return out


def split_indices(labels, train_size, val_size=1000, test_size=1000):

    total_indices = list(range(len(labels)))

    # split the indices into train
    class_indices = {0: [], 1: [], 2: []}
    for i, label in enumerate(labels):
        class_indices[label.item()].append(i)

    selected_indices = []
    for _, indices in class_indices.items():
        selected_indices.extend(np.random.choice(indices, train_size, replace=False))
    random.shuffle(selected_indices)

    # split the remaining indices into val
    remaining_indices = list(set(total_indices) - set(selected_indices))
    if len(remaining_indices) < val_size + test_size:
        raise ValueError("Not enough remaining indices to sample from.")

    val_indices = random.sample(remaining_indices, val_size)
    # print(torch.sum(labels[val_indices]==0))
    # print(torch.sum(labels[val_indices]==1))
    # print(torch.sum(labels[val_indices]==2))

    # split the remaining indices into test
    remaining_indices = list(set(remaining_indices) - set(val_indices))
    test_indices = random.sample(remaining_indices, test_size)
    # print(torch.sum(labels[test_indices]==0))
    # print(torch.sum(labels[test_indices]==1))
    # print(torch.sum(labels[test_indices]==2))

    np.save('./data/IMDB/train_20.npy', selected_indices)
    np.save('./data/IMDB/val_20.npy', val_indices)
    np.save('./data/IMDB/test_20.npy', test_indices)
    return selected_indices, val_indices, test_indices


def t_SNE(embeds: torch.Tensor, labels: torch.Tensor, dataset: str):
    """
    应用 t-SNE 将节点嵌入可视化到 2D 空间。

    Args:
        embeds (torch.Tensor): 节点嵌入矩阵，形状为 (num_nodes, embed_dim)
        labels (torch.Tensor): 节点标签，形状为 (num_nodes,)
        dataset (str): 数据集名称
    """

    tsne = TSNE(n_components=2, random_state=42)
    embeds_2d = tsne.fit_transform(embeds)
    num_class = len(np.unique(labels))
    one_hot_labels = np.eye(num_class)[labels]

    cmap = ListedColormap(['black', 'blue', 'green', 'red'])
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=one_hot_labels, cmap=cmap, alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(f'img/DBLP/t-SNE.png')


def visualize_masks(mask_anchor_f, mask_anchor_s):
    intra_mask_f = mask_anchor_f[0].sum(dim=1).cpu().numpy()
    inter_mask_f = mask_anchor_f[1].sum(dim=1).cpu().numpy()
    intra_mask_s = mask_anchor_s[0].sum(dim=1).cpu().numpy()
    inter_mask_s = mask_anchor_s[1].sum(dim=1).cpu().numpy()

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.bar(range(len(intra_mask_f)), intra_mask_f)
    plt.title('Intra Mask Positive Samples per Node (f)')
    plt.xlabel('Node')
    plt.ylabel('Positive Samples')

    plt.subplot(2, 2, 2)
    plt.bar(range(len(inter_mask_f)), inter_mask_f)
    plt.title('Inter Mask Positive Samples per Node (f)')
    plt.xlabel('Node')
    plt.ylabel('Positive Samples')

    plt.subplot(2, 2, 3)
    plt.bar(range(len(intra_mask_s)), intra_mask_s)
    plt.title('Intra Mask Positive Samples per Node (s)')
    plt.xlabel('Node')
    plt.ylabel('Positive Samples')

    plt.subplot(2, 2, 4)
    plt.bar(range(len(inter_mask_s)), inter_mask_s)
    plt.title('Inter Mask Positive Samples per Node (s)')
    plt.xlabel('Node')
    plt.ylabel('Positive Samples')

    plt.tight_layout()
    plt.savefig('img/mask_visualization_acm.png')


def plot_pos_neighbors(pos_neighbors_per_epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(pos_neighbors_per_epoch)), pos_neighbors_per_epoch, marker='o')
    plt.title('Positive Neighbors Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Positive Neighbors')
    plt.grid(True)
    plt.savefig('img/FreeBase/pos_neighbors_s_inter.png')


def visualize_graph(adj_matrix, path):
    """
    可视化二维邻接矩阵
    :param adj_matrix: 二维邻接矩阵，形状为 (num_nodes, num_nodes)
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    adj_matrix_np = adj_matrix.cpu().numpy()

    # 创建 NetworkX 图
    G = nx.from_numpy_array(adj_matrix_np)

    # 绘制图形
    pos = nx.spring_layout(G)  # 使用 spring 布局
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10, font_color='black')
    plt.title("Graph Visualization from Adjacency Matrix")
    plt.draw()
    plt.savefig(path)


def visualize_adjacency_matrix(adj_matrix, path):
    """
    可视化二维邻接矩阵
    :param adj_matrix: 二维邻接矩阵，形状为 (num_nodes, num_nodes)
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    adj_matrix_np = adj_matrix.cpu().numpy()

    # 绘制矩阵
    plt.figure(figsize=(12, 12))
    plt.imshow(adj_matrix_np, cmap='Blues', interpolation='none')
    plt.colorbar()
    # sns.heatmap(adj_matrix_np, cmap='Blues', cbar=True, square=True, linewidths=.5, linecolor='gray')
    plt.title("Adjacency Matrix Visualization")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")

    # 高亮数值为 1 的元素
    for (i, j), val in np.ndenumerate(adj_matrix_np):
        if val == 1:
            plt.text(j, i, '1', ha='center', va='center', color='red')

    plt.savefig(path)
