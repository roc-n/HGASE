import random
from tqdm import tqdm
import yaml
import logging
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch import optim as optim
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)


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
    # edge_left = to_undirected(edge_left)
    # edge_left, _ = add_self_loops(edge_left, num_nodes=num_nodes)

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



