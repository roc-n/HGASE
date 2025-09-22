import torch
import torch.nn as nn
from torch_geometric.datasets import DBLP, Planetoid, AMiner, IMDB
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling, to_dense_adj, to_edge_index
import pickle
import scipy.sparse as sp
import random


from utils.process import edgemask_um, edgemask_dm, edge_index2adj, split_indices

import os.path as osp
import numpy as np
import os

####################################################
# mps aka metapaths
####################################################

path = os.getcwd()


def load_dblp(ratio):

    mps = [
        [('author', 'paper'), ('paper', 'author')],
        [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')],
        [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')]
    ]

    dataset_path = osp.join(path, 'data/DBLP')
    dataset = DBLP(dataset_path)
    g_origin = dataset[0]

    # 提取需要的数据
    num_target_node = g_origin['author'].num_nodes
    data = process_data(g_origin, mps, num_target_node)

    # prepare_data(data, mps)

    # 检查内部元路径条数
    # edge_index = data.edge_index_dict['author', 'to', 'paper']
    # edge_index_ = data.edge_index_dict['paper', 'to', 'author']
    # mask = edge_index[0] == 0
    # mask_ = edge_index_[0] == 2364
    # mask__ = edge_index_[0] == 6457
    # print(edge_index[:, mask])
    # print(edge_index_[:, mask_])
    # print(edge_index_[:, mask__])

    multi_hop_dblp(data)
    load_meta_edges(data)
    hop_neighbors(data)

    # ratio = [0.2, 0.1, 0.05]
    # ratio = [0.05, 0.13, 0.18]
    data.top_adjs, data.max_neis, data.pathsim_ms = pathsim(data.meta_adjs, data.metapath_dict.keys(), ratio)

    return data


def sort_edge_index(edge_index):
    sorted_index = torch.argsort(edge_index[0])
    edge_index = edge_index[:, sorted_index]
    return edge_index


def multi_hop_dblp(data):

    if not osp.exists('./data/DBLP/big_adj_k_hop.pth'):
        # author, paper, conference, term
        nums = [data['author'].num_nodes, data['paper'].num_nodes, data['conference'].num_nodes, data['term'].num_nodes]
        adj_size = sum(nums)
        big_adj = torch.zeros((adj_size, adj_size))

        adj_a_p = torch.zeros((nums[0], nums[1]))
        edge_index = data['author', 'to', 'paper'].edge_index
        adj_a_p[edge_index[0], edge_index[1]] = 1

        adj_p_c = torch.zeros((nums[1], nums[2]))
        edge_index = data['paper', 'to', 'conference'].edge_index
        adj_p_c[edge_index[0], edge_index[1]] = 1

        adj_p_t = torch.zeros((nums[1], nums[3]))
        edge_index = data['paper', 'to', 'term'].edge_index
        adj_p_t[edge_index[0], edge_index[1]] = 1

        stop_author = nums[0]
        stop_paper = nums[0] + nums[1]
        stop_conference = nums[0] + nums[1] + nums[2]
        stop_term = sum(nums)
        big_adj[:stop_author, stop_author:stop_paper] = adj_a_p
        big_adj[stop_author:stop_paper, stop_paper:stop_conference] = adj_p_c
        big_adj[stop_author:stop_paper, stop_conference:stop_term] = adj_p_t

        upper_tri = torch.triu(big_adj, diagonal=1)
        big_adj = upper_tri + upper_tri.t()

        k = 4
        big_adj_k_hop = torch.matrix_power(big_adj, k)

        author_size = nums[0]
        matrix = big_adj_k_hop[:author_size, :author_size]
    else:
        matrix = torch.load('./data/DBLP/big_adj_k_hop.pth')

    m = filter_nodes_topk(matrix, k=50)
    edge_index_hop = torch.nonzero(m, as_tuple=False).t()
    data.edge_index_hop = edge_index_hop


def filter_nodes_topk(matrix, k=100):

    # non_zero_counts = (matrix > 0).sum(dim=1)

    # m = torch.zeros_like(matrix)

    # # 获取需要截取前 k 个元素的行索引
    # rows_to_topk = non_zero_counts > k

    # # 对需要截取前 k 个元素的行进行处理
    # topk_values, topk_indices = torch.topk(matrix[rows_to_topk], k, dim=1)
    # row_indices = torch.arange(matrix.size(0))[rows_to_topk].unsqueeze(1).expand(-1, k)
    # m[row_indices, topk_indices] = topk_values

    # # 对不需要截取前 k 个元素的行进行处理
    # m[~rows_to_topk] = matrix[~rows_to_topk]

    topk_values, topk_indices = torch.topk(matrix, k, dim=1)
    m = torch.zeros_like(matrix)
    row_indices = torch.arange(matrix.size(0)).unsqueeze(1).expand(-1, k)
    m[row_indices, topk_indices] = topk_values

    return m


def load_acm(ratio):
    x_dict = {
        'paper': sp.load_npz("./data/ACM/p_feat.npz").toarray(),
        'author': sp.load_npz("./data/ACM/a_feat.npz").toarray(),
        'subject': torch.eye(60)
    }

    p_a = np.genfromtxt("./data/ACM/pa.txt").T
    p_s = np.genfromtxt("./data/ACM/ps.txt").T
    a_p = np.array([p_a[1], p_a[0]])
    s_p = np.array([p_s[1], p_s[0]])

    label = np.load("./data/ACM/labels.npy")
    # uni_el,counts = np.unique(label,return_counts=True)
    # for element, count in zip(uni_el, counts):
    #     print(f"Element: {element}, Count: {count}")

    # 构建数据集
    data = HeteroData()
    data['paper'].x = torch.from_numpy(x_dict['paper']).to(torch.float32)
    data['author'].x = torch.from_numpy(x_dict['author']).to(torch.float32)
    data['subject'].x = x_dict['subject'].to(torch.float32)
    data['paper'].y = torch.from_numpy(label).to(torch.int64)
    data['paper', 'to', 'author'].edge_index = torch.tensor(p_a, dtype=torch.int64).contiguous()
    data['author', 'to', 'paper'].edge_index = sort_edge_index(torch.tensor(a_p, dtype=torch.int64))
    data['paper', 'to', 'subject'].edge_index = torch.tensor(p_s, dtype=torch.int64).contiguous()
    data['subject', 'to', 'paper'].edge_index = sort_edge_index(torch.tensor(s_p, dtype=torch.int64))

    mps = [[('paper', 'author'), ('author', 'paper')], [('paper', 'subject'), ('subject', 'paper')]]
    transform = T.AddMetaPaths(metapaths=mps, drop_orig_edge_types=False, drop_unconnected_node_types=False)
    data = transform(data)

    load_meta_edges(data, 'ACM')
    hop_neighbors(data, 'ACM')
    multi_hop_acm(data)

    # ratio = [0.6, 0.4]
    # ratio = [0.07, 0.3]
    data.top_adjs, data.max_neis, data.pathsim_ms = pathsim(data.meta_adjs, data.metapath_dict.keys(), ratio)

    return data


def multi_hop_acm(data):

    if not osp.exists('./data/ACM/big_adj_k_hop.pth'):
        # paper, author, subject
        nums = [data['paper'].num_nodes, data['author'].num_nodes, data['subject'].num_nodes]
        adj_size = sum(nums)
        big_adj = torch.zeros((adj_size, adj_size))

        adj_p_a = torch.zeros((nums[0], nums[1]))
        edge_index = data['paper', 'to', 'author'].edge_index
        adj_p_a[edge_index[0], edge_index[1]] = 1

        adj_p_s = torch.zeros((nums[0], nums[2]))
        edge_index = data['paper', 'to', 'subject'].edge_index
        adj_p_s[edge_index[0], edge_index[1]] = 1

        stop_paper = nums[0]
        stop_author = nums[0] + nums[1]
        stop_subject = nums[0] + nums[1] + nums[2]
        big_adj[:stop_paper, stop_paper:stop_author] = adj_p_a
        big_adj[:stop_paper, stop_author:stop_subject] = adj_p_s

        upper_tri = torch.triu(big_adj, diagonal=1)
        big_adj = upper_tri + upper_tri.t()

        k = 2
        big_adj_k_hop = torch.matrix_power(big_adj, k)

        paper_size = nums[0]
        matrix = big_adj_k_hop[:paper_size, :paper_size]

    else:
        matrix = torch.load('./data/ACM/big_adj_k_hop.pth')

    m = filter_nodes_topk(matrix, k=50)
    edge_index_hop = torch.nonzero(m, as_tuple=False).t()
    data.edge_index_hop = edge_index_hop


def load_aminer():

    # 构建数据集
    x_dict = {
        'paper': torch.eye(6564),
        'author': torch.eye(13329),
        'reference': torch.eye(35890)
    }

    p_a = np.genfromtxt("./data/AMiner/pa.txt").T
    p_r = np.genfromtxt("./data/AMiner/pr.txt").T
    a_p = np.array([p_a[1], p_a[0]])
    r_p = np.array([p_r[1], p_r[0]])

    label = np.load("./data/AMiner/labels.npy")
    uni_el, counts = np.unique(label, return_counts=True)
    for element, count in zip(uni_el, counts):
        print(f"Element: {element}, Count: {count}")

    data = HeteroData()
    data['paper'].x = x_dict['paper'].to(torch.float32)
    data['author'].x = x_dict['author'].to(torch.float32)
    data['reference'].x = x_dict['reference'].to(torch.float32)
    data['paper'].y = torch.from_numpy(label).to(torch.int64)
    data['paper', 'to', 'author'].edge_index = torch.tensor(p_a, dtype=torch.int64).contiguous()
    data['author', 'to', 'paper'].edge_index = sort_edge_index(torch.tensor(a_p, dtype=torch.int64))
    data['paper', 'to', 'reference'].edge_index = torch.tensor(p_r, dtype=torch.int64).contiguous()
    data['reference', 'to', 'paper'].edge_index = sort_edge_index(torch.tensor(r_p, dtype=torch.int64))

    # dataset_path = osp.join(path, 'data/Aminer')
    # dataset = AMiner(dataset_path)
    # g_origin = dataset[0]
    # num_target_node = g_origin['paper'].num_nodes
    mps = [[('paper', 'author'), ('author', 'paper')], [('paper', 'reference'), ('reference', 'paper')]]

    transform = T.AddMetaPaths(metapaths=mps, drop_orig_edge_types=False, drop_unconnected_node_types=False)
    data = transform(data)

    load_meta_edges(data, 'AMiner')
    hop_neighbors(data, 'AMiner')
    multi_hop_aminer(data)

    # ratio = [0.07, 0.3]
    ratio_ = [0.05, 0.07]
    # ratio = [0.1,0.07]

    # ratio=[0.15,0.18]
    # ratio_ = [0.06, 0.08]
    data.top_adjs, data.max_neis, data.pathsim_ms = pathsim(data.meta_adjs, data.metapath_dict.keys(), ratio_)
    return data


def load_imdb():

    dataset_path = osp.join(path, './data/IMDB')
    dataset = IMDB(dataset_path)
    data = dataset[0]

    print(torch.sum(data.edge_index_dict['movie', 'to', 'actor'][0] == 2000))
    print(torch.sum(data.edge_index_dict['movie', 'to', 'director'][1] == 172))
    print(data.edge_index_dict['movie', 'to', 'actor'].shape)
    print(data.edge_index_dict['movie', 'to', 'director'].shape)
    print(data.x_dict['movie'].shape)
    print(data.x_dict['director'].shape)
    print(data.x_dict['actor'].shape)

    def list_directors_with_multiple_connections(data):
        # 获取 edge_index
        edge_index = data.edge_index_dict['movie', 'to', 'director']

        # 获取 director 节点的索引
        director_indices = edge_index[1]

        # 计算每个 director 节点的度数
        num_directors = director_indices.max().item() + 1
        degree_counts = torch.bincount(director_indices, minlength=num_directors)

        # 筛选度数大于等于 2 的 director 节点
        directors_with_multiple_connections = (degree_counts >= 2).nonzero(as_tuple=True)[0]

        # 打印结果
        print("Degree counts:", degree_counts)
        print("Directors with 2 or more connections:", directors_with_multiple_connections)
        print("Number of directors with 2 or more connections:", directors_with_multiple_connections.size(0))

    def check_edge_index_coverage(data):
        # 获取 edge_index[0]
        movie_indices = data.edge_index_dict['movie', 'to', 'director'][0]

        # 创建一个包含范围 0-4277 的张量
        expected_indices = torch.arange(0, 4278, dtype=torch.long)

        # 检查 movie_indices 是否包含所有 expected_indices
        is_covered = torch.isin(expected_indices, movie_indices).all()

        # 打印结果
        print("Edge index[0] covers range 0-4277:", is_covered)

        if not is_covered:
            missing_indices = expected_indices[~torch.isin(expected_indices, movie_indices)]
            print("Missing indices:", missing_indices)

    # list_directors_with_multiple_connections(data)
    # check_edge_index_coverage(data)

    mps = [[('movie', 'actor'), ('actor', 'movie')], [('movie', 'director'), ('director', 'movie')]]
    transform = T.AddMetaPaths(metapaths=mps, drop_orig_edge_types=False, drop_unconnected_node_types=False)
    data = transform(data)

    load_meta_edges(data, 'IMDB')
    hop_neighbors(data, 'IMDB')
    multi_hop_imdb(data)

    ratio = [0.8, 0.8]
    data.top_adjs, data.max_neis, data.pathsim_ms = pathsim(data.meta_adjs, data.metapath_dict.keys(), ratio)

    return data


def multi_hop_imdb(data):

    if not osp.exists('./data/IMDB/big_adj_k_hop.pth'):
        # movie, actor, director
        nums = [data['movie'].num_nodes, data['actor'].num_nodes, data['director'].num_nodes]
        adj_size = sum(nums)
        big_adj = torch.zeros((adj_size, adj_size))

        adj_m_a = torch.zeros((nums[0], nums[1]))
        edge_index = data['movie', 'to', 'actor'].edge_index
        adj_m_a[edge_index[0], edge_index[1]] = 1

        adj_m_d = torch.zeros((nums[0], nums[2]))
        edge_index = data['movie', 'to', 'director'].edge_index
        adj_m_d[edge_index[0], edge_index[1]] = 1

        stop_movie = nums[0]
        stop_actor = nums[0] + nums[1]
        stop_director = nums[0] + nums[1] + nums[2]
        big_adj[:stop_movie, stop_movie:stop_actor] = adj_m_a
        big_adj[:stop_movie, stop_actor:stop_director] = adj_m_d

        upper_tri = torch.triu(big_adj, diagonal=1)
        big_adj = upper_tri + upper_tri.t()

        k = 2
        big_adj_k_hop = torch.matrix_power(big_adj, k)

        movie_size = nums[0]
        matrix = big_adj_k_hop[:movie_size, :movie_size]

        torch.save(matrix, './data/IMDB/big_adj_k_hop.pth')

    else:
        matrix = torch.load('./data/ACM/big_adj_k_hop.pth')

    m = filter_nodes_topk(matrix, k=50)
    edge_index_hop = torch.nonzero(m, as_tuple=False).t()
    data.edge_index_hop = edge_index_hop


def multi_hop_aminer(data):

    if not osp.exists('./data/AMiner/big_adj_k_hop.pth'):
        # paper, author, reference
        nums = [data['paper'].num_nodes, data['author'].num_nodes, data['reference'].num_nodes]
        adj_size = sum(nums)
        big_adj = torch.zeros((adj_size, adj_size))

        adj_p_a = torch.zeros((nums[0], nums[1]))
        edge_index = data['paper', 'to', 'author'].edge_index
        adj_p_a[edge_index[0], edge_index[1]] = 1

        adj_p_r = torch.zeros((nums[0], nums[2]))
        edge_index = data['paper', 'to', 'reference'].edge_index
        adj_p_r[edge_index[0], edge_index[1]] = 1

        stop_paper = nums[0]
        stop_author = nums[0] + nums[1]
        stop_reference = nums[0] + nums[1] + nums[2]
        big_adj[:stop_paper, stop_paper:stop_author] = adj_p_a
        big_adj[:stop_paper, stop_author:stop_reference] = adj_p_r

        upper_tri = torch.triu(big_adj, diagonal=1)
        big_adj = upper_tri + upper_tri.t()

        k = 2
        big_adj_k_hop = torch.matrix_power(big_adj, k)

        paper_size = nums[0]
        matrix = big_adj_k_hop[:paper_size, :paper_size]

    else:
        matrix = torch.load('./data/AMiner/big_adj_k_hop.pth')

    m = filter_nodes_topk(matrix, k=50)
    edge_index_hop = torch.nonzero(m, as_tuple=False).t()
    data.edge_index_hop = edge_index_hop


def load_freebase():
    x_dict = {
        'm': torch.eye(3492),
        'd': torch.eye(2502),
        'a': torch.eye(33401),
        'w': torch.eye(4459)
    }

    m_d = np.genfromtxt("./data/FreeBase/md.txt").T
    m_a = np.genfromtxt("./data/FreeBase/ma.txt").T
    m_w = np.genfromtxt("./data/FreeBase/mw.txt").T
    d_m = np.array([m_d[1], m_d[0]])
    a_m = np.array([m_a[1], m_a[0]])
    w_m = np.array([m_w[1], m_w[0]])

    label = np.load("./data/FreeBase/labels.npy")
    uni_el, counts = np.unique(label, return_counts=True)
    for element, count in zip(uni_el, counts):
        print(f"Element: {element}, Count: {count}")

    # 构建数据集
    data = HeteroData()
    data['m'].x = x_dict['m'].to(torch.float32)
    data['d'].x = x_dict['d'].to(torch.float32)
    data['a'].x = x_dict['a'].to(torch.float32)
    data['w'].x = x_dict['w'].to(torch.float32)
    data['m'].y = torch.from_numpy(label).to(torch.int64)
    data['m', 'to', 'd'].edge_index = torch.tensor(m_d, dtype=torch.int64).contiguous()
    data['d', 'to', 'm'].edge_index = sort_edge_index(torch.tensor(d_m, dtype=torch.int64))
    data['m', 'to', 'a'].edge_index = torch.tensor(m_a, dtype=torch.int64).contiguous()
    data['a', 'to', 'm'].edge_index = sort_edge_index(torch.tensor(a_m, dtype=torch.int64))
    data['m', 'to', 'w'].edge_index = torch.tensor(m_w, dtype=torch.int64).contiguous()
    data['w', 'to', 'm'].edge_index = sort_edge_index(torch.tensor(w_m, dtype=torch.int64))

    mps = [[('m', 'd'), ('d', 'm')], [('m', 'a'), ('a', 'm')], [('m', 'w'), ('w', 'm')]]
    transform = T.AddMetaPaths(metapaths=mps, drop_orig_edge_types=False, drop_unconnected_node_types=False)
    data = transform(data)

    load_meta_edges(data, 'FreeBase')
    hop_neighbors(data, 'FreeBase')
    multi_hop_freebase(data)

    # ratio = [0.2, 0.8, 0.4]
    # ratio = [0.5, 0.6, 0.4]

    # ratio = [0.05, 0.04, 0.05]
    # ratio = [0.1, 0.16, 0.18]
    ratio = [0.1, 0.16, 0.18]
    # ratio = [1, 1, 1]
    # data.top_adjs, data.pathsim_adj = pathsim(data.meta_adjs, data.metapath_dict.keys(), ratio)
    data.top_adjs, data.max_neis, data.pathsim_ms = pathsim(data.meta_adjs, data.metapath_dict.keys(), ratio)

    return data


def multi_hop_freebase(data):

    if not osp.exists('./data/FreeBase/big_adj_k_hop.pth'):
        # m,d,a,w
        nums = [data['m'].num_nodes, data['d'].num_nodes, data['a'].num_nodes, data['w'].num_nodes]
        adj_size = sum(nums)
        big_adj = torch.zeros((adj_size, adj_size))

        adj_m_d = torch.zeros((nums[0], nums[1]))
        edge_index = data['m', 'to', 'd'].edge_index
        adj_m_d[edge_index[0], edge_index[1]] = 1

        adj_m_a = torch.zeros((nums[0], nums[2]))
        edge_index = data['m', 'to', 'a'].edge_index
        adj_m_a[edge_index[0], edge_index[1]] = 1

        adj_m_w = torch.zeros((nums[0], nums[3]))
        edge_index = data['m', 'to', 'w'].edge_index
        adj_m_w[edge_index[0], edge_index[1]] = 1

        stop_m = nums[0]
        stop_d = nums[0] + nums[1]
        stop_a = nums[0] + nums[1] + nums[2]
        stop_w = nums[0] + nums[1] + nums[2] + nums[3]
        big_adj[:stop_m, stop_m:stop_d] = adj_m_d
        big_adj[:stop_m, stop_d:stop_a] = adj_m_a
        big_adj[:stop_m, stop_a:stop_w] = adj_m_w

        upper_tri = torch.triu(big_adj, diagonal=1)
        big_adj = upper_tri + upper_tri.t()

        k = 2
        big_adj_k_hop = torch.matrix_power(big_adj, k)

        paper_size = nums[0]
        matrix = big_adj_k_hop[:paper_size, :paper_size]

        torch.save(matrix, './data/FreeBase/big_adj_k_hop.pth')

    else:
        matrix = torch.load('./data/FreeBase/big_adj_k_hop.pth')

    m = filter_nodes_topk(matrix, k=50)
    edge_index_hop = torch.nonzero(m, as_tuple=False).t()
    data.edge_index_hop = edge_index_hop


def load_imdb_(ratio):
    # 4275 5432 2083 7313
    x_dict = {
        'm': sp.load_npz("./data/IMDB_/m_feat.npz").toarray(),
        'a': torch.eye(5432),
        'd': torch.eye(2083),
        'k': torch.eye(7313)
    }

    m_a = np.genfromtxt("./data/IMDB_/ma.txt").T
    m_d = np.genfromtxt("./data/IMDB_/md.txt").T
    m_k = np.genfromtxt("./data/IMDB_/mk.txt").T
    a_m = np.array([m_a[1], m_a[0]])
    d_m = np.array([m_d[1], m_d[0]])
    k_m = np.array([m_k[1], m_k[0]])

    label = np.load("./data/IMDB_/labels.npy") - 1
    # uni_el,counts = np.unique(label,return_counts=True)
    # for element, count in zip(uni_el, counts):
    #     print(f"Element: {element}, Count: {count}")

    # 构建数据集
    data = HeteroData()
    data['m'].x = torch.from_numpy(x_dict['m']).to(torch.float32)
    data['a'].x = x_dict['a'].to(torch.float32)
    data['d'].x = x_dict['d'].to(torch.float32)
    data['k'].x = x_dict['k'].to(torch.float32)
    data['m'].y = torch.from_numpy(label).to(torch.int64)
    data['m', 'to', 'a'].edge_index = torch.tensor(m_a, dtype=torch.int64).contiguous()
    data['a', 'to', 'm'].edge_index = sort_edge_index(torch.tensor(a_m, dtype=torch.int64))
    data['m', 'to', 'd'].edge_index = torch.tensor(m_d, dtype=torch.int64).contiguous()
    data['d', 'to', 'm'].edge_index = sort_edge_index(torch.tensor(d_m, dtype=torch.int64))
    data['m', 'to', 'k'].edge_index = torch.tensor(m_k, dtype=torch.int64).contiguous()
    data['k', 'to', 'm'].edge_index = sort_edge_index(torch.tensor(k_m, dtype=torch.int64))

    mps = [[('m', 'a'), ('a', 'm')], [('m', 'd'), ('d', 'm')], [('m', 'k'), ('k', 'm')]]
    transform = T.AddMetaPaths(metapaths=mps, drop_orig_edge_types=False, drop_unconnected_node_types=False)
    data = transform(data)

    load_meta_edges(data, 'IMDB_')
    hop_neighbors(data, 'IMDB_')
    multi_hop_imdb_(data)

    # ratio = [0.2, 0.8, 0.4]
    # ratio = [0.8, 0.3, 0.8]
    # data.top_adjs, data.pathsim_adj = pathsim(data.meta_adjs, data.metapath_dict.keys(), ratio)
    data.top_adjs, data.max_neis, data.pathsim_ms = pathsim(data.meta_adjs, data.metapath_dict.keys(), ratio)

    return data


def multi_hop_imdb_(data):

    if not osp.exists('./data/IMDB_/big_adj_k_hop.pth'):
        # m,a,d,k
        nums = [data['m'].num_nodes, data['a'].num_nodes, data['d'].num_nodes, data['k'].num_nodes]
        adj_size = sum(nums)
        big_adj = torch.zeros((adj_size, adj_size))

        adj_m_a = torch.zeros((nums[0], nums[1]))
        edge_index = data['m', 'to', 'a'].edge_index
        adj_m_a[edge_index[0], edge_index[1]] = 1

        adj_m_d = torch.zeros((nums[0], nums[2]))
        edge_index = data['m', 'to', 'd'].edge_index
        adj_m_d[edge_index[0], edge_index[1]] = 1

        adj_m_k = torch.zeros((nums[0], nums[3]))
        edge_index = data['m', 'to', 'k'].edge_index
        adj_m_k[edge_index[0], edge_index[1]] = 1

        stop_m = nums[0]
        stop_a = nums[0] + nums[1]
        stop_d = nums[0] + nums[1] + nums[2]
        stop_k = nums[0] + nums[1] + nums[2] + nums[3]
        big_adj[:stop_m, stop_m:stop_a] = adj_m_a
        big_adj[:stop_m, stop_a:stop_d] = adj_m_d
        big_adj[:stop_m, stop_d:stop_k] = adj_m_k

        upper_tri = torch.triu(big_adj, diagonal=1)
        big_adj = upper_tri + upper_tri.t()

        k = 2
        big_adj_k_hop = torch.matrix_power(big_adj, k)

        paper_size = nums[0]
        matrix = big_adj_k_hop[:paper_size, :paper_size]

        m = (matrix > 0).int()
        print(m)
        m_ = m.sum(dim=1)
        print(m_)
        print(((m_ < 50) & (m_ > 40)).int().sum())
        exit()

        torch.save(matrix, './data/IMDB_/big_adj_k_hop.pth')

    else:
        matrix = torch.load('./data/IMDB_/big_adj_k_hop.pth')

    m = filter_nodes_topk(matrix, k=51)
    edge_index_hop = torch.nonzero(m, as_tuple=False).t()
    data.edge_index_hop = edge_index_hop


def hop_neighbors(data, dataset='DBLP'):

    if dataset == 'DBLP':
        hop_adjs = {
            'paper': sp.load_npz("./data/DBLP/ap.npz").toarray(),
            'conference': sp.load_npz("./data/DBLP/apc.npz").toarray(),
            'term': sp.load_npz("./data/DBLP/apt.npz").toarray()
        }
    elif dataset == 'ACM':
        hop_adjs = {
            'author': sp.load_npz("./data/ACM/pa.npz").toarray(),
            'subject': sp.load_npz("./data/ACM/ps.npz").toarray()
        }
    elif dataset == 'AMiner':
        hop_adjs = {
            'author': sp.load_npz("./data/AMiner/pa.npz").toarray(),
            'reference': sp.load_npz("./data/AMiner/pr.npz").toarray()
        }
    elif dataset == 'IMDB':
        hop_adjs = {
            'actor': sp.load_npz("./data/IMDB/ma.npz").toarray(),
            'director': sp.load_npz("./data/IMDB/md.npz").toarray()
        }
    elif dataset == 'FreeBase':
        hop_adjs = {
            'd': sp.load_npz("./data/FreeBase/md.npz").toarray(),
            'a': sp.load_npz("./data/FreeBase/ma.npz").toarray(),
            'w': sp.load_npz("./data/FreeBase/mw.npz").toarray()
        }
    elif dataset == 'IMDB_':
        hop_adjs = {
            'a': sp.load_npz("./data/IMDB_/ma.npz").toarray(),
            'd': sp.load_npz("./data/IMDB_/md.npz").toarray(),
            'k': sp.load_npz("./data/IMDB_/mk.npz").toarray()
        }

    mps, mps_expand = data.metapath_dict.keys(), data.metapath_dict.values()
    hop_neighbors = {mp: list(mp_expand) for mp, mp_expand in zip(mps, mps_expand)}
    for mp, mps_expand in zip(mps, mps_expand):
        hops = len(mps_expand)
        for i in range(hops):
            type = mps_expand[i][1]
            if i < (hops / 2):
                hop_neighbors[mp][i] = torch.from_numpy(hop_adjs[type]).float()
            else:
                source, target = mps_expand[i][0], mps_expand[i][1]
                shape = (data.x_dict[source].shape[0], data.x_dict[target].shape[0])
                edge_index = data.edge_index_dict[source, 'to', target]
                adj = edge_index2adj(edge_index, shape)
                hop_adj = torch.matmul(hop_neighbors[mp][i - 1], adj)
                hop_neighbors[mp][i] = hop_adj

    data.hop_neighbors = hop_neighbors


def load_meta_edges(data, dataset='DBLP'):
    """
    载入基于元路径的邻接矩阵，数值代表节点间的元路径条数
    """

    if dataset == 'DBLP':
        list_meta_adj = [
            sp.load_npz("./data/DBLP/apa.npz").toarray(),
            sp.load_npz("./data/DBLP/apcpa.npz").toarray(),
            sp.load_npz("./data/DBLP/aptpa.npz").toarray()
        ]
    elif dataset == 'ACM':
        list_meta_adj = [
            sp.load_npz("./data/ACM/pap.npz").toarray(),
            sp.load_npz("./data/ACM/psp.npz").toarray()
        ]
    elif dataset == 'AMiner':
        list_meta_adj = [
            sp.load_npz("./data/AMiner/pap.npz").toarray(),
            sp.load_npz("./data/AMiner/prp.npz").toarray()
        ]
    elif dataset == 'IMDB':
        list_meta_adj = [
            sp.load_npz("./data/IMDB/mam.npz").toarray(),
            sp.load_npz("./data/IMDB/mdm.npz").toarray()
        ]
    elif dataset == 'FreeBase':
        list_meta_adj = [
            sp.load_npz("./data/FreeBase/mdm.npz").toarray(),
            sp.load_npz("./data/FreeBase/mam.npz").toarray(),
            sp.load_npz("./data/FreeBase/mwm.npz").toarray()
        ]
    elif dataset == 'IMDB_':
        list_meta_adj = [
            sp.load_npz("./data/IMDB_/mam.npz").toarray(),
            sp.load_npz("./data/IMDB_/mdm.npz").toarray(),
            sp.load_npz("./data/IMDB_/mkm.npz").toarray()
        ]

    meta_adjs = {}
    for i, mp in enumerate(data.metapath_dict.keys()):
        meta_adjs[mp] = torch.tensor(list_meta_adj[i], dtype=torch.float)

    data.meta_adjs = meta_adjs


def prepare_data(data, mps):

    edges_pos = []
    edges_neg = []

    for metapath in data.edge_types:
        edge_pos = data.edge_index_dict[metapath]
        edges_pos.append(edge_pos)
        edge_neg = negative_sampling(edges_pos[-1])
        edges_neg.append(edge_neg)

    data.edges_pos = edges_pos
    data.edges_neg = edges_neg


def fill_conference(data):

    egde_index = data.edge_index_dict['conference', 'to', 'paper']
    adj = edge_index2adj(egde_index, (data.x_dict['conference'].shape[0], data.x_dict['paper'].shape[0]))
    nei_counts = adj.sum(dim=1, keepdim=True)
    nei_counts[nei_counts == 0] = 1
    x = data.x_dict['paper']
    x = torch.matmul(adj, x)
    x = x / nei_counts
    data['conference'].x = x


def process_data(data_origin: HeteroData, mps, num_target_node):

    transform = T.AddMetaPaths(metapaths=mps, drop_orig_edge_types=False, drop_unconnected_node_types=False)
    data = transform(data_origin)

    data['conference'].x = torch.eye(20)
    # fill_conference(data)

    # edge_dict = load_dict('./data/PreProcess/edge_index_dict.pth')
    # for edge_type, edge_index in edge_dict.items():
    #     data[edge_type].edge_index = edge_index

    # data.edge_index_dict = load_dict('./data/PreProcess/edge_index_dict.pth')
    # data.meta_edges = [type for type in data.edge_index_dict.keys()]
    # data.edge_types = [type for type in data.edge_index_dict.keys()]

    # print(data)
    # print(data.metapath_dict)
    # print(data.x_dict)
    # print(data.edge_index_dict)

    return data


def load_dict(file):
    with open(file, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary


def mask_data(data, mps):

    edges_mask = {}
    edges_left = {}
    mask_edges = {'um': edgemask_um, 'dm': edgemask_dm}

    if mps:
        for metapath in mps:
            # edge_t = data[metapath].edge_index.t()
            edge_t = data.edge_index.t()
            num_nodes = data['author'].num_nodes
            edge_left, edge_mask = mask_edges['um'](0.7, edge_t, num_nodes)

            edges_mask[metapath] = edge_mask
            edges_left[metapath] = edge_left
    # edge_t = data.edge_index.t()
    # num_nodes = data.num_nodes
    # edges_left, edges_mask = mask_edges['um'](0.7, edge_t, num_nodes)

    return edges_mask, edges_left


def max_nei(adjs, mps, ratio):

    neis = {}
    top = 150
    # avg_neis=torch.zeros(random.choice(list(adjs.values())).shape[0])

    # neis_ = [20, 200, 200]
    # ratio = [0.1, 0.16, 0.18]
    B = torch.zeros(random.choice(list(adjs.values())).shape)
    for i, mp in zip(range(len(mps)), mps):
        A = adjs[mp]
        A = (A > 0).to(torch.int)
        B += A

        neis[mp] = A.sum(dim=1)

        max = neis[mp].max() * ratio[i]
        print(f'{mp}-max:', max)

        # min_neis.append(neis[mp])
        # avg_neis += neis[mp]
        # max = neis_[i]
        # print(neis[mp].max())
        # max = neis[mp].median() / 2
        # print(neis[mp].sum() / neis[mp].shape[0])

        mask = neis[mp] > max
        neis[mp][mask] = max
    #     print(neis[mp].sum())

    # avg_neis = avg_neis / len(mps)
    # min_neis = torch.stack(min_neis, dim=1)
    B = (B > 0).to(torch.int)
    top_nei = B.sum(dim=1)
    top_nei[top_nei > top] = top

    # min_neis = torch.min(min_neis, dim=1)[0]

    # exit()
    return neis, top_nei


def pathsim(adjs, mps, ratio):

    def pathsim_tops(A, pathsim_m, max_neis, mp):
        idx_x_list = [np.ones(max_neis[mp][i], dtype=np.int32) * i for i in range(A.shape[0])]
        idx_x = np.concatenate(idx_x_list)

        # m = np.argsort(pathsim_m, axis=1)[:, ::-1]
        m = np.argsort(pathsim_m, axis=1)[:, ::-1].copy()
        idx_y_list = [m[i, :max_neis[mp][i]] for i in range(A.shape[0])]
        idx_y_list = [np.sort(idx_y_list[i]) for i in range(A.shape[0])]
        idx_y = np.concatenate(idx_y_list)

        new = []
        for i, j in zip(idx_x, idx_y):
            new.append(A[i, j])
        new = np.array(new, dtype=np.int32)

        print(mp, len(new))

        adj_new = sp.coo_matrix((new, (idx_x, idx_y)), shape=A.shape).toarray()

        adj_new = torch.tensor(adj_new, dtype=torch.int)
        # return adj_new
        return adj_new, m

    def pathsim_sum_tops(total_m, min_neis):

        idx_x_list = [np.ones(min_neis[i], dtype=np.int32) * i for i in range(A.shape[0])]
        idx_x = np.concatenate(idx_x_list)

        m = np.argsort(total_m, axis=1)[:, ::-1]
        idx_y_list = [m[i, :min_neis[i]] for i in range(m.shape[0])]
        idx_y_list = [np.sort(idx_y_list[i]) for i in range(m.shape[0])]
        idx_y = np.concatenate(idx_y_list)

        new = []
        for i, j in zip(idx_x, idx_y):
            new.append(A[i, j])
        new = np.array(new, dtype=np.int32)
        adj_new = sp.coo_matrix((new, (idx_x, idx_y)), shape=m.shape).toarray()

        adj_new = torch.tensor(adj_new, dtype=torch.int)
        return adj_new

    top_adjs = {}
    pathsim_ms = {}
    total_m = np.zeros(shape=random.choice(list(adjs.values())).shape)

    max_neis, path_nei = max_nei(adjs, mps, ratio)
    for mp in mps:
        A = np.array(adjs[mp])

        x, y = A.nonzero()
        value = []
        for i, j in zip(x, y):
            value.append(2 * A[i, j] / (A[i, i] + A[j, j]))

        pathsim_m = sp.coo_matrix((value, (x, y)), shape=A.shape).toarray()

        total_m += pathsim_m
        top_adjs[mp], pathsim_ms[mp] = pathsim_tops(A, pathsim_m, max_neis, mp)
        pathsim_ms[mp] = torch.tensor(pathsim_ms[mp], dtype=torch.int)

    # path_adj = pathsim_sum_tops(total_m, path_nei)
    # return top_adjs, path_adj
    return top_adjs, max_neis, pathsim_ms


def load_data(dataset, ratio=None):
    if dataset == "DBLP":
        data = load_dblp(ratio)

        split_path = osp.join(path, 'data/DBLP/')
        train = [np.load(split_path + "train_" + str(i) + ".npy") for i in [20, 40, 60]]
        val = [np.load(split_path + "val_" + str(i) + ".npy") for i in [20, 40, 60]]
        test = [np.load(split_path + "test_" + str(i) + ".npy") for i in [20, 40, 60]]

        data.train_index = train
        data.valid_index = val
        data.test_index = test

    elif dataset == "ACM":
        data = load_acm(ratio)

        split_path = osp.join(path, 'data/ACM/')
        train = [np.load(split_path + "train_" + str(i) + ".npy") for i in [20, 40, 60]]
        val = [np.load(split_path + "val_" + str(i) + ".npy") for i in [20, 40, 60]]
        test = [np.load(split_path + "test_" + str(i) + ".npy") for i in [20, 40, 60]]

        data.train_index = train
        data.valid_index = val
        data.test_index = test

    elif dataset == "AMiner":
        data = load_aminer()

        split_path = osp.join(path, 'data/AMiner/')
        train = [np.load(split_path + "train_" + str(i) + ".npy") for i in [20, 40, 60]]
        val = [np.load(split_path + "val_" + str(i) + ".npy") for i in [20, 40, 60]]
        test = [np.load(split_path + "test_" + str(i) + ".npy") for i in [20, 40, 60]]

        data.train_index = train
        data.valid_index = val
        data.test_index = test

    elif dataset == "IMDB":
        data = load_imdb()

        # split_indices(data['movie'].y, train_size=20, val_size=1000, test_size=1000)

        split_path = osp.join(path, 'data/IMDB/')
        train = [np.load(split_path + "train_" + str(i) + ".npy") for i in [20]]
        val = [np.load(split_path + "val_" + str(i) + ".npy") for i in [20]]
        test = [np.load(split_path + "test_" + str(i) + ".npy") for i in [20]]

        data.train_index = train
        data.valid_index = val
        data.test_index = test

    elif dataset == "FreeBase":
        data = load_freebase()

        split_path = osp.join(path, 'data/FreeBase/')
        train = [np.load(split_path + "train_" + str(i) + ".npy") for i in [20, 40, 60]]
        val = [np.load(split_path + "val_" + str(i) + ".npy") for i in [20, 40, 60]]
        test = [np.load(split_path + "test_" + str(i) + ".npy") for i in [20, 40, 60]]

        data.train_index = train
        data.valid_index = val
        data.test_index = test

    elif dataset == "IMDB_":
        data = load_imdb_(ratio)

        split_path = osp.join(path, 'data/IMDB_/')
        train = [np.load(split_path + "train_" + str(i) + ".npy") for i in [20, 40, 60]]
        val = [np.load(split_path + "val_" + str(i) + ".npy") for i in [20, 40, 60]]
        test = [np.load(split_path + "test_" + str(i) + ".npy") for i in [20, 40, 60]]

        data.train_index = train
        data.valid_index = val
        data.test_index = test

    return data
