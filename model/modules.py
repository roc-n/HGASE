import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HANConv, SAGEConv
from torch_geometric.data import Data
from typing import Union, Dict
from torch.nn import Parameter, ParameterList, ModuleDict, ModuleList
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_undirected, add_self_loops, is_undirected, scatter, negative_sampling
from utils.process import adj2edge_index, edge_index2adj, cosine_similarity, cosine_similarity_matrix, adjacency_matrix_top_similarity, adjacency_matrix_from_similarity, aggregate_mean, pt_rand_mask
import torch.nn.init as init


class GEN(nn.Module):
    def __init__(self, edge_f, edge_s, x, device, mask_ratio, dim, num_node):
        super(GEN, self).__init__()

        self.device = device
        self.num_node = num_node
        self.dim = dim
        self.mask_ratio = mask_ratio
        self.x = x

        # self.enc_mask_token = nn.Parameter(torch.zeros(1, self.dim, device=device))
        # self.encoder = HAN(1433, 128, 128, 2, 0.5)
        # self.decoder = HAN(128, 256, 1, 2, 2, 0.5, de_v='v1')
        # self.encoder = MLP(self.dim, self.dim * 2, self.dim)
        # self.encoder = GraphSAGE(self.dim, self.dim * 2, self.dim)
        # self.decoder1 = MLP(self.dim, self.dim * 2, dim)
        # self.edge_index = edge_f

        # self.encoder = GraphSAGE(self.dim, self.dim * 2, self.dim)
        self.decoder2 = LinkPredictor(self.dim, self.dim * 2, 1, 2, 0.2)
        # self.edge_index = self.edges_gen(edge_f, edge_s)

        # self.encoder.to(device)
        # self.decoder1.to(device)
        self.decoder2.to(device)

        self.criterion = self.node_loss("mse")

    def forward(self, x_f, mask_nodes, x_s, edges):
        # x, (mask_nodes, keep_nodes) = self.node_mask(x)
        # loss1 = self.attr_predict(x_f, mask_nodes)

        # edge_mask, edge_left = self.edge_mask()
        loss2 = self.edge_predict(x_s, edges)

        # loss = self.edge_node_predict(x, edge_left, edge_mask, mask_nodes)
        return loss2

    def node_mask(self, x):

        ratio = self.mask_ratio

        num_node = x.shape[0]
        perm = torch.randperm(num_node, device=x.device)

        # random masking
        num_mask_nodes = int(ratio * num_node)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        x_ = x.clone()
        x_[mask_nodes] = 0.0
        x_[mask_nodes] += self.enc_mask_token

        return x_, (mask_nodes, keep_nodes)

    def edges_gen(self, edge_f, edge_s):
        num_node = self.num_node

        adj_f = edge_index2adj(edge_f, (num_node, num_node))
        adj_s = edge_index2adj(edge_s, (num_node, num_node))

        half_num_edges_f = torch.count_nonzero(adj_f) // 2
        half_num_edges_s = torch.count_nonzero(adj_s) // 2

        flat_adj_f = adj_f.view(-1)
        nonzero_indices_f = torch.nonzero(flat_adj_f, as_tuple=False).squeeze()
        random_indices_f = torch.randperm(len(nonzero_indices_f))
        selected_indices_f = random_indices_f[:half_num_edges_f]
        new_flat_adj_f = torch.zeros_like(flat_adj_f)
        new_flat_adj_f[nonzero_indices_f[selected_indices_f]] = flat_adj_f[nonzero_indices_f[selected_indices_f]]
        new_adj_f = new_flat_adj_f.view(num_node, num_node)

        flat_adj_s = adj_s.view(-1)
        nonzero_indices_s = torch.nonzero(flat_adj_s, as_tuple=False).squeeze()
        random_indices_s = torch.randperm(len(nonzero_indices_s))
        selected_indices_s = random_indices_s[:half_num_edges_s]
        new_flat_adj_s = torch.zeros_like(flat_adj_s)
        new_flat_adj_s[nonzero_indices_s[selected_indices_s]] = flat_adj_s[nonzero_indices_s[selected_indices_s]]
        new_adj_s = new_flat_adj_s.view(num_node, num_node)

        adj = ((new_adj_f + new_adj_s) > 0).int()
        return adj2edge_index(adj).to(self.device)

    def edge_mask(self):
        ratio = self.mask_ratio
        edge = self.edge_index

        num_edge = edge.shape[1]
        edge_t = edge.t()

        # random masking
        edge_mask, edge_left = pt_rand_mask(num_edge, int(num_edge * ratio), edge_t)

        return edge_mask, edge_left

    def get_embeds(self, x):

        x = self.encoder(x)
        # x = self.encoder(x, self.edge_index)
        return x

    def node_loss(self, loss_fn, alpha_l=None):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        # elif loss_fn == "sce":
        #     criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def attr_predict(self, x, mask_nodes):

        # ---- encoding ----
        # enc_rep = self.encoder(x)

        # ---- reconstruction ----
        recon = self.decoder(x)

        x_init = self.x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def edge_predict(self, x, edges):

        # ---- encoding ----
        # enc_rep = self.encoder(x, edge_left)

        # ---- reconstruction ----
        pos_out = self.decoder2(x, edges)

        # 可选部分:负边重建
        # new_edge_index, _ = add_self_loops(edge_mask.cpu())
        # edge = negative_sampling(new_edge_index, num_nodes=self.num_node,
        #                          num_neg_samples=edge_mask.shape[1]).to(self.device)
        # neg_out = self.decoder(enc_rep, edge)
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        pos_loss = -torch.log(pos_out + 1e-15).mean()
        loss = pos_loss
        return loss

    def edge_node_predict(self, x, edge_left, edge_mask, mask_nodes):
        # ---- encoding ----
        enc_rep = self.encoder(x, edge_left)

        # ---- reconstruction ----
        recon = self.decoder1(enc_rep)

        x_init = self.x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss1 = self.criterion(x_rec, x_init)

        pos_out = self.decoder2(enc_rep, edge_mask)
        loss2 = -torch.log(pos_out + 1e-15).mean()

        return loss1 + loss2


class FeatureView(nn.Module):
    def __init__(self, data, dim, threshold, lins, hop_dict, target_node='author'):
        super(FeatureView, self).__init__()

        self.x_dict = data.x_dict
        self.device = data.x_dict[target_node].device
        self.mps, self.mps_expand = data.metapath_dict.keys(), data.metapath_dict.values()
        self.edge_index_dict = data.edge_index_dict
        self.x_target = self.x_dict[target_node]
        self.threshold = threshold
        self.target_node = target_node
        self.dim = dim

        self.enc_mask_token = nn.Parameter(torch.zeros(1, dim, device=self.device))

        self.lins = lins

        self.convs = torch.nn.ModuleDict()
        x_dict_heter = {k: v for k, v in self.x_dict.items() if k != target_node}
        for node_type in x_dict_heter.keys():
            self.convs[node_type] = GATConv((dim, dim), dim, add_self_loops=False)

        self.att = AttentionInfo(dim, dim, len(self.x_dict), device=self.device)

        self.encoder = GraphSAGE(dim, dim * 2, dim)
        self.hop_dict = hop_dict

    def forward_(self, x_dict_align, edge_index):

        self.x_dict_from_target[self.target_node] = x_dict_align[self.target_node]
        self.x_from_target(self.edge_index_dict)
        x_heter = self.heterogeneous_view(x_dict_align, self.threshold)

        num_node = self.x_dict[self.target_node].shape[0]
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_node, num_node))
        x_dict_align[self.target_node] = self.encoder(x_dict_align[self.target_node], adj)
        x_mp = self.multi(x_heter, x_dict_align[self.target_node])

        x_all = {**x_heter, self.target_node: x_dict_align[self.target_node]}
        x = self.att(x_all)

        return x, x_mp

    def forward(self, x_dict_align, edge_index):

        self.x_dict_from_target[self.target_node] = x_dict_align[self.target_node]
        self.x_from_target(self.edge_index_dict)
        x_heter = self.heterogeneous_view(x_dict_align, self.threshold)

        num_node = self.x_dict[self.target_node].shape[0]
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_node, num_node))
        x_dict_align[self.target_node] = self.encoder(x_dict_align[self.target_node], adj)
        x_mp = self.multi(x_heter, x_dict_align[self.target_node])

        # ********
        # x_, (mask_nodes, _) = self.node_mask(x_dict_align[self.target_node])
        # x_ = self.encoder(x_, adj)
        # ********

        x_all = {**x_heter, self.target_node: x_dict_align[self.target_node]}
        x = self.att(x_all)

        return x, x_mp

    def multi(self, x_heter, x_target):
        x_mp = {mp: torch.zeros((x_target.shape[0], self.dim), device=self.device) for mp in self.mps}
        mps, mps_expand = self.mps, self.mps_expand
        for mp, mps_expand in zip(mps, mps_expand):
            hops = len(mps_expand)
            for i in range(hops - 1):
                type = mps_expand[i][1]
                x_mp[mp] += x_heter[type]
            x_mp[mp] += x_target
            x_mp[mp] = x_mp[mp] / hops
        return x_mp

    def preprocess(self, x_dict_align):
        self.x_dict_from_target = {key: x.detach().clone() for key, x in x_dict_align.items()}

    def heterogeneous_view(self, x_dict_ori, threshold):

        x_dict = self.x_dict_from_target
        target_node_type = self.target_node
        convs = self.convs

        x_target = x_dict[target_node_type]

        heter_info_dict = {}
        for node_type, x in x_dict.items():
            if node_type != target_node_type:
                similarity_matrix = cosine_similarity(x_target, x)

                if node_type == 'conference' or node_type == 'subject':
                    # adjacency_matrix = adjacency_matrix_from_similarity(similarity_matrix, threshold=0.1)
                    adjacency_matrix = similarity_matrix
                # elif node_type in ['a','d','k']:
                    # adjacency_matrix = adjacency_matrix_from_similarity(similarity_matrix, threshold=0.9)
                else:
                    adjacency_matrix = adjacency_matrix_top_similarity(similarity_matrix)

                adjacency_sparse = adjacency_matrix.to_sparse()
                edge_index = adjacency_sparse.indices()
                edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

                heter_info = convs[node_type]((x_dict_ori[node_type], x_target), edge_index)

                heter_info_dict[node_type] = heter_info

        return heter_info_dict

    def reset_parameters(model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                for param in layer.parameters():
                    if param.dim() > 1:
                        init.xavier_uniform_(param)
                    else:
                        init.zeros_(param)

    def x_from_target(self, edge_index_dict):

        hop_dict = self.hop_dict
        source = hop_dict[0]
        hop = 1
        while hop in hop_dict:
            for source_type in source:
                target = hop_dict[hop]

                for target_type in target:
                    key = (source_type, 'to', target_type)
                    if key not in edge_index_dict:
                        continue
                    edge_index = edge_index_dict[key]
                    index = edge_index[1]

                    x_source = self.x_dict_from_target[source_type][edge_index[0]]
                    out = scatter(x_source, index, dim=0, reduce='mean')
                    self.x_dict_from_target[target_type][index.unique()] = out[index.unique()]

            source = hop_dict[hop]
            hop += 1

    def encoding_mask_noise(self, x, mask_rate=0.3, replace_rate=0):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=self.x_target.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            # 开辟新的内存块存储张量
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def get_param(self):
        params = {0: {}, 1: {}}
        for name, param in self.encoder.conv1.named_parameters():
            params[0][name] = param
        for name, param in self.encoder.conv2.named_parameters():
            params[1][name] = param
        return params

    def set_param(self, params):
        # for name, _ in self.encoder.conv1.named_parameters():
        #     param = params[0][name]
        #     self.encoder.conv1.get_parameter(name).data = param.data
        for name, _ in self.encoder.conv2.named_parameters():
            param = params[1][name]
            self.encoder.conv2.get_parameter(name).data = param.data

    def node_mask(self, x):

        ratio = 0.6
        num_node = x.shape[0]
        perm = torch.randperm(num_node, device=x.device)

        # random masking
        num_mask_nodes = int(ratio * num_node)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        x_ = x.clone()
        x_[mask_nodes] = 0.0
        x_[mask_nodes] += self.enc_mask_token

        return x_, (mask_nodes, keep_nodes)


class SemanticView(nn.Module):
    def __init__(self, args, data, target_node, device):
        super(SemanticView, self).__init__()

        self.target_node = target_node
        self.device = device
        dim = args.dim
        self.dim = dim
        self.hop_neighbors = {key: [val.to(device) for val in vals] for key, vals in data.hop_neighbors.items()}

        self.mps, self.mps_expand = data.metapath_dict.keys(), data.metapath_dict.values()
        mps, mps_expand = self.mps, self.mps_expand
        self.hop_m = ModuleDict()
        for mp in mps:
            self.hop_m[str(mp)] = ModuleDict()
        for mp, mps_expand in zip(mps, mps_expand):
            hops = len(mps_expand)
            for i in range(hops):
                type = mps_expand[i][1]
                self.hop_m[str(mp)][str(type) + str(i)] = Linear(dim, dim)

        # self.encoder = GCN(args.in_channels, args.hidden_channels, args.out_channels)
        # self.encoder = GAT(args.in_channels, args.hidden_channels, dim)
        self.encoder = GraphSAGE(args.in_channels, args.hidden_channels, dim)
        # self.encoder_ = GraphSAGE(args.in_channels, args.hidden_channels, dim)

        score_m = sum(data.top_adjs[mp] * c for mp, c in zip(self.mps, args.co))
        pos_s = adjacency_matrix_from_similarity(score_m, args.ep_s)
        self.edge_index = adj2edge_index(pos_s)
        # self.encoder = ModuleDict()
        # for mp in mps:
        #     # self.encoder[str(mp)] = GAT(args.in_channels, args.hidden_channels, args.out_channels)
        #     self.encoder[str(mp)] = GraphSAGE(args.in_channels, args.hidden_channels, dim)
        # self.mp_m = ModuleDict()

        self.att = AttentionSemantic(args.dim, args.dim, len(mps), device)

    def forward_(self, x_dict, edge_index_dict):

        x_mp = self.hop_info(x_dict)
        x = self.att(x_mp)
        for mp in self.mps:
            num_node = x_dict[self.target_node].shape[0]
            adj = SparseTensor.from_edge_index(edge_index_dict[mp], sparse_sizes=(num_node, num_node))
            x_mp[mp] = self.encoder(x_mp[mp], adj)

        return x, x_mp

    def get_param(self):
        params = {0: {}, 1: {}}
        for name, param in self.encoder.conv1.named_parameters():
            params[0][name] = param
        for name, param in self.encoder.conv2.named_parameters():
            params[1][name] = param
        return params

    def set_param(self, params):
        # for name, _ in self.encoder.conv1.named_parameters():
        #     param = params[0][name]
        #     self.encoder.conv1.get_parameter(name).data = param.data
        for name, _ in self.encoder.conv2.named_parameters():
            param = params[1][name]
            self.encoder.conv2.get_parameter(name).data = param.data

    def forward(self, x_dict, edge_index_dict):

        # x_mp = {}
        # for mp in self.mps:

        #     num_node = x_dict[self.target_node].shape[0]
        #     adj = SparseTensor.from_edge_index(edge_index_dict[mp], sparse_sizes=(num_node, num_node))
        #     x_mp[mp] = self.encoder(x_dict[self.target_node], adj)

        # x_mp[mp] = self.encoder(x_mp[mp], edge_index_dict[mp])
        # x_dict[self.target_node] = x
        x_mp, x_mp_ = self.hop_info__(x_dict)

        x_ = {mp: x_mp[mp] + x_mp_[mp] for mp in self.mps}
        x = self.att(x_)

        for mp in self.mps:
            num_node = x_dict[self.target_node].shape[0]
            adj = SparseTensor.from_edge_index(edge_index_dict[mp], sparse_sizes=(num_node, num_node))
            x_mp[mp] = self.encoder(x_mp[mp], adj)
            # x_mp[mp] = self.encoder[str(mp)](x_mp[mp], adj)

        # x = self.att(x_dict_)
        # x_ = torch.sum(torch.stack([x_mp_[mp] for mp in self.mps]), dim=0)

        # x = self.att(x_mp) + self.att(x_mp_)
        # x = self.att(x_mp) + self.att(x_mp_)
        # x = self.att(x_mp_)
        # x_ = torch.sum(torch.stack([x_mp[mp] for mp in self.mps]), dim=0)
        # x += x_
        return x, x_mp

    def forward_ablation(self, x_dict, edge_index_dict):

        _, x_mp_ = self.hop_info__(x_dict)

        # x_ = {mp: x_mp[mp] + x_mp_[mp] for mp in self.mps}
        x = self.att(x_mp_)

        for mp in self.mps:
            num_node = x_dict[self.target_node].shape[0]
            adj = SparseTensor.from_edge_index(edge_index_dict[mp], sparse_sizes=(num_node, num_node))
            x_mp_[mp] = self.encoder(x_mp_[mp], adj)

        return x, x_mp_

    def forward_another(self, x_dict, edge_index_dict):

        x_mp, x_mp_ = self.hop_info__(x_dict)

        # x_ = {mp: x_mp[mp] + x_mp_[mp] for mp in self.mps}
        # x = self.att(x_)

        for mp in self.mps:
            num_node = x_dict[self.target_node].shape[0]
            adj = SparseTensor.from_edge_index(edge_index_dict[mp], sparse_sizes=(num_node, num_node))
            x_mp[mp] = self.encoder(x_mp[mp], adj)
            x_mp_[mp]=self.encoder_(x_mp_[mp],adj)

        # edges, edge_left = self.edge_mask()

        x = self.att(x_mp_)
        # x_ = self.encoder(x_, edge_left.to(self.device))

        return x, x_mp

    def edge_mask(self):
        ratio = 0.6
        edge = self.edge_index

        num_edge = edge.shape[1]
        edge_t = edge.t()

        # random masking
        edge_mask, edge_left = pt_rand_mask(num_edge, int(num_edge * ratio), edge_t)

        return edge_mask, edge_left

    def forward_aminer(self, x_dict, edge_index_dict):

        x_mp, x_mp_ = self.hop_info__(x_dict)
        x = self.att(x_mp)

        return x, x_mp

    def forward_freebase(self, x_dict, edge_index_dict):

        x_mp, x_mp_ = self.hop_info__(x_dict)

        x_ = {mp: x_mp[mp] + x_mp_[mp] for mp in self.mps}
        x = self.att(x_)

        return x, x_mp

    def hop_info(self, x_dict):

        target_node = self.target_node
        hop_neighbors = self.hop_neighbors
        mps, mps_expand = self.mps, self.mps_expand

        row = x_dict[target_node].shape[0]
        col = self.dim

        x_dict_mp = {}
        for mp, mps_expand in zip(mps, mps_expand):
            hops = len(mps_expand)
            x_hops = torch.zeros((row, col), device=self.device)
            for i in range(hops):
                type = mps_expand[i][1]
                trans_M = self.hop_m[str(mp)][type + str(i)]
                x = x_dict[type]
                adj = hop_neighbors[mp][i]

                x_ = torch.matmul(adj, x)
                x_hop = trans_M(x_)

                nei_counts = adj.sum(dim=1, keepdim=True)
                nei_counts[nei_counts == 0] = 1
                x_hop = x_hop / nei_counts

                # alpha = 1 / (i + 1)
                # x_hops += (x_hop * alpha)
                x_hops += x_hop

            x_dict_mp[mp] = x_hops + x_dict[target_node]
            # x_dict_mp[mp] = x_hops

        return x_dict_mp

    def hop_info_(self, x_dict):

        target_node = self.target_node
        hop_neighbors = self.hop_neighbors
        mps, mps_expand = self.mps, self.mps_expand

        row = x_dict[target_node].shape[0]
        col = self.dim

        x_dict_mp = {}
        x_dict_mp_ = {}
        for mp, mps_expand in zip(mps, mps_expand):
            hops = len(mps_expand)
            x_hops = torch.zeros((row, col), device=self.device)
            for i in range(hops - 1):
                type = mps_expand[i][1]
                trans_M = self.hop_m[str(mp)][type + str(i)]
                x = x_dict[type]
                adj = hop_neighbors[mp][i]

                x_ = torch.matmul(adj, x)
                x_hop = trans_M(x_)

                nei_counts = adj.sum(dim=1, keepdim=True)
                nei_counts[nei_counts == 0] = 1
                x_hop = x_hop / nei_counts

                # alpha = 1 / (i + 1)
                # x_hops += (x_hop * alpha)
                x_hops += x_hop

            # 末尾
            type = mps_expand[hops - 1][1]
            trans_M = self.hop_m[str(mp)][type + str(hops - 1)]
            x = x_dict[type]
            adj = hop_neighbors[mp][hops - 1]
            x_ = torch.matmul(adj, x)
            x_hop = trans_M(x_)
            nei_counts = adj.sum(dim=1, keepdim=True)
            nei_counts[nei_counts == 0] = 1
            x_hop = x_hop / nei_counts
            x_hop_target = x_hop

            # 起始
            # x_dict_mp[mp] = x_hops +x_dict[target_node]
            x_dict_mp[mp] = x_hops
            x_dict_mp_[mp] = x_hop_target + x_dict[target_node]

        return x_dict_mp, x_dict_mp_

    def hop_info__(self, x_dict):

        target_node = self.target_node
        hop_neighbors = self.hop_neighbors
        mps, mps_expand = self.mps, self.mps_expand

        row = x_dict[target_node].shape[0]
        col = self.dim

        x_dict_mp = {}
        x_dict_mp_ = {}
        for mp, mp_expand in zip(mps, mps_expand):
            hops = len(mp_expand)
            x_hops = torch.zeros((row, col), device=self.device)
            for i in range(hops):
                type = mp_expand[i][1]
                trans_M = self.hop_m[str(mp)][type + str(i)]
                x = x_dict[type]
                adj = hop_neighbors[mp][i]

                x_ = torch.matmul(adj, x)
                x_hop = trans_M(x_)

                nei_counts = adj.sum(dim=1, keepdim=True)
                nei_counts[nei_counts == 0] = 1
                x_hop = x_hop / nei_counts

                if i == hops - 1:
                    x_dict_mp_[mp] = x_hop + x_dict[target_node]
                else:
                    x_hops += x_hop
            x_dict_mp[mp] = x_hops

        return x_dict_mp, x_dict_mp_

    def reset_parameters(model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                for param in layer.parameters():
                    if param.dim() > 1:
                        init.xavier_uniform_(param)
                    else:
                        init.zeros_(param)

    def preprocess(self):
        pass


class AttentionSemantic(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_type, device):
        super(AttentionSemantic, self).__init__()

        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.atts = ParameterList()
        self.device = device
        self.num_type = num_type

        for _ in range(num_type):
            self.atts.append(Parameter(torch.empty(1, out_channels)))
            nn.init.xavier_uniform_(self.atts[-1])

        self.reset_parameters()

    def forward(self, x_dict):

        alpha_list = []
        for i, x in enumerate(x_dict.values()):
            out = self.lin(x)
            out = F.tanh(out)
            out = (out * self.atts[i]).sum(dim=-1)
            alpha = out.mean(dim=0)
            alpha_list.append(alpha.item())

        alpha = torch.tensor(alpha_list).to(self.device)
        alpha = F.softmax(alpha, dim=0)

        x_list = [x_dict[mp] for mp in x_dict.keys()]

        # [num_mp,4027,128]
        x = torch.stack(x_list, dim=0)
        alpha = alpha.view(self.num_type, 1, 1)

        x = x * alpha
        x = x.sum(dim=0)

        return x

    def reset_parameters(self):

        if self.lin is not None:
            self.lin.reset_parameters()

        for att in self.atts:
            glorot(att)


class AttentionInfo(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_type, device):
        super(AttentionInfo, self).__init__()

        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.atts = ParameterList()
        self.device = device
        self.num = num_type

        for _ in range(num_type):
            self.atts.append(Parameter(torch.empty(1, out_channels)))
            nn.init.xavier_uniform_(self.atts[-1])

        self.reset_parameters()

    def forward(self, x_dict):

        alpha_list = []
        for i, x in enumerate(x_dict.values()):
            out = self.lin(x)
            out = F.sigmoid(out)
            out = (out * self.atts[i]).sum(dim=-1)
            alpha = out.mean(dim=0)
            alpha_list.append(alpha.item())

        alpha = torch.tensor(alpha_list).to(self.device)
        alpha = F.softmax(alpha, dim=0)

        x_list = [x_dict[type] for type in x_dict.keys()]
        # [4,4027,128] for DBLP
        x = torch.stack(x_list, dim=0)
        alpha = alpha.view(self.num, 1, 1)

        x = x * alpha
        x = x.sum(dim=0)

        return x

    def reset_parameters(self):

        if self.lin is not None:
            self.lin.reset_parameters()

        for att in self.atts:
            glorot(att)


class HAN(nn.Module):
    def __init__(self, metadata, in_channels: Union[int, Dict[str, int]], out_channels: int,
                 hidden_channels=128, n_head=8, n_layer=1, negative_slope=0.2, dropout=0.2):
        super(HAN, self).__init__()

        self.HAN_layers = nn.ModuleList()

        if n_layer == 1:
            self.HAN_layers.append(HANConv(in_channels, out_channels, heads=n_head,
                                           dropout=dropout, metadata=metadata, negative_slope=negative_slope))
        else:
            self.HAN_layers.append(HANConv(in_channels, hidden_channels, heads=n_head,
                                           dropout=dropout, metadata=metadata, negative_slope=negative_slope))
            for _ in range(1, n_layer - 1):
                self.HAN_layers.append(HANConv(hidden_channels, hidden_channels, heads=n_head,
                                               dropout=dropout, metadata=metadata, negative_slope=negative_slope))
            self.HAN_layers.append(HANConv(hidden_channels, out_channels, heads=n_head,
                                           dropout=dropout, metadata=metadata, negative_slope=negative_slope))

    def reset_parameters(self):
        for gnn in self.HAN_layers:
            gnn.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        x_dicts = []
        for gnn in self.HAN_layers:
            x_dict = gnn(x_dict, edge_index_dict)
            x_dicts.append(x_dict)

        # out = self.lin(out['author'])

        return x_dicts[-1]['author']
        # return x_dicts


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layer,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layer - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, h, edge):
        src_x = h[edge[0]]
        dst_x = h[edge[1]]

        # src_x = h[1][edge[0]]
        # dst_x = h[1][edge[1]]
        x = src_x * dst_x
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class NodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_class):
        super(NodeClassifier, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_class)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))

        return xx

    # def out_embed(self, x, adj_t):
    #     xx = []
    #     for conv in self.convs[:-1]:
    #         x = conv(x, adj_t)
    #         x = F.relu(x)
    #         x = F.dropout(x, p=self.dropout, training=self.training)
    #         xx.append(x)
    #     x = self.convs[-1](x, adj_t)
    #     xx.append(F.relu(x))

    #     return xx

    def out_embed(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))

        xx[0].detach()
        xx[1].detach()
        return xx


class LPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout, de_v='v1'):
        super(LPDecoder, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        if de_v == 'v1':
            self.lins.append(torch.nn.Linear(
                in_channels * n_layer, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(
                    hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        elif de_v == 'v2':
            self.lins.append(torch.nn.Linear(
                in_channels * n_layer, in_channels * n_layer))
            self.lins.append(torch.nn.Linear(
                in_channels * n_layer, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(
                in_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(
                hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)


        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layer=2):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layer - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = 0.5

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

        # self.bn1 = nn.BatchNorm1d(hidden_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x=self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layer=2, heads=1):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))

        for _ in range(num_layer - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))

        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1))
        self.dropout = 0.5

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(nn.Linear(in_channels, hidden_channels))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))

        # 输出层
        self.layers.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

        self.bn1 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)

            x = self.bn1(x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layers[-1](x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
