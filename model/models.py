from itertools import chain
from functools import partial
import torch
import torch.nn as nn
from typing import Callable

from model.modules import HAN, SemanticView, FeatureView, AttentionInfo, AttentionSemantic, GraphSAGE, GAT, GCN, MLP, LinkPredictor, GEN
from torch_geometric.data import HeteroData
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Linear
from utils.process import adj2edge_index, edge_index2adj, cosine_similarity, cosine_similarity_matrix, adjacency_matrix_top_similarity, adjacency_matrix_from_similarity, aggregate_mean
from torch_sparse import SparseTensor
from utils.process import pt_rand_mask


class PreModel(nn.Module):
    def __init__(self, args, x_dict):
        super(PreModel, self).__init__()

        # self.num_metapath = num_metapath
        # self.focused_feature_dim = focused_feature_dim
        # self.hidden_dim = args.hidden_dim
        # self.num_layers = args.num_layers
        # self.num_heads = args.num_heads
        # self.num_out_heads = args.num_out_heads
        # self.activation = args.activation
        # self.feat_drop = args.feat_drop
        # self.attn_drop = args.attn_drop
        # self.negative_slope = args.negative_slope
        # self.residual = args.residual
        # self.norm = args.norm
        # self.feat_mask_rate = args.feat_mask_rate
        # self.encoder_type = args.encoder
        # self.decoder_type = args.decoder
        # self.loss_fn = args.loss_fn
        # self.enc_dec_input_dim = self.focused_feature_dim
        # assert self.hidden_dim % self.num_heads == 0
        # assert self.hidden_dim % self.num_out_heads == 0

        # num head: encoder
        # if self.encoder_type in ("gat", "dotgat", "han"):
        #     enc_num_hidden = self.hidden_dim // self.num_heads
        #     enc_nhead = self.num_heads
        # else:
        #     enc_num_hidden = self.hidden_dim
        #     enc_nhead = 1

        # num head: decoder
        # if self.decoder_type in ("gat", "dotgat", "han"):
        #     dec_num_hidden = self.hidden_dim // self.num_out_heads
        #     dec_nhead = self.num_out_heads
        # else:
        #     dec_num_hidden = self.hidden_dim
        #     dec_nhead = 1
        # dec_in_dim = self.hidden_dim

        # encoder
        # self.enc = setup_module(
        #     metadata=args.metadata,
        #     in_channels=args.in_channels,
        #     out_channels=args.out_channels,
        #     n_layer=args. n_layer,
        #     negative_slope=args.negative_slope,
        #     dropout=args.dropout,
        #     hidden_channels=args.hidden_channels,
        #     n_head=args.n_head
        # )

        # Cora
        # self.encoder = GCN(1433,128,128,2,0.5)
        # self.decoder = LPDecoder(128,256,1,2,2,0.5,de_v='v1')

        # DBLP
        # self.encoder = GCN(334,128,128,2,0.5)
        # self.decoder = LPDecoder(128, 256, 1, 2, 2, 0.5, de_v='v1')

        # decoder
        # self.decoder = LinkPredictor(
        #     in_channels=128, out_channels=1, num_layer=2, dropout=0.5, hidden_channels=256)

        # self.node_clf = NodeClassifier(
        #     in_channels=args.dim * 2,
        #     hidden_channels=args.hidden_channels,
        #     num_class=args.num_class)

        # self.node_clf = LogReg(ft_in=args.dim * 2, nb_classes=args.num_class)

        self.lins = torch.nn.ModuleDict()
        for node_type, x in x_dict.items():
            self.lins[node_type] = Linear(x.size(1), args.dim)


def setup_module(metadata, in_channels, out_channels, n_layer, negative_slope, dropout, hidden_channels, n_head) -> nn.Module:
    module = HAN(
        metadata=metadata,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layer=n_layer,
        negative_slope=negative_slope,
        dropout=dropout,
        n_head=n_head
    )

    return module


class MVHGCL(nn.Module):
    def __init__(self, args, data, hop_dict):
        super(MVHGCL, self).__init__()

        # 模型所需数据
        self.dataset = args.dataset
        self.target_node = args.target_node
        self.threshold = args.threshold
        self.data = data
        self.x_dict = data.x_dict
        self.mps = list(data.metapath_dict.keys())
        self.device = args.device
        self.device1 = args.device1

        # 对比学习参数
        self.nei_t = args.nei_t
        self.ep_f = args.ep_f
        self.ep_s = args.ep_s
        self.edge_index_hop = data.edge_index_hop
        self.co = args.co
        num_node = data.x_dict[self.target_node].shape[0]
        self.num_node = num_node
        self.adjs = {mp: edge_index2adj(data.edge_index_dict[mp], (num_node, num_node)).to(self.device) for mp in self.mps}
        self.dim = args.dim
        self.mask_ratio = args.mask_ratio

        # 模型初始化
        self.lins = torch.nn.ModuleDict()
        for node_type, x in self.x_dict.items():
            self.lins[node_type] = Linear(x.size(1), args.dim)

        self.SE = SemanticView(args, data, self.target_node, self.device1)
        self.FE = FeatureView(data, args.dim, args.threshold, self.lins, hop_dict, self.target_node)
        self.FE.to(self.device)
        self.SE.to(self.device1)


    def forward(self):
        data_s = self.data_s
        edge_index_sim = self.edge_index_sim

        x_dict_align = self.dimension_align(self.x_dict_mean)

        self.x_to(x_dict_align)
        z_s, z_s_mp = self.SE(data_s.x_dict, data_s.edge_index_dict)
        z_f, z_f_mp = self.FE(x_dict_align, edge_index_sim)

        z_s, z_s_mp = self.x_back(z_s, z_s_mp)

        loss = self.contrast(z_f, z_s, z_f_mp, z_s_mp)


        return loss

    def homogeneous_view(self, edge_index_hop=None, dataset='DBLP'):
        target_node = self.target_node
        threshold = self.threshold
        x_dict = self.x_dict

        x_dict_mean = {}
        if dataset == 'IMDB_':
            for node_type, x in x_dict.items():
                similarity_matrix = cosine_similarity_matrix(x)

                if node_type == target_node:
                    adjacency_matrix = adjacency_matrix_top_similarity(similarity_matrix, 50)
                    edge_index_target = adjacency_matrix.to_sparse().indices()
                else:
                    adjacency_matrix = adjacency_matrix_from_similarity(similarity_matrix, threshold=threshold)

                aggregates = aggregate_mean(adjacency_matrix, x)

                x_dict_mean[node_type] = x + aggregates
        elif dataset in ['AMiner', 'FreeBase']:
            for node_type, x in x_dict.items():
                aggregates = 0
                x_dict_mean[node_type] = x + aggregates
            edge_index_target = self.data.edge_index_hop
        else:
            for node_type, x in x_dict.items():
                similarity_matrix = cosine_similarity_matrix(x)

                device = similarity_matrix.device
                identity_matrix = torch.eye(similarity_matrix.size(0), device=device)
                if node_type == 'reference' or node_type == 'w':
                    x_dict_mean = x_dict
                    edge_index_target = edge_index_hop
                    adj = edge_index2adj(edge_index_hop, (x_dict[target_node].size(0), x_dict[target_node].size(0))).to(device)
                    x_dict_mean[target_node] = aggregate_mean(adj, x_dict_mean[target_node])
                    break

                if node_type == 'conference' or node_type == 'subject':
                    adjacency_matrix = adjacency_matrix_from_similarity(similarity_matrix, threshold=threshold)
                else:
                    adjacency_matrix = adjacency_matrix_top_similarity(similarity_matrix)

                if node_type == target_node:
                    adjacency_sparse = adjacency_matrix.to_sparse()
                    edge_index = adjacency_sparse.indices()
                    edge_index_target = edge_index

                aggregates = aggregate_mean(adjacency_matrix, x)

                x += aggregates

                x_dict_mean[node_type] = x

        return x_dict_mean, edge_index_target

    def dimension_align(self, x_dict):
        x_dict_align = {}
        for node_type, x in x_dict.items():
            x_dict_align[node_type] = self.lins[node_type](x)
        return x_dict_align

    def data4SE(self):
        device1 = self.device1
        data = self.data

        self.data_s = HeteroData()
        for type, x in self.x_dict.items():
            self.data_s[type].x = x.to(device1)
        for mp in self.mps:
            # 冗余边掩码策略
            self.data_s[mp].edge_index = adj2edge_index(data.top_adjs[mp].to(device1))
            print(self.data_s[mp].edge_index.shape)

    def mask_edges(self, mp, mask_ratio=0.6):
        data = self.data
        device1 = self.device1

        edge = data.edge_index_dict[mp]
        num_edge = edge.shape[1]
        edge_t = edge.t()

        _, edge_left = pt_rand_mask(num_edge, int(num_edge * mask_ratio), edge_t)
        return edge_left.to(device1)

    def x_to(self, x_dict):
        for node_type, x in x_dict.items():
            self.data_s[node_type].x = x.clone().to(self.device1)

    def x_back(self, z_s, z_s_mp):
        z_s = z_s.to(self.device)
        z_s_mp = {mp: z_s_mp[mp].to(self.device) for mp in self.mps}
        return z_s, z_s_mp

    def prep(self):
        self.x_dict_mean, self.edge_index_sim = self.homogeneous_view(None, self.dataset)
        x_dict_align = self.dimension_align(self.x_dict_mean)
        self.FE.preprocess(x_dict_align)
        self.data4SE()

        return self.x_dict_mean

    # def get_embeds(self):
    #     data_s = self.data_s
    #     edge_index_sim = self.edge_index_sim

    #     x_dict_align = self.dimension_align(self.x_dict_mean)

    #     self.x_to(x_dict_align)
    #     z_s, z_s_mp = self.SE(data_s.x_dict, data_s.edge_index_dict)
    #     z_f, z_f_mp = self.FE(x_dict_align, edge_index_sim)

    #     z_s, z_s_mp = self.x_back(z_s, z_s_mp)
    #     return z_s, z_s_mp, z_f, z_f_mp

    def get_embeds(self):
        data_s = self.data_s
        edge_index_sim = self.edge_index_sim

        x_dict_align = self.dimension_align(self.x_dict_mean)

        self.x_to(x_dict_align)
        z_s, z_s_mp = self.SE(data_s.x_dict, data_s.edge_index_dict)
        z_f, z_f_mp = self.FE(x_dict_align, edge_index_sim)

        z_s, z_s_mp = self.x_back(z_s, z_s_mp)
        return z_s, z_s_mp, z_f, z_f_mp

    def contrast(self, z_f, z_s, z_f_mp, z_s_mp):

        mask_f = self.get_mask(z_f, z_s, 'f')
        mask_s = self.get_mask(z_s, z_f, 's')
        mask_mp = self.get_mask_mp(z_f_mp, z_s_mp)
        # mask_mp, mask_mp_ = self.get_mask_mp_(z_f_mp, z_s_mp)

        # 主对比
        mask = [mask_f, mask_s]
        loss = self.loss_cl(z_f, z_s, mask, 0.5, self.infonce)

        # 元路径对比
        loss_mp = 0
        for mp in self.mps:
            mask = mask_mp[mp]
            loss_mp += self.loss_cl(z_f_mp[mp], z_s_mp[mp], mask, 0.5, self.infonce_mp)

        loss += loss_mp
        return loss

    def contrast_(self, z_f, z_s, z_f_mp, z_s_mp, z):

        mask_f = self.get_mask(z_f, z_s, 'f')
        mask_s = self.get_mask(z_s, z_f, 's')
        mask_mp = self.get_mask_mp(z_f_mp, z_s_mp)
        # mask_mp, mask_mp_ = self.get_mask_mp_(z_f_mp, z_s_mp)

        # 主对比
        mask = [mask_f, mask_s]
        loss = self.loss_cl(z_f, z_s, mask, 0.5, self.infonce)

        # 元路径对比
        loss_mp = 0
        for mp in self.mps:
            mask = mask_mp[mp]
            loss_mp += self.loss_cl(z_f_mp[mp], z_s_mp[mp], mask, 0.5, self.infonce_mp)

        mask = torch.eye(self.x_dict[self.target_node].shape[0], device=self.device)
        loss_ = self.loss_cl_(z_f, z_s, z, mask, 0.5, self.infonce_mp)

        loss = loss + loss_mp + loss_
        return loss

    def get_mask(self, z1, z2, anchor):

        ep_f = self.ep_f
        ep_s = self.ep_s
        mps = self.mps
        co = self.co
        nei_t = self.nei_t
        edge_index_hop = self.edge_index_hop
        adjs = self.adjs

        # feature
        sim_intra = cosine_similarity(z1, z1)
        pos_f_intra = adjacency_matrix_from_similarity(sim_intra, ep_f)

        sim_inter = cosine_similarity(z1, z2)
        pos_f_inter = adjacency_matrix_from_similarity(sim_inter, ep_f)
        pos_f_inter.fill_diagonal_(1)

        # semantic
        score_m = sum(adjs[mp] * c for mp, c in zip(mps, co))
        pos_s = adjacency_matrix_from_similarity(score_m, ep_s)
        pos_s_hop = edge_index2adj(edge_index_hop, (z1.size(0), z1.size(0))).to(z1.device)

        if anchor == 'f':
            pos_s_intra = pos_s_hop

            pos_s_inter = pos_s
            pos_s_inter.fill_diagonal_(1)

            pos_intra = torch.logical_and(pos_f_intra, pos_s_intra)
            pos_inter = torch.logical_and(pos_f_inter, pos_s_inter)

            # expadn pos_inter for nodes with few neighbors
            nei_counts = pos_inter.sum(dim=1)
            nodes_to_expand = nei_counts < nei_t
            expanded_neighbors = torch.logical_or(pos_inter, pos_s)
            pos_inter = torch.where(nodes_to_expand.unsqueeze(1), expanded_neighbors, pos_inter)
        elif anchor == 's':

            pos_s_intra = pos_s
            pos_s_inter = pos_s_hop
            pos_s_inter.fill_diagonal_(1)

            pos_intra = torch.logical_and(pos_f_intra, pos_s_intra)
            pos_inter = torch.logical_and(pos_f_inter, pos_s_inter)

            # 针对邻居数过少的问题拓展pos_inter
            nei_counts = pos_inter.sum(dim=1)
            nodes_to_expand = nei_counts < nei_t
            expanded_neighbors = torch.logical_or(pos_inter, pos_s)
            pos_inter = torch.where(nodes_to_expand.unsqueeze(1), expanded_neighbors, pos_inter)

        return [pos_intra, pos_inter]

    def get_mask_mp(self, z1_mp, z2_mp):
        mps = self.mps
        ep_f = self.ep_f

        adjs = self.data.top_adjs

        # adjs = {}
        # for mp in self.mps:
        #     adjs[mp] = edge_index2adj(self.data_s[mp].edge_index, (z1_mp[mp].size(0), z1_mp[mp].size(0))).to(self.device)

        mask_mp = {}

        for i, mp in enumerate(mps):
            z1, z2 = z1_mp[mp], z2_mp[mp]

            # feature
            sim_cos_z1 = cosine_similarity(z1, z1)
            pos_f_z1 = adjacency_matrix_from_similarity(sim_cos_z1, ep_f)
            pos_f_z1.fill_diagonal_(1)

            sim_cos_z2 = cosine_similarity(z2, z2)
            pos_f_z2 = adjacency_matrix_from_similarity(sim_cos_z2, ep_f)
            pos_f_z2.fill_diagonal_(1)

            # semantic
            pos_s = adjs[mp].to_dense()
            pos_s.fill_diagonal_(1)

            pos_z1_mask = torch.logical_and(pos_f_z2, pos_s)
            pos_z2_mask = torch.logical_and(pos_f_z1, pos_s)

            mask_mp[mp] = [pos_z1_mask, pos_z2_mask]
        return mask_mp

    def get_mask_mp_(self, z1_mp, z2_mp):
        mps = self.mps
        ep_f = self.ep_f
        adjs = self.data.top_adjs

        mask_mp = {}

        mask_mp_ = {}
        total = torch.ones(z1_mp[mps[0]].size(0), z1_mp[mps[0]].size(0), dtype=torch.int).to(z1_mp[mps[0]].device)

        for i, mp in enumerate(mps):
            z1, z2 = z1_mp[mp], z2_mp[mp]

            # feature
            sim_cos_z1 = cosine_similarity(z1, z1)
            pos_f_z1 = adjacency_matrix_from_similarity(sim_cos_z1, ep_f)
            pos_f_z1.fill_diagonal_(1)

            sim_cos_z2 = cosine_similarity(z2, z2)
            pos_f_z2 = adjacency_matrix_from_similarity(sim_cos_z2, ep_f)
            pos_f_z2.fill_diagonal_(1)

            # semantic
            pos_s = adjs[mp].to_dense()
            pos_s.fill_diagonal_(1)

            pos_z1_mask = torch.logical_and(pos_f_z2, pos_s)
            pos_z2_mask = torch.logical_and(pos_f_z1, pos_s)
            mask_mp[mp] = [pos_z1_mask, pos_z2_mask]

            neg_z1_mask_extra = pos_z2_mask
            neg_z2_mask_extra = pos_z1_mask
            mask_mp_[mp] = [neg_z1_mask_extra, neg_z2_mask_extra]
        return mask_mp, mask_mp_

    def loss_cl(self, z1, z2, mask, l, infonce: Callable):

        node_loss = self.node_node(z1, z2, mask, l, infonce)
        # graph_loss = node_graph(z1, z2, l, infonce_graph)

        loss = node_loss
        return loss

    def loss_cl_(self, z1, z2, z, mask, l, infonce: Callable):

        loss1 = infonce(z1, z, mask)
        loss2 = infonce(z2, z, mask)

        loss = l * loss1 + (1 - l) * loss2
        return loss

    def node_node(self, z1, z2, mask, l, nce: Callable):

        loss1 = nce(z1, z2, mask[0])
        loss2 = nce(z2, z1, mask[1])

        loss = l * loss1 + (1 - l) * loss2
        return loss

    def infonce(self, z1, z2, mask):
        def f(x): return torch.exp(x / 0.2)
        sim_intra = f(cosine_similarity(z1, z1))
        sim_inter = f(cosine_similarity(z1, z2))

        x = (sim_intra * mask[0]).sum(1) + (sim_inter * mask[1]).sum(1)
        x = torch.where(x == 0, torch.tensor(1e-8, device=x.device), x)
        y = (sim_intra.sum(1) + sim_inter.sum(1))
        y = torch.where(y == 0, torch.tensor(1e-8, device=y.device), y)

        loss = -torch.log(x / y)
        return loss.mean()

    def infonce_mp(self, z1, z2, mask):
        def f(x): return torch.exp(x / 0.2)
        sim_inter = f(cosine_similarity(z1, z2))

        x = (sim_inter * mask).sum(1)
        x = torch.where(x == 0, torch.tensor(1e-8, device=x.device), x)
        y = sim_inter.sum(1)
        y = torch.where(y == 0, torch.tensor(1e-8, device=y.device), y)

        loss = -torch.log(x / y)
        return loss.mean()

    def infonce_mp_(self, z1, z2, mask):
        def f(x): return torch.exp(x / 0.2)
        sim_inter = f(cosine_similarity(z1, z2))

        sim_intra = f(cosine_similarity(z1, z1))

        x = (sim_inter * mask[0]).sum(1) + (sim_intra * mask[1]).sum(1)
        x = torch.where(x == 0, torch.tensor(1e-8, device=x.device), x)
        y = sim_inter.sum(1) + sim_intra.sum(1)
        y = torch.where(y == 0, torch.tensor(1e-8, device=y.device), y)

        loss = -torch.log(x / y)
        return loss.mean()


class HGME(nn.Module):
    def __init__(self, edge_index, device, mask_ratio, dim):
        super(HGME, self).__init__()

        self.device = device
        self.dim = dim * 2
        self.edge_index = edge_index
        self.mask_ratio = mask_ratio

        self.encoder = MLP(self.dim, self.dim * 2, self.dim)
        self.decoder = MLP(self.dim, self.dim * 2, self.dim)

        self.encoder.to(device)
        self.decoder.to(device)

        self.criterion = self.node_loss("mse")

    def forward(self, x):

        edge_mask, edge_left = self.edge_mask()
        loss = self.edge_predict(x, edge_left, edge_mask)

        return loss

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

    def edge_mask(self):
        ratio = self.mask_ratio
        edge = self.edge_index

        num_edge = edge.shape[1]
        edge_t = edge.t()

        # random masking
        edge_mask, edge_left = pt_rand_mask(num_edge, int(num_edge * ratio), edge_t)

        return edge_mask, edge_left

    def get_embeds(self, x):
        x = self.encoder(x, self.edge_index)
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
        enc_rep = self.encoder(x)

        # ---- reconstruction ----
        recon = self.decoder(enc_rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def edge_predict(self, x, edge_left, edge_mask):

        # ---- encoding ----
        enc_rep = self.encoder(x, edge_left)

        # ---- reconstruction ----
        pos_out = self.decoder(enc_rep, edge_mask)

        pos_loss = -torch.log(pos_out + 1e-15).mean()
        loss = pos_loss
        return loss
