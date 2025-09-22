import time
import pickle
import torch
import torch.nn as nn
from torch.nn import Parameter, ParameterList
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.data import Data, HeteroData

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import KFold
from sklearn import svm

from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
import copy
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from torch_geometric.nn import MeanAggregation
from torch_geometric.utils import to_undirected, add_self_loops, is_undirected
from utils.data import load_data
from torch_geometric.nn import GATConv
from utils.loss import sce_loss

from torch_sparse import SparseTensor
from model.models import PreModel
from utils.params import build_args


class FeatureView(nn.Module):
    def __init__(self, x_dict, threshold, target_node_type='author'):
        super(FeatureView, self).__init__()

        self.x_dict = x_dict
        self.x_target = x_dict[target_node_type]
        self.threshold = threshold
        self.target_node_type = target_node_type

        self.enc_mask_token = nn.Parameter(torch.zeros(1, 668))

        self.lin_target = Linear(x_dict[target_node_type].size(1), 128)
        self.lins = torch.nn.ModuleDict()
        for node_type, x in x_dict.items():
            self.lins[node_type] = Linear(x.size(1) * 2, 128)

        self.convs = torch.nn.ModuleDict()
        x_dict_heter = {k: v for k, v in x_dict.items() if k != target_node_type}
        for node_type, x in x_dict_heter.items():
            self.convs[node_type] = GATConv((128, 128), 128, add_self_loops=False)

        self.att = AttentionInfo(128, 64, 4)

        self.encoder = GraphSAGE(128, 256, 64)

    def forward(self, x_dict_mean, edge_index):

        x_dict = x_dict_mean.copy()
        # x_dict['author'], (mask_nodes, keep_nodes) = self.encoding_mask_noise(x_dict_mean[self.target_node_type])

        x_dict_homo, x_target = dimension_transform(x_dict, self.x_target, self.lins, self.lin_target)

        x_heter = heterogeneous_view(x_dict_homo, self.convs, self.target_node_type, self.threshold)

        x_all = {**x_heter, 'author': x_target}

        x = self.att(x_all)
        x = self.encoder(x_dict_homo['author'], edge_index)

        # return x, (mask_nodes, keep_nodes)
        return x

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


class NodeDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layer, dropout):
        super(NodeDecoder, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layer - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train_node(view, decoder, x_dict_mean, edge_index, optimizer):

    optimizer.zero_grad()

    x, (mask_nodes, keep_nodes) = view(x_dict_mean, edge_index)
    x_rec = decoder(x)

    x_init = x_dict_mean['author'][mask_nodes]
    x_rec = x_rec[mask_nodes]
    loss = sce_loss(x_rec, x_init, alpha=3)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(view.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def dimension_transform(x_dict, x_target, lins, lin_target):
    x_dict_result = {}
    for node_type, x in x_dict.items():
        x_dict_result[node_type] = lins[node_type](x)

    x_target_result = lin_target(x_target)
    return x_dict_result, x_target_result


def cosine_similarity_matrix(features):
    # similarity = 1 - squareform(pdist(features, metric='cosine'))
    # print(similarity)

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


def mean_aggregate(edge_index, x):
    # edge_index[0] 是源节点，edge_index[1] 是目标节点
    row, col = edge_index

    # 初始化 MeanAggregation
    aggr = MeanAggregation()

    out = aggr(x[row], col, dim_size=x.size(0))

    return out


def aggregate_mean(adjacency_matrix, x):
    # 将 adjacency_matrix 转换为稀疏张量
    adjacency_sparse = adjacency_matrix.to_sparse()

    # 使用稀疏矩阵乘法计算聚合特征
    out = torch.sparse.mm(adjacency_sparse, x)

    # 计算每个节点的度数
    degrees = adjacency_matrix.sum(dim=1).unsqueeze(1)

    # 避免除以零
    degrees[degrees == 0] = 1

    # 归一化聚合特征
    out = out / degrees

    return out


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class AttentionInfo(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_type):
        super(AttentionInfo, self).__init__()

        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.atts = ParameterList()

        for _ in range(num_type):
            self.atts.append(Parameter(torch.empty(1, out_channels)))

        self.reset_parameters()

    def reset_parameters(self):

        if self.lin is not None:
            self.lin.reset_parameters()

        for att in self.atts:
            glorot(att)

    def forward(self, x_dict):

        alpha_list = []
        for i, x in enumerate(x_dict.values()):
            out = self.lin(x)
            out = F.sigmoid(out)
            out = (out * self.atts[i]).sum(dim=-1)
            alpha = out.mean(dim=0)
            alpha_list.append(alpha.item())

        alpha = torch.tensor(alpha_list).to(x_dict['author'].device)
        alpha = F.softmax(alpha, dim=0)

        x_list = [x_dict[node_type] for node_type in x_dict.keys()]
        # [4,4027,128]
        x = torch.stack(x_list, dim=0)
        alpha = alpha.view(4, 1, 1)

        x = x * alpha
        x = x.sum(dim=0)

        return x


def homogeneous_view(x_dict, threshold):

    x_dict_mean = {}
    edge_index_target = None
    for node_type, x in x_dict.items():
        similarity_matrix = cosine_similarity_matrix(x)
        adjacency_matrix = adjacency_matrix_from_similarity(similarity_matrix, threshold=threshold)

        adjacency_sparse = adjacency_matrix.to_sparse()
        edge_index = adjacency_sparse.indices()
        if node_type == 'author':
            edge_index_target = edge_index

        # adjacency_sparse = csr_matrix(adjacency_matrix)
        # 无自环，无向图
        # edge_index, _ = from_scipy_sparse_matrix(adjacency_sparse)

        aggregates = aggregate_mean(adjacency_matrix, x)

        x = torch.cat([x, aggregates], dim=1)
        x_dict_mean[node_type] = x

    return x_dict_mean, edge_index_target


def heterogeneous_view(x_dict, convs, target_node_type, threshold):

    x_dict_heter = {k: v for k, v in x_dict.items() if k != target_node_type}
    x_target = x_dict[target_node_type]

    heter_info_dict = {}
    for node_type, x in x_dict_heter.items():

        similarity_matrix = cosine_similarity(x_target, x)
        # similarity_matrix = torch.load(f'./similarity_matrix_{node_type}.pth')

        adjacency_matrix = adjacency_matrix_from_similarity(similarity_matrix, threshold=threshold)
        adjacency_sparse = adjacency_matrix.to_sparse()
        edge_index = adjacency_sparse.indices()
        # source-->target
        edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        heter_info = convs[node_type]((x, x_target), edge_index)
        heter_info_dict[node_type] = heter_info
    return heter_info_dict


def semantic_view(data, device, target_node_type='author'):

    args = build_args()
    args.metadata = data.metadata()

    models = PreModel(args)
    encoder = models.encoder
    encoder = encoder.to(device)

    encoder.train()

    x_dict = {target_node_type: data.x_dict[target_node_type]}

    out = encoder(x_dict, data.edge_index_dict)

    return None


def feature_view(x_dict: dict, device, threshold, target_node_type='author'):

    view = FeatureView(x_dict, threshold, target_node_type)
    decoder = NodeDecoder(128, 256, 668, 2, 0.2)

    view = view.to(device)
    decoder = decoder.to(device)

    # preprocess
    x_dict_mean, edge_index = homogeneous_view(x_dict, threshold)

    # x_dict_homo, x_target = dimension_transform(x_dict_mean, x_dict[target_node_type], view.lins, 128)

    # x_heter = heterogeneous_view(x_dict_homo, view.convs, target_node_type, threshold)

    # att = view.att
    # x_heter['author'] = x_target
    # x = att(x_heter)

    # encoder = view.encoder
    # x = encoder(x, edge_index)

    # print(x.shape)
    # print("Mean Aggregation:\n", x)

    optimizer = torch.optim.Adam(list(view.parameters()) + list(decoder.parameters()), lr=0.01, weight_decay=0.001)
    count_wait = 0
    best_loss = 1e9
    best_epoch = 0
    patience = 30
    best_model_state_dict = None

    for epoch in range(1, 200):
        start_time = time.time()
        loss = train_node(view, decoder, x_dict_mean, edge_index, optimizer)
        end_time = time.time()

        epoch_time = end_time - start_time
        print(f"{epoch} epoch time: {epoch_time} seconds")
        print("loss: ", loss)

        if loss < (best_loss - 0.03):
            best_loss = loss
            best_epoch = epoch
            count_wait = 0
            best_model_state_dict = view.state_dict()
            torch.save(best_model_state_dict, './model/snapshots/feature_view.pth')
        else:
            count_wait += 1

        if count_wait == patience:
            print('-----Early stopping-----')
            print('The best epoch is: ', best_epoch)
            break


def save_dict(dic, file):
    with open(file, 'wb') as file:
        pickle.dump(dic, file)


def contrast(z1, z2, mask, l):

    loss1 = infonce(z1, z2, mask)
    loss2 = infonce(z2, z1, mask)

    loss = l * loss1 + (1 - l) * loss2
    return loss


def correlation(size, mps, adjs, co):
    rows, cols, vals = [], [], []
    for i, mp in enumerate(mps):
        adjs[mp].set_value_(adjs[mp].storage.value() * co[i])
        row, col, val = adjs[mp].coo()
        rows.append(row)
        cols.append(col)
        vals.append(val)

    # 合并 row, col 和 value
    row = torch.cat([row for row in rows])
    col = torch.cat([col for col in cols])
    val = torch.cat([val for val in vals])

    adj = SparseTensor(row=row, col=col, value=val, sparse_sizes=(size, size)).coalesce()
    score_m = adj.to_dense()

    return score_m


def get_mask(z1, z2, ep_f, ep_s, mps, adjs, co=[0.4, 0.6]):

    # feature
    sim_cos_intra = cosine_similarity(z1, z1)
    pos_f_intra = adjacency_matrix_from_similarity(sim_cos_intra, ep_f)

    sim_cos_inter = cosine_similarity(z1, z2)
    pos_f_inter = adjacency_matrix_from_similarity(sim_cos_inter, ep_f)
    pos_f_inter.fill_diagonal_(1)

    # semantic
    score_m = correlation(z1.size(0), mps, adjs, co)
    pos_s = adjacency_matrix_from_similarity(score_m, ep_s)
    pos_s.fill_diagonal_(1)

    pos_intra = torch.logical_and(pos_f_intra, pos_s)
    pos_inter = torch.logical_and(pos_f_inter, pos_s)

    return [pos_intra, pos_inter]


def infonce(z1, z2, mask):
    def f(x): return torch.exp(x / 0.2)
    sim_intra = f(cosine_similarity(z1, z1))
    sim_inter = f(cosine_similarity(z1, z2))

    loss = -torch.log(
        ((sim_intra * mask[0]).sum(1) + (sim_inter * mask[1]).sum(1)) / (sim_intra.sum(1) + sim_inter.sum(1))
    )

    return loss.mean()


def evaluate_nc(encoder_s, encoder_f, node_classifier, data, x_dict_target, x_dict_mean, edge_index, data_s):

    labels = data['author'].y
    # labels = data.y

    # print(torch.cuda.memory_allocated(2))
    # print(torch.cuda.memory_allocated(2)/(1024 ** 3))
    # print('##########')

    embeds = None
    with torch.no_grad():
        embeds_s = encoder_s(data_s.x_dict, data_s.edge_index_dict).to(data.x_dict['author'].device)
        embeds_f = encoder_f(x_dict_mean, edge_index)
        embeds = torch.cat([embeds_s, embeds_f], dim=1)

    # train_mask = data.train_mask
    # valid_mask = data.val_mask
    # test_mask = data.test_mask
    # train_mask = data['author'].train_mask
    # valid_mask = data['author'].val_mask
    # test_mask = data['author'].test_mask
    train_mask = data.train_index
    valid_mask = data.val_index
    test_mask = data.test_index

    train_labels = labels[train_mask]
    valid_labels = labels[valid_mask]
    test_labels = labels[test_mask]
    train_embeds = embeds[train_mask]
    valid_embeds = embeds[valid_mask]
    test_embeds = embeds[test_mask]

    cross_ent = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(node_classifier.parameters(), lr=0.01, weight_decay=0)

    valid_accs = []
    val_macro_f1s = []
    val_micro_f1s = []
    test_accs = []
    test_macro_f1s = []
    test_micro_f1s = []

    for epoch in range(0, 200):
        # train
        node_classifier.train()
        opt.zero_grad()

        logits = node_classifier(train_embeds)
        loss = cross_ent(logits, train_labels)
        loss.backward()
        opt.step()

        # validate
        logits = node_classifier(valid_embeds)
        preds = torch.argmax(logits, dim=1)

        val_acc = torch.sum(preds == valid_labels).float() / valid_labels.numel()
        val_f1_macro = f1_score(valid_labels.cpu(), preds.cpu(), average='macro')
        val_f1_micro = f1_score(valid_labels.cpu(), preds.cpu(), average='micro')

        # test
        logits = node_classifier(test_embeds)
        preds = torch.argmax(logits, dim=1)

        test_acc = torch.sum(preds == test_labels).float() / test_labels.numel()
        test_f1_macro = f1_score(test_labels.cpu(), preds.cpu(), average='macro')
        test_f1_micro = f1_score(test_labels.cpu(), preds.cpu(), average='micro')

        valid_accs.append(val_acc)
        val_macro_f1s.append(val_f1_macro)
        val_micro_f1s.append(val_f1_micro)
        test_accs.append(test_acc)
        test_macro_f1s.append(test_f1_macro)
        test_micro_f1s.append(test_f1_micro)

        print(f'epoch: {epoch:03d}, loss: {loss:.4f}\n'
              f'val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}\n'
              f'val_f1_macro: {val_f1_macro:.4f}, test_f1_macro: {test_f1_macro:.4f}\n'
              f'val_f1_micro: {val_f1_micro:.4f}, test_f1_micro: {test_f1_micro:.4f}\n'
              '----------------------------')

    # select best ACC-Macro-Micro based on validation
    max_acc_iter = valid_accs.index(max(valid_accs))
    max_macro_iter = val_macro_f1s.index(max(val_macro_f1s))
    max_micro_iter = val_micro_f1s.index(max(val_micro_f1s))
    acc = test_accs[max_acc_iter]
    macro_f1 = test_macro_f1s[max_macro_iter]
    micro_f1 = test_micro_f1s[max_micro_iter]

    print(max_acc_iter)
    print(max_macro_iter)
    print(max_micro_iter)

    print(f'ACC: {acc:.4f}, Macro-F1: {macro_f1:.4f}, Micro-F1: {micro_f1:.4f}')


# def split_data(x, ratio):
#     perm = torch.randperm(x.size(0))
#     num_train = x.size(0) * ratio

#     index_train = perm[:num_train]
#     index_test = perm[num_train:]

#     return index_train, index_test

def accuracy(preds, labels):
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)


def SVM(x, y):
    f1_mac = []
    f1_mic = []
    accs = []

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)

        micro = f1_score(y_test, preds, average='micro')
        macro = f1_score(y_test, preds, average='macro')
        acc = accuracy(preds, y_test)
        accs.append(acc)
        f1_mac.append(macro)
        f1_mic.append(micro)
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    accs = np.array(accs)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    accs = np.mean(accs)
    print('Testing based on svm: ',
          'f1_micro=%.4f' % f1_mic,
          'f1_macro=%.4f' % f1_mac,
          'acc=%.4f' % accs)
    return f1_mic, f1_mac, accs


def train_cl(data, device, args):

    device_1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    target_node = args.target_node
    threshold = args.threshold

    model = PreModel(args)
    semantic_encoder = model.encoder
    feature_encoder = FeatureView(data.x_dict, threshold, target_node)

    encoder_f, encoder_s = feature_encoder.to(device), semantic_encoder.to(device_1)
    node_classifier = model.node_classifier.to(device)

    # preprocess
    x_dict_mean, edge_index = homogeneous_view(data.x_dict, threshold)
    x_dict_target = {target_node: data.x_dict[target_node]}

    # 多GPU并行-----------------------------------
    data_s = HeteroData()
    data_s[target_node].x = data.x_dict[target_node].to(device_1)
    for edge_type, edge_index in data.edge_index_dict.items():
        data_s[edge_type].edge_index = edge_index.to(device_1)
    # -------------------------------------------

    encoder_f.train()
    encoder_s.train()

    optimizer = torch.optim.Adam(list(semantic_encoder.parameters()) + list(feature_encoder.parameters()),
                                 lr=0.01, weight_decay=0.001)
    patience = 40
    count_wait = 0
    best_loss = 1e9
    best_epoch = 0
    z_s_list = []
    for epoch in range(1, 600):
        start_time = time.time()

        optimizer.zero_grad()

        z_f = encoder_f(x_dict_mean, edge_index)
        z_s = encoder_s(data_s.x_dict, data_s.edge_index_dict).to(device)
        # z_s = encoder_s(x_dict_target, data.edge_index_dict).to(device)

        adjs = {}
        for mp in data.edge_types:
            num_edges = data.edge_index_dict[mp].shape[1]
            adjs[mp] = SparseTensor.from_edge_index(data.edge_index_dict[mp], edge_attr=torch.ones(num_edges, device=device))

        # co = att_dict[target_node].tolist()
        mask = get_mask(z_f, z_s, 0.4, 0.5, data.edge_types, adjs)
        loss = contrast(z_f, z_s, mask, 0.5)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(semantic_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 1.0)
        optimizer.step()

        end_time = time.time()
        epoch_time = end_time - start_time
        # print(f"{epoch} epoch time: {epoch_time} seconds")
        print("loss: ", loss.item())

        if loss < (best_loss - 0.03):
            best_loss = loss
            best_epoch = epoch
            count_wait = 0
            # best_model_state_dict = view.state_dict()
            # torch.save(best_model_state_dict, './model/snapshots/feature_view.pth')
        else:
            count_wait += 1

        if count_wait == patience:
            print('-----Early stopping-----')
            print('The best epoch is: ', best_epoch)
            break

    semantic_encoder.eval()
    feature_encoder.eval()

    labels = data['author'].y
    embeds = None
    with torch.no_grad():
        embeds_s = encoder_s(data_s.x_dict, data_s.edge_index_dict).to(data.x_dict['author'].device)
        embeds_f = encoder_f(x_dict_mean, edge_index)
        embeds = torch.cat([embeds_s, embeds_f], dim=1)
        # embeds = embeds_f+embeds_s

    evaluate_nc(semantic_encoder, feature_encoder, node_classifier, data, x_dict_target, x_dict_mean, edge_index, data_s)
    # SVM(embeds.data.cpu().numpy(), labels.data.cpu().numpy())


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(42)

args = build_args()
data = load_data('DBLP')
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

args.threshold = 0.5
args.target_node = 'author'
args.metadata = data.metadata()
train_cl(data, device, args)

# feature_view(x_dict=data.x_dict, device=device, threshold=0.2)
# semantic_view(data, device)