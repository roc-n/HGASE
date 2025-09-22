import pickle
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold

import numpy as np
import os
from utils.data import load_data

from model.models import MVHGCL
from utils.params import build_args
from utils.process import pt_rand_mask, find_hop_nodes
from utils.evaluate import MLP, evaluate_cluster

EPS = 1e-15


def save_dict(dic, file):
    with open(file, 'wb') as file:
        pickle.dump(dic, file)


def main(data, device, args):

    target_node = args.target_node
    hop_dict = find_hop_nodes(data.edge_index_dict, target_node)
    patience = args.patience

    # load model
    MVHG = MVHGCL(args, data, hop_dict)

    # preprocess
    MVHG.prep()

    # model train
    results = []

    for _ in range(10):
        MVHG.train()

        params = list(MVHG.SE.parameters()) + list(MVHG.FE.parameters())
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

        count_wait = 0
        best_loss = 1e9
        best_epoch = 0
        best_state_dict = None

        for epoch in range(1, 1000):
            opt.zero_grad()

            loss = MVHG()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MVHG.SE.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(MVHG.FE.parameters(), 1.0)
            opt.step()

            print("loss: ", loss.item())
            if loss < (best_loss):
                best_loss = loss
                best_epoch = epoch
                count_wait = 0
                best_state_dict = MVHG.state_dict()
            else:
                count_wait += 1

            if count_wait == patience:
                print('-----early stopping-----')
                print('the best epoch is: ', best_epoch)
                break

        MVHG.load_state_dict(best_state_dict)
        MVHG.eval()
        with torch.no_grad():
            embeds_s, z_s_mp, embeds_f, z_f_mp = MVHG.get_embeds()
            embeds = torch.cat([embeds_f, embeds_s], dim=1)


        if args.task == 'classification':
            split = [20, 40, 60]
            labels = data[target_node].y
            for i in range(len(data.train_index)):
                train_mask = data.train_index[i]
                valid_mask = data.valid_index[i]
                test_mask = data.test_index[i]
                acc, _, micro, micro_, macro, macro_ = MLP(
                    embeds, labels, train_mask, valid_mask, test_mask, args.num_class, device)
                print(f'Test Results via MLP({split[i]}): ', f'Mi-F1={micro:.4f},{micro_:.4f}', f'Ma-F1={macro:.4f},{macro_:.4f}')
        else:
            labels = data[target_node].y.cpu().data.numpy()
            nmi_list, ari_list = [], []
            embeds = embeds.cpu().data.numpy()
            for kmeans_random_state in range(10):
                nmi, ari = evaluate_cluster(embeds, labels, args.num_class, kmeans_random_state)
                nmi_list.append(nmi)
                ari_list.append(ari)
            print("\t[clustering] nmi: [{:.4f}, {:.4f}] ari: [{:.4f}, {:.4f}]".format(
                np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)))
        # exit()
        # accumulate for further evaluate:
    #     results.append((acc, micro, macro))
    #     MVHG.FE.reset_parameters()
    #     MVHG.SE.reset_parameters()
    # display(results)


def display(results):

    acc_values = [result[0] for result in results]
    micro_values = [result[1] for result in results]
    macro_values = [result[2] for result in results]

    acc_mean = np.mean(acc_values)
    acc_std = np.std(acc_values)
    micro_mean = np.mean(micro_values)
    micro_std = np.std(micro_values)
    macro_mean = np.mean(macro_values)
    macro_std = np.std(macro_values)

    print("  acc list: ", [f'{value:.4f}' for value in acc_values])
    print("micro list: ", [f'{value:.4f}' for value in micro_values])
    print("macro list: ", [f'{value:.4f}' for value in macro_values])

    print(f'  acc={acc_mean: .4f}±{acc_std: .4f}',
          f'Mi-F1={micro_mean:.4f}±{micro_std:.4f}',
          f'Ma-F1={macro_mean:.4f}±{macro_std:.4f}')


def load_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(66666)

datasets = ['DBLP', 'ACM', 'IMDB_']
dataset = datasets[2]

args = build_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
args.device = device
args.device1 = device1
args.dataset = dataset
args.task = 'classification'

args.lr = 0.01
args.weight_decay = 0.001
args.patience = 30

if dataset == 'DBLP':
    if args.task == 'classification':

        args.mask_ratio = 0.6
        args.ratio = [0.05, 0.13, 0.18]
        data = load_data('DBLP', args.ratio)
        args.threshold = 0.5
        args.target_node = 'author'
        args.metadata = data.metadata()
        args.neighbor_threshold = 100
        args.num_class = 4
        args.dim = 64
        args.dataset = 'DBLP'

        args.ep_f = 0.4
        # args.ep_s = 0.6
        # args.co = [0.2, 0.7, 0.1]
        args.ep_s = 0.1
        args.co = [0.3, 0.3, 0.3]
        args.nei_t = 2
        args.patience = 25
    else:
        cfg_name = "clustering.yml"
        args = load_configs(args, cfg_name)
        data = load_data(args.dataset, args.ratio)
        args.metadata = data.metadata()
elif dataset == 'ACM':
    if args.task == 'classification':
        args.ep_f = 0.4  # 0.4
        # args.ep_s = 0.9
        # args.co = [0.4, 0.6]
        args.ep_s = 0.7
        args.co = [0.4, 0.6]
        args.nei_t = 10

        args.mask_ratio = 0.6
        args.ratio = [0.07, 0.3]

        data = load_data('ACM', args.ratio)
        args.threshold = 0.5
        args.target_node = 'paper'
        args.metadata = data.metadata()
        args.neighbor_threshold = 100
        args.num_class = 3
        args.dim = 128
        args.patience = 25
    else:
        cfg_name = "clustering.yml"
        args = load_configs(args, cfg_name)
        data = load_data(args.dataset, args.ratio)
        args.metadata = data.metadata()
elif dataset == 'IMDB_':
    if args.task == 'classification':
        args.ep_f = 0.4
        args.ep_s = 0.4
        args.co = [0.3, 0.3, 0.3]  # [0.2, 0.7, 0.1]
        args.ini_max = [60, 26, 360]  # [90, 20, 500],[50, 22, 200],[60,16,360],[[10, 2, 53]]
        args.ratio = [0.1, 0.1, 0.1]
        args.mask_ratio = 0.2

        data = load_data(args.dataset, args.ratio)

        args.threshold = 0.5
        args.target_node = 'm'
        args.metadata = data.metadata()
        args.neighbor_threshold = 100
        args.num_class = 3
        args.dim = 64
        args.nei_t = 8
        args.patience = 30
    else:
        cfg_name = "clustering.yml"
        args = load_configs(args, cfg_name)
        data = load_data(args.dataset, args.ratio)
        args.metadata = data.metadata()

data = data.to(device)
if __name__ == '__main__':
    main(data, device, args)