import argparse

datasets_args = {
    "dblp": {
        "type_num": [4057, 14328, 7723, 20],  # the number of every node type
        "nei_num": 1,  # the number of neighbors' types
        "n_labels": 4,
    },
    "aminer": {
        "type_num": [6564, 13329, 35890],
        "nei_num": 2,
        "n_labels": 4,
    },
    "freebase": {
        "type_num": [3492, 2502, 33401, 4459],
        "nei_num": 3,
        "n_labels": 3,
    },
    "acm": {
        "type_num": [4019, 7167, 60],
        "nei_num": 2,
        "n_labels": 3,
    },
}

def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    parser.add_argument("--n_head", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--n_layer", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--dropout", type=int, default=0.2)
    parser.add_argument("--feat_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--in_channels", type=int, default=-1)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--feat_mask_rate", type=str, default="0.5,0.005,0.8",
                        help="""The mask rate. If provide a float like '0.5', mask rate is static over the training. 
                        If provide two number connected by '~' like '0.4~0.9', mask rate is uniformly sampled over the training.
                        If Provide '0.7,-0.1,0.5', mask rate starts from 0.7, ends at 0.5 and reduce 0.1 for each epoch.""")
    parser.add_argument("--replace_rate", type=float, default=0.0,
                        help="The replace rate. The ratio of nodes that is replaced by random nodes.")
    parser.add_argument("--leave_unchanged", type=float, default=0.0,
                        help="The ratio of nodes left unchanged (no mask), but is asked to reconstruct.")

    parser.add_argument("--encoder", type=str, default="han")
    parser.add_argument("--decoder", type=str, default="han")
    parser.add_argument("--loss_fn", type=str, default="mse")
    parser.add_argument("--alpha_l", type=float, default=2,
                        help="pow index for sce loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--scheduler_gamma", type=float, default=0.99,
                        help="decay the lr by gamma for ExponentialLR scheduler")

    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--batch_size', type=int, default=102400)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=512)  # 64
    parser.add_argument('--mae_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)  # 0.05
    parser.add_argument('--eva_wd', type=float, default=0, help="weight decay")

    # The parameters of learning process
    parser.add_argument('--patience', type=int,
                        default=5)  # 5. we can change the value. this might impact the performance.
    parser.add_argument('--l2_coef', type=float, default=0)

    args, _ = parser.parse_known_args()
    for key, value in datasets_args[args.dataset].items():
        setattr(args, key, value)
    return args