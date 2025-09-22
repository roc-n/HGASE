import numpy as np
import torch
from model.modules import LogReg, NodeClassifier
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as accuracy
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

def SVM(x, y, splits=None, train_ratio=0.2):
    if splits:
        train_mask = splits.train_index
        test_mask = splits.test_index
        x_train, y_train = x[train_mask], y[train_mask]
        x_test, y_test = x[test_mask], y[test_mask]
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=42)

    # rbf , linear
    clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)

    acc = accuracy(y_test, preds)
    micro = f1_score(y_test, preds, average='micro')
    macro = f1_score(y_test, preds, average='macro')

    return acc, micro, macro


def MLP(x, y, train_mask, valid_mask, test_mask, num_classes, device):

    x_train = x[train_mask]
    x_valid = x[valid_mask]
    x_test = x[test_mask]
    y_train = y[train_mask]
    y_valid = y[valid_mask]
    y_test = y[test_mask]

    accs = []
    micro_f1s = []
    macro_f1s = []
    for _ in range(50):
        acc_valids = []
        macro_valids = []
        micro_valids = []
        acc_tests = []
        macro_tests = []
        micro_tests = []

        node_clf = LogReg(x_train.shape[1], num_classes).to(device)
        cross_ent = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(node_clf.parameters(), lr=0.01, weight_decay=0)

        for epoch in range(0, 150):
            # train
            node_clf.train()
            opt.zero_grad()

            logits = node_clf(x_train)
            loss = cross_ent(logits, y_train)
            loss.backward()
            opt.step()

            # validate
            node_clf.eval()
            logits = node_clf(x_valid)
            preds = torch.argmax(logits, dim=1)

            acc_valid = torch.sum(preds == y_valid).float() / y_valid.numel()
            macro_valid = f1_score(y_valid.cpu(), preds.cpu(), average='macro')
            micro_valid = f1_score(y_valid.cpu(), preds.cpu(), average='micro')

            # test
            logits = node_clf(x_test)
            preds = torch.argmax(logits, dim=1)

            acc_test = torch.sum(preds == y_test).float() / y_test.numel()
            macro_test = f1_score(y_test.cpu(), preds.cpu(), average='macro')
            micro_test = f1_score(y_test.cpu(), preds.cpu(), average='micro')

            acc_valids.append(acc_valid)
            macro_valids.append(macro_valid)
            micro_valids.append(micro_valid)
            acc_tests.append(acc_test)
            macro_tests.append(macro_test)
            micro_tests.append(micro_test)

        # print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}\n'
        #       f'ACC-Valid: {acc_valid:.4f}, ACC-Test: {acc_test:.4f}\n'
        #       f'Ma-F1-Valid: {macro_valid:.4f}, Ma-F1-Test: {macro_test:.4f}\n'
        #       f'Mi-F1-Valid: {micro_valid:.4f}, Mi-F1-Test: {micro_test:.4f}\n'
        #       '----------------------------')

        # select best ACC-Micro-Macro based on validation
        max_acc_iter = acc_valids.index(max(acc_valids))
        max_micro_iter = micro_valids.index(max(micro_valids))
        max_macro_iter = macro_valids.index(max(macro_valids))
        acc = acc_tests[max_acc_iter]
        micro = micro_tests[max_micro_iter]
        macro = macro_tests[max_macro_iter]

        accs.append(acc.cpu())
        micro_f1s.append(micro)
        macro_f1s.append(macro)

    acc = np.mean(accs)
    acc_ = np.std(accs)
    micro = np.mean(micro_f1s)
    micro_ = np.std(micro_f1s)
    macro = np.mean(macro_f1s)
    macro_ = np.std(macro_f1s)
    return acc, acc_, micro, micro_, macro, macro_

def evaluate_cluster(embeds, y, n_labels, kmeans_random_state):
    Y_pred = KMeans(n_labels, random_state=kmeans_random_state).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    return nmi, ari