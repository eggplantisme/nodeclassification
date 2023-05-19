import numpy as np
import sys
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from nodeSetSplit import divide
from net import EmpiricalNet, SyntheticNet
from nodeEmbedding import Embedding
from sklearn.linear_model import LogisticRegression


class NodeInfer:
    def __init__(self, net, attr):
        """
        The base class for node attribute inference.
        :param net: An EmpiricalNet or other Variable which has mat, nodeAttr, nodeAttrMeta.
        :param attr: attribute name
        """
        self.net = net
        self.attr = attr
        self.node_list = np.arange(net.n)
        if attr in net.nodeAttr.keys() and attr in net.nodeAttrMeta.keys():
            self.attr_vec = net.nodeAttr[attr]
        else:
            print(f'Net:{net.net_name}, Attribute Name:{self.attr}, Attribute Name Wrong!')
            self.attr_vec = None
        self.train_index, self.test_index = None, None

    def divide(self, train_ratio):
        self.train_index, self.test_index = divide(self.node_list, train_ratio)

    def BASIC(self, is_random=False):
        """
        BASIC infer node attribute without links and groups (Zheleva and Getoor, 2009)
        :param is_random: random guess or not (base on probability in train set)
        :return: accuracy
        """
        # Prepare Tr/Te Data
        train_attr_vec = self.attr_vec[self.train_index]
        test_attr_vec = self.attr_vec[self.test_index]
        # Prepare method
        attr_kinds = list(self.net.nodeAttrMeta[self.attr].keys())
        attr_num = len(attr_kinds)
        if is_random:
            p = np.array([1/attr_num] * attr_num)
        else:
            p = np.zeros(attr_num)
            for ai, a in enumerate(attr_kinds):
                p[ai] = np.sum(train_attr_vec == a) / np.size(train_attr_vec)
        # Predict
        predict_attr_vec = np.zeros(np.size(test_attr_vec))
        for i in range(np.size(test_attr_vec)):
            predict_attr_vec[i] = np.random.choice(attr_kinds, p=p)
        # Accuracy
        results = test_attr_vec == predict_attr_vec
        accuracy = np.sum(results) / np.size(results)
        return accuracy

    def LogisticRegression(self, embedding_mat):
        """
        Logistic Regression infer node attribute
        :param embedding_mat: feature matrix
        :return: accuracy
        """
        train_x = embedding_mat[self.train_index, :]
        train_y = self.attr_vec[self.train_index]
        clf = LogisticRegression().fit(train_x, train_y)
        test_x = embedding_mat[self.test_index, :]
        test_y = self.attr_vec[self.test_index]
        accuracy = clf.score(test_x, test_y)
        return accuracy


def main_get_accuracy(times=1):
    net = EmpiricalNet("highSchool")
    ni = NodeInfer(net, attr="gender")
    emb_mat = Embedding(net)
    emb_mat.node2vec()
    train_ratios = np.arange(0.05, 0.9, 0.05)
    BnR = []
    BR = []
    LR_adj = []
    LR_node2vec = []
    for r in tqdm(train_ratios):
        BnR.append([])
        BR.append([])
        LR_adj.append([])
        LR_node2vec.append([])
        for t in range(times):
            ni.divide(r)
            BnR[-1].append(ni.BASIC(is_random=False))
            BR[-1].append(ni.BASIC(is_random=True))
            LR_adj[-1].append(ni.LogisticRegression(embedding_mat=emb_mat.embedding["adj"]))
            LR_node2vec[-1].append(ni.LogisticRegression(embedding_mat=emb_mat.embedding["node2vec"]))
        # print("BASIC accuracy(NoRandom) is", ni.BASIC(is_random=False))
        # print("BASIC accuracy(Random) is", ni.BASIC(is_random=True))
        # print("LR accuracy(adj) is", ni.LogisticRegression(embedding_mat=emb_mat.embedding["adj"]))
        # print("LR accuracy(node2vec) is", ni.LogisticRegression(embedding_mat=emb_mat.embedding["node2vec"]))
    return train_ratios, BnR, BR, LR_adj, LR_node2vec


def main_symmetric_sbm(times=1):
    """
    For specific sbm network, accuracy of different method for different train ratio
    :param times:
    :return:
    """
    n = 2 ** 10
    k = 2
    c = 10
    epsilon = 0.3
    net_name = f"SSBM_{n}_{c}_{k}_{epsilon}"
    attr_name = "block"
    net = SyntheticNet(net_name)
    ni = NodeInfer(net, attr=attr_name)
    emb_mat = Embedding(net)
    emb_mat.node2vec()
    train_ratios = np.arange(0.05, 0.9, 0.05)
    BnR = []
    BR = []
    LR_adj = []
    LR_node2vec = []
    for r in tqdm(train_ratios):
        BnR.append([])
        BR.append([])
        LR_adj.append([])
        LR_node2vec.append([])
        for t in range(times):
            ni.divide(r)
            BnR[-1].append(ni.BASIC(is_random=False))
            BR[-1].append(ni.BASIC(is_random=True))
            LR_adj[-1].append(ni.LogisticRegression(embedding_mat=emb_mat.embedding["adj"]))
            LR_node2vec[-1].append(ni.LogisticRegression(embedding_mat=emb_mat.embedding["node2vec"]))
        # print("BASIC accuracy(NoRandom) is", ni.BASIC(is_random=False))
        # print("BASIC accuracy(Random) is", ni.BASIC(is_random=True))
        # print("LR accuracy(adj) is", ni.LogisticRegression(embedding_mat=emb_mat.embedding["adj"]))
        # print("LR accuracy(node2vec) is", ni.LogisticRegression(embedding_mat=emb_mat.embedding["node2vec"]))
    return train_ratios, BnR, BR, LR_adj, LR_node2vec


def main1_symmetric_sbm(times=1):
    """
    accuracy of specific method for different train ratio and different sbm's epsilon
    :param times:
    :return:
    """
    n = 2 ** 10
    k = 2
    c = 10
    epsilons = np.arange(0.1, 0.9, 0.05)
    train_ratios = np.arange(0.05, 0.9, 0.05)
    accus = dict()
    for epsilon in epsilons:
        print("epsilon:"+str(round(epsilon, 3)), "Begin...")
        accus[epsilon] = dict()
        net_name = f"SSBM_{n}_{c}_{k}_{epsilon}"
        attr_name = "block"
        for t in tqdm(range(times), desc="t"):
            net = SyntheticNet(net_name, verbose=False)
            ni = NodeInfer(net, attr=attr_name)
            emb_mat = Embedding(net)
            emb_mat.node2vec(force_calc=True, version="t"+str(t))
            for r in train_ratios:
                accus[epsilon][r] = [] if r not in accus[epsilon].keys() else accus[epsilon][r]
                ni.divide(r)
                accus[epsilon][r].append(ni.LogisticRegression(embedding_mat=emb_mat.embedding["node2vec"]))
    return epsilons, train_ratios, accus


if __name__ == '__main__':
    main1_symmetric_sbm(times=2)
