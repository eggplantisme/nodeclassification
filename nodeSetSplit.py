import random


def divide(node_list, train_ratio):
    """
    random choose a part of nodes as train set
    :param node_list: a list of node id
    :param train_ratio: the ratio of train set
    :return: train set and left set of node id list
    """
    nodes = node_list.copy()
    random.shuffle(nodes)
    n = len(node_list)
    train_n = int(n * train_ratio)
    train = nodes[:train_n]
    left = nodes[train_n:]
    return train, left
