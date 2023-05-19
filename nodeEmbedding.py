import os
import numpy as np
from net import EmpiricalNet
from __UTIL import state
from embedding.node2vec.main import my_main


class Embedding:
    def __init__(self, net):
        """
        @Jiaze
        prepare node embedding for net. all embedding save in embedding variable(a dict)
        default have a embedding: a row of adjacent matrix for one node.
        for other embedding, need run corresponding method to get the embedding
        (Generally, saved key of embedding is the method name, such as node2vec,
        but every method's function have a parameter key to specify that, for the case that
        user want run the same method embedding many times.)
        :param net: a prepared EmpricalNet object
        """
        self.net = net
        self.embedding = dict()  # {embedding_name:embedding_mat(n*embed_dim), }
        self.adjacent()

    def adjacent(self):
        self.embedding['adj'] = self.net.mat

    # @state(start="node2vec start", end="node2vec end!")
    def node2vec(self, key="node2vec", directed=False, p=1, q=1, num_walks=10, walk_length=80, dimensions=128,
                 window_size=10, workers=8, _iter=1, force_calc=False, version=None):
        # set sl path
        path = "./embedding/node2vec/" + self.net.net_name + "/"
        os.makedirs(path) if os.path.exists(path) is False else None
        filename = path + self.net.net_name + '_' + str(p) + '_' + str(q) + '_' + str(num_walks) + '_' + \
                   str(walk_length) + '_' + str(window_size) + '_' + str(_iter) + '_' + str(dimensions)
        filename += "" if version is None else "_" + str(version)
        filename += '.emd'
        # run node2vec to the file
        if os.path.exists(filename) and force_calc is False:
            pass
        else:
            edges = self.net.edges
            my_main(edges, filename, directed, p, q, num_walks, walk_length, dimensions, window_size, workers, _iter)
        # extract from file
        with open(filename, 'r') as fn:
            # get node_vec_len
            line = fn.readline()
            line = line.strip()
            s = line.split(' ')
            node_vec_len = int(s[1])
            embedding_array = np.zeros((self.net.n, node_vec_len))
            # get node_vec
            line = fn.readline()
            while line:
                s = line.strip().split(' ')
                embedding_array[int(s[0]), :] = [float(x) for x in s[1:]]
                line = fn.readline()
        self.embedding[key] = embedding_array
        return embedding_array


if __name__ == '__main__':
    net = EmpiricalNet("highSchool")
    emb = Embedding(net)
    emb.node2vec()
