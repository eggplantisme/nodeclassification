import time

import numpy as np
from scipy.sparse.linalg import eigsh, inv
from scipy.sparse import eye, diags, issparse, csr_array, find, hstack, vstack
import itertools
import random
import hypernetx as hnx
from tqdm import tqdm


class HyperSBM:
    def __init__(self, sizes, ps_dict):
        """
        Initial hyper SBM
        :param sizes: the sizes of each community
        :param ps_dict: the link probability tensors, key is order of edges m, value is m-order tensor,
                        dimension is the number of communities
        """
        self.n = np.sum(sizes)
        self.sizes = sizes
        self.ps_dict = ps_dict
        self.A = dict()  # adjacent list for different order of edges
        self.groupId = []
        self.H = None
        self.hyper_g = None
        self.bipartite_g = None
        self.bipartite_A = None
        self.e = 0  # number of edges
        self.construct()

    def construct(self):
        gid = 0
        for s in self.sizes:
            self.groupId += [gid] * int(s)
            gid += 1
        self.groupId = np.array(self.groupId)
        data = []
        row_ind = []
        col_ind = []
        for key in self.ps_dict.keys():
            print(f"Generating hyper edges for order {key}...")
            self.A[key] = []
            # temp_edge_set = set()
            # Use combination iterable to consider only undirected edges
            for index in tqdm(itertools.combinations(range(self.n), r=key)):
                if np.size(np.unique(index)) < len(index):
                    # pass when exist same node
                    pass
                else:
                    gindex = tuple(self.groupId[i] for i in index)
                    p = np.array(self.ps_dict[key])[gindex]
                    r = random.random()
                    if r < p:
                        self.A[key].append(index)
                        # temp_edge_set.add(index)
                        data += [1] * key
                        row_ind += list(index)
                        col_ind += [self.e] * key
                        self.e += 1
        self.H = csr_array((data, (row_ind, col_ind)))
        # print(np.shape(self.H))
        self.hyper_g = hnx.Hypergraph.from_numpy_array(self.H.toarray())
        lefttop = csr_array(np.zeros((self.n, self.n)))
        rightbottom = csr_array(np.zeros((self.e, self.e)))
        self.bipartite_A = csr_array(vstack([hstack([lefttop, self.H]), hstack([self.H.transpose(), rightbottom])]))
        # print(np.shape(self.bipartite_A))
        # self.bipartite_g = self.hyper_g.bipartite()
        # self.bipartite_A = nx.to_scipy_sparse_array(self.bipartite_g)

    def get_operator(self, operator='B', r=0):
        """
        operator
        :param operator: Bi, NB_Bi, BH_Bi
        :param r: parameter for BH
        """
        if operator == 'Bi':
            return self.bipartite_A
        elif operator == 'NB_Bi':
            edges = []
            x, y, _ = find(self.bipartite_A)
            for i, x_i in enumerate(x):
                edges.append((x_i, y[i]))
            e = len(edges)
            B = np.zeros((e, e))  # TODO maybe 2e*2e
            for i in range(len(edges)):
                for j in range(len(edges)):
                    if edges[i][1] == edges[j][0] and edges[i][0] != edges[j][1]:
                        B[i, j] = 1
                    else:
                        B[i, j] = 0
            return csr_array(B)
        elif operator == 'BH_Bi':
            D = diags(self.bipartite_A.sum(axis=1).flatten().astype(float))
            B = eye(self.bipartite_A.shape[0]) * (r ** 2 - 1) - r * self.bipartite_A + D
            return B
        elif operator == "NB":
            directed_hyperedge_size = self.H.sum()
            directed_hyperedges = []
            B = np.zeros((directed_hyperedge_size, directed_hyperedge_size))
            for mu in range(self.e):
                for i in range(self.n):
                    if self.H[i, mu] == 1:
                        directed_hyperedges.append((i, mu))
            print(f'Non-backtrack constructing for {directed_hyperedge_size} directed node-hyperEdge pairs...')
            for index in tqdm(itertools.product(range(directed_hyperedge_size), repeat=2)):
                i = index[0]
                j = index[1]
                node_i = directed_hyperedges[i][0]
                node_j = directed_hyperedges[j][0]
                edge_i = self.H[:, [directed_hyperedges[i][1]]].nonzero()[0]
                edge_j = self.H[:, [directed_hyperedges[j][1]]].nonzero()[0]
                if node_j in edge_i and node_j != node_i and ((np.size(edge_j) != np.size(edge_i)) or ((edge_j == edge_i).all()) is np.False_):
                    B[i, j] = 1
                else:
                    B[i, j] = 0
            return csr_array(B)
        else:
            pass

    def getA_2order_edges(self):
        """
        :return: an adjacent matrix only based on 2-order edges in hypergraph
        """
        data = []
        row_ind = []
        col_ind = []
        for edge in self.A[2]:
            data.append(1)
            row_ind.append(edge[0])
            col_ind.append(edge[1])
            data.append(1)
            row_ind.append(edge[1])
            col_ind.append(edge[0])
        return csr_array((data, (row_ind, col_ind)))


class UniformSymmetricHSBM(HyperSBM):
    def __init__(self, n, q, k, cin, cout):
        """
        HyperGraph generated by HyperSBM:
        Uniform: with only one specific type of k-order edges
        Symmetric: The link probability is symmetric(
                    if k nodes have same community label, then cin/n^{k-1}. Otherwise cout/n^{k-1},
                    n^{k-1} to promise the network is sparse)
        :param n: Number of nodes
        :param q: Number of communities
        :param k: Order number of hyper edges
        :param cin: cin/n^{k-1} is the link probability if all k nodes have same community label
        :param cout: cout/n^{k-1} is the link probability if not all k nodes have same community label
        """
        self.n = n
        self.q = q
        self.k = k
        self.cin = cin
        self.cout = cout
        sizes = [int(n/q)] * q
        ps_dict = dict({k: None})
        cs = np.zeros(tuple([q]*k))
        for index in itertools.product(range(self.q), repeat=k):
            if np.size(np.unique(index)) == 1:
                cs[index] = cin
            else:
                cs[index] = cout
        ps = cs / (n ** (k-1))
        ps_dict[k] = ps
        super().__init__(sizes, ps_dict)

    def get_crit_epsilon(self):
        d = self.H.sum() / self.n
        epsilon_c = (np.sqrt(d * (self.k - 1)) - 1) / (np.sqrt(d * (self.k - 1)) + self.q - 1)
        epsilon = epsilon_c / (self.q ** (self.k - 2) - (self.q ** (self.k - 2) - 1) * epsilon_c)
        return epsilon

    def get_operator(self, operator='BH', r=0):
        if operator == "BH":
            D = diags(self.H.sum(axis=1).flatten().astype(float))
            A = self.H.dot(self.H.T) - diags(self.H.dot(self.H.T).diagonal())
            B = (r - 1) * (r + self.k - 1) * eye(D.shape[0]) + (self.k - 1) * D - r * A
            return B
        else:
            return super().get_operator(operator, r)


class UnUniformSymmetricHSBM(HyperSBM):
    def __init__(self, n, q, Ks, cin, cout):
        """
        HyperGraph generated by UnUniformSymmetricHyperSBM:
        """
        self.n = n
        self.q = q
        self.Ks = Ks
        self.cin = cin
        self.cout = cout
        sizes = [int(n/q)] * q
        ps_dict = dict()
        for k in self.Ks:
            cs = np.zeros(tuple([q]*k))
            for index in itertools.product(range(self.q), repeat=k):
                if np.size(np.unique(index)) == 1:
                    cs[index] = cin
                else:
                    cs[index] = cout
            ps = cs / (n ** (k-1))
            ps_dict[k] = ps
        super().__init__(sizes, ps_dict)

    def get_operator(self, operator='BH', r=0):
        if operator == "BH":
            edge_order = self.H.sum(axis=0).flatten()
            D = None
            A = None
            for k in self.Ks:
                edge_index = np.where(edge_order == k)[0]
                Hk = self.H[:, edge_index]
                Dk = diags(Hk.sum(axis=1).flatten().astype(float))
                Ak = Hk.dot(Hk.T) - diags(Hk.dot(Hk.T).diagonal())
                if D is None:
                    D = (k-1)/((1-r)*(r+k-1))*Dk
                else:
                    D += (k-1)/((1-r)*(r+k-1))*Dk
                if A is None:
                    A = r/((1-r)*(r+k-1))*Ak
                else:
                    A += r/((1-r)*(r+k-1))*Ak
            B = eye(D.shape[0]) - D + A
            return B
        else:
            return super().get_operator(operator, r)


def main_test():
    n = 100
    q = 2
    Ks = [2, 3]
    cin = 20
    cout = 2
    hsbm = UnUniformSymmetricHSBM(n, q, Ks, cin, cout)
    print(f'# of nodes {hsbm.n}, # of edges {hsbm.e}')
    start = time.time()
    NB = hsbm.get_operator('NB')
    # hsbm.get_operator("BH", r=2.8)
    print(f"Time for constructing operator: {time.time() - start}")


if __name__ == '__main__':
    main_test()
