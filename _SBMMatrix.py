import numpy as np
from scipy.sparse.linalg import eigsh, inv
from scipy.sparse import eye, diags, issparse, csr_array, find, hstack, vstack
import networkx as nx
import sys
from numpy.linalg import eig
import matplotlib.pyplot as plt
import itertools
import random
import hypernetx as hnx


class Matrix:
    def __init__(self, n):
        """
        :param n: Network (Matrix) size
        """
        self.n = n
        self.A = None

    def construct(self):
        pass


class SBMMatrix(Matrix):
    def __init__(self, sizes, ps):
        """
        :param sizes: the sizes of each block
        :param ps: the link probability matrix: len(sizes) * len(sizes)
        """
        super().__init__(np.sum(sizes))
        self.sizes = sizes
        self.ps = ps
        if len(sizes) == len(ps):
            self.g = None
            self.groupId = []
            self.construct()
        else:
            print("Parameter Wrong: please check sizes or ps!")
            sys.exit()

    def construct(self):
        if self.g is None:
            self.g = nx.stochastic_block_model(sizes=self.sizes, p=self.ps)
        self.A = nx.to_scipy_sparse_array(self.g)
        gid = 0
        for s in self.sizes:
            self.groupId += [gid] * int(s)
            gid += 1
        self.groupId = np.array(self.groupId)

    def get_operator(self, operator='A', r=0):
        """
        :param operator: A, L, NB, BH
        :param r: parameter for BH
        """
        if operator == 'A':
            return self.A
        elif operator == 'L':
            D = diags(self.A.sum(axis=1).flatten().astype(float))
            L = D - self.A
            return L
        elif operator == 'NB':
            edges = []
            x, y, _ = find(self.A)
            for i, x_i in enumerate(x):
                edges.append((x_i, y[i]))
            e = len(edges)
            B = np.zeros((e, e))
            for i in range(len(edges)):
                for j in range(len(edges)):
                    if edges[i][1] == edges[j][0] and edges[i][0] != edges[j][1]:
                        B[i, j] = 1
                    else:
                        B[i, j] = 0
            return csr_array(B)
        elif operator == 'BH':
            D = diags(self.A.sum(axis=1).flatten().astype(float))
            B = eye(self.A.shape[0]) * (r ** 2 - 1) - r * self.A + D
            return B
        else:
            pass

    def get_SNR(self):
        P = np.diag(np.array(self.sizes).flatten()) / self.n
        Q = self.n * np.array(self.ps)
        eigva, _ = eig(P.dot(Q))
        sorteigva = sorted(np.around(np.real(eigva), 5), key=lambda v: np.abs(v))
        snr = (sorteigva[-2] ** 2) / sorteigva[-1]
        return snr


class SymmetricSBM(SBMMatrix):
    def __init__(self, n, k, pin, pout):
        """
        Symmetric SBM with k equal size community,
        :param n: the network size
        :param k: the number of communities
        :param pin:  link probability in-community pin
        :param pout:  link probability between-communities pout
        """
        self.n = n
        self.k = k
        self.pin = pin
        self.pout = pout
        self.sizes = [int(n / k)] * (k - 1) + [n - (k - 1) * int(n / k)]
        self.ps = []
        for i in range(k):
            p_i = [self.pin if j == i else self.pout for j in range(k)]
            self.ps.append(p_i)
        super().__init__(self.sizes, self.ps)

    @classmethod
    def init_epsc(cls, n, k, c, epsilon):
        """
        Initial with epsilon-c mode
        :param n: the network size
        :param k: the number of communities
        :param c: the average degree, c = (cin + (k-1)cout)/k
        :param epsilon: cout/cin
        :return:
        """
        pin = c*k/(n*(epsilon*(k-1)+1))
        pout = pin * epsilon
        print(f'pin={pin}, pout={pout}')
        return cls(n, k, pin, pout)

    def get_SNR(self):
        din = self.n * self.pin / self.k
        dout = self.n * self.pout / self.k
        d = din + (self.k - 1) * dout
        return (din - dout) ** 2 / d


class BipartiteSBM(SBMMatrix):
    def __init__(self, k1, k2, sizes, H):
        """
        Bipartite SBM with k1 communities in 1st type, k2 communities in 2nd type
        :param k1: k1 communities in 1st type
        :param k2: k2 communities in 2nd type
        :param sizes: community size for k1+k2 communities
        :param H: k1*k2, link probability between nodes in 1st type and in 2nd type
        """
        self.k1 = k1
        self.k2 = k2
        self.H = np.array(H)
        if len(sizes) == k1 + k2 and np.shape(H) == (k1, k2):
            ps = []
            for i in range(k1 + k2):
                p_i = []
                for j in range(k1 + k2):
                    if (i < k1 and j < k1) or (i >= k1 and j >= k1):
                        p_i.append(0)
                    elif i < k1 <= j:
                        p_i.append(self.H[i, j-k1])
                    else:
                        p_i.append(self.H.T[i-k1, j])
                ps.append(p_i)
            super().__init__(sizes, ps)
        else:
            print("Parameter Wrong: please check sizes or omegas!")
            sys.exit()

    def getSingulars(self):
        _, s, _ = np.linalg.svd(self.H)
        return s


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
            self.A[key] = []
            for index in itertools.product(range(self.n), repeat=key):
                if np.size(np.unique(index)) < len(index):
                    # pass when exist same node
                    pass
                else:
                    gindex = tuple(self.groupId[i] for i in index)
                    p = np.array(self.ps_dict[key])[gindex]
                    r = random.random()
                    if r < p:
                        self.A[key].append(index)
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
        operator corresponding with bipartite form
        :param operator: B, NB, BH
        :param r: parameter for BH
        """
        if operator == 'B':
            return self.bipartite_A
        elif operator == 'NB':
            edges = []
            x, y, _ = find(self.bipartite_A)
            for i, x_i in enumerate(x):
                edges.append((x_i, y[i]))
            e = len(edges)
            B = np.zeros((e, e))
            for i in range(len(edges)):
                for j in range(len(edges)):
                    if edges[i][1] == edges[j][0] and edges[i][0] != edges[j][1]:
                        B[i, j] = 1
                    else:
                        B[i, j] = 0
            return csr_array(B)
        elif operator == 'BH':
            D = diags(self.bipartite_A.sum(axis=1).flatten().astype(float))
            B = eye(self.bipartite_A.shape[0]) * (r ** 2 - 1) - r * self.bipartite_A + D
            return B
        else:
            pass


def main():
    # hierarchy = generation.create2paramGHRG(n=3**9, snr=25, c_bar=38, n_levels=3, groups_per_level=3)
    # A = hierarchy.sample_network()
    #
    # Laplacian = spectral_operators.Laplacian(A)
    # BethsHessian = spectral_operators.BetheHessian(A)

    # rm = RandomMatrix(5)
    # print(rm.A, np.sum(rm.A))
    # rm.spy()
    # rm.plot_eigenvalue()
    # sizes = [25, 25, 25]
    # ps = [[0.9, 0.2, 0.1], [0.2, 0.8, 0.3], [0.1, 0.3, 0.9]]
    # sbm = SBMMatrix(sizes, ps)
    # sbm.change_operator('NB')
    # sbm.plot_eigenvalue()
    n = 2 ** 6
    k = 3
    d = 25
    epsilon = 0
    net = SymmetricSBM.init_epsc(n, k, d, epsilon)


if __name__ == '__main__':
    main()
