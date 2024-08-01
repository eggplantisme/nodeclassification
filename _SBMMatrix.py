import numpy as np
from scipy.sparse.linalg import eigsh, inv
from scipy.sparse import eye, diags, issparse, csr_array, find, hstack, vstack
import networkx as nx
import sys
from numpy.linalg import eig
import matplotlib.pyplot as plt
import itertools
import random
from tqdm import tqdm
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

    def get_projection_operator(self, projection_matrix, operator='WNB', r=0):
        """
        :param operator: WNB, WBH
        :param r: parameter for WBH
        """
        if operator == 'WNB':
            edges = []
            x, y, _ = find(projection_matrix)
            for i, x_i in enumerate(x):
                edges.append((x_i, y[i]))
            e = len(edges)
            B = np.zeros((e, e))
            for i in range(len(edges)):
                for j in range(len(edges)):
                    if edges[i][1] == edges[j][0] and edges[i][0] != edges[j][1]:
                        B[i, j] = projection_matrix[edges[i][0], edges[i][1]]
                    else:
                        B[i, j] = 0
            return csr_array(B)
        elif operator == 'WBH':
            n1 = np.shape(projection_matrix)[0]
            BBT = projection_matrix
            # BBT = BBT / BBT.max() # Normalize
            # r = np.sqrt(BBT.sum() / BBT.shape[0])
            # BBT = r * BBT.tanh()
            d = csr_array(BBT ** 2 / (csr_array(r ** 2 * np.ones((n1, n1))) - BBT ** 2)).sum(axis=1).flatten().astype(
                float)
            d = diags(d, 0)
            d = d + csr_array(np.identity(n1))
            BH = d - csr_array((r * BBT) / (csr_array(r ** 2 * np.ones((n1, n1))) - BBT ** 2))
            return BH
        else:
            pass


class PoissonSBM:
    def __init__(self, sizes, omegas):
        """
        generate a weighted graph by poisson SBM in paper: <Stochastic blockmodels and community structure in networks>
        :param sizes: the sizes of each block
        :param omegas: the expect weight matrix: len(sizes) * len(sizes)
        """
        self.n = np.sum(sizes)
        self.sizes = sizes
        self.omegas = omegas
        self.groupId = []
        self.A = np.zeros((self.n, self.n))
        self.construct()

    def construct(self):
        gid = 0
        for s in self.sizes:
            self.groupId += [gid] * int(s)
            gid += 1
        self.groupId = np.array(self.groupId)
        for index in tqdm(itertools.combinations(range(self.n), r=2)):
            if np.size(np.unique(index)) < len(index):
                # pass when exist same node
                pass
            else:
                gindex = tuple(self.groupId[i] for i in index)
                omega = self.omegas[gindex]
                w = np.random.poisson(omega, 1)
                self.A[index] = w
        self.A = self.A + self.A.T
        self.A = csr_array(self.A)
        # self.A = self.A.tanh()

    def get_operator(self, operator='WNB', r=0):
        """
        :param operator: WNB, WBH
        :param r: parameter for WBH
        """
        if operator == 'WNB':
            edges = []
            x, y, _ = find(self.A)
            for i, x_i in enumerate(x):
                edges.append((x_i, y[i]))
            e = len(edges)
            B = np.zeros((e, e))
            for i in range(len(edges)):
                for j in range(len(edges)):
                    if edges[i][1] == edges[j][0] and edges[i][0] != edges[j][1]:
                        B[i, j] = self.A[edges[i][0], edges[i][1]]
                    else:
                        B[i, j] = 0
            return csr_array(B)
        elif operator == 'WBH':
            n1 = np.shape(self.A)[0]
            d = csr_array(self.A ** 2 / (csr_array(r ** 2 * np.ones((n1, n1))) - self.A ** 2)).sum(axis=1).flatten().astype(
                float)
            d = diags(d, 0)
            d = d + csr_array(np.identity(n1))
            BH = d - csr_array((r * self.A) / (csr_array(r ** 2 * np.ones((n1, n1))) - self.A ** 2))
            return BH
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
    # n = 2 ** 6
    # k = 3
    # d = 25
    # epsilon = 0
    # net = SymmetricSBM.init_epsc(n, k, d, epsilon)
    n = 1000
    q = 2
    d = 5
    snr = 3
    sizes = [int(n / q)] * q
    ps_dict = dict({2: None})  # only have 2-order edges
    pout = (d - np.sqrt(snr * d)) / n
    pin = pout + q * np.sqrt(snr * d) / n
    ps_dict[2] = (pin - pout) * np.identity(q) + pout * np.ones((q, q))
    hsbm = HyperSBM(sizes, ps_dict)


if __name__ == '__main__':
    main()
