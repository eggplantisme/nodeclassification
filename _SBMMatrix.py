import numpy as np
import scipy as sp
import networkx as nx
import sys
import matplotlib.pyplot as plt


class Matrix:
    def __init__(self, n):
        self.n = n
        self.A = None

    def construct(self):
        pass


class RandomMatrix(Matrix):
    def __init__(self, n):
        super().__init__(n)
        self.construct()

    def construct(self):
        """
        random matrix with row sum 1
        """
        self.A = np.random.dirichlet(np.ones(self.n), size=self.n)


class SBMMatrix(Matrix):
    def __init__(self, sizes, ps, operator="A"):
        """
        :param sizes: the sizes of each block
        :param ps: the link probability matrix: len(sizes) * len(sizes)
        :param operator:
        """
        super().__init__(np.sum(sizes))
        self.g = None
        self.sizes = sizes
        self.ps = ps
        self.operator = operator
        if len(sizes) == len(ps):
            self.construct()
        else:
            print("Parameter Wrong: please check sizes or ps!")

    def construct(self):
        if self.g is None:
            self.g = nx.stochastic_block_model(sizes=self.sizes, p=self.ps)
        self.change_operator(self.operator)

    def change_operator(self, operator='A', r=0):
        """ operator: A, L, L_rw, L_sym, NB, BH
                r special for BH
        """
        self.operator = operator
        A = nx.to_numpy_array(self.g)
        if self.operator == 'A':
            self.A = A
        elif self.operator == 'L':
            L = np.diag(np.sum(A, 0)) - A
            self.A = L
        elif self.operator == 'L_rw':
            self.A = np.linalg.inv(np.diag(np.sum(A, 0))) @ A
        elif self.operator == 'L_sym':
            D = np.sum(A, 0)
            _D = np.diag(D)
            L = _D @ A @ _D
            self.A = L
        elif self.operator == 'NB':
            edges = []
            x, y = np.where(A == 1)
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
            self.A = B
        elif self.operator == 'BH':
            D = np.sum(A, 0)
            _D = np.diag(D)
            B = np.eye(np.shape(A)[0]) * (r**2 - 1) - r * A + _D
            self.A = B
        else:
            pass


class SymmetricSBM(SBMMatrix):
    def __init__(self, n, c, k, epsilon):
        """
        :param n: the network size
        :param c: the average degree of the network c < n
        :param k: the block number of the SBM k >= 2
        :param epsilon: cout/cin > 0
        """
        self.n = n
        self.k = k
        self.c = c
        self.epsilon = epsilon
        self.cin = c*k/(epsilon*(k-1)+1)
        self.cout = c*k*epsilon/(epsilon*(k-1)+1)
        if self.cin > n or self.cout > n:
            print("Because the setting of epsilon, cin/cout > n!", end=" ")
            a = c*k/(k-1)
            b = c*k
            if n > b:
                s = "epsilon should greater than 0"
            elif n >= a:
                lower_bound = round((c*k-1)/n*(k-1), 3)
                s = "epsilon should greater than " + str(lower_bound)
            else:
                lower_bound = round((c * k - 1) / n * (k - 1), 3)
                upper_bound = round(n/(c*k-n*(k-1)), 3)
                s = "epsilon should greater than " + str(lower_bound) + " smaller than " + str(upper_bound)
            print(s)
            sys.exit()
        self.sizes = [int(n/k)] * (k-1) + [n - (k-1)*int(n/k)]
        pin, pout = self.cin/n, self.cout/n
        self.ps = []
        for i in range(k):
            p_i = [pin if j == i else pout for j in range(k)]
            self.ps.append(p_i)
        super(SymmetricSBM, self).__init__(self.sizes, self.ps)


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
    sizes = [25, 25, 25]
    ps = [[0.9, 0.2, 0.1], [0.2, 0.8, 0.3], [0.1, 0.3, 0.9]]
    sbm = SBMMatrix(sizes, ps)
    # sbm.change_operator('NB')
    # sbm.plot_eigenvalue()


if __name__ == '__main__':
    main()
