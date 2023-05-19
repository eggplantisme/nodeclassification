import numpy as np
import random
import networkx as nx


class MetaSBM:
    def __init__(self, n, rho, ps=None, sizes=None):
        """
        A stochastic block model with metadata generated based on groups
        :param n: number of nodes
        :param rho: alignment probability between meta and group
        :param ps: edge probability between different groups
        :param sizes: number of nodes in each group of each meta, a list of lists, for example:
                      [[10, 10, 10], [10, 10, 10], [10, 10 ,10]] means 3 metadata, each metadata has 3 group,
                      each group 10 nodes.
        """
        self.N = n
        self.rho = rho
        self.sizes = None
        self.ps = None
        self.get_sbm_parameter(sizes, ps)
        self.groupId = None
        self.metaId = None
        self.generate_meta()

    def get_sbm_parameter(self, sizes, ps):
        if sizes is None and ps is None:
            # default sizes [[N/4, N/4], [N/4, N/4]]
            number_of_Meta = 2
            number_of_GroupPerMeta = 2
            number_of_group = number_of_Meta * number_of_GroupPerMeta
            if self.N < number_of_group:
                print("Number of Node Too less!")
            else:
                number_of_NodesPerGroup = int(self.N / number_of_group)
                self.sizes = [[number_of_NodesPerGroup] * number_of_GroupPerMeta
                              for i in range(number_of_Meta)]
            # default ps pin=0.8, pout=0.2
            pin = 0.8
            pout = 0.2
            self.ps = (pin - pout) * np.identity(number_of_group) + \
                       pout * np.ones((number_of_group, number_of_group))
        elif sizes is None or ps is None:
            print("Please define both sizes and ps!")
        else:
            self.sizes = sizes
            self.ps = ps

    def generate_meta(self):
        number_of_group = np.sum([len(s) for s in self.sizes])
        number_of_meta = len(self.sizes)
        self.groupId = np.arange(0, number_of_group).repeat(np.array([gs for ms in self.sizes for gs in ms]))
        self.metaId = np.arange(0, number_of_meta).repeat(np.array([np.sum(ms) for ms in self.sizes]))
        metaIdClass = np.unique(self.metaId)
        for i in range(self.N):
            r = random.random()
            if r > self.rho:
                self.metaId[i] = np.random.choice(np.setdiff1d(metaIdClass, np.array([self.metaId[i]])))
        print("Metadata generation done!")

    def sample(self):
        group_sizes = np.array(self.sizes).flatten()
        g = nx.stochastic_block_model(sizes=group_sizes, p=self.ps)
        A = nx.to_scipy_sparse_array(g)  # csr sparse matrix
        return A

    def filter(self, A, metaId):
        """
        filter the network with specific metaId
        :param A: adjacent matrix by sample
        :param metaId: specific metaId
        :return: filtered matrix and corresponding groupId
        """
        metaIndex = np.where(self.metaId == metaId)[0]
        filterA = A[np.ix_(metaIndex, metaIndex)]
        filterGroupId = self.groupId[metaIndex]
        return filterA, filterGroupId

    @staticmethod
    def snr(_rho=0.5, _delta=0, N=600, X=2, Z=3, d=300, withMeta=False):
        if withMeta is False:
            snr = N**2 * _delta**2 / ((X*Z)**2*d)
        else:
            # only know for X=2
            if _rho <= 1 / 2:
                # SNR1
                snr = 2*(N**2)*((1-_rho)**2)*(_delta**2)/(Z*(2*Z*d+N*_delta+np.sqrt((2*Z*d-N*_delta)**2+(8*N*Z*d*_delta)*((1-2*_rho)**2))))
            elif _rho > 1 / 2:
                # SNR2
                snr = 2*(N**2)*(_rho**2)*(_delta**2)/(Z*(2*Z*d+N*_delta+np.sqrt((2*Z*d-N*_delta)**2+(8*N*Z*d*_delta)*((1-2*_rho)**2))))
        return snr


class SymMetaSBM(MetaSBM):
    def __init__(self, n, X, Z, rho, pin, pout):
        sizes = [[int(n/(X*Z)) for z in range(Z)] for x in range(X)]
        ps = (pin - pout) * np.identity(X*Z) + pout * np.ones((X*Z, X*Z))
        super().__init__(n, rho, ps, sizes)


def test_main():
    n = 2 ** 10
    rho = 0.8
    MetaSBM(n, rho)


if __name__ == '__main__':
    test_main()
