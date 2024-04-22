import numpy as np
from scipy.sparse import csr_array
import time
from _SBMMatrix import HyperSBM
from _CommunityDetect import CommunityDetect, adjusted_mutual_info_score
from _FigureJiazeHelper import get_confusionmatrix


def testSSBM(n=3000, q=3, d=15):
    sizes = [int(n / q)] * q
    ps_dict = dict({2: None})  # only have 2-order edges
    SNRs = np.concatenate((np.linspace(0.1, 1, 10), np.linspace(2, 10, 9)), axis=None)
    A_BHamis = []
    A_BHnumbers = []
    BA_BHamis = []
    BA_BHnumbers = []
    for snr in SNRs:
        start = time.time()
        # Consider assortative case
        pout = (d - np.sqrt(snr * d)) / n
        pin = pout + q * np.sqrt(snr * d) / n
        ps_dict[2] = (pin - pout) * np.identity(q) + pout * np.ones((q, q))
        hsbm = HyperSBM(sizes, ps_dict)
        print(f'SNR={snr} start. pin={pin}, pout={pout}, hsbm construct time={time.time()-start}')
        # Construct adjacent matrix A
        data = []
        row_ind = []
        col_ind = []
        for edge in hsbm.A[2]:
            data.append(1)
            row_ind.append(edge[0])
            col_ind.append(edge[1])
            data.append(1)
            row_ind.append(edge[1])
            col_ind.append(edge[0])
        A = csr_array((data, (row_ind, col_ind)))
        # BH on A
        start = time.time()
        cd = CommunityDetect(A)
        A_BHpartition, A_BHnumgroups = cd.BetheHessian()
        true_numberpartition = q
        node_partition = A_BHpartition
        node_numberpartition = np.size(np.unique(node_partition))
        A_cm, _ = get_confusionmatrix(hsbm.groupId, node_partition, true_numberpartition, node_numberpartition)
        A_ami = adjusted_mutual_info_score(hsbm.groupId, node_partition)
        print(f"BH result in A: {A_ami}. Time={time.time() - start}. Confusion Matrix({np.shape(A_cm)}) is: \n{A_cm}")
        A_BHamis.append(A_ami)
        A_BHnumbers.append(node_numberpartition)
        # BH on bipartite_A (R^{(n+e) * (n+e)})
        start = time.time()
        cd = CommunityDetect(hsbm.bipartite_A)
        BA_BHpartition, BA_BHnumgroups = cd.BetheHessian()
        true_numberpartition = q
        node_partition = BA_BHpartition[:hsbm.n]
        node_numberpartition = np.size(np.unique(node_partition))
        BA_cm, _ = get_confusionmatrix(hsbm.groupId, node_partition, true_numberpartition, node_numberpartition)
        BA_ami = adjusted_mutual_info_score(hsbm.groupId, node_partition)
        print(f"BH result in bipartiteA: {BA_ami}. Time={time.time() - start}. Confusion Matrix({np.shape(BA_cm)}) is: \n{BA_cm}")
        BA_BHamis.append(BA_ami)
        BA_BHnumbers.append(node_numberpartition)
    return SNRs, A_BHamis, A_BHnumbers, BA_BHamis, BA_BHnumbers


if __name__ == '__main__':
    testSSBM(n=1000, q=2, d=10)
