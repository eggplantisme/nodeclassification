import numpy as np
import time
import os
from spectralOperator import BetheHessian
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from _DetectabilityWithMeta import *
from multiprocessing import Pool


class CommunityDetect:
    def __init__(self, A):
        """
        do community detection to network represented by adjacency matrix A
        :param A: sparse csr matrix
        """
        self.A = A

    def BetheHessian(self, num_groups=None):
        BHa_pos = BetheHessian(self.A, regularizer='BHa')
        BHa_neg = BetheHessian(self.A, regularizer='BHan')
        if num_groups is None:
            Kpos = BHa_pos.find_negative_eigenvectors()
            Kneg = BHa_neg.find_negative_eigenvectors()
            num_groups = Kpos + Kneg
            print(f'number of groups = {num_groups}')
            if num_groups == 0 or num_groups == 1:
                print("no indication for grouping -- return all in one partition")
                partition_vector = np.zeros(self.A.shape[0], dtype='int')
                return partition_vector, num_groups
            # construct combined_evecs to cluster
            combined_evecs = np.hstack([BHa_pos.evecs, BHa_neg.evecs])
        else:
            # If num_group is given, cluster evec corresonding with the first num_group eval of BHa_pos and BHa_neg
            BHa_pos.find_k_eigenvectors(num_groups, which='SA')
            BHa_neg.find_k_eigenvectors(num_groups, which='SA')
            # combine both sets of eigenvales and eigenvectors and take first k
            combined_evecs = np.hstack([BHa_pos.evecs, BHa_neg.evecs])
            combined_evals = np.hstack([BHa_pos.evals, BHa_neg.evals])
            index = np.argsort(combined_evals)
            combined_evecs = combined_evecs[:, index[:num_groups]]
        # cluster with Kmeans
        cluster = KMeans(n_clusters=num_groups, n_init=20)
        cluster.fit(combined_evecs)
        partition_vecs = cluster.predict(combined_evecs)
        return partition_vecs, num_groups

    @staticmethod
    def run_exp(rhos, deltas, times, save_path=None, X=2, Z=3, n=600, d=300, Withsnr=False):
        for rho in rhos:
            for delta in deltas:
                pin = d / n + delta * (1 - 1 / (X * Z))
                pout = d / n - delta / (X * Z)
                pin = 0 if pin < 1e-10 else pin
                pout = 0 if pout < 1e-10 else pout
                # pout = 2 * Z * dout / n
                # pin = Z * pout * z + pout
                msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
                for t in range(times):
                    start = time.time()
                    print(f"EXP begin... rho={rho}, delta={delta}, times={t}, pin={pin}, pout={pout}")
                    A = msbm.sample()
                    fullBHpartition = CommunityDetect(A).BetheHessian()
                    full_ami = adjusted_mutual_info_score(msbm.groupId, fullBHpartition)
                    filterA, filterGroupId = msbm.filter(A, metaId=0)
                    subBHpartition = CommunityDetect(filterA).BetheHessian()
                    sub_ami = adjusted_mutual_info_score(filterGroupId, subBHpartition)
                    if Withsnr and X == 2:
                        snr_nm = msbm.snr(rho, delta, N=n, X=X, Z=Z, d=d, withMeta=False)
                        snr_m = msbm.snr(rho, delta, N=n, X=X, Z=Z, d=d, withMeta=True)
                        print(f"EXP end. full_ami={full_ami}, sub_ami={sub_ami}. snr_nm={snr_nm}, snr_m={snr_m}. "
                              f"Time:{np.around(time.time()-start, 3)}")
                        if save_path is not None:
                            with open(save_path, 'a') as fw:
                                fw.write(f'{rho} {delta} {t} {full_ami} {sub_ami} {snr_nm} {snr_m}\n')
                    else:
                        print(f"EXP end. full_ami={full_ami}, sub_ami={sub_ami}. "
                              f"Time:{np.around(time.time()-start, 3)}")
                        if save_path is not None:
                            with open(save_path, 'a') as fw:
                                fw.write(f'{rho} {delta} {t} {full_ami} {sub_ami}\n')

    @staticmethod
    def read_exp(load_path, Withsnr=False):
        with open(load_path, 'r') as f:
            results = np.float64([row.strip().split() for row in f.readlines()])
            rhos = np.unique(results[:, 0])
            zs = np.unique(results[:, 1])
            full_ami = np.zeros(np.size(zs) * np.size(rhos))
            sub_ami = np.zeros(np.size(zs) * np.size(rhos))
            snr_nm = np.zeros(np.size(zs) * np.size(rhos)) if Withsnr else None
            snr_m = np.zeros(np.size(zs) * np.size(rhos)) if Withsnr else None
            i = 0
            for _rho in rhos:
                for _z in zs:
                    ami_results = results[np.squeeze(np.argwhere(np.logical_and(results[:, 0]==_rho, results[:, 1]==_z)))]
                    mean_ami = np.mean(ami_results, 0)[3:]
                    full_ami[i] = mean_ami[0]
                    sub_ami[i] = mean_ami[1]
                    if Withsnr:
                        snr_nm[i] = mean_ami[2]
                        snr_m[i] = mean_ami[3]
                    i += 1
            plot_rhos = np.repeat(rhos, np.size(zs))
            plot_zs = np.tile(zs, np.size(rhos))
        return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m


def test_main():
    # TEST 1
    fileID = 'amiExp4.20'
    load_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    plot_rhos, plot_zs, full_ami, sub_ami = CommunityDetect.read_exp(load_path=load_path)
    import _FigureJiazeHelper
    _FigureJiazeHelper.color_scatter_2d(plot_rhos, plot_zs, full_ami, title="AMI for full graph", xlabel=r'$\rho$',
                                        ylabel=r'$z$', save_path=None)
    # TEST 0
    # rho = 0.5
    # n = 600
    # X = 2  # Number of Meta
    # Z = 3  # Number of Group in each Meta
    # pin = 0.6
    # pout = 0.3
    # msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    # A = msbm.sample()
    # cd = CommunityDetect(A)
    # BHpartition = cd.BetheHessian()
    # print("FULL, real labels:", np.unique(msbm.groupId))
    # print("FULL, detect labels:", np.unique(BHpartition))
    # print(adjusted_mutual_info_score(msbm.groupId, BHpartition))
    # filterA, filterGroupId = msbm.filter(A, metaId=0)
    # # print("SUB graph size:", np.size(filterGroupId))
    # # print("SUB, real labels:", np.unique(filterGroupId0))
    # cd = CommunityDetect(filterA)
    # BHpartition = cd.BetheHessian()
    # # print("SUB, detect labels:", np.unique(BHpartition))
    # # print("SUB graph size:", np.size(filterGroupId))
    # # print("SUB, real labels:", np.unique(filterGroupId0))
    # print(adjusted_mutual_info_score(filterGroupId, BHpartition))


if __name__ == '__main__':
    test_main()
