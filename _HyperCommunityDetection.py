# from _HyperSBM import *
from spectralOperator import *
from sklearn.cluster import KMeans
from _CommunityDetect import CommunityDetect


class HyperCommunityDetect:
    def __init__(self):
        pass

    @staticmethod
    def BetheHessian(hsbm, num_groups=None, only_assortative=False):
        d = hsbm.H.sum() / hsbm.n
        if type(hsbm) is UniformSymmetricHSBM:
            bulk = np.sqrt(d * (hsbm.k - 1))
        else:
            # For Nonuniform Hsbm or empirical hyper graph
            edge_order, edge_count = np.unique(hsbm.H.sum(axis=0).flatten(), return_counts=True)
            order_count = dict(zip(edge_order, edge_count))
            print(order_count)
            ds = dict()
            for o in order_count:
                ds[o] = o * order_count[o] / hsbm.n
            bulk = 0
            for k in hsbm.Ks:
                bulk += ds[k] * (k - 1)
            bulk = np.sqrt(bulk)
        BHa_pos = SpectralOperator()
        BHa_pos.operator = hsbm.get_operator("BH", r=bulk)
        if only_assortative is False:
            BHa_neg = SpectralOperator()
            BHa_neg.operator = hsbm.get_operator("BH", r=-bulk)
        if num_groups is None:
            Kpos = BHa_pos.find_negative_eigenvectors()
            if only_assortative is False:
                Kneg = BHa_neg.find_negative_eigenvectors()
                num_groups = Kpos + Kneg if Kpos + Kneg < hsbm.n else hsbm.n  # max number of group should be N
                print(f'number of groups = {num_groups}, Kpos={Kpos}, Kneg={Kneg}')
            else:
                num_groups = Kpos
                print(f'number of groups = {num_groups}, Kpos={Kpos}')
            if num_groups == 0 or num_groups == 1:
                print("no indication for grouping -- return all in one partition")
                partition_vecs = np.zeros(hsbm.n, dtype='int')
                return partition_vecs, num_groups
            # construct combined_evecs to cluster
            if only_assortative is False:
                combined_evecs = np.hstack([BHa_pos.evecs, BHa_neg.evecs])
            else:
                combined_evecs = BHa_pos.evecs
        else:
            # If num_group is given, cluster evec corresonding with the first num_group eval of BHa_pos and BHa_neg
            BHa_pos.find_k_eigenvectors(num_groups, which='SA')
            if only_assortative is False:
                BHa_neg.find_k_eigenvectors(num_groups, which='SA')
            # combine both sets of eigenvales and eigenvectors and take first k
            if only_assortative is False:
                combined_evecs = np.hstack([BHa_pos.evecs, BHa_neg.evecs])
                combined_evals = np.hstack([BHa_pos.evals, BHa_neg.evals])
            else:
                combined_evecs = BHa_pos.evecs
                combined_evals = BHa_pos.evals
            index = np.argsort(combined_evals)
            combined_evecs = combined_evecs[:, index[:num_groups]]
        # cluster with Kmeans
        if num_groups < hsbm.n:
            cluster = KMeans(n_clusters=num_groups, n_init=20)
            cluster.fit(combined_evecs)
            partition_vecs = cluster.predict(combined_evecs)
        else:
            partition_vecs = np.array(list(range(hsbm.n)))
        return partition_vecs, num_groups

    @staticmethod
    def BipartiteBH(hsbm, num_groups=None):
        projA = hsbm.H.dot(hsbm.H.T)
        projA = projA - diags(projA.diagonal())
        return CommunityDetect(projA).BetheHessian(num_groups, weighted=True)
