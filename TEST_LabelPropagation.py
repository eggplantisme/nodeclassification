import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from _CommunityDetect import *
from _VisNetwork import *
from propagation import *
from EXPERIMENT import get_range_delta
import random
import os


def prepare():
    # Parameter prepare
    X = 2
    Z = 3
    n = X * Z * 2000  # 12000 nodes
    d = 50
    min_delta, max_delta = get_range_delta(d, n, X, Z)
    print(f'The range of delta is {min_delta}~{max_delta}')
    # delta = random.choice(np.setdiff1d(np.around(np.linspace(min_delta, max_delta, int((max_delta-min_delta)/0.01)+1), 5), np.array([0])))
    delta = 0.004
    rho = 0.1
    pin = d / n + delta * (1 - 1 / (X * Z))
    pout = d / n - delta / (X * Z)
    pin = 0 if pin < 1e-10 else pin
    pout = 0 if pout < 1e-10 else pout
    print(f"EXP pid={os.getpid()} begin... n={n}, X={X}, Z={Z}, d={d}, rho={rho}, delta={delta}, pin={pin}, pout={pout}")

    # Create model and get snr for full and sub case.
    msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    snr_nm = msbm.snr(rho, delta, n, X, Z, d, withMeta=False)
    snr_m = msbm.snr(rho, delta, n, X, Z, d, withMeta=True)
    print(f'SNR for full graph(without meta)={snr_nm}, SNR for sub graph(with meta)={snr_m}')

    # Calculate AMI for full and sub graph
    A = msbm.sample()
    fullBHpartition, _ = CommunityDetect(A).BetheHessian()
    full_ami = adjusted_mutual_info_score(msbm.groupId, fullBHpartition)

    filterA0, filterGroupId0 = msbm.filter(A, metaId=0)
    sub0BHpartition, _ = CommunityDetect(filterA0).BetheHessian()
    sub0_ami = adjusted_mutual_info_score(filterGroupId0, sub0BHpartition)

    filterA1, filterGroupId1 = msbm.filter(A, metaId=1)
    sub1BHpartition, _ = CommunityDetect(filterA1).BetheHessian()
    sub1_ami = adjusted_mutual_info_score(filterGroupId1, sub1BHpartition)
    print(f'AMI for full = {full_ami}, AMI for sub0 = {sub0_ami}, AMI for sub1 = {sub1_ami}')
    return msbm, A, filterA0, filterA1, filterGroupId0, filterGroupId1, sub0BHpartition, sub1BHpartition


def infer_fullpartition(msbm, A, filterA0, filterA1, filterGroupId0, filterGroupId1, sub0BHpartition, sub1BHpartition):
    # Concatenate the 2 sub graph real group and detected partition
    # filterGroupId = np.concatenate((filterGroupId0, filterGroupId1))
    sub1BHpartition = np.size(np.unique(sub0BHpartition)) + sub1BHpartition
    # subBHpartition = np.concatenate((sub0BHpartition, sub1BHpartition))
    meta0Index = np.where(msbm.metaId == 0)[0]
    meta1Index = np.where(msbm.metaId == 1)[0]
    partitionConcatenatedBysub = np.zeros(msbm.N)
    for i, p in enumerate(sub0BHpartition):
        partitionConcatenatedBysub[meta0Index[i]] = p
    for i, p in enumerate(sub1BHpartition):
        partitionConcatenatedBysub[meta1Index[i]] = p
    subPropagationPartition, num_group = CommunityDetect(A).TwoStepLabelPropagate(partitionConcatenatedBysub)
    sub_ami = adjusted_mutual_info_score(msbm.groupId, subPropagationPartition)
    print(f'sub_ami={sub_ami}, num_label={num_group}')
    # B construct from 2 subgraph result
    # k = len(np.unique(sub0BHpartition)) + len(np.unique(sub1BHpartition))
    # nr_nodes = msbm.N
    # H = sparse.coo_matrix((np.ones(nr_nodes), (np.arange(nr_nodes), partitionConcatenatedBysub)), shape=(nr_nodes, k)).tocsr()
    # B construct from 1 subgraph result
    # k = len(np.unique(sub0BHpartition))
    # k = 6
    # nr_nodes = msbm.N
    # nr0_nodes = np.size(sub0BHpartition)
    # H = sparse.coo_matrix((np.ones(nr0_nodes), (meta0Index, sub0BHpartition)), shape=(nr_nodes, k)).tocsr()

    # Derive the full network partition by sub graph partition (By propagation)
    # alpha = 0.1
    # propagation = TwoStepLabelPropagation(A, k, H, alpha=alpha)
    # visNodeGroup(msbm.groupId)
    # visNodeGroup(partitionConcatenatedBysub)
    # iter_i = 0
    # while True:
    #     last_F = np.copy(propagation.signal)
    #     propagation.propagate()
    #     subPropagationPartition = propagation.result()
    #     sub_ami = adjusted_mutual_info_score(msbm.groupId, subPropagationPartition)
    #     visNodeGroup(subPropagationPartition)
    #     F = propagation.signal
    #     diff = np.sum(np.abs(F-last_F))
    #     print(f'iter {iter_i}, sub_ami={sub_ami}, num_label={np.unique(subPropagationPartition)}, diff={diff}')
    #     if diff < 1e-6 or iter_i > 200:
    #         break
    #     iter_i += 1


def main():
    msbm, A, filterA0, filterA1, filterGroupId0, filterGroupId1, sub0BHpartition, sub1BHpartition = prepare()
    infer_fullpartition(msbm, A, filterA0, filterA1, filterGroupId0, filterGroupId1, sub0BHpartition, sub1BHpartition)


if __name__ == '__main__':
    main()

# Results
# The range of delta is -0.005~0.025
# EXP pid=31472 begin... n=12000, X=2, Z=3, d=50, rho=0.1, delta=0.004, pin=0.0075, pout=0.0035
# Metadata generation done!
# SNR for full graph(without meta)=1.28, SNR for sub graph(with meta)=1.8714545532805171
# number of groups = 6
# number of groups = 3
# number of groups = 3
# AMI for full = 0.22933833472017287, AMI for sub0 = 0.3403169188791801, AMI for sub1 = 0.3353095691075711
# iter 0, sub_ami=0.4459110645706607, num_label=[0 1 2 3 4 5]
# iter 1, sub_ami=0.4459110645706607, num_label=[0 1 2 3 4 5]
# iter 2, sub_ami=0.4459110645706607, num_label=[0 1 2 3 4 5]
# iter 3, sub_ami=0.4459110645706607, num_label=[0 1 2 3 4 5]
# iter 4, sub_ami=0.4459110645706607, num_label=[0 1 2 3 4 5]
# iter 5, sub_ami=0.4459110645706607, num_label=[0 1 2 3 4 5]
# iter 6, sub_ami=0.4459110645706607, num_label=[0 1 2 3 4 5]
