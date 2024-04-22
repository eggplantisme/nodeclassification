from EXPERIMENT import get_range_delta
from _CommunityDetect import *


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
    rho = 0.5
    pin = d / n + delta * (1 - 1 / (X * Z))
    pout = d / n - delta / (X * Z)
    pin = 0 if pin < 1e-10 else pin
    pout = 0 if pout < 1e-10 else pout
    print(f"EXP pid={os.getpid()} begin... n={n}, X={X}, Z={Z}, d={d}, rho={rho}, delta={delta}, pin={pin}, pout={pout}")

    # Create model and get snr for full and subcase.
    msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    snr_nm = msbm.snr(rho, delta, n, X, Z, d, withMeta=False)
    snr_m = msbm.snr(rho, delta, n, X, Z, d, withMeta=True)
    print(f'SNR for full graph(without meta)={snr_nm}, SNR for sub graph(with meta)={snr_m}')

    # Calculate AMI for full and sub graph
    A = msbm.sample()
    fullBHpartition, _ = CommunityDetect(A).BetheHessian()
    fullBHami = adjusted_mutual_info_score(msbm.groupId, fullBHpartition)

    fullDCBHpartition, _, _ = CommunityDetect(A).DCBetheHessian()
    fullDCBHami = adjusted_mutual_info_score(msbm.groupId, fullDCBHpartition)

    filterA0, filterGroupId0 = msbm.filter(A, metaId=0)
    sub0BHpartition, _ = CommunityDetect(filterA0).BetheHessian()
    sub0BHami = adjusted_mutual_info_score(filterGroupId0, sub0BHpartition)

    sub0DCBHpartition, _, _ = CommunityDetect(filterA0).DCBetheHessian()
    sub0DCBHami = adjusted_mutual_info_score(filterGroupId0, sub0DCBHpartition)

    help_evec, partition_vecs = CommunityDetect(A).BetheHessian(return_evec=True)
    meta0Index = np.where(msbm.metaId == 0)[0]
    help_evec = help_evec[meta0Index, :]
    help_num_groups = np.size(np.unique(partition_vecs[meta0Index]))
    sub0BHwithFULLpartition, _ = CommunityDetect(filterA0).BetheHessian(help_evec=help_evec, help_num_groups=help_num_groups)
    sub0BHwithFULLami = adjusted_mutual_info_score(filterGroupId0, sub0BHwithFULLpartition)

    print(f'AMI for full by BH = {fullBHami}, AMI for full by DCBH = {fullDCBHami}, \n '
          f'AMI for sub0 by BH = {sub0BHami}, AMI for sub0 by DCBH = {sub0DCBHami}, '
          f'AMI for sub0 by BH help by FULL = {sub0BHwithFULLami}')


if __name__ == '__main__':
    prepare()
