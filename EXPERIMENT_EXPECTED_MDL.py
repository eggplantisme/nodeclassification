import numpy as np
import time
from EXPERIMENT_MINORITY import get_range_delta
from _CommunityDetect import *


def run_exp(rhos, deltas, times, save_path=None, q=3, n=600, d=300, Z_s=None, Z_b=None, multiprocessing=True):
    rho_delta_pair = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                rho_delta_pair.add((round(float(row[0]), 5), round(float(row[1]), 5)))
    if multiprocessing:
        p = Pool(10)
        for rho in rhos:
            for delta in deltas:
                if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                    print(f'rho={rho}, delta={delta} has been run!')
                    continue
                p.apply_async(exp_subprocess, args=(n, q, Z_s, Z_b, d, rho, delta, times, save_path,),
                              callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for rho in rhos:
            for delta in deltas:
                if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                    print(f'rho={rho}, delta={delta} has been run!')
                    continue
                savepath, results = exp_subprocess(n, q, Z_s, Z_b, d, rho, delta, times, save_path)
                write_results((savepath, results))


def read_exp(load_path, exclude_rho=None, add_path=None):
    """
    read the results file from run_exp
    :param load_path:
    :return:
    """
    exclude_rho = [] if exclude_rho is None else exclude_rho
    max_lambda_num = 4
    with open(load_path, 'r') as f:
        results_list = [row.strip().split() for row in f.readlines()]
        if add_path is not None:
            with open(add_path, 'r') as add_f:
                results_list = results_list + [row.strip().split() for row in add_f.readlines()]
        results = np.round(np.float64(results_list), decimals=5)
        rhos = np.setdiff1d(np.unique(results[:, 0]), np.array(exclude_rho))
        zs = np.unique(results[:, 1])
        learnq = np.zeros(np.size(zs) * np.size(rhos))
        i = 0
        for _rho in rhos:
            for _z in zs:
                q_results = results[
                    np.squeeze(np.argwhere(np.logical_and(results[:, 0] == _rho, results[:, 1] == _z)))]
                if np.size(q_results) == 0:
                    print(f"Some parameter rho={_rho}, z={_z} didn't run!")
                mean_q = np.mean(q_results, 0)[-1] if len(np.shape(q_results)) == 2 else q_results[-1]
                learnq[i] = mean_q
                i += 1
        plot_rhos = np.repeat(rhos, np.size(zs))
        plot_zs = np.tile(zs, np.size(rhos))
    # return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group
    return plot_rhos, plot_zs, learnq


def write_results(arg):
    """
    :param arg: savepath, results
    :return:
    """
    if arg[0] is not None:
        with open(arg[0], 'a') as fw:
            fw.write(arg[1])


def print_error(value):
    print(value)


def exp_subprocess(n, q, Z_s, Z_b, d, rho, delta, times, save_path):
    start = time.time()
    # generate subgraph
    pout = d / n - ((1 - rho) ** 2 / Z_b + rho ** 2 / Z_s) * delta
    pin = pout + delta
    pin = 0 if pin < 1e-10 else pin
    pout = 0 if pout < 1e-10 else pout
    ps = (pin - pout) * np.identity(q) + pout * np.ones((q, q))
    n_f = int(n * (Z_s + Z_b) * (Z_b * rho + Z_s * (1 - rho)) / (Z_s * Z_b))
    rho_f = Z_b * rho / (Z_b * rho + Z_s * (1 - rho))
    n_fq = int(n_f / q)
    n_f = int(n_fq * q)
    sizes = [[n_fq] * Z_s, [n_fq] * Z_b]
    msbm = MetaSBM(n_f, rho_f, ps, sizes)
    A = msbm.sample()
    metaIdSelect = 0
    filterA, filterGroupId = msbm.filter(A, metaId=metaIdSelect)
    N = np.size(filterGroupId)
    E = np.sum(filterA) / 2
    print(f'rho={rho}, delta={delta}, generation time:{time.time() - start}')
    # calculate e_rs of True
    unique_gid, unique_gid_count = np.unique(filterGroupId, return_counts=True)
    real_q = np.size(unique_gid)
    Ters = np.zeros((real_q, real_q))
    for r in unique_gid:
        for s in unique_gid:
            r_index = np.where(filterGroupId == r)[0]
            s_index = np.where(filterGroupId == s)[0]
            ers = np.sum(filterA[np.ix_(r_index, s_index)])
            ers = ers if r != s else ers / 2
            Ters[r, s] = ers
    # for q=1, 2, 3, 4: get expected confusion matrix, calculate e_rs of expected, get MDL and record
    results = f'{rho} {delta} '
    Epsilons = []
    for q in [1, 2, 3, 4]:
        if q == 1:
            Ecm = np.array([[1], [1], [1], [1]])
            Enrs = np.array([np.sum(unique_gid_count)])
        elif q == 2:
            Ecm = np.array([[0.5, 0.5], [0.5, 0.5], [1, 0], [0, 1]]) if rho < 0.5 else \
                np.array([[1, 0], [0, 1], [0.5, 0.5], [0.5, 0.5]])
            Enrs = np.array([(unique_gid_count[0] + unique_gid_count[1]) / 2 + unique_gid_count[2],
                             (unique_gid_count[0] + unique_gid_count[1]) / 2 + unique_gid_count[3]]) if rho < 0.5 else \
                np.array([unique_gid_count[0] + (unique_gid_count[2] + unique_gid_count[3]) / 2,
                          unique_gid_count[1] + (unique_gid_count[2] + unique_gid_count[3]) / 2])
        elif q == 3:
            Ecm = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) if rho < 0.5 else \
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
            Enrs = np.array(
                [unique_gid_count[0] + unique_gid_count[1], unique_gid_count[2], unique_gid_count[3]]) if rho < 0.5 else \
                np.array([unique_gid_count[0], unique_gid_count[1], unique_gid_count[2] + unique_gid_count[3]])
        else:
            Ecm = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            Enrs = unique_gid_count
        Eers = np.dot(np.dot(np.transpose(Ecm), Ters), Ecm)
        x = q * (q + 1) / (2 * E)
        hx = ((1 + x) * np.log(1 + x) - x * np.log(x))
        Lt = E * hx + N * np.log(q)
        # It = 0
        St = E
        unique_partition = list(range(q))
        for r in unique_partition:
            for s in unique_partition:
                n_r = Enrs[r]
                n_s = Enrs[s]
                ers = Eers[r, s]
                St -= 1 / 2 * ers * np.log(ers / (n_r * n_s))
                # mrs = ers / (2 * E)
                # wr = n_r / N
                # ws = n_s / N
                # It += mrs * np.log(mrs / (wr * ws)) if mrs != 0 else 0
        Epsilon = Lt + St
        # Epsilonb = Lt - E * It
        results += f'{Epsilon} '
        Epsilons.append(Epsilon)
    q_minMDL = np.argmin(Epsilons) + 1
    results += f'{q_minMDL}\n'
    print(f'rho={rho}, delta={delta}, DL calculation done time:{time.time() - start}')
    return save_path, results


def exp0():
    times = 1
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    rhos = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([0, 1]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    deltas = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, int((max_delta - min_delta) / 0.0005) + 1), 5),
                          np.array([0]))
    save_path = "./result/expectedMDL/eMDL.2.1.txt"
    run_exp(rhos, deltas, times, save_path, q, n, d, Z_s, Z_b, multiprocessing=False)


if __name__ == '__main__':
    exp0()
