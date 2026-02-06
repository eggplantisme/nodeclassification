import numpy as np
from _SBMMatrix import BipartiteSBM
from _CommunityDetect import CommunityDetect
from _FigureJiazeHelper import get_confusionmatrix
from EXPERIMENT_HYPER_EMPIRICAL import EmpiricalHyperGraph
from _HyperCommunityDetection import HyperCommunityDetect
from sklearn.metrics.cluster import adjusted_mutual_info_score
from scipy.sparse import diags
import time
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import Pool
import sys


def symBSBM(n1=100, q1=2, delta=0.02, d=1, verbose=True):
    # Construct BSBM with symmetric B
    n2 = n1
    q2 = q1
    sizes = [int(n1 / q1)] * q1 + [int(n2 / q2)] * q2
    a = d/n1-delta/q1+delta
    b = d/n1-delta/q1
    if a < 0 or a > 1 or b < 0 or b > 1:
        return None, None, -1, -1, -1, -1, -1, -1
    print(f'a={a}, b={b}') if verbose else None
    H = (a - b) * np.identity(q1) + b * np.ones((q1, q2))
    bsbm = BipartiteSBM(q1, q2, sizes, H)
    # Get SNR
    H_svalues = bsbm.getSingulars()
    print(H_svalues) if verbose else None
    full_snr = n1 / q1 * H_svalues[0]
    sub1_snr = n1 / q1* H_svalues[1]**2 / H_svalues[0]
    print(f'SNR of full bipartite network is {full_snr}, the first partite is {sub1_snr}') if verbose else None
    # CD
    try:
        full_partition, full_numgroups = CommunityDetect(bsbm.A).BetheHessian(num_groups=None)
    except Exception as e:
        print(e)
        return bsbm, None, -1, -1, full_snr, -1, -1, sub1_snr
    full_ami = adjusted_mutual_info_score(bsbm.groupId, full_partition)
    sub1_numgroups = np.size(np.unique(full_partition[:n1]))
    sub1_ami = adjusted_mutual_info_score(bsbm.groupId[:n1], full_partition[:n1])
    print(f'Detect {full_numgroups} communities, the AMI of full is {full_ami}, of sub1 is {sub1_ami}') if verbose else None
    return bsbm, full_partition, full_numgroups, full_ami, full_snr, sub1_numgroups, sub1_ami, sub1_snr

def one_mode_projection_cd(bsbm, n1, given_q, only_assortative=False, proj_way="BBT"):
    start = time.time()
    if bsbm is None:
        print("BSBM is None, which means parameter is wrong.")
        return -1, -1
    AA = bsbm.A.dot(bsbm.A)
    if proj_way == "BBT":
        pass
    elif proj_way == "BBT-diag":
        AA = AA - diags(np.diag(AA.toarray()), 0)  # remove diagonal
    BBT = AA[:n1, :n1]
    try:
        BBT_BHpartition, BBT_BHnumgroups = CommunityDetect(BBT).BetheHessian(num_groups=given_q, weighted=True, only_assortative=only_assortative)
    except Exception as e:
        print(f"Error in Community Detection: {e}")
        return None, None
    BBT_BH_ami = adjusted_mutual_info_score(bsbm.groupId[:n1], BBT_BHpartition)
    BBT_BH_numgroups = BBT_BHnumgroups
    return BBT_BH_ami, BBT_BH_numgroups

def HyperBH_cd(bsbm, n1):
    start = time.time()
    if bsbm is None:
        print("BSBM is None, which means parameter is wrong.")
        return -1, -1
    H = bsbm.A[:n1, n1:]
    hsbm = EmpiricalHyperGraph(name="temp")
    hsbm.H = H
    hsbm.Ks = np.unique(H.sum(axis=0).flatten())
    hsbm.n = n1
    hsbm.e = H.shape[1]
    try:
        Hyper_BHpartition, Hyper_BHnumgroups = HyperCommunityDetect().BetheHessian(hsbm)
    except Exception as e:
        print(f"Error in Community Detection: {e}")
        return None, None
    Hyper_BH_ami = adjusted_mutual_info_score(bsbm.groupId[:n1], Hyper_BHpartition)
    Hyper_BH_numgroups = Hyper_BHnumgroups
    return Hyper_BH_ami, Hyper_BH_numgroups

def exp_subprocess(n1, q1, d, delta, times, save_path, proj_way="BBT", HyperBH=False):
    results = ""
    for t in range(times):
        start = time.time()
        print(f"EXP pid={os.getpid()} begin... d={d}, delta={delta}, times={t}")
        bsbm, full_partition, full_numgroups, full_ami, full_snr, sub1_numgroups, sub1_ami, sub1_snr = \
            symBSBM(n1=n1, q1=q1, delta=delta, d=d, verbose=True)
        if bsbm is not None and full_partition is None:
            print(f"Some error in CD, skip this run.")
            continue
        if HyperBH:
            Hyper_BH_ami, Hyper_BH_numgroups = HyperBH_cd(bsbm, n1)
        else:
            BBT_BH_ami, BBT_BH_numgroups = one_mode_projection_cd(bsbm, n1, given_q=q1, only_assortative=False, proj_way=proj_way)
        if HyperBH:
            results += f'{d} {delta} {t} {full_numgroups} {full_ami} {full_snr} '
            results += f'{sub1_numgroups} {sub1_ami} {sub1_snr}'
            results += f' {Hyper_BH_numgroups} {Hyper_BH_ami}'
        else:
            results += f'{d} {delta} {t} {full_numgroups} {full_ami} {full_snr} '
            results += f'{sub1_numgroups} {sub1_ami} {sub1_snr}'
            results += f' {BBT_BH_numgroups} {BBT_BH_ami}'
        results += "\n"
        print(f"EXP pid={os.getpid()} end. d={d}, delta={delta}, times={t}, Time:{np.around(time.time() - start, 3)}")
    return save_path, results


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


def run_exp(ds, deltas, times, save_path=None, n1=100, q1=2, proj_way="BBT", HyperBH=False, multiprocessing=True):
    d_delta_pair = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                d_delta_pair.add((round(float(row[0]), 5), round(float(row[1]), 5)))
    if multiprocessing:
        p = Pool(4)
        for d in ds:
            for delta in deltas:
                if (round(d, 5), round(delta, 5)) in d_delta_pair:
                    print(f'rho={d}, delta={delta} has been run!')
                    continue
                p.apply_async(exp_subprocess, args=(n1, q1, d, delta, times, save_path, proj_way, HyperBH),
                              callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for d in ds:
            for delta in deltas:
                if (round(d, 5), round(delta, 5)) in d_delta_pair:
                    print(f'rho={d}, delta={delta} has been run!')
                    continue
                savepath, results = exp_subprocess(n1, q1, d, delta, times, save_path, proj_way, HyperBH)
                write_results((savepath, results))


def read_exp(load_path, add_paths=None, only_one_mode=False, exclude_d=[]):
    """
    read the results file from run_exp
    :param load_path:
    :return:
    """
    with open(load_path, 'r') as f:
        results_list = [row.strip().split() for row in f.readlines()]
        if add_paths is not None:
            for add_path in add_paths:
                print("Additional result adding...")
                with open(add_path, 'r') as add_f:
                    results_list = results_list + [row.strip().split() for row in add_f.readlines()]
        # for i in range(len(results_list)):
        #     if len(results_list[i]) < 9:
        #         results_list[i] += ["0"] * (9 - len(results_list[i]))
        if len(exclude_d) > 0:
            results_list = [row for row in results_list if float(row[0]) not in exclude_d]
        if only_one_mode:
            # print(results_list[0])
            results_list = [row[:3] + row[9:] for row in results_list if len(row) > 9]
            # print(results_list[0])
            results = np.round(np.float64(results_list), decimals=5)
            # print(results[0, :])
        else:
            results = np.round(np.float64(results_list), decimals=5)
        ds = np.unique(results[:, 0])
        deltas = np.unique(results[:, 1])
        Results = []
        i = 0
        for d in ds:
            for delta in deltas:
                _results = results[np.squeeze(np.argwhere(np.logical_and(results[:, 0]==d, results[:, 1]==delta)))]
                if np.size(_results) == 0:
                    print(f"Some parameter d={d}, delta={delta} didn't run!")
                mean_results = np.mean(_results, 0)[3:] if len(np.shape(_results)) == 2 else _results[3:]
                Results.append(mean_results)
                i += 1
        plot_ds = np.repeat(ds, np.size(deltas))
        plot_deltas = np.tile(deltas, np.size(ds))
    # return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group
    return plot_ds, plot_deltas, Results


def exp0():
    n1 = 100
    q1 = 2
    times = 15 # 5
    max_d = 5
    deltas = np.linspace(0, min((q1 - q1 / n1 * max_d) / (q1 - 1), q1 / n1 * max_d), 41)
    ds = np.hstack((np.linspace(0.5, 1, 21), np.linspace(1.2, max_d, 20)))
    tag = "symmetricB_acrossD=1"
    # fileID = 'amiExp25.2.1' + f'_n1={n1}_q1={q1}_{tag}'
    fileID = 'amiExp25.8.15' + f'_n1={n1}_q1={q1}_{tag}'
    save_path = "./result/detectabilityBipartite/" + fileID + ".txt"
    run_exp(ds, deltas, times, save_path=save_path, n1=n1, q1=q1, multiprocessing=False)


def exp1():
    n1 = 2000
    q1 = 2
    times = 15
    max_d = 10
    deltas = np.linspace(0, min((q1 - q1 / n1 * max_d) / (q1 - 1), q1 / n1 * max_d), 58)
    ds = np.linspace(5, max_d, 58)
    tag = "symmetricB_largeN1_15more"
    fileID = 'amiExp25.8.16' + f'_n1={n1}_q1={q1}_{tag}'
    save_path = "./result/detectabilityBipartite/" + fileID + ".txt"
    run_exp(ds, deltas, times, save_path=save_path, n1=n1, q1=q1, multiprocessing=False)

def exp2():
    n1 = 2000
    q1 = 2
    times = 10
    max_d = 10
    deltas = np.linspace(0, min((q1 - q1 / n1 * max_d) / (q1 - 1), q1 / n1 * max_d), 58)
    ds = np.linspace(5, max_d, 58)
    proj_way = "BBT-diag"
    tag = f"symmetricB_largeN1_incluOneMode_{proj_way}"
    fileID = 'amiExp25.8.18' + f'_n1={n1}_q1={q1}_{tag}'
    save_path = "./result/detectabilityBipartite/" + fileID + ".txt"
    run_exp(ds, deltas, times, save_path=save_path, n1=n1, q1=q1, proj_way=proj_way, multiprocessing=True)

def exp3():
    n1 = 2000
    q1 = 2
    times = 10
    max_d = 10
    deltas = np.linspace(0, min((q1 - q1 / n1 * max_d) / (q1 - 1), q1 / n1 * max_d), 58)
    ds = np.linspace(5, max_d, 58)
    HyperBH = True
    tag = f"symmetricB_largeN1_incluHyperBH"
    fileID = 'amiExp25.8.18' + f'_n1={n1}_q1={q1}_{tag}'
    save_path = "./result/detectabilityBipartite/" + fileID + ".txt"
    run_exp(ds, deltas, times, save_path=save_path, n1=n1, q1=q1, HyperBH=HyperBH, multiprocessing=True)


if __name__ == '__main__':
    # exp0()
    # exp1()
    # exp2()
    exp3()
