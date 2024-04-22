import numpy as np
from _SBMMatrix import BipartiteSBM
from _CommunityDetect import CommunityDetect
from _FigureJiazeHelper import get_confusionmatrix
from sklearn.metrics.cluster import adjusted_mutual_info_score
from scipy.sparse import diags
import time
import os
from multiprocessing import Pool
import sys


def range_delta(n1, k1, d):
    min_delta = max(- k1 / (k1 - 1) * d / n1, k1 * (d / n1 - 1))
    max_delta = min(k1 / (k1 - 1) * (1 - d / n1), k1 * d / n1)
    return min_delta, max_delta


def symmetric_bipartite(n1, k1, pBd, pBo):
    k2 = k1
    sizes = [int(n1 / k1)] * k1 + [int(n1 / k1)] * k1
    H = (pBd - pBo) * np.identity(k1) + pBo * np.ones((k1, k2))
    return BipartiteSBM(k1, k2, sizes, H)


def get_SNR_bbipartite(n1, k1, d, delta):
    SNR_A = d
    # SNR_AA = (n1**3) / (k1**4) * (delta**4) / (d**2)
    # SNR_AA = (k1 / n1) * (d**2)
    SNR_AA = -1  # Unknown
    SNR_H = (n1**2) / (k1**2) * (delta**2) / d
    SNR_HH = SNR_AA
    return SNR_A, SNR_AA, SNR_H, SNR_HH


def synthetic_exp_sym_bipartite(bsbm):
    k = bsbm.k1 + bsbm.k2
    n1 = np.sum(bsbm.sizes[:bsbm.k1])
    result_data = []
    A = bsbm.A
    A_BHpartition, A_BHnumgroups = CommunityDetect(A).BetheHessian(num_groups=k)
    result_data.append(adjusted_mutual_info_score(bsbm.groupId, A_BHpartition))
    result_data.append(A_BHnumgroups)
    result_data.append(adjusted_mutual_info_score(bsbm.groupId[:n1], A_BHpartition[:n1]))
    result_data.append(np.size(np.unique(A_BHpartition[:n1])))

    AA = bsbm.A.dot(bsbm.A)
    AA = AA - diags(np.diag(AA.toarray()), 0)  # remove diagonal
    AA_WBHpartition, AA_WBHnumgroups = CommunityDetect(AA).BetheHessian(num_groups=k, weighted=True)
    result_data.append(adjusted_mutual_info_score(bsbm.groupId, AA_WBHpartition))
    result_data.append(AA_WBHnumgroups)
    result_data.append(adjusted_mutual_info_score(bsbm.groupId[:n1], AA_WBHpartition[:n1]))
    result_data.append(np.size(np.unique(AA_WBHpartition[:n1])))

    # x, y = np.nonzero(AA.toarray())
    # nonzeroAA = np.zeros(np.shape(AA))
    # for _x, _y in zip(x, y):
    #     nonzeroAA[_x, _y] = 1
    # NonZeroAA_BHpartition, NonZeroAA_BHnumgroups = CommunityDetect(nonzeroAA).BetheHessian()
    # result_data.append(adjusted_mutual_info_score(bsbm.groupId, NonZeroAA_BHpartition))
    # result_data.append(NonZeroAA_BHnumgroups)

    B = bsbm.A[:n1, n1:]  # TODO B is not undirected
    B_BHpartition, B_BHnumgroups = CommunityDetect(B).BetheHessian(num_groups=bsbm.k1)
    result_data.append(adjusted_mutual_info_score(bsbm.groupId[:n1], B_BHpartition))
    result_data.append(B_BHnumgroups)

    # BB = B.dot(B.T)  # The detect result of BB should be similar as AA (The distinct eigenvalue of PQ is same.)
    # BB_WBHpartition, BB_WBHnumgroups = CommunityDetect(BB).BetheHessian(weighted=True)
    # result_data.append(adjusted_mutual_info_score(bsbm.groupId[:n1], BB_WBHpartition))
    # result_data.append(BB_WBHnumgroups)
    # x, y = np.nonzero(BB.toarray())
    # nonzeroBB = np.zeros(np.shape(BB))
    # for _x, _y in zip(x, y):
    #     nonzeroBB[_x, _y] = 1
    # NonZeroBB_BHpartition, NonZeroBB_BHnumgroups = CommunityDetect(nonzeroBB).BetheHessian()
    # result_data.append(adjusted_mutual_info_score(bsbm.groupId[:n1], NonZeroBB_BHpartition))
    # result_data.append(NonZeroBB_BHnumgroups)

    BBT = AA[:n1, :n1]
    BBT_BHpartition, BBT_BHnumgroups = CommunityDetect(BBT).BetheHessian(num_groups=bsbm.k1, weighted=True)
    result_data.append(adjusted_mutual_info_score(bsbm.groupId[:n1], BBT_BHpartition))
    result_data.append(BBT_BHnumgroups)

    BTB = AA[n1:, n1:]
    BTB_BHpartition, BTB_BHnumgroups = CommunityDetect(BTB).BetheHessian(num_groups=bsbm.k1, weighted=True)
    result_data.append(adjusted_mutual_info_score(bsbm.groupId[:n1], BTB_BHpartition))
    result_data.append(BTB_BHnumgroups)
    return result_data


def exp_subprocess(n1, n2, k1, k2, d, delta, times, save_path, WithlambdaB=True):
    if n1 != n2 or k1 != k2:
        print("No consider asymmetric case...")
        sys.exit()
    else:
        min_delta, max_delta = range_delta(n1, k1, d)
        pBo = d / n1 - delta / k1
        pBd = pBo + delta
        print(f'--For case n1={n1}, k1={k1}, d={d}, Range of delta ({min_delta}, {max_delta}). Now delta={delta} '
              f'which make p_diagonal_B={pBd}, p_offdiagonal_B={pBo}')
        SNRs = get_SNR_bbipartite(n1, k1, d, delta)
        print(f'--In this case by theory, SNR_A=d={SNRs[0]}, SNR_AA={SNRs[1]}, SNR_H={SNRs[2]}, SNR_HH=SNR_AA')
        bsbm = symmetric_bipartite(n1, k1, pBd, pBo)
        results = ""
        if WithlambdaB:
            lambdas = bsbm.getSingulars()
            lambdas = np.array(lambdas.tolist() + [0] * (min(k1, k2) - np.size(lambdas)))  # Expand 0s
            print(f"--Singular value of B is {lambdas}")
            print(f"--Verified SNR_A={n1 / k1 * lambdas[0]} "
                  # f"SNR_AA={n1/k1 * lambdas[1]**4 / lambdas[0]**2} "
                  f"SNR_H={n1/k1*lambdas[1]**2 / lambdas[0]}")
        for t in range(times):
            start = time.time()
            print(f"EXP pid={os.getpid()} begin... d={d}, delta={delta}, times={t}")
            result_data = synthetic_exp_sym_bipartite(bsbm)
            results += f'{d} {delta} {t} {result_data[0]} {result_data[1]} '
            results += f'{result_data[2]} {result_data[3]} '
            results += f'{result_data[4]} {result_data[5]} '
            results += f'{result_data[6]} {result_data[7]} '
            results += f'{result_data[8]} {result_data[9]} '
            results += f'{result_data[10]} {result_data[11]}'
            results += f'{result_data[12]} {result_data[13]}'
            if WithlambdaB:
                for l in lambdas:
                    results += f' {l}'
            results += "\n"
            print(f"EXP pid={os.getpid()} end. AMI_# result {result_data}. Time:{np.around(time.time() - start, 3)}")
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


def run_exp(ds, deltas, times, save_path=None, n1=3000, n2=3000, k1=3, k2=3, WithlambdaB=True, multiprocessing=True):
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
                p.apply_async(exp_subprocess, args=(n1, n2, k1, k2, d, delta, times, save_path, WithlambdaB, ),
                              callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for d in ds:
            for delta in deltas:
                if (round(d, 5), round(delta, 5)) in d_delta_pair:
                    print(f'rho={d}, delta={delta} has been run!')
                    continue
                savepath, results = exp_subprocess(n1, n2, k1, k2, d, delta, times, save_path, WithlambdaB,)
                write_results((savepath, results))


def read_exp(load_path, WithlambdaB=True, exclude_rho=None, add_path=None, num_result=10):
    """
    read the results file from run_exp
    :param load_path:
    :return:
    """
    exclude_rho = [] if exclude_rho is None else exclude_rho
    with open(load_path, 'r') as f:
        results_list = [row.strip().split() for row in f.readlines()]
        if add_path is not None:
            print("Additional result adding...")
            with open(add_path, 'r') as add_f:
                results_list = results_list + [row.strip().split() for row in add_f.readlines()]
        # for i in range(len(results_list)):
        #     if len(results_list[i]) < 9:
        #         results_list[i] += ["0"] * (9 - len(results_list[i]))
        results = np.round(np.float64(results_list), decimals=5)
        ds = np.setdiff1d(np.unique(results[:, 0]), np.array(exclude_rho))
        deltas = np.unique(results[:, 1])
        AMIResults = []
        for i in range(num_result):
            AMIResults.append(np.zeros(np.size(deltas) * np.size(ds)))
        # A_ami = np.zeros(np.size(deltas) * np.size(ds))
        # Ahalf_ami = np.zeros(np.size(deltas) * np.size(ds))
        # AA_ami = np.zeros(np.size(deltas) * np.size(ds))
        # AAhalf_ami = np.zeros(np.size(deltas) * np.size(ds))
        # H_ami = np.zeros(np.size(deltas) * np.size(ds))
        lambdas = None
        # A_num_group = np.zeros(np.size(deltas) * np.size(ds))
        # Ahalf_num_group = np.zeros(np.size(deltas) * np.size(ds))
        # AA_num_group = np.zeros(np.size(deltas) * np.size(ds))
        # AAhalf_num_group = np.zeros(np.size(deltas) * np.size(ds))
        # H_num_group = np.zeros(np.size(deltas) * np.size(ds))
        i = 0
        for d in ds:
            for delta in deltas:
                ami_results = results[np.squeeze(np.argwhere(np.logical_and(results[:, 0]==d, results[:, 1]==delta)))]
                if np.size(ami_results) == 0:
                    print(f"Some parameter d={d}, delta={delta} didn't run!")
                mean_ami = np.mean(ami_results, 0)[3:] if len(np.shape(ami_results)) == 2 else ami_results[3:]
                for nr in range(num_result):
                    AMIResults[nr][i] = mean_ami[nr]
                # full_ami[i] = mean_ami[0]
                # A_ami[i] = mean_ami[0]
                # A_num_group[i] = mean_ami[1]
                # Ahalf_ami[i] = mean_ami[2]
                # Ahalf_num_group[i] = mean_ami[3]
                # AA_ami[i] = mean_ami[4]
                # AA_num_group[i] = mean_ami[5]
                # AAhalf_ami[i] = mean_ami[6]
                # AAhalf_num_group[i] = mean_ami[7]
                # H_ami[i] = mean_ami[8]
                # H_num_group[i] = mean_ami[9]
                if WithlambdaB:
                    if lambdas is None:
                        lambdas = np.zeros((np.size(deltas) * np.size(ds), np.size(mean_ami)-num_result))
                    lambdas[i] = mean_ami[num_result:]
                i += 1
        plot_ds = np.repeat(ds, np.size(deltas))
        plot_deltas = np.tile(deltas, np.size(ds))
    # return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group
    return plot_ds, plot_deltas, AMIResults, lambdas


def exp0():
    n1 = 3000
    k1 = 3
    d = 10
    min_delta, max_delta = range_delta(n1, k1, d)
    print(f'Range of delta ({min_delta}, {max_delta})')
    delta = 0.004
    pBo = d / n1 - delta / k1
    pBd = pBo + delta
    # SNR_AA = 1
    # Assortative case
    # pBo = d / n1 - np.sqrt(SNR_AA * d / n1)
    # pBd = pBo + np.sqrt(SNR_AA * d / n1) * k1
    # Disassortative case
    # pBo = d / n1 + np.sqrt(SNR_AA * d / n1)
    # pBd = pBo - np.sqrt(SNR_AA * d / n1) * k1
    SNR_AA = 1/k1 * (pBd-pBo)**4 / (pBd+(k1-1)*pBo)**2
    SNR_A = d
    print(f'pB of diagonal is {pBd}, off-diagonal is {pBo}, SNR_AA={SNR_AA}, SNR_A={SNR_A}')
    start = time.time()
    bsbm = symmetric_bipartite(n1, k1, pBd, pBo)
    print(f'bsbm SNR is {bsbm.get_SNR()}')
    cd = CommunityDetect(bsbm.A)
    BHpartition, BHnumgroups = cd.BetheHessian()
    Acm, _ = get_confusionmatrix(bsbm.groupId, BHpartition, k1 * 2, BHnumgroups)
    ami = adjusted_mutual_info_score(bsbm.groupId, BHpartition)
    print(f"BH result: {ami}. Time={time.time() - start}. Confusion Matrix({np.shape(Acm)}) is: \n{Acm}")

    AA = bsbm.A.dot(bsbm.A)
    # TODO Try weighted BH in AA
    cd = CommunityDetect(AA)
    WBHpartition, WBHnumgroups = cd.BetheHessian(weighted=True)
    AAcm, _ = get_confusionmatrix(bsbm.groupId, WBHpartition, k1 * 2, WBHnumgroups)
    ami = adjusted_mutual_info_score(bsbm.groupId, WBHpartition)
    print(f"WBH result: {ami}. Time={time.time() - start}. Confusion Matrix({np.shape(AAcm)}) is: \n{AAcm}")
    pass


def exp1():
    times = 1
    n1 = n2 = 3000
    k1 = k2 = 3
    ds = np.around(np.linspace(10, 20, 11), 2)
    # ds = [10, 1700]
    min_delta, max_delta = range_delta(n1, k1, d=min(ds))
    deltas = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 29), 5), np.array([]))
    print(deltas.tolist())

    WithlambdaB = True
    multiprocessing = True
    fileID = 'amiExp24.2.29' + f'_n1={n1}n2={n2}_k1={k1}k2={k2}_{"lambda" if WithlambdaB else ""}_givenNumgroups'
    save_path = "./result/detectabilityBipartite/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(ds) * np.size(deltas) * times}")
    run_exp(ds, deltas, times, save_path=save_path, n1=n1, n2=n2, k1=k1, k2=k2, WithlambdaB=WithlambdaB,
            multiprocessing=multiprocessing)


def exp2():
    times = 2
    n1 = n2 = 3000
    k1 = k2 = 3
    ds = np.around(np.linspace(10, 30, 21), 2)
    # ds = [10, 1700]
    min_delta, max_delta = range_delta(n1, k1, d=min(ds))
    deltas = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 29), 5), np.array([]))
    print(deltas.tolist())

    WithlambdaB = True
    multiprocessing = True
    fileID = 'amiExp24.2.30' + f'_n1={n1}n2={n2}_k1={k1}k2={k2}_{"lambda" if WithlambdaB else ""}_givenNumgroups'
    save_path = "./result/detectabilityBipartite/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(ds) * np.size(deltas) * times}")
    run_exp(ds, deltas, times, save_path=save_path, n1=n1, n2=n2, k1=k1, k2=k2, WithlambdaB=WithlambdaB,
            multiprocessing=multiprocessing)


def exp3():
    times = 2
    n1 = n2 = 3000
    k1 = k2 = 3
    ds = np.around(np.linspace(10, 30, 21), 2)
    # ds = [10, 1700]
    min_delta, max_delta = range_delta(n1, k1, d=min(ds))
    deltas = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 29), 5), np.array([]))
    print(deltas.tolist())

    WithlambdaB = True
    multiprocessing = True
    fileID = 'amiExp24.3.4' + f'_n1={n1}n2={n2}_k1={k1}k2={k2}_{"lambda" if WithlambdaB else ""}_givenNumgroups_HHT'
    save_path = "./result/detectabilityBipartite/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(ds) * np.size(deltas) * times}")
    run_exp(ds, deltas, times, save_path=save_path, n1=n1, n2=n2, k1=k1, k2=k2, WithlambdaB=WithlambdaB,
            multiprocessing=multiprocessing)


def exp4():
    times = 10
    n1 = n2 = 3000
    k1 = k2 = 3
    ds = np.around(np.linspace(10, 30, 21), 2)
    # ds = [10, 1700]
    min_delta, max_delta = range_delta(n1, k1, d=min(ds))
    deltas = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 29), 5), np.array([]))
    print(deltas.tolist())

    WithlambdaB = True
    multiprocessing = False
    fileID = 'amiExp24.3.6' + f'_n1={n1}n2={n2}_k1={k1}k2={k2}_{"lambda" if WithlambdaB else ""}_givenNumgroups_modifyWBH'
    save_path = "./result/detectabilityBipartite/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(ds) * np.size(deltas) * times}")
    run_exp(ds, deltas, times, save_path=save_path, n1=n1, n2=n2, k1=k1, k2=k2, WithlambdaB=WithlambdaB,
            multiprocessing=multiprocessing)


if __name__ == '__main__':
    # exp0()
    # exp1()
    # exp2()
    # exp3()
    exp4()
