import time
import numpy as np
import os
from scipy.special import comb
from scipy.sparse import diags
from _HyperCommunityDetection import HyperCommunityDetect
from _FigureJiazeHelper import get_confusionmatrix
from _HyperSBM import *
from sklearn.metrics.cluster import adjusted_mutual_info_score
from multiprocessing import Pool


def BHCD(hsbm):
    start = time.time()
    BH_Partition, BH_NumGroup = HyperCommunityDetect().BetheHessian(hsbm)
    cd_time = time.time() - start
    ami = adjusted_mutual_info_score(hsbm.groupId, BH_Partition)
    print(f"BH result AMI: {ami}. Time={cd_time}.")
    # cm, _ = get_confusionmatrix(hsbm.groupId, BH_Partition, hsbm.q, BH_NumGroup)
    # print(f"Confusion Matrix({np.shape(cm)}) is: \n{cm}")
    return ami, BH_NumGroup, cd_time


def exp_subprocess(rho, epsilon, fix_parameter, save_path=None):
    n = fix_parameter["n"]
    q_s = fix_parameter["q_s"]
    q_b = fix_parameter["q_b"]
    q = q_s + q_b
    d = fix_parameter["d"]
    Ks = fix_parameter["Ks"]
    times = fix_parameter["times"]

    n_s = int(n * rho / q_s)  # size of minority community
    n_b = int(n * (1 - rho) / q_b)  # size of majority community
    sizes = [n_s] * q_s + [n_b] * q_b
    ps_dict = dict()
    temp = 0
    for k in Ks:
        temp += q_s * comb(n_s, k) * k / (n**k) + q_b * comb(n_b, k) * k / (n**k) + epsilon * (comb(n, k) - q_s * comb(n_s, k) - q_b * comb(n_b, k)) * k / (n**k)
    cin = d / temp
    cout = epsilon * cin
    results = ""
    for t in range(times):
        start = time.time()
        if len(Ks) > 1:
            hsbm = UnUniformSymmetricHSBM(n, q, Ks, cin, cout)
        elif len(Ks) == 1:
            hsbm = UniformSymmetricHSBM(n, q, Ks[0], cin, cout)
        print(f'rho={rho}, epsilon={epsilon}, times={t} start. cin={cin}, cout={cout}, hsbm construct time={time.time()-start}')
        if cin < 0 or cout < 0 or cin > n**min(Ks) or cout > n**min(Ks):
            print(f'cin={cin}, cout={cout} is invalid!')
            results += f'{rho} {epsilon} {t} -1 -1 -1\n'
            continue
        # Community Detection
        result = BHCD(hsbm)
        results += f'{rho} {epsilon} {t} {result[0]} {result[1]} {result[2]}\n'
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


def run_exp(variable_parameter, fix_parameter, save_path=None, multiprocessing=True):
    variable_parameter_done = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                pair = tuple([row[0] for i in range(len(variable_parameter))])
                variable_parameter_done.add(pair)
    if multiprocessing:
        p = Pool(4)
        for rho in variable_parameter["rhos"]:
            for epsilon in variable_parameter["epsilons"]:
                pair = (rho, epsilon)
                if pair in variable_parameter_done:
                    print(f'rho={rho}, epsilon={epsilon} has been run!')
                    continue
                p.apply_async(exp_subprocess, args=(rho, epsilon, fix_parameter, save_path),
                          callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for rho in variable_parameter["rhos"]:
            for epsilon in variable_parameter["epsilons"]:
                pair = (rho, epsilon)
                if pair in variable_parameter_done:
                    print(f'rho={rho}, epsilon={epsilon} has been run!')
                    continue
                savepath, results = exp_subprocess(rho, epsilon, fix_parameter, save_path)
                write_results((savepath, results))


def read_exp(load_path, add_paths=None):
    """
    read the results file from run_exp
    :param load_path:
    :param Withsnr:
    :return:
    """
    with open(load_path, 'r') as f:
        results_list = [row.strip().split() for row in f.readlines()]
        if add_paths is not None:
            print("Additional result adding...")
            for add_path in add_paths:
                with open(add_path, 'r') as add_f:
                    results_list = results_list + [row.strip().split() for row in add_f.readlines()]
        results = np.round(np.float64(results_list), decimals=5)
        rhos = np.unique(results[:, 0])
        zs = np.unique(results[:, 1])
        ami = np.zeros(np.size(zs) * np.size(rhos))
        num_group = np.zeros(np.size(zs) * np.size(rhos))
        i = 0
        for _rho in rhos:
            for _z in zs:
                ami_results = results[np.squeeze(np.argwhere(np.logical_and(results[:, 0]==_rho, results[:, 1]==_z)))]
                if np.size(ami_results) == 0:
                    print(f"Some parameter rho={_rho}, z={_z} didn't run!")
                mean_ami = np.mean(ami_results, 0)[3:]
                ami[i] = mean_ami[0]
                num_group[i] = mean_ami[1]
                i += 1
        # print(ami)
        # print(num_group)
        plot_rhos = np.repeat(rhos, np.size(zs))
        plot_zs = np.tile(zs, np.size(rhos))
    # return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group
    return plot_rhos, plot_zs, ami, num_group


def exp0():
    fix_parameter = {
        "n": 5000,
        "q_s": 2,
        "q_b": 2,
        "q": 4,
        "d": 10,
        "times": 10,
        "Ks": (2, 3),
    }
    variable_parameter = {
        "epsilons": np.around(np.linspace(0.1, 0.6, 26), 5),  # cout/cin ratio
        "rhos": np.around(np.linspace(0.01, fix_parameter["q_s"]/fix_parameter["q"], 26), 5),  # the proportion of minority communities
    }
    multiprocessing = False
    addtionTag = ""
    fileId = 'amiExpMinorityHyperBH25.12.26' + f'_n={fix_parameter["n"]}_q={fix_parameter["q"]}_d={round(fix_parameter["d"])}_Ks={fix_parameter["Ks"]}_{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId}, rows={np.size(variable_parameter['epsilons']) * np.size(variable_parameter['rhos']) * fix_parameter['times']}")
    run_exp(variable_parameter, fix_parameter, save_path, multiprocessing)


if __name__ == "__main__":
    exp0()