import os
import numpy as np
from multiprocessing import Pool
from sklearn.metrics.cluster import adjusted_mutual_info_score
import time
from datetime import date
from _SBMMatrix import SymmetricSBM
from _CommunityDetect import CommunityDetect


def exp_subprocess(parameters, snr, times, save_path):
    """
    :param parameters:
    :param snr:
    :param times:
    :return:
    """
    n = parameters.get('n', 3000)
    q = parameters.get('q', 3)
    d = parameters.get('d', 10)
    method = parameters.get('method', 'BH')
    pout = (d - np.sqrt(snr * d)) / n
    pin = (d + (q-1) * np.sqrt(snr * d)) / n
    results = ""
    for t in range(times):
        start = time.time()
        print(f"EXP pid={os.getpid()} begin... snr={snr}, times={t}")
        sbm = SymmetricSBM(n=n, k=q, pin=pin, pout=pout)
        if method == 'BH':
            partition, num_groups = CommunityDetect(sbm.A).BetheHessian()
        elif method == 'Leiden':
            partition, num_groups = CommunityDetect(sbm.A).leiden()
        elif method == 'Louvain':
            partition, num_groups = CommunityDetect(sbm.A).louvain()
        else:
            partition, num_groups = None, None
        ami = adjusted_mutual_info_score(sbm.groupId, partition)
        results += f'{snr} {t} {ami} {num_groups}\n'
        print(f"EXP pid={os.getpid()} end. snr={snr}, ami={ami}, Time:{np.around(time.time() - start, 3)}")
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


def run_exp(snrs, times, parameters={}, save_path=None, multiprocessing=True):
    snr_done = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                snr_done.add(round(float(row[0]), 5))
    if multiprocessing:
        p = Pool(4)
        for snr in snrs:
            if round(snr, 5) in snr_done:
                print(f'snr={snr} has been run!')
                continue
            p.apply_async(exp_subprocess, args=(parameters, snr, times, save_path, ),
                          callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for snr in snrs:
            if round(snr, 5) in snr_done:
                print(f'snr={snr} has been run!')
                continue
            savepath, results = exp_subprocess(parameters, snr, times, save_path)
            write_results((savepath, results))


def read_exp(load_path, add_paths=None, num_result=2):
    """
    read the results file from run_exp
    :param load_path:
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
        snrs = np.unique(results[:, 0])
        Results = []
        for i in range(num_result):
            Results.append(np.zeros(np.size(snrs)))
        i = 0
        for snr in snrs:
            ami_results = results[np.squeeze(np.argwhere(results[:, 0] == snr))]
            if np.size(ami_results) == 0:
                print(f"Some parameter epsilon={snr} didn't run!")
            mean_ami = np.mean(ami_results, 0)[2:]
            for nr in range(num_result):
                Results[nr][i] = mean_ami[nr]
            i += 1
    return snrs, Results


def main0():
    snrs = np.arange(0.1, 6.1, 0.1)
    times = 10
    parameters = {
        'n': 3000,
        'q': 3,
        'd': 10,
        'method': 'Louvain'  # BH, Leiden, Louvain
    }
    timestramp = date.today().strftime("%y.%m.%d")
    save_path = f"./result/detectability/EXP{timestramp}_detectability_{parameters['method']}.txt"
    run_exp(snrs, times, parameters, save_path, multiprocessing=True)


def main1():
    snrs = np.arange(0.1, 6.1, 0.1)
    times = 90
    parameters = {
        'n': 3000,
        'q': 3,
        'd': 10,
        'method': 'Louvain'  # BH, Leiden, Louvain
    }
    timestramp = date.today().strftime("%y.%m.%d")
    save_path = f"./result/detectability/EXP{timestramp}_detectability_{parameters['method']}_{times}more.txt"
    run_exp(snrs, times, parameters, save_path, multiprocessing=True)


def main2():
    snrs = np.arange(0.1, 6.1, 0.1)
    times = 90
    parameters = {
        'n': 3000,
        'q': 3,
        'd': 10,
        'method': 'BH'  # BH, Leiden, Louvain
    }
    timestramp = date.today().strftime("%y.%m.%d")
    save_path = f"./result/detectability/EXP{timestramp}_detectability_{parameters['method']}_{times}more.txt"
    run_exp(snrs, times, parameters, save_path, multiprocessing=True)


if __name__ == "__main__":
    # main0()
    main1()
    main2()
