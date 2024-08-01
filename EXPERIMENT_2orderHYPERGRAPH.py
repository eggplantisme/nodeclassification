import numpy as np
import os
from scipy.sparse import csr_array
import time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import Pool
from _HyperSBM import HyperSBM
from _CommunityDetect import CommunityDetect, adjusted_mutual_info_score
from _FigureJiazeHelper import get_confusionmatrix


def CDwithBH(A, hsbm, operator='A'):
    start = time.time()
    cd = CommunityDetect(A)
    A_BHpartition, A_BHnumgroups = cd.BetheHessian()
    true_numberpartition = len(hsbm.sizes)
    node_partition = A_BHpartition[:hsbm.n]
    node_numberpartition = np.size(np.unique(node_partition))
    A_cm, _ = get_confusionmatrix(hsbm.groupId, node_partition, true_numberpartition, node_numberpartition)
    A_ami = adjusted_mutual_info_score(hsbm.groupId, node_partition)
    print(f"BH result in {operator}: {A_ami}. Time={time.time() - start}. Confusion Matrix({np.shape(A_cm)}) is: \n{A_cm}")
    return A_ami, node_numberpartition


def exp_subprocess(n=3000, q=3, d=15, snr=1, times=1, save_path=None, assortative=True):
    sizes = [int(n / q)] * q
    ps_dict = dict({2: None})  # only have 2-order edges
    # SNRs = np.concatenate((np.linspace(0.1, 1, 10), np.linspace(2, 10, 9)), axis=None)
    # A_BHamis = []
    # A_BHnumbers = []
    # BA_BHamis = []
    # BA_BHnumbers = []
    results = ""
    for t in range(times):
        start = time.time()
        if assortative:
            # Consider assortative case
            pout = (d - np.sqrt(snr * d)) / n
            pin = pout + q * np.sqrt(snr * d) / n
        else:
            # Consider disassortative case
            pin = (d - (q-1) * np.sqrt(snr * d)) / n
            pout = pin + q * np.sqrt(snr * d) / n
        ps_dict[2] = (pin - pout) * np.identity(q) + pout * np.ones((q, q))
        hsbm = HyperSBM(sizes, ps_dict)
        print(f'SNR={snr} times={t} start. pin={pin}, pout={pout}, hsbm construct time={time.time()-start}')
        # Construct adjacent matrix A
        A = hsbm.getA_2order_edges()
        # BH on A
        result = CDwithBH(A, hsbm, operator="A")
        # A_BHamis.append(result[0])
        # A_BHnumbers.append(result[1])
        results += f'{snr} {t} {result[0]} {result[1]} '
        # BH on bipartite_A (R^{(n+e) * (n+e)})
        result = CDwithBH(hsbm.bipartite_A, hsbm, operator="BipartiteA")
        # BA_BHamis.append(result[0])
        # BA_BHnumbers.append(result[1])
        results += f'{result[0]} {result[1]}\n'
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


def run_exp(snrs, times, save_path=None, q=3, n=3000, d=15, assortative=True, multiprocessing=True):
    snr_done = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                snr_done.add(round(float(row[0]), 5))
    if multiprocessing:
        p = Pool(2)
        for snr in snrs:
            if round(snr, 5) in snr_done:
                print(f'snr={snr} has been run!')
                continue
            p.apply_async(exp_subprocess, args=(n, q, d, round(snr, 5), times, save_path, assortative, ),
                          callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for snr in snrs:
            if round(snr, 5) in snr_done:
                print(f'snr={snr} has been run!')
                continue
            savepath, results = exp_subprocess(n, q, d, round(snr, 5), times, save_path, assortative)
            write_results((savepath, results))


def read_exp(load_path, add_paths=None, num_result=4):
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
                print(f"Some parameter snr={snr} didn't run!")
            mean_ami = np.mean(ami_results, 0)[2:]
            for nr in range(num_result):
                Results[nr][i] = mean_ami[nr]
            i += 1
    return snrs, Results


def exp0():
    n = 1000
    q = 2
    d = 10
    times = 10
    SNRs = np.concatenate((np.linspace(0.1, 1, 10), np.linspace(2, 10, 9)), axis=None)
    assortative = False
    multiprocessing = True
    fileId = 'amiExp24.5.5' + f'_n={n}_q={q}_d={round(d)}_{"assortative" if assortative else "disassortative"}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(SNRs) * times}")
    run_exp(SNRs, times, save_path, q, n, d, assortative, multiprocessing)


if __name__ == '__main__':
    exp0()
