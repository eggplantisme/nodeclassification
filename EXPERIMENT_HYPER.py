import time
import os
import numpy as np
from scipy.special import comb
from sklearn.metrics.cluster import adjusted_mutual_info_score
from _HyperSBM import *
from _HyperCommunityDetection import *
from _FigureJiazeHelper import get_confusionmatrix
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import Pool


def CDwithBH(hsbm, bipartite=False):
    start = time.time()
    if bipartite is False:
        BH_Partition, BH_NumGroup = HyperCommunityDetect().BetheHessian(hsbm)
    else:
        BH_Partition, BH_NumGroup = HyperCommunityDetect().BipartiteBH(hsbm, num_groups=hsbm.q)
    cd_time = time.time() - start
    cm, _ = get_confusionmatrix(hsbm.groupId, BH_Partition, hsbm.q, BH_NumGroup)
    ami = adjusted_mutual_info_score(hsbm.groupId, BH_Partition)
    print(f"BH result AMI: {ami}. Time={cd_time}. Confusion Matrix({np.shape(cm)}) is: \n{cm}")
    return ami, BH_NumGroup, cd_time


def exp_subprocess(n=3000, q=3, d=15, Ks=(2, ), epsilon=1, times=1, save_path=None, bipartite=False):
    sizes = [int(n / q)] * q
    ps_dict = dict()
    temp = 0
    for k in Ks:
        temp += q * comb(int(n/q), k) * k / (n**k) + epsilon * (comb(n, k) - q * comb(int(n/q), k)) * k / (n**k)
    cin = d / temp
    cout = epsilon * cin
    results = ""
    for t in range(times):
        start = time.time()
        if len(Ks) > 1:
            hsbm = UnUniformSymmetricHSBM(n, q, Ks, cin, cout)
        elif len(Ks) == 1:
            hsbm = UniformSymmetricHSBM(n, q, Ks[0], cin, cout)
        print(f'epsilon={epsilon} times={t} start. cin={cin}, cout={cout}, hsbm construct time={time.time()-start}')
        # Community Detection
        result = CDwithBH(hsbm, bipartite)
        results += f'{epsilon} {t} {result[0]} {result[1]} {result[2]}\n'
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


def run_exp(epsilons, times, save_path=None, n=3000, q=3, d=15, Ks=(2, ), multiprocessing=True, bipartite=False):
    epsilon_done = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                epsilon_done.add(round(float(row[0]), 5))
    if multiprocessing:
        p = Pool(4)
        for epsilon in epsilons:
            if round(epsilon, 5) in epsilon_done:
                print(f'snr={epsilon} has been run!')
                continue
            p.apply_async(exp_subprocess, args=(n, q, d, Ks, epsilon, times, save_path, bipartite, ),
                          callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for epsilon in epsilons:
            if round(epsilon, 5) in epsilon_done:
                print(f'snr={epsilon} has been run!')
                continue
            savepath, results = exp_subprocess(n, q, d, Ks, epsilon, times, save_path, bipartite)
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
        epsilons = np.unique(results[:, 0])
        Results = []
        for i in range(num_result):
            Results.append(np.zeros(np.size(epsilons)))
        i = 0
        for epsilon in epsilons:
            ami_results = results[np.squeeze(np.argwhere(results[:, 0] == epsilon))]
            if np.size(ami_results) == 0:
                print(f"Some parameter epsilon={epsilon} didn't run!")
            mean_ami = np.mean(ami_results, 0)[2:]
            for nr in range(num_result):
                Results[nr][i] = mean_ami[nr]
            i += 1
    return epsilons, Results


def exp0():
    n = 100
    q = 2
    d = 15
    times = 5
    epsilons = np.concatenate((np.linspace(0.1, 1, 10), np.linspace(2, 10, 9)), axis=None)
    Ks = (2, 3)
    multiprocessing = False
    fileId = 'amiExpHyper24.5.23' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp1():
    n = 100
    q = 2
    d = 15
    times = 5
    epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    Ks = (3, )
    multiprocessing = True
    addtionTag = "_uniform"
    fileId = 'amiExpHyper24.5.24' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp2():
    n = 100
    q = 2
    d = 15
    times = 40
    epsilons = np.linspace(0.4, 0.8, 40)
    Ks = (2, 3)
    multiprocessing = True
    addStrId = f'_40more0.4~0.8'
    fileId = 'amiExpHyper24.5.30' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH' + addStrId
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp3():
    n = 100
    q = 2
    d = 15
    times = 40
    epsilons = np.linspace(0.1, 1, 20)
    Ks = (2, 3)
    bipartite = True
    multiprocessing = True
    addStrId = f'_0.4~0.8_40more'
    fileId = 'amiExpHyper24.5.31' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{"_bi" if bipartite else ""}' + addStrId
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bipartite)


def exp4():
    """ Exp q>2 """
    n = 150
    q = 3
    d = 15
    times = 40
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    bipartite = False
    multiprocessing = True
    # addStrId = f'_higher_q'
    addStrId = f'_40more'
    # fileId = 'amiExpHyper24.6.5' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{"_bi" if bipartite else ""}' + addStrId
    fileId = 'amiExpHyper24.6.30' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{"_bi" if bipartite else ""}' + addStrId
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing, bipartite)


def exp5():
    n = 100
    q = 2
    d = 10
    times = 40
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (3, )
    multiprocessing = True
    # addtionTag = ""
    addtionTag = "_40more"
    fileId = 'amiExpHyper24.6.29' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def exp6():
    n = 100
    q = 2
    d = 10
    times = 50
    # epsilons = np.concatenate((np.linspace(0.1, 1, 51), np.linspace(1.4, 10, 21)), axis=None)
    epsilons = np.linspace(0.1, 1, 46)
    Ks = (2, 3)
    multiprocessing = True
    addtionTag = ""
    fileId = 'amiExpHyper24.6.30' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH{addtionTag}'
    save_path = "./result/detectabilityHyper/" + fileId + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileId} size={np.size(epsilons) * times}")
    run_exp(epsilons, times, save_path, n, q, d, Ks, multiprocessing)


def debug():
    n = 100
    q = 2
    d = 15
    Ks = (2, 3)
    fileId = 'amiExpHyper24.5.23' + f'_n={n}_q={q}_d={round(d)}_Ks={Ks}BH'
    load_path = "./result/detectabilityHyper/" + fileId + ".txt"
    epsilons, results = read_exp(load_path=load_path)


if __name__ == '__main__':
    # exp0()
    # debug()
    # exp1()
    # exp2()
    # exp3()
    exp4()
    # exp5()
    # exp6()
