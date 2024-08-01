import os
import numpy as np
import time
from EXPERIMENT_MinoritySameDegree import get_ps
from _CommunityDetect import CommunityDetect
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_array, triu
from EXPERIMENT_MINORITY import get_confusionmatrix
from sklearn.metrics.cluster import adjusted_mutual_info_score
from _DetectabilityWithMeta import MetaSBM
from multiprocessing import Pool


def compute_f(A, partition):
    n = np.shape(A)[0]
    c = np.sum(A) / n
    uniquePartition, ns = np.unique(partition, return_counts=True)
    partition_counts = dict(zip(uniquePartition, ns))
    numGroup = np.size(uniquePartition)
    cab = np.zeros((numGroup, numGroup))
    ha = np.zeros(numGroup)
    for ia in partition_counts.keys():
        a_index = np.where(partition == ia)[0]
        for ib in partition_counts.keys():
            b_index = np.where(partition == ib)[0]
            if ia != ib:
                cab[ia, ib] = np.sum(A[a_index, :][:, b_index]) / (partition_counts[ia] * partition_counts[ib]) * n
            else:
                cab[ia, ib] = np.sum(A[a_index, :][:, b_index]) / (partition_counts[ia] * (partition_counts[ia]-1)) * n
            ha[ia] += partition_counts[ib] * cab[ia, ib] / n
    Zi = np.zeros(n)
    for i in range(n):
        for ia in partition_counts.keys():
            temp = 1
            for ib in partition[np.nonzero(A[[i], :])[0]]:
                temp *= cab[ia, ib]
            Zi[i] += partition_counts[ia] * np.exp(-ha[ia]) * temp / n
    f = 0
    for (i, j) in zip(*triu(A).nonzero()):
        f += np.log(cab[partition[i], partition[j]]) / n
    f -= c / 2
    for i in range(n):
        f -= np.log(Zi[i]) / n
    return f


def synthetic_exp(msbm, init_epsilon=None, strId="", writeCM=False, stop_when_increasing_f=True, init_BH=False, elbow_learn='f_compare'):
    A = msbm.sample()
    metaIdSelect = 0
    filterA, filterGroupId = msbm.filter(A, metaId=metaIdSelect)
    result_data = []
    pid = os.getpid()
    learnq_path = f'./result/test_record_f/{strId}_BPlearnq_FreeEnergy_{str(pid) if pid is not None else ""}_inite{init_epsilon}.txt'
    max_learn_q = 7
    # First Run BP learn q
    return_all_partitions = True if stop_when_increasing_f is False else False
    subpartition, sub_num_groups, all_partitions = CommunityDetect(filterA).BP_FE_learnq(groupId=filterGroupId,
                                                                                    processId=str(pid) + "SUB",
                                                                                    strId=strId,
                                                                                    init_epsilon=init_epsilon,
                                                                                    max_learn_q=max_learn_q,
                                                                                    stop_when_increasing_f=stop_when_increasing_f, 
                                                                                    learnq_path=learnq_path, 
                                                                                    return_all_partitions=return_all_partitions, 
                                                                                    elbow_learn=elbow_learn)
    if elbow_learn=='f_compare' and return_all_partitions and writeCM:
        for q in all_partitions.keys():
            with open(f"./result/confusionMatrix/{strId}_BP_q={q}.txt", 'w') as fw:
                trueNumgroup = np.size(np.unique(filterGroupId))
                cdNumgroup = np.size(np.unique(all_partitions[q]))
                cm, _ = get_confusionmatrix(filterGroupId, all_partitions[q], trueNumgroup, cdNumgroup)
                fw.write(str(cm))
                fw.write('\n')
                ami = adjusted_mutual_info_score(filterGroupId, all_partitions[q])
                fw.write(f'{ami}\n')
    if writeCM:
        with open(f"./result/confusionMatrix/{strId}.txt", 'w') as fw:
            trueNumgroup = np.size(np.unique(filterGroupId))
            cdNumgroup = sub_num_groups
            cm, _ = get_confusionmatrix(filterGroupId, subpartition, trueNumgroup, cdNumgroup)
            fw.write(str(cm))
    # Then run BP init by BH
    if init_BH is True:
        # BP initial with BetheHessian result
        for q in range(2, max_learn_q+1):
            # BH
            subBHpartition, subBHNumGroup = CommunityDetect(filterA).BetheHessian(num_groups=q)
            if writeCM:
                with open(f"./result/confusionMatrix/{strId}_initBH_q={q}.txt", 'w') as fw:
                    trueNumgroup = np.size(np.unique(filterGroupId))
                    cm, _ = get_confusionmatrix(filterGroupId, subBHpartition, trueNumgroup, subBHNumGroup)
                    fw.write(str(cm))
                    fw.write('\n')
                    ami = adjusted_mutual_info_score(filterGroupId, subBHpartition)
                    fw.write(f'{ami}\n')
            # Compute BH f
            # f = CommunityDetect(filterA).BP(num_groups=subBHNumGroup, na=None, cab=None, 
            #     groupId=subBHpartition, processId=str(pid) + "SUB", infermode=7, init_epsilon=init_epsilon, 
            #     learn_conv_crit=None, learn_max_time=None, message_init_flag=2)
            f = compute_f(filterA, partition=subBHpartition)
            log = f'BHq={subBHNumGroup} f={f}\n'
            with open(learnq_path, 'a') as fw:
                fw.write(log)
            # BPinitBH
            subpartition, f = CommunityDetect(filterA).BP(num_groups=subBHNumGroup, na=None, cab=None, 
                groupId=subBHpartition, processId=str(pid) + "SUB", infermode=2, init_epsilon=init_epsilon, 
                learn_conv_crit=None, learn_max_time=None, message_init_flag=2)
            sub_num_groups = np.size(np.unique(subpartition))
            log = f'BPinitBHq={subBHNumGroup} f={f} detect_q={sub_num_groups}\n'
            with open(learnq_path, 'a') as fw:
                fw.write(log)
            if writeCM:
                with open(f"./result/confusionMatrix/{strId}_BPinitBH_q={q}.txt", 'w') as fw:
                    trueNumgroup = np.size(np.unique(filterGroupId))
                    cdNumgroup = sub_num_groups
                    cm, _ = get_confusionmatrix(filterGroupId, subpartition, trueNumgroup, cdNumgroup)
                    fw.write(str(cm))
                    fw.write('\n')
                    ami = adjusted_mutual_info_score(filterGroupId, subpartition)
                    fw.write(f'{ami}\n')
    sub_ami = adjusted_mutual_info_score(filterGroupId, subpartition)
    result_data.append(sub_ami)
    result_data.append(sub_num_groups)
    return result_data


def exp_subprocess(n, q, Z_s, Z_b, d, rho, delta, times, savepath, Withlambda=True, writeCM=False, stop_when_increasing_f=True,
                    additionId="", Type=2, init_BH=False, elbow_learn='f_compare'):
    ps, pin, pout1, pout2 = get_ps(n, d, Z_s, Z_b, rho, delta, Type)

    n_f = int(n * (Z_s + Z_b) * (Z_b * rho + Z_s * (1 - rho)) / (Z_s * Z_b))
    rho_f = Z_b * rho / (Z_b * rho + Z_s * (1 - rho))
    n_fq = int(n_f / q)
    n_f = int(n_fq * q)
    sizes = [[n_fq] * Z_s, [n_fq] * Z_b]
    msbm = MetaSBM(n_f, rho_f, ps, sizes)

    results = ""
    # Calc eig of subgraph first
    if Withlambda:
        lambdas = msbm.get_lambdas_homoDegree(n, d, Z_s, Z_b, rho, delta, Type=Type)
        SNR = lambdas[1]**2/lambdas[0]
        print(f'lambdas_PQ={lambdas}')
    for t in range(times):
        start = time.time()
        print(f"EXP pid={os.getpid()} begin... rho={rho}, delta={delta}, times={t}")
        print(f'pin_s={pin}, pin_b={pout1}, pout={pout2}')
        if Withlambda and SNR < 1:
            sub_ami = 0
            sub_num_groups = 1
        else:
            init_epsilon = None
            if delta < 0:
                init_epsilon = 5
            elif delta > 0:
                init_epsilon = 0.2
            strId = f'{additionId}_n={n}d={d}Zs={Z_s}Zb={Z_b}rho={rho}delta={delta}t={t}'
            if pin < 0 or pin > 1 or pout1 < 0 or pout1 > 1 or pout2 < 0 or pout2 > 1:
                # for delta let pin pout not in [0, 1], return (-1,0)
                result_data = (-1, 0)
            else:
                result_data = synthetic_exp(msbm, init_epsilon=init_epsilon, strId=strId, writeCM=writeCM, 
                    stop_when_increasing_f=stop_when_increasing_f, init_BH=init_BH, elbow_learn=elbow_learn)
            sub_ami, sub_num_groups = result_data[0], result_data[1]

        results += f'{rho} {delta} {t} {sub_ami} {sub_num_groups}'
        if Withlambda:
            for l in lambdas:
                results += f' {l}'
        results += '\n'
        print(f"EXP pid={os.getpid()} end. sub_ami={sub_ami}, Time:{np.around(time.time() - start, 3)}")
    return savepath, results


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


def run_exp(rho_deltas, times, save_path=None, q=3, n=600, d=300, Z_s=None, Z_b=None, Withlambda=False,
            multiprocessing=True,
            writeCM=False, stop_when_increasing_f=True,
            additionId="", Type=2, init_BH=False, elbow_learn='f_compare'):
    rho_delta_pair = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                rho_delta_pair.add((round(float(row[0]), 5), round(float(row[1]), 5)))
    if multiprocessing:
        p = Pool(8)
        for rho_delta in rho_deltas:
            rho = rho_delta[0]
            delta = rho_delta[1]
            if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                print(f'rho={rho}, delta={delta} has been run!')
                continue
            p.apply_async(exp_subprocess,
                            args=(n, q, Z_s, Z_b, d, rho, delta, times, save_path, Withlambda, 
                            writeCM, stop_when_increasing_f, additionId, Type, init_BH, elbow_learn, ),
                            callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for rho_delta in rho_deltas:
            rho = rho_delta[0]
            delta = rho_delta[1]
            if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                print(f'rho={rho}, delta={delta} has been run!')
                continue
            savepath, results = exp_subprocess(n, q, Z_s, Z_b, d, rho, delta, times, save_path, Withlambda,
                                                writeCM, stop_when_increasing_f, additionId, Type, init_BH, elbow_learn)
            write_results((savepath, results))


def read_exp(load_path, Withlambda=False, exclude_rho=None, add_paths=None):
    """
    read the results file from run_exp
    :param load_path:
    :param Withsnr:
    :return:
    """
    exclude_rho = [] if exclude_rho is None else exclude_rho
    max_lambda_num = 4
    with open(load_path, 'r') as f:
        results_list = [row.strip().split() for row in f.readlines()]
        if add_paths is not None:
            print("Additional result adding...")
            for add_path in add_paths:
                with open(add_path, 'r') as add_f:
                    results_list = results_list + [row.strip().split() for row in add_f.readlines()]
        for i in range(len(results_list)):
            if len(results_list[i]) < 9:
                results_list[i] += ["0"] * (9 - len(results_list[i]))
        results = np.round(np.float64(results_list), decimals=5)
        rhos = np.setdiff1d(np.unique(results[:, 0]), np.array(exclude_rho))
        zs = np.unique(results[:, 1])
        # full_ami = np.zeros(np.size(zs) * np.size(rhos))
        sub_ami = np.zeros(np.size(zs) * np.size(rhos))
        # snr_nm = np.zeros(np.size(zs) * np.size(rhos)) if Withsnr else None
        lambdas = None
        # full_num_group = np.zeros(np.size(zs) * np.size(rhos))
        sub_num_group = np.zeros(np.size(zs) * np.size(rhos))
        i = 0
        for _rho in rhos:
            for _z in zs:
                ami_results = results[
                    np.squeeze(np.argwhere(np.logical_and(results[:, 0] == _rho, results[:, 1] == _z)))]
                if np.size(ami_results) == 0:
                    print(f"Some parameter rho={_rho}, z={_z} didn't run!")
                mean_ami = np.mean(ami_results, 0)[3:] if len(np.shape(ami_results)) == 2 else ami_results[3:]
                # full_ami[i] = mean_ami[0]
                sub_ami[i] = mean_ami[0]
                sub_num_group[i] = mean_ami[1]
                if Withlambda:
                    if lambdas is None:
                        lambdas = np.zeros((np.size(zs) * np.size(rhos), max_lambda_num))
                        # print(np.shape(lambdas))
                    # l = np.unique(np.around(mean_ami[2:], 5))
                    # l = sorted(l, key=lambda v: np.abs(v), reverse=True)
                    lambdas[i] = mean_ami[2:]

                i += 1
        plot_rhos = np.repeat(rhos, np.size(zs))
        plot_zs = np.tile(zs, np.size(rhos))
    # return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group
    return plot_rhos, plot_zs, sub_ami, sub_num_group, lambdas


def exp0():
    """
        Try BP
    """
    times = 1
    n = 6000
    d = 5
    Z_s = 2
    Z_b = 3
    q = Z_s + Z_b
    Type = 2
    rho_deltas = [(0.12, 0.00161), (0.24, 0.0022), (0.38, 0.00293)]
    print(rho_deltas)
    Withlambda = True
    writeCM = True
    stop_when_increasing_f = False
    # additionId = "2ndType_3or2decimal_10more"
    additionId = "2ndType_BPinitBH"
    initBH = True
    multiprocessing = False
    fileID = 'amiExp24.6.6' + f'_n={n}_Zs={Z_s}_Zb={Z_b}_d={round(d)}_{"lambda" if Withlambda else ""}_' \
                              f'{"writeCM" if writeCM else ""}_' \
                              f'{"initBH" if initBH else ""}_' \
                              f'{additionId}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho_deltas) * times}")
    run_exp(rho_deltas, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            writeCM=writeCM, stop_when_increasing_f=stop_when_increasing_f, additionId=additionId, Type=Type, init_BH=initBH)


def exp1():
    """
        Try BP
    """
    times = 1
    n = 6000
    d = 5
    Z_s = 2
    Z_b = 3
    q = Z_s + Z_b
    Type = 2
    # rho_deltas = [(0.12, 0.00161), (0.24, 0.0022), (0.38, 0.00293)]
    rho_deltas = [(0.24, 0.0022)]
    print(rho_deltas)
    Withlambda = True
    writeCM = True
    stop_when_increasing_f = False
    # additionId = "2ndType_3or2decimal_10more"
    # additionId = "6.15_2ndType_SameSampleNet"
    additionId = "6.17_2ndType_multiInitBH"
    initBH = True
    multiprocessing = False
    fileID = 'amiExp24.6.17' + f'_n={n}_Zs={Z_s}_Zb={Z_b}_d={round(d)}_{"lambda" if Withlambda else ""}_' \
                              f'{"writeCM" if writeCM else ""}_' \
                              f'{"initBH" if initBH else ""}_' \
                              f'{additionId}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho_deltas) * times}")
    run_exp(rho_deltas, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            writeCM=writeCM, stop_when_increasing_f=stop_when_increasing_f, additionId=additionId, Type=Type, init_BH=initBH)


def exp2():
    """
        Try BP
    """
    times = 1
    n = 6000
    d = 5
    Z_s = 2
    Z_b = 3
    q = Z_s + Z_b
    Type = 2
    # rho_deltas = [(0.12, 0.00161), (0.24, 0.0022), (0.38, 0.00293)]
    rho_deltas = [(0.24, 0.0022)]
    print(rho_deltas)
    Withlambda = True
    writeCM = True
    stop_when_increasing_f = False
    # additionId = "2ndType_3or2decimal_10more"
    # additionId = "6.15_2ndType_CrossElbowPoint"
    additionId = "7.1_2ndType_CrossElbowPoint"
    initBH = False
    elbow_learn = 'cross_point'
    multiprocessing = False
    fileID = 'amiExp24.7.1' + f'_n={n}_Zs={Z_s}_Zb={Z_b}_d={round(d)}_{"lambda" if Withlambda else ""}_' \
                              f'{"writeCM" if writeCM else ""}_' \
                              f'{"initBH" if initBH else ""}_' \
                              f'{elbow_learn}_' \
                              f'{additionId}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho_deltas) * times}")
    run_exp(rho_deltas, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            writeCM=writeCM, stop_when_increasing_f=stop_when_increasing_f, additionId=additionId, Type=Type, init_BH=initBH, 
            elbow_learn=elbow_learn)

def exp3():
    """
        Try BP
    """
    times = 1
    n = 6000
    d = 5
    Z_s = 2
    Z_b = 3
    q = Z_s + Z_b
    Type = 2
    # rho_deltas = [(0.12, 0.00161), (0.24, 0.0022), (0.38, 0.00293)]
    rho_deltas = [(0.24, 0.0022)]
    print(rho_deltas)
    Withlambda = True
    writeCM = True
    stop_when_increasing_f = False
    # additionId = "2ndType_3or2decimal_10more"
    # additionId = "6.15_2ndType_SameSampleNet"
    # additionId = "7.25_2ndType_multiInitBH"
    additionId = "7.30_2ndType_initBHdirectF"
    initBH = True
    multiprocessing = False
    fileID = 'amiExp24.7.30' + f'_n={n}_Zs={Z_s}_Zb={Z_b}_d={round(d)}_{"lambda" if Withlambda else ""}_' \
                              f'{"writeCM" if writeCM else ""}_' \
                              f'{"initBH" if initBH else ""}_' \
                              f'{additionId}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho_deltas) * times}")
    run_exp(rho_deltas, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            writeCM=writeCM, stop_when_increasing_f=stop_when_increasing_f, additionId=additionId, Type=Type, init_BH=initBH)



if __name__ == '__main__':
    # exp0()
    # exp1()
    # exp2()
    exp3()
