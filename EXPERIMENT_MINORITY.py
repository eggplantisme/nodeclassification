import os
import numpy as np
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import Pool
from _CommunityDetect import *
from scipy.optimize import linear_sum_assignment


# def get_range_delta(d, n, q):
#     if 0 < d < n / q:
#         min_delta = q * d / ((1 - q) * n)
#         max_delta = q * d / n
#     elif n / q <= d < (1 - 1 / q) * n:
#         min_delta = q * d / ((1 - q) * n)
#         max_delta = (q / (1 - q)) * (d / n - 1)
#     elif (1 - 1 / q) * n <= d < n:
#         min_delta = q * (d / n - 1)
#         max_delta = (q / (1 - q)) * (d / n - 1)
#     else:
#         min_delta, max_delta = None, None
#     return min_delta, max_delta
def get_confusionmatrix(truePartition, cdPartition, trueNumgroup, cdNumgroup):
    confusionMatrix = np.zeros((trueNumgroup, cdNumgroup))
    uniqueTpartition = np.unique(truePartition)
    uniqueDpartition = np.unique(cdPartition)
    for iTrue in uniqueTpartition:
        trueIndex = np.where(truePartition == iTrue)
        for iCD in uniqueDpartition:
            i = np.where(uniqueTpartition == iTrue)
            j = np.where(uniqueDpartition == iCD)
            confusionMatrix[i, j] = np.size(np.where(cdPartition[trueIndex]==iCD))
#     print(confusionMatrix)
    true_ind, CD_ind = linear_sum_assignment(confusionMatrix, maximize=True)
    print(f'True index is {true_ind}, Community detected index is {CD_ind}')
    confusionMatrix[:, np.sort(CD_ind)] = confusionMatrix[:, CD_ind]
    return confusionMatrix, CD_ind


def get_range_delta(d, n, Z_s, Z_b):
    """
    For sparse network d << n
    :param d:
    :param n:
    :param Z_s:
    :param Z_b:
    :return:
    """
    if 0 < d < n / (Z_s + Z_b):
        min_delta = (Z_s + Z_b) * d / ((1 - Z_s - Z_b) * n)
        max_delta = min(Z_s, Z_b) * d / n
    else:
        min_delta, max_delta = None, None
    return min_delta, max_delta


def synthetic_exp_full2sub(msbm, givenNumGroup=False, DC=False, BP=False, init_epsilon=None, learnqby=None, givenNacab=True, strId="", writeCM=False):
    A = msbm.sample()
    metaIdSelect = 0
    filterA, filterGroupId = msbm.filter(A, metaId=metaIdSelect)
    result_data = []
    if not givenNumGroup:
        if BP is True:
            pid = os.getpid()
            if learnqby == "MDL":
                subpartition, sub_num_groups = CommunityDetect(filterA).BP_MDL_learnq(groupId=filterGroupId, processId=str(pid)+"SUB", strId=strId, init_epsilon=init_epsilon)
            elif learnqby == "FE":
                subpartition, sub_num_groups = CommunityDetect(filterA).BP_FE_learnq(groupId=filterGroupId, processId=str(pid)+"SUB", strId=strId, init_epsilon=init_epsilon, max_learn_q=7, stop_when_increasing_f=False)
            else:
                print("Wrong Parameter!!!")
        elif DC is False:
            # fullpartition, full_num_groups = CommunityDetect(A).BetheHessian()
            if learnqby is None:
                subpartition, sub_num_groups = CommunityDetect(filterA).BetheHessian()
            elif learnqby == 'MDL':
                subpartition, sub_num_groups = CommunityDetect(filterA).BH_MDL_learnq()
        else:
            # fullpartition, full_num_groups, full_zetas = CommunityDetect(A).DCBetheHessian()
            subpartition, sub_num_groups, sub_zetas = CommunityDetect(filterA).DCBetheHessian()
    else:
        # full_num_groups_given = np.size(np.unique(msbm.groupId))
        sub_num_groups_given = np.size(np.unique(filterGroupId))
        if BP is True:
            pid = os.getpid()
            # FULL
            # na = (np.array(msbm.sizes).flatten() / msbm.N).tolist()
            # cab = []
            # for x in np.nditer(msbm.ps):
            #     cab.append(np.around(x*msbm.N, 2))
            # fullpartition = CommunityDetect(A).BP(full_num_groups_given, na, cab, msbm.rho, msbm.groupId, msbm.metaId, processId=str(pid)+"FULL")
            # full_num_groups = full_num_groups_given
            # SUB
            if givenNacab:
                subsize = np.size(filterGroupId)
                v, counts = np.unique(filterGroupId, return_counts=True)
                na = counts / np.size(filterGroupId)
                cab = []
                for x in np.nditer(msbm.ps[v, :][:, v]):
                    cab.append(np.around(x*subsize, 2))
                subpartition, _ = CommunityDetect(filterA).BP(sub_num_groups_given, na, cab, filterGroupId, processId=str(pid)+"SUB")
            else:
                na = None
                cab = None
                subpartition, _ = CommunityDetect(filterA).BP(sub_num_groups_given, na, cab, filterGroupId, processId=str(pid)+"SUB", infermode=2, init_epsilon=init_epsilon)
            sub_num_groups = np.size(np.unique(subpartition))
        elif DC is False:
            # fullpartition, full_num_groups = CommunityDetect(A).BetheHessian(full_num_groups_given)
            subpartition, sub_num_groups = CommunityDetect(filterA).BetheHessian(sub_num_groups_given)
        else:
            # fullpartition, full_num_groups, full_zetas = CommunityDetect(A).DCBetheHessian(full_num_groups_given)
            subpartition, sub_num_groups, sub_zetas = CommunityDetect(filterA).DCBetheHessian(sub_num_groups_given)
    if writeCM:
        with open(f"./result/confusionMatrix/{strId}.txt", 'w') as fw:
            trueNumgroup = np.size(np.unique(filterGroupId))
            cdNumgroup = sub_num_groups
            cm, _ = get_confusionmatrix(filterGroupId, subpartition, trueNumgroup, cdNumgroup)
            fw.write(str(cm))
    sub_ami = adjusted_mutual_info_score(filterGroupId, subpartition)
    result_data.append(sub_ami)
    result_data.append(sub_num_groups)
    
    return result_data


def exp_subprocess(n, q, Z_s, Z_b, d, rho, delta, times, savepath, Withlambda=True, givenNumGroup=False, DC=False,
                     BP=False, learnqby=None, givenNacab=True, givenTrueEpsilon=False, writeCM=False):
    """
    Here n, rho, d is sub parameter. The full parameter n_f, rho_f, pin, pout should be calculated base on sub
    :param n:
    :param q:
    :param Z_s:
    :param Z_b:
    :param d:
    :param rho:
    :param delta:
    :param times:
    :param savepath:
    :param Withlambda:
    :param givenNumGroup:
    :param DC:
    :param BP:
    :return:
    """
    pout = d / n - ((1-rho)**2 / Z_b + rho**2 / Z_s) * delta
    pin = pout + delta
    pin = 0 if pin < 1e-10 else pin
    pout = 0 if pout < 1e-10 else pout
    if BP:
        # For BP not use two small pin and pout
        pin = 1e-5 if pin < 1e-5 else pin
        pout = 1e-5 if pout < 1e-5 else pout
    ps = (pin - pout) * np.identity(q) + pout * np.ones((q, q))
    n_f = int(n * (Z_s + Z_b) * (Z_b * rho + Z_s * (1 - rho)) / (Z_s * Z_b))
    rho_f = Z_b * rho / (Z_b * rho + Z_s * (1 - rho))
    n_fq = int(n_f / q)
    n_f = int(n_fq * q)
    sizes = [[n_fq] * Z_s, [n_fq] * Z_b]
    msbm = MetaSBM(n_f, rho_f, ps, sizes)

    results = ""
    # Calc eig of subgraph first
    if Withlambda:
        lambdas = msbm.get_lambdas(n, rho, Z_s, Z_b, pin, pout)
        SNR = lambdas[1]**2 / lambdas[0]
    for t in range(times):
        start = time.time()
        print(f"EXP pid={os.getpid()} begin... rho={rho}, delta={delta}, times={t}, pin={pin}, pout={pout}")
        # if BP and SNR < 1:
        #     sub_ami = 0
        #     sub_num_groups = 1
        # else:
        if givenTrueEpsilon is False:
            init_epsilon = None
            if BP and delta < 0:
                init_epsilon = 5
            elif BP and delta > 0:
                init_epsilon = 0.2
        else:
            init_epsilon = np.around(pout / pin, 6)
        strId = f'n={n}d={d}Zs={Z_s}Zb={Z_b}rho={rho}delta={delta}t={t}{"givenNumGroup" if givenNumGroup else ""}_'\
                f'{"DC" if DC else ""}_{"BP" if BP else ""}_{"MDL" if learnqby == "MDL" else "FE"}_'\
                f'{"TrueEpsilon" if givenTrueEpsilon else ""}_moreq'
        result_data = synthetic_exp_full2sub(msbm, givenNumGroup=givenNumGroup, DC=DC, BP=BP, init_epsilon=init_epsilon, learnqby=learnqby, givenNacab=False, strId=strId, writeCM=writeCM)
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


def run_exp(rhos, deltas, times, save_path=None, q=3, n=600, d=300, Z_s=None, Z_b=None, Withlambda=False, multiprocessing=True,
            givenNumGroup=False, DC=False, BP=False, learnqby=None, givenNacab=True, givenTrueEpsilon=False, writeCM=False):
    rho_delta_pair = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                rho_delta_pair.add((round(float(row[0]), 5), round(float(row[1]), 5)))
    if multiprocessing:
        p = Pool(8)
        for rho in rhos:
            for delta in deltas:
                if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                    print(f'rho={rho}, delta={delta} has been run!')
                    continue
                p.apply_async(exp_subprocess, args=(n, q, Z_s, Z_b, d, rho, delta, times, save_path, Withlambda, givenNumGroup,
                                                    DC, BP, learnqby, givenNacab, givenTrueEpsilon, writeCM),
                              callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for rho in rhos:
            for delta in deltas:
                if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                    print(f'rho={rho}, delta={delta} has been run!')
                    continue
                savepath, results = exp_subprocess(n, q, Z_s, Z_b, d, rho, delta, times, save_path, Withlambda, givenNumGroup, DC,
                                                   BP, learnqby, givenNacab, givenTrueEpsilon, writeCM)
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
                ami_results = results[np.squeeze(np.argwhere(np.logical_and(results[:, 0]==_rho, results[:, 1]==_z)))]
                if np.size(ami_results) == 0:
                    print(f"Some parameter rho={_rho}, z={_z} didn't run!")
                mean_ami = np.mean(ami_results, 0)[3:] if len(np.shape(ami_results)) == 2 else ami_results[3:]
                # full_ami[i] = mean_ami[0]
                sub_ami[i] = mean_ami[0]
                sub_num_group[i] = mean_ami[1]
                if Withlambda:
                    if lambdas is None:
                        lambdas = np.zeros((np.size(zs) * np.size(rhos), max_lambda_num))
                    lambdas[i] = mean_ami[2:]
                i += 1
        plot_rhos = np.repeat(rhos, np.size(zs))
        plot_zs = np.tile(zs, np.size(rhos))
    # return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group
    return plot_rhos, plot_zs, sub_ami, sub_num_group, lambdas


def exp0():
    times = 50
    n_q = 4000
    q = 3
    n = q * n_q
    d = 50
    Z_s = 1
    Z_b = 2
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([0, 1]))
    min_delta, max_delta = get_range_delta(d, n, q)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, int(58)), 5), np.array([0]))
    print(f'delta={delta}')
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    multiprocessing = True
    fileID = 'amiExp12.28' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)

def exp1():
    times = 20
    n_q = 2000
    q = 6
    n = q * n_q
    d = 50
    Z_s = 3
    Z_b = 3
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, q)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, int((max_delta - min_delta) / 0.0005) + 1), 5),
                         np.array([0]))
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    multiprocessing = False
    fileID = 'amiExp24.1.2' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)

def exp2():
    times = 50
    n_q = 6000
    q = 2
    n = q * n_q
    d = 50
    Z_s = 1
    Z_b = 1
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([0, 1]))
    min_delta, max_delta = get_range_delta(d, n, q)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, int(58)), 5), np.array([0]))
    print(f'delta={delta}')
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    multiprocessing = False
    fileID = 'amiExp24.1.4' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)

def exp3():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    :return:
    """
    times = 40
    n = 6000
    d = 50
    Z_s = 1
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([0, 1]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 58), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    multiprocessing = True
    fileID = 'amiExp24.1.10' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_fixsubparameter_more'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)


def exp4():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    :return:
    """
    times = 50
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    multiprocessing = True
    fileID = 'amiExp24.1.19' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_fixsubparameter'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)

def exp5():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    :return:
    """
    times = 20
    n = 6000
    d = 50
    Z_s = 1
    Z_b = 1
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([0, 1]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 58), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    multiprocessing = True
    fileID = 'amiExp24.1.17' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_fixsubparameter'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)


def exp6():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    :return:
    """
    times = 50
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 3
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    multiprocessing = True
    fileID = 'amiExp24.1.23' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_fixsubparameter_more'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)


def exp7():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP learnq with MDL
    :return:
    """
    times = 1
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = True
    multiprocessing = True
    fileID = 'amiExp24.1.27' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_fixsubparameter'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)


def exp8():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    :return:
    """
    times = 50
    n = 6000
    d = 50
    Z_s = 3
    Z_b = 3
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    multiprocessing = True
    fileID = 'amiExp24.1.29' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_fixsubparameter'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)


def exp9():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP learnq with MDL
    :return:
    """
    times = 5
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = True
    multiprocessing = True
    fileID = 'amiExp24.2.11' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_fixsubparameter_reducetimecost_farinitialepsilon'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)


def exp10():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP learnq with MDL
    :return:
    """
    times = 15
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = True
    multiprocessing = True
    fileID = 'amiExp24.2.12' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_fixsubparameter_reducetimecost_0.2_5_Sigmab_more'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)


def exp11():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP given True # of communities
    :return:
    """
    times = 5
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = True
    DC = False
    BP = True
    multiprocessing = True
    fileID = 'amiExp24.2.15' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_fixsubparameter'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP)


def exp12():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP learnq by FreeEnergy
    :return:
    """
    times = 5
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = True
    learnqby = 'FE'
    multiprocessing = True
    fileID = 'amiExp24.2.15' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_learnqby{"MDL" if learnqby == "MDL" else "FE"}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP, learnqby=learnqby)


def exp13():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BH learnq by MDL
    :return:
    """
    times = 5
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    learnqby = 'MDL'
    multiprocessing = True
    fileID = 'amiExp24.2.19' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_learnqby{"MDL" if learnqby == "MDL" else "FE"}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP, learnqby=learnqby)


def exp14():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP given True # of communities
    :return:
    """
    times = 5
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = True
    DC = False
    BP = True
    givenNacab = False
    multiprocessing = True
    fileID = 'amiExp24.2.20' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_{"givenNacab" if givenNacab else "learnNacab"}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP, givenNacab=givenNacab)


def exp15():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP learnq by FreeEnergy 10 times for each q
    :return:
    """
    times = 5
    n = 6000
    d = 50
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    rho = rho[:9]
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = True
    learnqby = 'FE'
    multiprocessing = False
    fileID = 'amiExp24.2.21' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_learnqby{"MDL" if learnqby == "MDL" else "FE"}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP, learnqby=learnqby)


def exp16():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP learnq by FreeEnergy 10 times for each q
    :return:
    """
    times = 10
    n = 6000
    d = 15
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    # rho = rho[:9]
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = True
    learnqby = None
    multiprocessing = True
    fileID = 'amiExp24.3.28' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP, learnqby=learnqby)


def exp17():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP learnq by FreeEnergy 10 times for each q
    :return:
    """
    times = 5
    n = 6000
    d = 15
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    # rho = rho[:9]
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = True
    learnqby = 'FE'
    multiprocessing = True
    fileID = 'amiExp24.3.30' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_learnqby{"MDL" if learnqby == "MDL" else "FE"}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP, learnqby=learnqby)


def exp18():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP learnq by FreeEnergy 10 times for each q
    :return:
    """
    times = 10
    n = 6000
    d = 15
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    # rho = rho[:9]
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))\
    delta = np.array([0.004])
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = True
    learnqby = 'FE'
    multiprocessing = True
    fileID = 'amiExp24.4.15' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_learnqby{"MDL" if learnqby == "MDL" else "FE"}_'\
                              f'10exp20learn1delta_10more'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP, learnqby=learnqby)


def exp19():
    """
    sub parameter is fix as n, d. rho, delta belong to sub
    Try BP learnq by FreeEnergy 10 times for each q
    :return:
    """
    times = 1
    n = 10000
    d = 16
    Z_s = 2
    Z_b = 2
    q = Z_s + Z_b
    # sizes = [[n_q] * Z_s, [n_q] * Z_b]
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    rho = np.array([0.5])
    # rho = np.setdiff1d(np.around(np.linspace(0, 1, 26), 2), np.array([]))
    # rho = rho[:9]
    min_delta, max_delta = get_range_delta(d, n, Z_s, Z_b)
    # delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 30), 5), np.array([0]))
    delta = delta[17:]
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = True
    learnqby = 'FE'
    givenTrueEpsilon = True
    writeCM = True
    multiprocessing = True
    fileID = 'amiExp24.4.21' + f'_n={n}_q={q}_d={round(d)}_{"lambda" if Withlambda else ""}_'\
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_learnqby{"MDL" if learnqby == "MDL" else "FE"}_'\
                              f'{"givenTrueEpsilon" if givenTrueEpsilon else ""}_{"writeCM" if writeCM else ""}'\
                              f'1exp20learn1rho_recordf'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP, learnqby=learnqby, givenTrueEpsilon=givenTrueEpsilon, writeCM=writeCM)


if __name__ == '__main__':
    # exp0()
    # exp1()
    # exp2()
    # exp3()
    # exp4()
    # exp5()
    # exp6()
    # exp7()
    # exp8()
    # exp9()
    # exp10()
    # exp11()
    # exp12()
    # exp13()
    # exp14()
    # exp15()
    # exp16()
    # exp17()
    # exp18()
    exp19()
