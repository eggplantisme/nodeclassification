import os
import numpy as np
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import Pool
from _CommunityDetect import *
from scipy.optimize import linear_sum_assignment
from EXPERIMENT_MINORITY import get_confusionmatrix


def get_ps(n, d, Z_s, Z_b, rho, delta, Type=2):
    q = Z_s + Z_b
    if Type == 1:
        pout = d / n - (rho / Z_s) * delta
        pin_s = pout + delta
        pin_b = pout + Z_b * rho / ((1 - rho) * Z_s) * delta
        pin_s = 0 if pin_s < 1e-10 else pin_s
        pin_b = 0 if pin_b < 1e-10 else pin_b
        pout = 0 if pout < 1e-10 else pout
        ps = pout * np.ones((q, q))
        for i in range(Z_s):
            ps[i, i] = pin_s
        for i in range(Z_b):
            ps[Z_s + i, Z_s + i] = pin_b
        return ps, pin_s, pin_b, pout
    elif Type == 2:
        pout1 = d / n - (1 / (1 - 2 * rho)) * ((1 - rho) ** 2 / Z_b - rho ** 2 / Z_s) * delta
        pin = pout1 + delta
        pout2 = 2 * d / n - ((1 - rho) / Z_b + rho / Z_s) * delta - pout1
        pout1 = 0 if abs(pout1) < 1e-10 else pout1
        pout2 = 0 if abs(pout2) < 1e-10 else pout2
        pin = 0 if abs(pin) < 1e-10 else pin
        ps = np.zeros((q, q))
        for i in range(q):
            for j in range(q):
                if i == j:
                    ps[i, i] = pin
                elif i < Z_s and j < Z_s:
                    ps[i, j] = pout1
                elif i >= Z_s and j >= Z_s:
                    ps[i, j] = pout1
                else:
                    ps[i, j] = pout2
        return ps, pin, pout1, pout2
    else:
        return None


def synthetic_exp_full2sub(msbm, givenNumGroup=False, DC=False, BP=False, init_epsilon=None, learnqby=None,
                           givenNacab=True, strId="", writeCM=False):
    A = msbm.sample()
    metaIdSelect = 0
    filterA, filterGroupId = msbm.filter(A, metaId=metaIdSelect)
    result_data = []
    if not givenNumGroup:
        if BP is True:
            pid = os.getpid()
            if learnqby == "MDL":
                subpartition, sub_num_groups = CommunityDetect(filterA).BP_MDL_learnq(groupId=filterGroupId,
                                                                                      processId=str(pid) + "SUB",
                                                                                      strId=strId,
                                                                                      init_epsilon=init_epsilon)
            elif learnqby == "FE":
                subpartition, sub_num_groups = CommunityDetect(filterA).BP_FE_learnq(groupId=filterGroupId,
                                                                                     processId=str(pid) + "SUB",
                                                                                     strId=strId,
                                                                                     init_epsilon=init_epsilon,
                                                                                     max_learn_q=7,
                                                                                     stop_when_increasing_f=False)
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
                    cab.append(np.around(x * subsize, 2))
                subpartition, _ = CommunityDetect(filterA).BP(sub_num_groups_given, na, cab, filterGroupId,
                                                              processId=str(pid) + "SUB")
            else:
                na = None
                cab = None
                subpartition, _ = CommunityDetect(filterA).BP(sub_num_groups_given, na, cab, filterGroupId,
                                                              processId=str(pid) + "SUB", infermode=2,
                                                              init_epsilon=init_epsilon)
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
                   BP=False, learnqby=None, givenNacab=True, givenTrueEpsilon=False, writeCM=False, additionId="",
                   Type=2, checkSNR=False):
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
        P = np.diag([rho / Z_s] * Z_s + [(1 - rho) / Z_b] * Z_b)
        Q = n * ps
        lambdas = msbm.get_lambdas_homoDegree(n, d, Z_s, Z_b, rho, delta, Type=Type)
        print(f'lambdas_PQ={lambdas}')
    for t in range(times):
        start = time.time()
        print(f"EXP pid={os.getpid()} begin... rho={rho}, delta={delta}, times={t}")
        print(f'pin_s={pin}, pin_b={pout1}, pout={pout2}')
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
            pass
            # init_epsilon = np.around(pout / pin, 6)
        strId = f'n={n}d={d}Zs={Z_s}Zb={Z_b}rho={rho}delta={delta}t={t}{"givenNumGroup" if givenNumGroup else ""}_' \
                f'{"DC" if DC else ""}_{"BP" if BP else ""}_{"MDL" if learnqby == "MDL" else "FE"}_' \
                f'{"TrueEpsilon" if givenTrueEpsilon else ""}_{additionId}'
        if pin < 0 or pin > 1 or pout1 < 0 or pout1 > 1 or pout2 < 0 or pout2 > 1:
            # for delta let pin pout not in [0, 1], return (-1,0)
            result_data = (-1, 0)
        elif checkSNR:
            result_data = (0, 1)
        else:
            result_data = synthetic_exp_full2sub(msbm, givenNumGroup=givenNumGroup, DC=DC, BP=BP,
                                                 init_epsilon=init_epsilon, learnqby=learnqby, givenNacab=False,
                                                 strId=strId, writeCM=writeCM)
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


def run_exp(rhos, deltas, times, save_path=None, q=3, n=600, d=300, Z_s=None, Z_b=None, Withlambda=False,
            multiprocessing=True,
            givenNumGroup=False, DC=False, BP=False, learnqby=None, givenNacab=True, givenTrueEpsilon=False,
            writeCM=False,
            additionId="", Type=2, checkSNR=False):
    rho_delta_pair = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for row in f.readlines():
                row = row.strip().split()
                rho_delta_pair.add((round(float(row[0]), 5), round(float(row[1]), 5)))
    if multiprocessing:
        p = Pool(4)
        for rho in rhos:
            for delta in deltas:
                if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                    print(f'rho={rho}, delta={delta} has been run!')
                    continue
                p.apply_async(exp_subprocess,
                              args=(n, q, Z_s, Z_b, d, rho, delta, times, save_path, Withlambda, givenNumGroup,
                                    DC, BP, learnqby, givenNacab, givenTrueEpsilon, writeCM, additionId, Type,
                                    checkSNR),
                              callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for rho in rhos:
            for delta in deltas:
                if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                    print(f'rho={rho}, delta={delta} has been run!')
                    continue
                savepath, results = exp_subprocess(n, q, Z_s, Z_b, d, rho, delta, times, save_path, Withlambda,
                                                   givenNumGroup, DC,
                                                   BP, learnqby, givenNacab, givenTrueEpsilon, writeCM, additionId,
                                                   Type, checkSNR)
                write_results((savepath, results))


def read_exp(load_path, Withlambda=False, exclude_rho=None, exclude_z=None, add_paths=None):
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
        zs = np.setdiff1d(np.unique(results[:, 1]), np.array(exclude_z))
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
        Try homo average degree for different part
    """
    times = 5
    n = 6000
    d = 5
    Z_s = 2
    Z_b = 3
    q = Z_s + Z_b
    rho = np.setdiff1d(np.around(np.linspace(0, 0.5, 52), 2), np.array([0, 0.5]))
    Type = 2
    print("Selecting delta...")
    min_delta, max_delta = None, None
    for r in rho:
        for delta in np.setdiff1d(np.around(np.linspace(-1, 1, 10000), 6), np.array([])):
            _, pin, pout1, pout2 = get_ps(n, d, Z_s, Z_b, r, delta, Type)
            if 0 <= pin <= 1 and 0 <= pout1 <= 1 and 0 <= pout2 <= 1:
                min_delta = delta if min_delta is None or delta < min_delta else min_delta
                max_delta = delta if max_delta is None or delta > max_delta else max_delta
    min_delta = 0  # only consider assortative case
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, 60), 5), np.array([0]))
    print(delta)
    Withlambda = True
    givenNumGroup = False
    DC = False
    BP = False
    learnqby = None
    givenTrueEpsilon = False
    writeCM = False
    additionId = "2ndType"
    checkSNR = False
    multiprocessing = True
    fileID = 'amiExp24.5.14' + f'_n={n}_Zs={Z_s}_Zb={Z_b}_d={round(d)}_{"lambda" if Withlambda else ""}_' \
                              f'{"givenNumGroup" if givenNumGroup else ""}_' \
                              f'{"DC" if DC else ""}_{"BP" if BP else ""}_' \
                              f'{"givenTrueEpsilon" if givenTrueEpsilon else ""}_{"writeCM" if writeCM else ""}_' \
                              f'{"CheckSNR" if checkSNR else ""}_' \
                              f'{additionId}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withlambda={Withlambda}, givenNumberGroup={givenNumGroup}, '
          f'DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, q, n, d, Z_s, Z_b, Withlambda=Withlambda, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, BP=BP, learnqby=learnqby, givenTrueEpsilon=givenTrueEpsilon,
            writeCM=writeCM, additionId=additionId, Type=Type, checkSNR=checkSNR)


if __name__ == '__main__':
    exp0()
