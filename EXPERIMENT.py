import os

import numpy as np

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from _CommunityDetect import *
from multiprocessing import Pool


def print_error(value):
    print(value)


def get_range_delta(d, n, X, Z):
    if 0 < d < n / (X * Z):
        min_delta = X * Z * d / ((1 - X * Z) * n)
        max_delta = X * Z * d / n
    elif n / (X * Z) <= d < (1 - 1 / (X * Z)) * n:
        min_delta = X * Z * d / ((1 - X * Z) * n)
        max_delta = (X * Z / (1 - X * Z)) * (d / n - 1)
    elif (1 - 1 / (X * Z)) * n <= d < n:
        min_delta = X * Z * (d / n - 1)
        max_delta = (X * Z / (1 - X * Z)) * (d / n - 1)
    else:
        min_delta, max_delta = None, None
    return min_delta, max_delta


def synthetic_exp_full2full(msbm, HelpWithFull=False):
    result_data = []
    A = msbm.sample()
    num_of_group = np.size(np.unique(msbm.groupId))
    fullBHpartition, full_num_groups = CommunityDetect(A).BetheHessian()
    # 1. propagation from all subgraph and see the number of group detected
    partitionConcatenatedBysub = -1 + np.zeros(msbm.N)
    num_partition = 0
    for metaId in np.unique(msbm.metaId):
        filterA, filterGroupId = msbm.filter(A, metaId=metaId)
        if HelpWithFull is False:
            subBHpartition, _ = CommunityDetect(filterA).BetheHessian()
        else:
            help_evec, fullBHpartition, full_num_groups = CommunityDetect(A).BetheHessian(return_evec=True)
            meta0Index = np.where(msbm.metaId == metaId)[0]
            help_evec = help_evec[meta0Index, :]
            help_num_groups = np.size(np.unique(fullBHpartition[meta0Index]))
            subBHpartition, sub_num_groups = CommunityDetect(filterA).BetheHessian(help_evec=help_evec,
                                                                                   help_num_groups=help_num_groups)
        subBHpartition = num_partition + subBHpartition
        num_partition += np.size(np.unique(subBHpartition))
        metaIndex = np.where(msbm.metaId == metaId)[0]
        for i, p in enumerate(subBHpartition):
            partitionConcatenatedBysub[metaIndex[i]] = p
    B = np.zeros((msbm.N, num_partition))
    for i, p in enumerate(partitionConcatenatedBysub):
        if p == -1:
            B[i, :] = B[i, :] + 1 / num_partition
        else:
            B[i, int(p)] = 1
    subPropagationPartition, full_num_groups_propagate = CommunityDetect(A).TwoStepLabelPropagate(B, operator_name='W^2')
    if full_num_groups_propagate > num_of_group:
        print(f'All subgraph propagation get {full_num_groups_propagate} more than True {num_of_group} communities')
        # 2. if "the number of group detected by propagation from all subgraph" > # of group from BH,
        # then propagate from 1st subgraph
        metaId = 0
        filterA, filterGroupId = msbm.filter(A, metaId=metaId)
        if HelpWithFull is False:
            subBHpartition, _ = CommunityDetect(filterA).BetheHessian()
        else:
            help_evec, fullBHpartition, full_num_groups = CommunityDetect(A).BetheHessian(return_evec=True)
            meta0Index = np.where(msbm.metaId == metaId)[0]
            help_evec = help_evec[meta0Index, :]
            help_num_groups = np.size(np.unique(fullBHpartition[meta0Index]))
            subBHpartition, sub_num_groups = CommunityDetect(filterA).BetheHessian(help_evec=help_evec,
                                                                                   help_num_groups=help_num_groups)
        fullHalfPartition = -1 + np.zeros(msbm.N)
        metaIndex = np.where(msbm.metaId == metaId)[0]
        for i, p in enumerate(subBHpartition):
            fullHalfPartition[metaIndex[i]] = p
        # Here set real number of group. If want number of group in subgraph, set np.size(np.unique(subBHpartition))
        B = np.zeros((msbm.N, num_of_group))
        for i, p in enumerate(fullHalfPartition):
            if p == -1:
                B[i, :] = B[i, :] + 1 / num_of_group
            else:
                B[i, int(p)] = 1
        subPropagationPartition, full_num_groups_propagate = CommunityDetect(A).TwoStepLabelPropagate(B, operator_name='W^2')
    full_ami = adjusted_mutual_info_score(msbm.groupId, fullBHpartition)
    full_ami_propagate = adjusted_mutual_info_score(msbm.groupId, subPropagationPartition)
    result_data.append(full_ami)
    result_data.append(full_ami_propagate)
    result_data.append(full_num_groups)
    result_data.append(full_num_groups_propagate)
    return result_data

def synthetic_exp_full2sub(msbm, givenNumGroup=True, DC=False, HelpWithFull=False, BP=False, oriBP=False):
    A = msbm.sample()
    metaIdSelect = 0
    filterA, filterGroupId = msbm.filter(A, metaId=metaIdSelect)
    result_data = []
    if not givenNumGroup:
        if DC is False:
            if HelpWithFull is False:
                fullpartition, full_num_groups = CommunityDetect(A).BetheHessian()
                subpartition, sub_num_groups = CommunityDetect(filterA).BetheHessian()
            else:
                help_evec, fullpartition, full_num_groups = CommunityDetect(A).BetheHessian(return_evec=True)
                meta0Index = np.where(msbm.metaId == 0)[0]
                help_evec = help_evec[meta0Index, :]
                help_num_groups = np.size(np.unique(fullpartition[meta0Index]))
                subpartition, sub_num_groups = CommunityDetect(filterA).BetheHessian(help_evec=help_evec,
                                                                                       help_num_groups=help_num_groups)
        else:
            if HelpWithFull is False:
                fullpartition, full_num_groups, full_zetas = CommunityDetect(A).DCBetheHessian()
                subpartition, sub_num_groups, sub_zetas = CommunityDetect(filterA).DCBetheHessian()
            else:
                help_evec, fullpartition, full_num_groups = CommunityDetect(A).DCBetheHessian(return_evec=True)
                meta0Index = np.where(msbm.metaId == 0)[0]
                help_evec = help_evec[meta0Index, :]
                help_num_groups = np.size(np.unique(fullpartition[meta0Index]))
                subpartition, sub_num_groups = CommunityDetect(filterA).DCBetheHessian(help_evec=help_evec,
                                                                                       help_num_groups=help_num_groups)
    else:
        full_num_groups_given = np.size(np.unique(msbm.groupId))
        sub_num_groups_given = np.size(np.unique(filterGroupId))
        if BP is True:
            pid = os.getpid()
            # FULL
            na = (np.array(msbm.sizes).flatten() / msbm.N).tolist() 
            cab = []
            for x in np.nditer(msbm.ps):
                cab.append(np.around(x*msbm.N, 2))
            fullpartition = CommunityDetect(A).BP_meta(full_num_groups_given, na, cab, msbm.rho, msbm.groupId, msbm.metaId, processId=str(pid)+"FULL")
            full_num_groups = full_num_groups_given
            # SUB BP_with_meta can't be used in subgraph, only select sub part in fullpartition
            metaIndex = np.where(msbm.metaId == metaIdSelect)[0]
            subpartition = np.copy(fullpartition[metaIndex])
            sub_num_groups = np.size(np.unique(subpartition))
            # Below is subgraph detected by BP_with_meta
            # subsize = np.size(filterGroupId)
            # v, counts = np.unique(filterGroupId, return_counts=True)
            # na = counts / np.size(filterGroupId)
            # cab = []
            # for x in np.nditer(msbm.ps[v, :][:, v]):
            #     cab.append(np.around(x*subsize, 2))
            # subpartition = CommunityDetect(filterA).BP(sub_num_groups_given, na, cab, msbm.rho, filterGroupId, [metaIdSelect] * subsize, processId=str(pid)+"SUB")
            # sub_num_groups = sub_num_groups_given
        elif oriBP is True:
            pid = os.getpid()
            # FULL
            na = (np.array(msbm.sizes).flatten() / msbm.N).tolist() 
            cab = []
            for x in np.nditer(msbm.ps):
                cab.append(np.around(x*msbm.N, 2))
            fullpartition = CommunityDetect(A).BP(full_num_groups_given, na, cab, msbm.groupId, processId=str(pid)+"FULL")
            full_num_groups = full_num_groups_given
            # SUB BP
            subsize = np.size(filterGroupId)
            v, counts = np.unique(filterGroupId, return_counts=True)
            na = counts / np.size(filterGroupId)
            cab = []
            for x in np.nditer(msbm.ps[v, :][:, v]):
                cab.append(np.around(x*subsize, 2))
            subpartition = CommunityDetect(filterA).BP(sub_num_groups_given, na, cab, filterGroupId, processId=str(pid)+"SUB")
            sub_num_groups = sub_num_groups_given
        elif DC is False:
            fullpartition, full_num_groups = CommunityDetect(A).BetheHessian(full_num_groups_given)
            subpartition, sub_num_groups = CommunityDetect(filterA).BetheHessian(sub_num_groups_given)
        else:
            fullpartition, full_num_groups, full_zetas = CommunityDetect(A).DCBetheHessian(full_num_groups_given)
            subpartition, sub_num_groups, sub_zetas = CommunityDetect(filterA).DCBetheHessian(sub_num_groups_given)
    full_ami = adjusted_mutual_info_score(msbm.groupId, fullpartition)
    sub_ami = adjusted_mutual_info_score(filterGroupId, subpartition)
    result_data.append(full_ami)
    result_data.append(sub_ami)
    result_data.append(full_num_groups)
    result_data.append(sub_num_groups)
    if DC:
        result_data.append(full_zetas)
        result_data.append(sub_zetas)
    return result_data


def exp_subprocess(n, X, Z, d, rho, delta, times, savepath, Withsnr=True, givenNumGroup=False, DC=False,
                   HelpWithFull=False, LabelPropagate=False, BP=False, oriBP=False):
    pin = d / n + delta * (1 - 1 / (X * Z))
    pout = d / n - delta / (X * Z)
    pin = 0 if pin < 1e-10 else pin
    pout = 0 if pout < 1e-10 else pout
    if BP or oriBP:
        # For BP not use two small pin and pout
        pin = 1e-5 if pin < 1e-5 else pin
        pout = 1e-5 if pout < 1e-5 else pout
    # pout = 2 * Z * dout / n
    # pin = Z * pout * z + pout
    msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    results = ""
    # Calc snr first
    if Withsnr and X == 2:
        snr_nm = msbm.snr(rho, delta, N=n, X=X, Z=Z, d=d, withMeta=False)
        snr_m = msbm.snr(rho, delta, N=n, X=X, Z=Z, d=d, withMeta=True)
    for t in range(times):
        start = time.time()
        print(f"EXP pid={os.getpid()} begin... rho={rho}, delta={delta}, times={t}, pin={pin}, pout={pout}")
        if LabelPropagate is False:
            result_data = synthetic_exp_full2sub(msbm, givenNumGroup=givenNumGroup, DC=DC, HelpWithFull=HelpWithFull, BP=BP, oriBP=oriBP)
            full_ami, sub_ami, full_num_groups, sub_num_groups = result_data[0], result_data[1], result_data[2], result_data[3]
            if DC:
                full_zetas, sub_zetas = result_data[4], result_data[5]
            # combine results str
            results += f'{rho} {delta} {t} {full_ami} {sub_ami} '
            if Withsnr and X == 2:
                results += f'{snr_nm} {snr_m} '
            results += f'{full_num_groups} {sub_num_groups}'
            if DC:
                results += f' {full_zetas[0]}'
                for zeta in full_zetas[1:]:
                    results += f' {zeta}'
                results += f' {sub_zetas[0]}'
                for zeta in sub_zetas[1:]:
                    results += f' {zeta}'
            results += '\n'
            print(f"EXP pid={os.getpid()} end. full_ami={full_ami}, sub_ami={sub_ami}, "
                  f"Time:{np.around(time.time() - start, 3)}")
        else:
            result_data = synthetic_exp_full2full(msbm, HelpWithFull=HelpWithFull)
            full_ami, full_ami_propagate, full_num_groups, full_num_groups_propagate = result_data[0], result_data[1], \
                result_data[2], result_data[3]
            results += f'{rho} {delta} {t} {full_ami} {full_ami_propagate} '
            if Withsnr and X == 2:
                results += f'{snr_nm} {snr_m} '
            results += f'{full_num_groups} {full_num_groups_propagate}'
            results += '\n'
            print(f"EXP pid={os.getpid()} end. full_ami={full_ami}, full_ami from propagate={full_ami_propagate}, "
                  f"Time:{np.around(time.time() - start, 3)}")
    return savepath, results


def write_results(arg):
    """
    :param arg: savepath, results
    :return:
    """
    if arg[0] is not None:
        with open(arg[0], 'a') as fw:
            fw.write(arg[1])


def run_exp(rhos, deltas, times, save_path=None, X=2, Z=3, n=600, d=300, Withsnr=False, multiprocessing=True,
            givenNumGroup=False, DC=False, HelpWithFull=False, LabelPropagate=False, BP=False, oriBP=False):
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
                p.apply_async(exp_subprocess, args=(n, X, Z, d, rho, delta, times, save_path, Withsnr, givenNumGroup,
                                                    DC, HelpWithFull, LabelPropagate, BP, oriBP),
                              callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for rho in rhos:
            for delta in deltas:
                if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                    print(f'rho={rho}, delta={delta} has been run!')
                    continue
                savepath, results = exp_subprocess(n, X, Z, d, rho, delta, times, save_path, Withsnr, givenNumGroup, DC,
                                                   HelpWithFull, LabelPropagate, BP, oriBP)
                write_results((savepath, results))


def read_exp(load_path, Withsnr=False, givenNumGroup=False, old=False):
    """
    read the results file from run_exp
    :param load_path:
    :param Withsnr:
    :param givenNumGroup:
    :param old: in old version, the result for givenNumGroup is in the same file with noGivenNumGroup, for amiExp5.17
    :return:
    """
    with open(load_path, 'r') as f:
        # for row in f.readlines():
        #     print(row.strip().split()[:9])
        if givenNumGroup and old:
            results = np.round(np.float64([row.strip().split()[:13] for row in f.readlines()]), decimals=5)
        else:
            results = np.round(np.float64([row.strip().split()[:9] for row in f.readlines()]), decimals=5)
        rhos = np.unique(results[:, 0])
        zs = np.unique(results[:, 1])
        full_ami = np.zeros(np.size(zs) * np.size(rhos))
        sub_ami = np.zeros(np.size(zs) * np.size(rhos))
        snr_nm = np.zeros(np.size(zs) * np.size(rhos)) if Withsnr else None
        snr_m = np.zeros(np.size(zs) * np.size(rhos)) if Withsnr else None
        full_num_group = np.zeros(np.size(zs) * np.size(rhos))
        sub_num_group = np.zeros(np.size(zs) * np.size(rhos))
        full_ami_givenNumGroup = np.zeros(np.size(zs) * np.size(rhos))
        sub_ami_givenNumGroup = np.zeros(np.size(zs) * np.size(rhos))
        full_num_groups_given = np.zeros(np.size(zs) * np.size(rhos))
        sub_num_groups_given = np.zeros(np.size(zs) * np.size(rhos))
        i = 0
        for _rho in rhos:
            for _z in zs:
                ami_results = results[np.squeeze(np.argwhere(np.logical_and(results[:, 0]==_rho, results[:, 1]==_z)))]
                if np.size(ami_results) == 0:
                    print("Some parameter didn't run!")
                mean_ami = np.mean(ami_results, 0)[3:]
                full_ami[i] = mean_ami[0]
                sub_ami[i] = mean_ami[1]
                if Withsnr:
                    snr_nm[i] = mean_ami[2]
                    snr_m[i] = mean_ami[3]
                    full_num_group[i] = mean_ami[4]
                    sub_num_group[i] = mean_ami[5]
                    if givenNumGroup and old:
                        full_ami_givenNumGroup[i] = mean_ami[6]
                        sub_ami_givenNumGroup[i] = mean_ami[7]
                        full_num_groups_given[i] = mean_ami[8]
                        sub_num_groups_given[i] = mean_ami[9]
                else:
                    full_num_group[i] = mean_ami[2]
                    sub_num_group[i] = mean_ami[3]
                    if givenNumGroup and old:
                        full_ami_givenNumGroup[i] = mean_ami[4]
                        sub_ami_givenNumGroup[i] = mean_ami[5]
                        full_num_groups_given[i] = mean_ami[6]
                        sub_num_groups_given[i] = mean_ami[7]
                i += 1
        plot_rhos = np.repeat(rhos, np.size(zs))
        plot_zs = np.tile(zs, np.size(rhos))
    if old:
        return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group, full_ami_givenNumGroup, sub_ami_givenNumGroup, full_num_groups_given, sub_num_groups_given
    else:
        return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group


def exp2_subprocess(d, X, Z, n, times, fileId, Withsnr=False, givenNumGroup=False, delta=None, step=0.01, rho=None, DC=False, HelpWithFull=False, multiprocessing=True, LabelPropagate=False, BP=False, oriBP=False):
    min_delta, max_delta = get_range_delta(d, n, X, Z)
    if rho is None:
        rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    if delta is None:
        delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, int((max_delta-min_delta)/step)+1), 5),
                             np.array([0]))
        # min_delta, max_delta = get_range_delta(d, n, X, Z)
        print(f"Delta in this case is from {min_delta} to {max_delta}, the size with step {step} is {np.size(delta)}")
    else:
        min_delta, max_delta = np.min(delta), np.max(delta)
    fileID = fileId
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta) * times}",
          f'min_delta={min_delta} max_delta={max_delta}, Withsnr={Withsnr}, givenNumberGroup={givenNumGroup}, DC={DC}, BP={BP}')
    run_exp(rho, delta, times, save_path, X, Z, n, d, Withsnr=Withsnr, multiprocessing=multiprocessing,
            givenNumGroup=givenNumGroup, DC=DC, HelpWithFull=HelpWithFull, LabelPropagate=LabelPropagate, BP=BP, oriBP=oriBP)


def exp2():
    times = 10
    X = 2
    Z = 3
    n = 600
    # ds = np.linspace(n/(2*Z), (1-1/(2*Z))*n, int(((1-1/(2*Z))*n-n/(2*Z))/50)+1)
    ds = np.array([300])
    p = Pool(1)
    for d in ds:
        fileID = 'amiExp4.24' + f'_d={d}'
        p.apply_async(exp2_subprocess, args=(d, X, Z, n, times, fileID, ), error_callback=print_error)
    p.close()
    p.join()
    print('All subprocesses done.')


def exp3():
    """ TEST for X > 2 """
    times = 10
    X = 3
    Z = 3
    n = 3**6
    d = 300
    fileId = 'amiExp4.24' + f'_d={d}_X={X}_n={n}'
    exp2_subprocess(d, X, Z, n, times, fileId)


def exp4():
    """ For big network size n """
    times = 10
    X = 2
    Z = 3
    n = X*Z*500
    d = n / 2
    Withsnr = True
    fileID = 'amiExp5.4' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}'
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr)
    print('All subprocesses done.')


def exp5():
    """ For more big network size n """
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 300
    Withsnr = True
    givenNumGroup = False
    fileID = 'amiExp5.8' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.03, 0.03, int(0.06 / 0.002) + 1), 5), np.array([0]))
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta)
    print('All subprocesses done.')


def exp6():
    """ For more big network size n """
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 300
    Withsnr = True
    givenNumGroup = True
    fileID = 'amiExp5.8' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_{"givenNumGroup" if givenNumGroup else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.03, 0.03, int(0.06 / 0.002) + 1), 5), np.array([0]))
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta)
    print('All subprocesses done.')


def exp7():
    """ Smaller average degree"""
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = True
    fileID = 'amiExp5.17' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.001) + 1), 5), np.array([0]))
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta)
    print('All subprocesses done.')


def exp8():
    """ Smaller average degree"""
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = True
    fileID = 'amiExp5.25' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_{"givenNumGroup" if givenNumGroup else ""}_{"DC" if DC else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.001) + 1), 5), np.array([0]))
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, DC=DC, multiprocessing=False)
    print('All subprocesses done.')


def exp9():
    """ subgraph Help With Full"""
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = False
    HelpWithFull = True
    fileID = 'amiExp6.12' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, DC=DC,
                    HelpWithFull=HelpWithFull, multiprocessing=False)
    print('All subprocesses done.')


def exp10():
    """ compare full_ami and full_ami from sub_propagation"""
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = False
    HelpWithFull = False
    LabelPropagate = True
    fileID = 'amiExp6.12' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}_' \
                            f'{"LabelPropagate" if LabelPropagate else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, DC=DC,
                    HelpWithFull=HelpWithFull, multiprocessing=False, LabelPropagate=LabelPropagate)
    print('All subprocesses done.')


def exp11():
    """ compare full_ami and full_ami from sub_propagation which sub result from helping with fullgraph"""
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = False
    HelpWithFull = True
    LabelPropagate = True
    fileID = 'amiExp6.21' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}_' \
                            f'{"LabelPropagate" if LabelPropagate else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, DC=DC,
                    HelpWithFull=HelpWithFull, multiprocessing=False, LabelPropagate=LabelPropagate)
    print('All subprocesses done.')


def exp12():
    times = 40
    X = 2
    Z = 3
    n = X * Z * 2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = False
    HelpWithFull = False
    LabelPropagate = False
    multiprocessing = True
    for HelpWithFull, LabelPropagate in [(False, False)]: #[(False, False), (True, False), (False, True), (True, True)]:
        fileID = 'amiExp8.8' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' + \
                 f'{"givenNumGroup" if givenNumGroup else ""}_' + \
                 f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}_' + \
                 f'{"LabelPropagate" if LabelPropagate else ""}' + '_add'
        delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
        # delta = delta[:21]
        # delta = None
        exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, DC=DC,
                        HelpWithFull=HelpWithFull, multiprocessing=multiprocessing, LabelPropagate=LabelPropagate)
    print('All subprocesses done.')


def exp13():
    times = 10
    X = 2
    Z = 3
    n = X * Z * 2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = True
    BP = True
    multiprocessing = True
    fileID = 'amiExp10.9' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' + \
                 f'{"givenNumGroup" if givenNumGroup else ""}_' + \
                 f'{"BP" if BP else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = delta[:21]
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, 
                    delta=delta, multiprocessing=multiprocessing, BP=BP)


def exp14():
    times = 10
    X = 2
    Z = 3
    n = X * Z * 2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = True
    BP = True
    multiprocessing = True
    fileID = 'amiExp10.28' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' + \
                 f'{"givenNumGroup" if givenNumGroup else ""}_' + \
                 f'{"BP" if BP else ""}_adjustrho'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = delta[:21]
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, 
                    delta=delta, multiprocessing=multiprocessing, BP=BP)


def exp15():
    times = 50
    X = 2
    Z = 3
    n = X * Z * 2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = False
    HelpWithFull = False
    LabelPropagate = False
    multiprocessing = True
    for HelpWithFull, LabelPropagate in [(False, False)]: #[(False, False), (True, False), (False, True), (True, True)]:
        fileID = 'amiExp11.1' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' + \
                 f'{"givenNumGroup" if givenNumGroup else ""}_' + \
                 f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}_' + \
                 f'{"LabelPropagate" if LabelPropagate else ""}' + '_add_exactpart'
        delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
        delta = delta[38:]  # 0.0145~0.025
        # delta = None
        exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, DC=DC,
                        HelpWithFull=HelpWithFull, multiprocessing=multiprocessing, LabelPropagate=LabelPropagate)
    print('All subprocesses done.')


def exp16():
    """ compare full_ami and full_ami from sub_propagation"""
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = False
    HelpWithFull = False
    LabelPropagate = True
    multiprocessing = True
    fileID = 'amiExp11.5' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}_' \
                            f'{"LabelPropagate" if LabelPropagate else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, DC=DC,
                    HelpWithFull=HelpWithFull, multiprocessing=multiprocessing, LabelPropagate=LabelPropagate)
    print('All subprocesses done.')


def exp17():
    """ compare full_ami and full_ami from sub_propagation"""
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = False
    HelpWithFull = False
    LabelPropagate = True
    multiprocessing = True
    fileID = 'amiExp11.9' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}_' \
                            f'{"LabelPropagate" if LabelPropagate else ""}_somebaseon1sub'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, DC=DC,
                    HelpWithFull=HelpWithFull, multiprocessing=multiprocessing, LabelPropagate=LabelPropagate)
    print('All subprocesses done.')


def exp18():
    times = 10
    X = 2
    Z = 3
    n = X * Z * 2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = True
    BP = True
    multiprocessing = True
    fileID = 'amiExp11.14' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' + \
                 f'{"givenNumGroup" if givenNumGroup else ""}_' + \
                 f'{"BP" if BP else ""}_corrected'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = delta[:21]
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, 
                    delta=delta, multiprocessing=multiprocessing, BP=BP)


def exp19():
    """ compare full_ami and full_ami from sub_propagation"""
    times = 10
    X = 2
    Z = 3
    n = X*Z*2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = False
    HelpWithFull = False
    LabelPropagate = True
    multiprocessing = True
    fileID = 'amiExp11.18' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}_' \
                            f'{"LabelPropagate" if LabelPropagate else ""}_somebaseon1sub_W2'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, DC=DC,
                    HelpWithFull=HelpWithFull, multiprocessing=multiprocessing, LabelPropagate=LabelPropagate)
    print('All subprocesses done.')


def exp20():
    """ A simple network with 2meta and 2community"""
    times = 10
    X = 2
    Z = 1
    n = X * Z * 2000  # 4000 nodes
    d = 50
    Withsnr = False
    givenNumGroup = False
    DC = False
    HelpWithFull = False
    LabelPropagate = False
    multiprocessing = True
    fileID = 'amiExp11.21' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}_' \
                            f'{"LabelPropagate" if LabelPropagate else ""}'
    min_delta, max_delta = get_range_delta(d=d, n=n, X=X, Z=Z)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, int((max_delta-min_delta) / 0.0005) + 1), 5), np.array([min_delta, 0]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([0, 1]))
    # print(f'delta number is {np.size(delta)}')
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, rho=rho, DC=DC,
                    HelpWithFull=HelpWithFull, multiprocessing=multiprocessing, LabelPropagate=LabelPropagate)
    print('All subprocesses done.')


def exp21():
    times = 10
    X = 2
    Z = 3
    n = X * Z * 2000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = True
    oriBP = True
    multiprocessing = True
    fileID = 'amiExp11.14' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' + \
                 f'{"givenNumGroup" if givenNumGroup else ""}_' + \
                 f'{"oriBP" if oriBP else ""}'
    delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = delta[:21]
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, 
                    delta=delta, multiprocessing=multiprocessing, oriBP=oriBP)


def exp22():
    """ A simple network with 2meta and 2community"""
    times = 10
    X = 2
    Z = 1
    n = X * Z * 2000  # 4000 nodes
    d = 50
    Withsnr = False
    givenNumGroup = False
    DC = False
    HelpWithFull = False
    LabelPropagate = False
    oriBP = True
    multiprocessing = True
    fileID = 'amiExp12.07' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' \
                            f'{"givenNumGroup" if givenNumGroup else ""}_' \
                            f'{"oriBP" if oriBP else ""}'
    min_delta, max_delta = get_range_delta(d=d, n=n, X=X, Z=Z)
    delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, int((max_delta-min_delta) / 0.0005) + 1), 5), np.array([min_delta, 0]))
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([0, 1]))
    # print(f'delta number is {np.size(delta)}')
    # delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, rho=rho, DC=DC,
                    HelpWithFull=HelpWithFull, multiprocessing=multiprocessing, LabelPropagate=LabelPropagate, oriBP=oriBP)
    print('All subprocesses done.')


def exp23():
    times = 10
    X = 2
    Z = 2
    n = X * Z * 3000  # 12000 nodes
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = False
    HelpWithFull = False
    LabelPropagate = False
    multiprocessing = True
    fileID = 'amiExp12.22' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_' + \
                f'{"givenNumGroup" if givenNumGroup else ""}_' + \
                f'{"DC" if DC else ""}_{"HelpWithFull" if HelpWithFull else ""}_' + \
                f'{"LabelPropagate" if LabelPropagate else ""}'
    # delta = np.setdiff1d(np.around(np.linspace(-0.005, 0.025, int(0.03 / 0.0005) + 1), 5), np.array([0]))
    # delta = delta[38:]  # 0.0145~0.025
    delta = None
    exp2_subprocess(d, X, Z, n, times, fileID, Withsnr=Withsnr, givenNumGroup=givenNumGroup, delta=delta, step=0.0003, DC=DC,
                    HelpWithFull=HelpWithFull, multiprocessing=multiprocessing, LabelPropagate=LabelPropagate)
    print('All subprocesses done.')


def debug():
    # For big n
    # X = 2
    # Z = 3
    # n = X * Z * 2000
    # d = 50
    # Withsnr = True
    # givenNumGroup = False
    # rho = 0
    # delta = 0.025
    # exp_subprocess(n, X, Z, d, rho, delta, times=1, savepath=None, Withsnr=True, givenNumGroup=False, DC=True)
    # fileId = 'amiExp5.8' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}' \
    #                        f'{"_givenNumGroup" if givenNumGroup else ""}'
    # load_path = "./result/detectabilityWithMeta/" + fileId + ".txt"
    # plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group = read_exp(load_path=load_path,
    #                                                                                                Withsnr=Withsnr)
    # print(f'min delta={np.min(plot_zs)}, max delta={np.max(plot_zs)}')
    # save_path = "./_Figure/AMI_fullgraph_d_" + str(d) + ".png"
    X = 2
    Z = 3
    n = X * Z * 2000
    d = 50
    Withsnr = True
    givenNumGroup = True
    fileId = 'amiExp5.17' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}'
    load_path = "./result/detectabilityWithMeta/" + fileId + ".txt"
    plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group, full_ami_givenNumGroup, sub_ami_givenNumGroup, full_num_groups_given, sub_num_groups_given = read_exp(
        load_path=load_path, Withsnr=Withsnr, givenNumGroup=givenNumGroup, old=True)

    save_path = None
    X = 2
    Z = 3
    n = X * Z * 2000
    d = 50
    Withsnr = True
    givenNumGroup = False
    DC = True
    fileId = 'amiExp5.25' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}_{"givenNumGroup" if givenNumGroup else ""}_{"DC" if DC else ""}'
    load_path = "./result/detectabilityWithMeta/" + fileId + ".txt"
    plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group = read_exp(load_path=load_path,
                                                                                                   Withsnr=Withsnr)


if __name__ == '__main__':
    # exp2()
    # debug()
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
    # exp19()
    # exp20()
    # exp21()
    # exp22()
    exp23()
