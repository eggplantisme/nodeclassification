import os
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


def exp_subprocess(n, X, Z, d, rho, delta, times, Withsnr, savepath, givenNumGroup=False):
    pin = d / n + delta * (1 - 1 / (X * Z))
    pout = d / n - delta / (X * Z)
    pin = 0 if pin < 1e-10 else pin
    pout = 0 if pout < 1e-10 else pout
    # pout = 2 * Z * dout / n
    # pin = Z * pout * z + pout
    msbm = SymMetaSBM(n, X, Z, rho, pin, pout)
    results = ""
    for t in range(times):
        start = time.time()
        print(f"EXP pid={os.getpid()} begin... rho={rho}, delta={delta}, times={t}, pin={pin}, pout={pout}")
        A = msbm.sample()
        filterA, filterGroupId = msbm.filter(A, metaId=0)
        # givenNumGroup
        full_num_groups_given = np.size(np.unique(msbm.groupId))
        sub_num_groups_given = np.size(np.unique(filterGroupId))
        fullBHpartition_givenNumGroup, full_num_groups_given = CommunityDetect(A).BetheHessian(full_num_groups_given)
        subBHpartition_givenNumGroup, sub_num_groups_given = CommunityDetect(filterA).BetheHessian(sub_num_groups_given)
        full_ami_givenNumGroup = adjusted_mutual_info_score(msbm.groupId, fullBHpartition_givenNumGroup)
        sub_ami_givenNumGroup = adjusted_mutual_info_score(filterGroupId, subBHpartition_givenNumGroup)
        # else
        fullBHpartition, full_num_groups = CommunityDetect(A).BetheHessian()
        subBHpartition, sub_num_groups = CommunityDetect(filterA).BetheHessian()
        full_ami = adjusted_mutual_info_score(msbm.groupId, fullBHpartition)
        sub_ami = adjusted_mutual_info_score(filterGroupId, subBHpartition)
        if Withsnr and X == 2:
            snr_nm = msbm.snr(rho, delta, N=n, X=X, Z=Z, d=d, withMeta=False)
            snr_m = msbm.snr(rho, delta, N=n, X=X, Z=Z, d=d, withMeta=True)
            results += f'{rho} {delta} {t} {full_ami} {sub_ami} {snr_nm} {snr_m} {full_num_groups} {sub_num_groups} ' \
                       f'{full_ami_givenNumGroup} {sub_ami_givenNumGroup} {full_num_groups_given} {sub_num_groups_given}\n'
            print(f"EXP pid={os.getpid()} end. full_ami={full_ami}, sub_ami={sub_ami}. snr_nm={snr_nm}, snr_m={snr_m}. "
                  f"Time:{np.around(time.time()-start, 3)}")
        else:
            results += f'{rho} {delta} {t} {full_ami} {sub_ami} {full_num_groups} {sub_num_groups} ' \
                       f'{full_ami_givenNumGroup} {sub_ami_givenNumGroup} {full_num_groups_given} {sub_num_groups_given}\n'
            print(f"EXP pid={os.getpid()} end. full_ami={full_ami}, sub_ami={sub_ami}. "
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
            givenNumGroup=False):
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
                p.apply_async(exp_subprocess, args=(n, X, Z, d, rho, delta, times, Withsnr, save_path, givenNumGroup, ),
                              callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for rho in rhos:
            for delta in deltas:
                if (round(rho, 5), round(delta, 5)) in rho_delta_pair:
                    print(f'rho={rho}, delta={delta} has been run!')
                    continue
                savepath, results = exp_subprocess(n, X, Z, d, rho, delta, times, Withsnr, save_path, givenNumGroup)
                write_results((savepath, results))


def read_exp(load_path, Withsnr=False, givenNumGroup=False):
    with open(load_path, 'r') as f:
        results = np.round(np.float64([row.strip().split() for row in f.readlines()]), decimals=5)
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
                    if givenNumGroup:
                        full_ami_givenNumGroup[i] = mean_ami[6]
                        sub_ami_givenNumGroup[i] = mean_ami[7]
                        full_num_groups_given[i] = mean_ami[8]
                        sub_num_groups_given[i] = mean_ami[9]
                else:
                    full_num_group[i] = mean_ami[2]
                    sub_num_group[i] = mean_ami[3]
                    if givenNumGroup:
                        full_ami_givenNumGroup[i] = mean_ami[4]
                        sub_ami_givenNumGroup[i] = mean_ami[5]
                        full_num_groups_given[i] = mean_ami[6]
                        sub_num_groups_given[i] = mean_ami[7]
                i += 1
        plot_rhos = np.repeat(rhos, np.size(zs))
        plot_zs = np.tile(zs, np.size(rhos))
    return plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group, \
           full_ami_givenNumGroup, sub_ami_givenNumGroup, full_num_groups_given, sub_num_groups_given


def exp2_subprocess(d, X, Z, n, times, fileId, Withsnr=False, givenNumGroup=False, delta=None):
    min_delta, max_delta = get_range_delta(d, n, X, Z)
    rho = np.setdiff1d(np.around(np.linspace(0, 1, 51), 2), np.array([]))
    if delta is None:
        delta = np.setdiff1d(np.around(np.linspace(min_delta, max_delta, int((max_delta-min_delta)/0.01)+1), 5),
                             np.array([0]))
    fileID = fileId
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for file={fileID} size={np.size(rho) * np.size(delta)}", f'min_delta={min_delta}, '
            f'max_delta={max_delta}, Withsnr={Withsnr}, givenNumberGroup={givenNumGroup}')
    run_exp(rho, delta, times, save_path, X, Z, n, d, Withsnr=Withsnr, givenNumGroup=givenNumGroup)


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


def debug():
    # For big n
    X = 2
    Z = 3
    n = X * Z * 2000
    d = 300
    Withsnr = True
    givenNumGroup = False
    fileId = 'amiExp5.8' + f'_n={n}_X={X}_Z={Z}_d={round(d)}_{"snr" if Withsnr else ""}' \
                           f'{"_givenNumGroup" if givenNumGroup else ""}'
    load_path = "./result/detectabilityWithMeta/" + fileId + ".txt"
    plot_rhos, plot_zs, full_ami, sub_ami, snr_nm, snr_m, full_num_group, sub_num_group = read_exp(load_path=load_path,
                                                                                                   Withsnr=Withsnr)
    print(f'min delta={np.min(plot_zs)}, max delta={np.max(plot_zs)}')
    # save_path = "./_Figure/AMI_fullgraph_d_" + str(d) + ".png"
    save_path = None


if __name__ == '__main__':
    # exp2()
    # debug()
    # exp3()
    # exp4()
    # exp5()
    # exp6()
    exp7()
