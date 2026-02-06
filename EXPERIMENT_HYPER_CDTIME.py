from EXPERIMENT_HYPER import *

class ExperimentHyperCDTime:
    def __init__(self, n, q, d, Ks, epsilon):
        self.n = n
        self.q = q
        self.d = d
        self.Ks = Ks
        self.epsilon = epsilon
        self.hsbm = None
        self.generate()
    
    def generate(self):
        sizes = [int(self.n / self.q)] * self.q
        ps_dict = dict()
        temp = 0
        for k in self.Ks:
            temp += self.q * comb(int(self.n/self.q), k) * k / (self.n**k) + self.epsilon * (comb(self.n, k) - self.q * comb(int(self.n/self.q), k)) * k / (self.n**k)
        cin = self.d / temp
        cout = self.epsilon * cin
        if len(self.Ks) > 1:
            self.hsbm = UnUniformSymmetricHSBM(self.n, self.q, self.Ks, cin, cout)
        elif len(self.Ks) == 1:
            self.hsbm = UniformSymmetricHSBM(self.n, self.q, self.Ks[0], cin, cout)
    
def exp_subprocess(exp_cdtime, t, save_path, method):
    results = ''
    # Community Detection
    for m in method:
        if m == "HyBH":
            result = CDwithBH(exp_cdtime.hsbm)
            results += f'{m} {exp_cdtime.n} {exp_cdtime.q} {exp_cdtime.d} {exp_cdtime.Ks} {exp_cdtime.epsilon} {t} {result[0]} {result[1]} {result[2]}\n'
        elif m == "HyBP":
            hypergraph_save_name = f'amiexp_n={exp_cdtime.n}_q={exp_cdtime.q}_d={exp_cdtime.d}_Ks={exp_cdtime.Ks}_epsilon={exp_cdtime.epsilon}_times={t}'
            result = CDwithBP(exp_cdtime.hsbm, hypergraph_save_name=hypergraph_save_name)
            results += f'{m} {exp_cdtime.n} {exp_cdtime.q} {exp_cdtime.d} {exp_cdtime.Ks} {exp_cdtime.epsilon} {t} {result[0]} {result[1]} {result[2]}\n'
        else:
            pass
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

def run_exp(parameters, times, save_path=None, method=["HyBH"], multiprocessing=True):
    n = parameters['n']
    q = parameters['q']
    d = parameters['d']
    Ks = parameters['Ks']
    epsilon = parameters['epsilon']
    if multiprocessing:
        p = Pool(4)
        for t in range(times):
            exp_cdtime = ExperimentHyperCDTime(n, q, d, Ks, epsilon)
            p.apply_async(exp_subprocess, args=(exp_cdtime, t, save_path, method ), 
                          callback=write_results, error_callback=print_error)
        p.close()
        p.join()
    else:
        for t in range(times):
            exp_cdtime = ExperimentHyperCDTime(n, q, d, Ks, epsilon)
            savepath, results = exp_subprocess(exp_cdtime, t, save_path, method)
            write_results((savepath, results))

def exp0():
    ns = [500, 1000, 5000, 10000, 30000]  # 50, 100, 
    for n in ns:
        parameters = {
            'n': n,
            'q': 3,
            'd': 10,
            'Ks': [2, 3],
            'epsilon': 0.2
        }
        times = 5  # 10
        methods = ["HyBH", "HyBP"]
        save_path = f'./result/hyperCDtime/{"_".join(methods)}_cdtime.txt'
        run_exp(parameters, times, save_path=save_path, method=methods, multiprocessing=False)

if __name__ == '__main__':
    exp0()
