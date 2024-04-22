from other.EffectiveResistanceSampling import *
import networkx as nx
import sys
import time
sys.path.append("other/EffectiveResistanceSampling")
from other.EffectiveResistanceSampling.Network import *
from _SBMMatrix import BipartiteSBM
from EXPERIMENT_BIPARTITE import symmetric_bipartite, range_delta

start = time.time()
# From Bipartite network generate a weighted dense network
n1 = 3000
k1 = 3
d = 15
min_delta, max_delta = range_delta(n1, k1, d)
print(f'min_delta={min_delta}, max_delta={max_delta}')
delta = 0.01
pBo = d / n1 - delta / k1
pBd = pBo + delta
bsbm = symmetric_bipartite(n1, k1, pBd, pBo)
AA = bsbm.A.dot(bsbm.A)
BBT = AA[:n1, :n1]
print(f"Construct BBT Time:{time.time()-start}")
start = time.time()

# Construct network in EffectiveResistanceSampling
x, y = BBT.nonzero()
nonzero_size = np.size(x)
print(f'size of nonzeros:{nonzero_size}')
E_list = np.zeros((nonzero_size, 2), dtype=int)
for i in range(nonzero_size):
    E_list[i, 0] = x[i]
    E_list[i, 1] = y[i]
weights = [BBT[x[i], y[i]] for i in range(nonzero_size)]
network = Network(E_list, weights)
print(f"Construct Network Time:{time.time()-start}")
start = time.time()

# Compute EffectiveResistance
epsilon=0.1
method='kts'
Effective_R = network.effR(epsilon, method)
print(f'EffectiveR shape is {np.shape(Effective_R)}')
print(f"Construct ER Time:{time.time()-start}")
start = time.time()

# Sparsifying
q = 10000
seed = 2020
EffR_Sparse = network.spl(q, Effective_R, seed=2020)
print(f"Sparsify Time:{time.time()-start}")