import networkx as nx
import numpy as np
import random
import time


class Ising:
    def __init__(self, A):
        """
        Ising model
        :param A: adjacent matrix
        """
        self.A = A
        self.g = nx.from_numpy_array(A)
        self.n = self.g.number_of_nodes()
        self.s = [1] * self.n

    def init_s(self):
        """
        random initial states
        """
        for i in range(self.n):
            r = random.random()
            self.s[i] = 1 if r < 0.5 else -1

    def energy(self, J, H, S=None):
        """
        system energy
        :param J: interaction
        :param H: external magnetic strength
        :param S: states
        :return: energy
        """
        if S is None:
            S = self.s
        e = 0
        e -= ((J * np.array(S)) @ np.triu(self.A, 1) @ (np.array(S).reshape(-1, 1)))[0]
        e -= H * np.sum(S)
        return e

    def random_flip(self):
        i = np.random.randint(self.n)
        update_s = self.s.copy()
        update_s[i] = 1 if update_s[i] == -1 else -1
        return update_s

    def iter(self, J, H, T):
        """
        Iteration 1 step to lower energy
        :param J: interaction
        :param H: external magnetic strength
        :param T: temperature
        :return: is update?
        """
        update_s = self.random_flip()
        e_ori = self.energy(J, H)
        e_update = self.energy(J, H, update_s)
        miu = min(np.exp((e_ori - e_update) / T), 1)
        r = random.random()
        if r < miu:
            self.s = update_s
            return True
        else:
            return False


if __name__ == '__main__':
    A = np.random.randint(low=0, high=2, size=(100, 100))
    A = np.triu(A, 1) + np.triu(A, 1).transpose()
    ising = Ising(A)
    ising.init_s()
    J = 1
    H = 0.5
    T = 1
    print("INITIAL:", ising.energy(J=J, H=H))
    epoch = 0
    while True:
        epoch += 1
        if ising.iter(J, H, T):
            print("EPOCHS:", epoch, "UPDATED:", ising.energy(J=J, H=H))

