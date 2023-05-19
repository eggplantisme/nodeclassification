import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


class RandomGraph:
    def __init__(self, n, p) -> None:
        self.n = n
        self.p = p
        self.A = np.zeros((n, n))
        self.edge = []
        self.message_edge = []
        self.generate()
        self.message_neighbor = None
    
    def generate(self):
        for i in range(self.n):
            for j in range(i+1, self.n):
                if random.random() < self.p:
                    self.A[i, j] = 1
                    self.A[j, i] = 1
                    self.edge.append((i, j))
                    self.message_edge.append((i, j))
                    self.message_edge.append((j, i))

    def draw(self, pos=None, ax=None, node_color='#1f78b4'):
        G = nx.from_numpy_array(self.A)
        if pos is None:
            pos = nx.spring_layout(G, k=0.1)
        if ax is None:
            nx.draw(G, pos=pos, node_size=10, node_color=node_color)
        else:
            nx.draw(G, pos=pos, node_size=10, ax=ax, node_color=node_color)
        return pos

    def bp_init(self):
        # Construct Message Neighbor advance
        message_neighbor = dict()
        for m_edge in self.message_edge:
            i, j = m_edge[0], m_edge[1]
            neighbor = set(np.where(self.A[i, :] == 1)[0].flatten())
            neighbor.remove(j)
            message_neighbor[m_edge] = [(k, i) for k in neighbor]
        self.message_neighbor = message_neighbor
        # Construct initial message and return
        psi_init = dict()
        for m_edge in self.message_edge:
            psi_init[m_edge] = random.random()
        return psi_init

    def bp_iter(self, psi_before):
        psi_after = {m_edge: 0 for m_edge in self.message_edge}
        if self.message_neighbor is None:
            print("Please do bp_init!")
            return None
        for m_edge in psi_after:
            psi_after[m_edge] = 1
            for m_neighbor in self.message_neighbor[m_edge]:
                psi_after[m_edge] *= psi_before[m_neighbor]
        return psi_after

    def bp_marginal(self, psi):
        psi_i = {i: 1 for i in range(self.n)}
        for i in range(self.n):
            neighbor = set(np.where(self.A[i, :] == 1)[0].flatten())
            for i_n in neighbor:
                psi_i[i] *= psi[(i_n, i)]
        return psi_i


def main():
    n, p = 100, 0.01
    rg = RandomGraph(n, p)
    pos = rg.draw()
    plt.show()


if __name__ == '__main__':
    main()


