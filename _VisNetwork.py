import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from _FigureJiazeHelper import *


def visNodeSignal(A, signal, colors=None, layout=None):
    """
    visualization of network with node color represent node signal (label, signal)
    :param A: sparse matrix for adjacent network
    :param signal:
    :param colors:
    :param layout:
    :return:
    """
    g = nx.from_scipy_sparse_matrix(A)
    if layout is None:
        layout_pos = nx.spring_layout(g)
        print("1st time Layout Done!")
    else:
        layout_pos = layout
    # colored by signal
    start_color = colors_red[0]
    end_color = colors_blue[0]
    colors = gen_colors(start_color, end_color, np.size(np.unique(signal)))
    color_map = dict()
    for sig_i, sig in enumerate(np.sort(np.unique(signal))):
        color_map[sig] = colors[sig_i]
    node_colors = []
    for i in range(np.size(signal)):
        g.nodes[i]['signal'] = signal[i]
        node_colors.append(color_map[signal[i]])
    options = {
        'node_color': node_colors,
        'node_size': 100,
        'width': 0.5,
    }
    nx.draw(g, pos=layout_pos, **options)
    # plt.legend()
    plt.show()
    return layout_pos


def visNodeGroup(groupId):
    plt.figure(figsize=(10, 4))
    plt.imshow(np.atleast_2d(groupId), cmap=mpl.colormaps["tab20c"], origin='lower', extent=(0, np.size(groupId), 0, 1), aspect='auto')
    plt.show()
