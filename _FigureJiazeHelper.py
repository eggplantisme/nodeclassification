import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.patches as patches
import colour
from scipy.sparse.linalg import eigsh
from scipy import sparse
from scipy.sparse import eye, diags, issparse, csr_matrix


# color set
basic_line_color = '#808080'
black = '#222222'
darkgray = '#A9A9A9'
highlight = '#00B2EE'
gray = "#C0C0C0"
colors_red = ['#F01F1F', '#F385EE']
colors_blue = ["#130DF7", '#2D83E9', '#a6cee3', '#15D4B7']
colors_green = ['#13B632', '#13F121', '#BAEB34']
colors_yellow = ['#F0FA0D']


def gen_colors(start, end, n):
    start = colour.Color(start)
    end = colour.Color(end)
    colors = [c.hex_l for c in list(start.range_to(end, n))]
    return colors


def plot_block_matrix(reorder_A, partition_names=None, partition_counts=None, colors=None, ms=1,
                      save_path=None, label='Adjacency matrix', show_legend=True,
                      show_thislevel_partition=False, thislevel_partition_counts=None,
                      show_highlevel_partition=False, highlevel_partition_counts=None):
    if colors is None:
        colors = []
    if partition_counts is None:
        partition_counts = []
    if partition_names is None:
        partition_names = []
    plt.spy(reorder_A, markersize=ms, rasterized=True, color=gray)
    accumulate_count = 0
    for i in range(len(partition_counts)):
        cur_counts = int(partition_counts[i])
        accumulate_count += cur_counts
        temp_A = np.copy(reorder_A)
        temp_A[accumulate_count:, :] = 0
        temp_A[:, accumulate_count:] = 0
        temp_A[:accumulate_count-cur_counts, :] = 0
        temp_A[:, :accumulate_count-cur_counts] = 0
        plt.spy(temp_A, markersize=ms, rasterized=True, color=colors[i], label=partition_names[i])
    if show_thislevel_partition:
        counts_sum = 0
        for count in thislevel_partition_counts[:-1]:
            counts_sum += count
            plt.axvline(counts_sum, color='grey', lw=1)
            plt.axhline(counts_sum, color='grey', lw=1)
    if show_highlevel_partition:
        counts_sum = 0
        for count in highlevel_partition_counts[:-1]:
            counts_sum += count
            plt.axvline(counts_sum, color='k', lw=2)
            plt.axhline(counts_sum, color='k', lw=2)
    plt.xticks([])
    plt.yticks([])
    plt.title(label)
    if show_legend:
        plt.legend(loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=10, markerscale=2/ms)
    if save_path is not None:
        plt.savefig(save_path, dpi=600)


def color_scatter_2d(x, y, z, z_center, title, xlabel, ylabel, min_z=None, max_z=None, cmap=cm.coolwarm,
                     save_path=None, ax=None, fig=None):
    """
    scatter z for x,y-axis. Divided by z_center, z is colored different. all value in z > 0
    """
    C = [0] * np.shape(z)[0]
    minz = min_z if min_z is not None else np.min(z)
    maxz = max_z if max_z is not None else np.max(z)
    print(f'min={minz}, max={maxz}')
    for i in range(np.size(z)):
        if minz <= z[i] <= z_center:
            C[i] = cmap((z[i] - minz) / (z_center - minz) * 0.5)
        elif z[i] > z_center:
            C[i] = cmap((z[i] - z_center) / (maxz - z_center) * 0.5 + 0.5)
        else:
            pass
    if ax is None or fig is None:
        fig = plt.figure(figsize=(10, 10))
        widths = [4]
        heights = [4]
        spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths, height_ratios=heights)
        row = 0
        col = 0
        ax = fig.add_subplot(spec5[row, col])
    # plt.axhline(0, linestyle=':')
    p = ax.scatter(x, y, s=10, c=C)
    ax.set_title(title, fontsize=30)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, rotation='horizontal', fontsize=20)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    print(f'{np.min(y)}, {np.max(y)}')
    ax.set_xticks(np.arange(np.min(x), np.max(x)+0.1, 0.1))
    ax.set_yticks(np.linspace(np.min(y), np.max(y), np.size(np.unique(y))))
    # plt.axhline(-1/3, color='k', lw=1)
    # plt.axvline(0.5, linestyle=':')
    # ax.set_zlabel(r'$\frac{SNR_2}{SNR_3}$')
    cticks = [0, 0.25, 0.5, 0.75, 1]
    clabels = [str(np.round(2 * x * (z_center - minz) + minz, 3)) if x < 0.5 else str(
        np.round((2 * x - 1) * (maxz - z_center) + z_center, 3)) for x in cticks]
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), ticks=cticks, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.set_yticklabels(clabels)
    # plt.legend(loc=1, fontsize=10, markerscale=4)
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
