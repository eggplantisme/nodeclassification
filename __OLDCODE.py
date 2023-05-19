def create_gexf(self):
    """
    create a gephi file for visualization
    :return:
    """
    gexf_file_path = self.gexf_path + self.net_name + ".gexf"
    _edges = self.get_edges()
    # xml 编写
    nsmap = {'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}
    gexf = etree.Element('gexf', nsmap=nsmap)
    gexf.set('xmlns', 'http://www.gexf.net/1.1draft')
    gexf.set('version', '1.1')

    graph = etree.SubElement(gexf, 'graph', attrib={'mode': 'static', 'defaultedgetype': 'undirected'})
    nodes = etree.SubElement(graph, 'nodes')
    edges = etree.SubElement(graph, 'edges')
    node_list = []
    for edge in _edges:
        if edge[0] not in node_list:
            node_list.append(edge[0])
            xml_node = etree.Element('node', attrib={'id': str(edge[0]), 'label': str(edge[0])})
            nodes.append(xml_node)
        if edge[1] not in node_list:
            node_list.append(edge[1])
            xml_node = etree.Element('node', attrib={'id': str(edge[1]), 'label': str(edge[1])})
            nodes.append(xml_node)
        xml_edge = etree.Element('edge', attrib={'source': str(edge[0]), 'target': str(edge[1])})
        edges.append(xml_edge)
    gexf_tree = etree.ElementTree(gexf)
    gexf_tree.write(gexf_file_path, pretty_print=True, xml_declaration=True, encoding='utf-8')
    print(self.net_name, "has created gephi file at", gexf_file_path)

class zachary(OriNet):
    def __init__(self, ori_path):
        self.ori_path = ori_path

    def get_mat(self):
        # origin from 1 start -----> from 0 start
        mat = np.zeros((34, 34))
        with open(self.ori_path + "out.ucidata-zachary", "r") as f:
            f.readline()
            f.readline()
            line = f.readline()
            while line:
                line = line.strip().split(' ')
                edge = (int(line[0]) - 1, int(line[1]) - 1)
                if edge[0] != edge[1]:
                    # remove self loop
                    mat[edge[0], edge[1]] = 1
                    mat[edge[1], edge[0]] = 1
                line = f.readline()
        return mat

    def get_node_attr(self):
        return dict(), dict()

def plot_eigenvalue(self, ax):
    w, _ = np.linalg.eig(self.A)
    w = np.sort(w)
    w = np.around(w, decimals=3)
    x = []
    y = []
    for _w in w:
        _x = _w.real if isinstance(_w, complex) else _w
        _y = _w.imag if isinstance(_w, complex) else 0
        x.append(_x)
        y.append(_y)
    # print(w)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.scatter(x, y, label=str(self.n))
    ax.set_xlabel('Spectrum')
    return w


def exp1_subprocess(dout, X, Z, n, times):
    pout = 2 * Z * dout / n
    max_z = (1 - pout) / (Z * pout)
    min_z = -1 / Z
    rho = np.setdiff1d(np.around(np.arange(0, 1, 0.01), 2), np.array([]))
    z = np.setdiff1d(np.around(np.arange(min_z, max_z, 0.02), 2), np.array([]))
    fileID = 'amiExp4.21' + f'_dout={dout}'
    save_path = "./result/detectabilityWithMeta/" + fileID + ".txt"
    print(f"EXP pid={os.getpid()} for dout={dout} size", np.size(rho) * np.size(z))
    CommunityDetect.run_exp(rho, z, times, save_path, X, Z, n, dout)


def exp1():
    times = 10
    X = 2
    Z = 3
    n = 600
    douts = np.arange(5, 100, 10)
    p = Pool(5)
    for dout in douts:
        p.apply_async(exp1_subprocess, args=(dout, X, Z, n, times, ), error_callback=print_error)
    p.close()
    p.join()
    print('All subprocesses done.')