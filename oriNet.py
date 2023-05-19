import numpy as np
import os
from lxml import etree
import scipy.io as scio
from scipy import sparse
from __UTIL import save_edge_list
from _SBMMatrix import SymmetricSBM


class OriNet:
    def get_mat(self):
        """get adjacent matrix (Sparse matrix)"""
        pass

    def get_node_attr(self):
        """
        return node attributes: {"attrName": [attrV[0], ...], ...}
        node attributes meta: {"attrName": {attrV:attrMeta, ...}, ...}
        attrV is 1-dim numpy array
        """
        pass

    @staticmethod
    def relabel_nodes_id(edges, nodes):
        # relabel nodes' id to a consecutive form from 0, 0,1,2,3,...
        old_ids = np.unique(np.array(list(nodes.keys())))
        remap = dict((k, v) for v, k in enumerate(old_ids))
        relabeled_edges = []
        relabeled_nodes = dict()
        for edge in edges:
            relabeled_edges.append((remap[edge[0]], remap[edge[1]]))
        for node in nodes.keys():
            relabeled_nodes[remap[node]] = nodes[node]
        return relabeled_edges, relabeled_nodes, remap


class SyntheticSSBM(OriNet):
    def __init__(self, n, c, k, epsilon):
        self.sbm = SymmetricSBM(n, c, k, epsilon)

    def get_mat(self):
        return sparse.csr_matrix(self.sbm.A)

    def get_node_attr(self):
        attrName = "block"
        blockVec = []
        for bi, bs in enumerate(self.sbm.sizes):
            blockVec.extend([bi] * bs)
        node_attr = {attrName: np.array(blockVec)}
        node_attr_meta = {attrName: {k: k for k in range(len(self.sbm.sizes))}}
        return node_attr, node_attr_meta


class highSchool(OriNet):
    def __init__(self, ori_path):
        self.ori_path = ori_path
        self.re_edges, self.re_nodes, self.re_map = None, None, None
        self.str_classes, self.str_gender = None, None

    def get_mat(self):
        edges = []
        with open(self.ori_path + "Facebook-known-pairs_data_2013.csv", 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                if line[2] == '0':
                    pass
                else:
                    edge = (int(line[0]), int(line[1]))
                    edges.append(edge)
        _nodes = dict()
        with open(self.ori_path + "metadata_2013.txt", 'r') as f:
            for line in f:
                line = line.strip().split('\t')  # node_id \t class \t gender
                _nodes[int(line[0])] = {"class": line[1], "gender": line[2]}
        # exclude node not in edges
        nodes = dict()
        for edge in edges:
            nodes[edge[0]] = _nodes[edge[0]] if edge[0] not in nodes else nodes[edge[0]]
            nodes[edge[1]] = _nodes[edge[1]] if edge[1] not in nodes else nodes[edge[1]]
        re_edges, re_nodes, re_map = self.relabel_nodes_id(edges, nodes)
        self.re_edges, self.re_nodes, self.re_map = re_edges, re_nodes, re_map
        # construct adjacent matrix
        # A = np.zeros((len(re_nodes), len(re_nodes)))
        A = sparse.lil_matrix((len(re_nodes), len(re_nodes)), dtype=np.int8)
        for edge in re_edges:
            A[edge[0], edge[1]], A[edge[1], edge[0]] = 1, 1
        A = A.tocsr()
        # save node mapping
        save_edge_list(list(zip(list(re_map), list(re_map.values()))), self.ori_path + "node_mapping.txt")
        return A

    def get_node_attr(self):
        self.get_mat()
        # construct partition (attribute) vec
        class_vec = np.zeros(len(self.re_nodes))
        gender_vec = np.zeros(len(self.re_nodes))
        str_classes = dict.fromkeys([self.re_nodes[node]["class"] for node in self.re_nodes.keys()])  # dict reserve order
        str_gender = dict.fromkeys([self.re_nodes[node]["gender"] for node in self.re_nodes.keys()])
        for ci, c in enumerate(str_classes.keys()):
            str_classes[c] = ci
        for gi, g in enumerate(str_gender.keys()):
            str_gender[g] = gi
        # print(str_classes)
        # print(str_gender)
        self.str_classes, self.str_gender = str_classes, str_gender
        for node in self.re_nodes.keys():
            class_vec[node] = str_classes[self.re_nodes[node]["class"]]
            gender_vec[node] = str_gender[self.re_nodes[node]["gender"]]
        node_attr = {"class": class_vec, "gender": gender_vec}
        node_attr_meta = {"class": {v: k for k, v in str_classes.items()},
                          "gender": {v: k for k, v in str_gender.items()}}
        return node_attr, node_attr_meta


class primarySchool(OriNet):
    def __init__(self, ori_path):
        self.ori_path = ori_path
        self.re_edges, self.re_nodes, self.re_map = None, None, None
        self.str_classes, self.str_gender = None, None

    def get_mat(self):
        with open(self.ori_path + "sp_data_school_day_1_g.gexf", 'r') as f:
            gexf_tree = etree.parse(f)
            root = gexf_tree.getroot()
            cur_nsmap = root.nsmap
            # node attributes
            nodes = dict()
            # set node attributes
            for node in root.iter('{' + cur_nsmap[None] + '}' + 'node'):
                node_id = int(node.get('id'))
                nodes[node_id] = dict()
                for attr in node.iter('{' + cur_nsmap[None] + '}' + 'attvalue'):
                    if attr.get('for') == '0':
                        nodes[node_id]['classname'] = attr.get('value')
                    elif attr.get('for') == '1':
                        nodes[node_id]['gender'] = attr.get('value')
                    else:
                        pass
            # set edge list
            edges = []
            for edge in root.iter('{' + cur_nsmap[None] + '}' + 'edge'):
                edges.append((int(edge.get('source')), int(edge.get('target'))))
            re_edges, re_nodes, re_map = self.relabel_nodes_id(edges, nodes)
            self.re_edges, self.re_nodes, self.re_map = re_edges, re_nodes, re_map
            # construct adjacent matrix
            # A = np.zeros((len(re_nodes), len(re_nodes)))
            A = sparse.lil_matrix((len(re_nodes), len(re_nodes)), dtype=np.int8)
            for edge in re_edges:
                A[edge[0], edge[1]], A[edge[1], edge[0]] = 1, 1
            A = A.tocsr()
            # save node mapping
            save_edge_list(list(zip(list(re_map), list(re_map.values()))), self.ori_path + "node_mapping.txt")
        return A

    def get_node_attr(self):
        self.get_mat()
        # construct partition (attribute) vec
        class_vec = np.zeros(len(self.re_nodes))
        gender_vec = np.zeros(len(self.re_nodes))
        str_classes = dict.fromkeys([self.re_nodes[node]["classname"] for node in self.re_nodes.keys()])  # dict reserve order
        str_gender = dict.fromkeys([self.re_nodes[node]["gender"] for node in self.re_nodes.keys()])
        for ci, c in enumerate(str_classes.keys()):
            str_classes[c] = ci
        for gi, g in enumerate(str_gender.keys()):
            str_gender[g] = gi
        for node in self.re_nodes.keys():
            class_vec[node] = str_classes[self.re_nodes[node]["classname"]]
            gender_vec[node] = str_gender[self.re_nodes[node]["gender"]]
        node_attr = {"class": class_vec, "gender": gender_vec}
        node_attr_meta = {"class": {v: k for k, v in str_classes.items()},
                          "gender": {v: k for k, v in str_gender.items()}}
        return node_attr, node_attr_meta


class facebook100(OriNet):
    def __init__(self, ori_path, region_name):
        self.ori_path = ori_path
        self.region_name = region_name
        self.data = None
        self.str_classes, self.str_gender = None, None

    def get_mat(self):
        mat_path = self.ori_path + self.region_name + ".mat"
        self.data = scio.loadmat(mat_path)
        A = self.data['A'].tocsr()
        return A

    def get_node_attr(self):
        self.get_mat()
        info = self.data['local_info']
        # 6 attributes vec
        attr_name = ["gender", "major", "minor", "dorm", "year", "highschool"]
        node_attr = dict()
        node_attr_meta = dict()

        def get_attr(index, info):
            attr_vec = info[:, index]
            attrs = dict.fromkeys(info[:, index].tolist())
            for gi, g in enumerate(attrs.keys()):
                attrs[g] = gi
            attr_vec = [attrs[v] for v in info[:, index]]
            return attr_vec, attrs

        for i, attr in enumerate(attr_name):
            attr_vec, attrs = get_attr(i+1, info)
            node_attr[attr] = attr_vec
            node_attr_meta[attr] = {v: k for k, v in attrs.items()}
        return node_attr, node_attr_meta

    def get_region_list(self):
        """  get all facebook100 regions in format of list"""
        files = []
        for filename in os.listdir(self.ori_path):
            if filename != "meta_file":
                files.append("facebook100_" + filename.split('.')[0])
        print(files)


if __name__ == '__main__':
    # facebook100("./net_data/facebook100/origin/", "American75").get_region_list()
    n = 2 ** 8
    k = 2
    c = 10
    epsilon = 0.3
    net = SyntheticSSBM(n, c, k, epsilon)
