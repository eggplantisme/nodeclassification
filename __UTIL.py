import time


def load_edge_list(path):
    edges = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            edge = (int(line[0]), int(line[1]))
            edges.append(edge)
    return edges


def save_edge_list(edges, path):
    with open(path, 'w') as f:
        for edge in edges:
            f.write(str(edge[0]) + " " + str(edge[1]) + "\n")


def state(**kwds):
    """
    This is an annotate function. Only fit with un recursion function
    """
    def decorate(f):
        def run(*args):
            print(kwds['start'])
            start = time.time()
            f(*args)
            print(kwds['end'] + " time:" + str(time.time() - start))
        return run
    return decorate
