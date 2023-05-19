import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, diags, issparse, csr_matrix


class SpectralOperator(object):
    def __init__(self):
        self.operator = None
        self.evals = None
        self.evecs = None

    def find_k_eigenvectors(self, K, which='SA'):
        M = self.operator
        if M.shape[0] == 1:
            print('WARNING: Matrix is a single element')
            evals = np.array([1])
            evecs = np.array([[1]])
        elif K < M.shape[0]:
            evals, evecs = eigsh(M, K, which=which)
        else:
            evals, evecs = eigsh(M, M.shape[0] - 1, which=which)
        self.evals, self.evecs = evals, evecs

    def find_negative_eigenvectors(self):
        """
        Find negative eigenvectors.

        Given a matrix M, find all the eigenvectors associated to negative
        eigenvalues and return number of negative eigenvalues
        """
        M = self.operator
        Kmax = M.shape[0] - 1
        K = min(10, Kmax)
        if self.evals is None:
            self.find_k_eigenvectors(K, which='SA')
        elif len(self.evals) < K:
            self.find_k_eigenvectors(K, which='SA')
        relevant_ev = np.nonzero(self.evals < 0)[0]
        while relevant_ev.size == K:
            K = min(2 * K, Kmax)
            self.find_k_eigenvectors(K, which='SA')
            relevant_ev = np.nonzero(self.evals < 0)[0]
            # search negative eigenvectors from 10, 20, 30, ..., use this way to save the memory @jiaze
        self.evals = self.evals[relevant_ev]
        self.evecs = self.evecs[:, relevant_ev]
        return len(relevant_ev)


class BetheHessian(SpectralOperator):

    def __init__(self, A, r=None, regularizer='BHa'):
        super(BetheHessian, self).__init__()
        self.A = A
        self.r = r
        self.calc_r(regularizer)
        self.build_operator()

    def calc_r(self, regularizer='BHa'):
        if self.r is None:
            A = self.A
            if regularizer.startswith('BHa'):
                # set r to square root of average degree
                self.r = np.sqrt(A.sum() / A.shape[0])
        # if last character is 'n' then use the negative version of the BetheHessian
        if regularizer[-1] == 'n':
            self.r = -self.r

    def build_operator(self):
        """
        Construct Standard Bethe Hessian as discussed, e.g., in Saade et al
        B = (r^2-1)*I-r*A+D
        """
        A = self.A
        r = self.r
        A = test_sparse_and_transform(A)
        d = A.sum(axis=1).flatten().astype(float)
        B = eye(A.shape[0]) * (r ** 2 - 1) - r * A + diags(d, 0)
        self.operator = B


def test_sparse_and_transform(A):
    """ Check if matrix is sparse and if not, return it as sparse matrix"""
    if not issparse(A):
        print("""Input matrix not in sparse format, transforming to sparse matrix""")
        A = csr_matrix(A)
    return A
