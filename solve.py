"""
Function to solve the linear system Ku = F
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def check_symmetric(a, tol=1.):
    """Check if matrix a is symmetric withing a given tolerance tol (default 1)
    """
    return np.all(np.abs(a - a.T) < tol)


def solve_KU(K, F, free_dofs, fixed_dofs, Ub_vect, nen, method="reduce"):
    """Solve the system Ku = F using the scipy sparse solver

    :param free_dofs: dofs id where no dirichlet BC was imposed
    :param fixed_dofs: dofs id where dirichlet BC was imposed
    :param Ub_vect: Dirichlet BC list
    :param method: "reduce" (default) for solving only on unconstrained dofs
    :param nen: number of nodes
    :return: displacement u and reaction forces f_rea
    """
    if not check_symmetric(K, 1.):
        print("Warning: K was not symmetric")
        K = K + K.T
    # ------------------- Solve
    u = np.matrix(np.zeros((2 * nen, 1)))
    f_rea = np.matrix(np.zeros((2 * nen, 1)))
    K2 = 0
    if method == "reduce":
        K2 = K[free_dofs, :]
        K2 = K2[:, free_dofs]
    elif method == "penalty":
        K2 = K
        for elem in fixed_dofs:
            K2[elem, elem] = 1e7 * K2[elem, elem]

    if method == "reduce":
        K2_sparse = csc_matrix(K2)
        u[free_dofs, 0] = spsolve(K2_sparse, np.matrix(F[free_dofs]).T)
        u[fixed_dofs] = Ub_vect.transpose()
        f_rea = K * u
    if method == "penalty":
        u = np.linalg.pinv(L.T) * np.linalg.pinv(L) * np.matrix(F).T
        f_rea = K2 * u

    return u, f_rea
