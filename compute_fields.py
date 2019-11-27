import math
import numpy as np

def compute_stress(toPlot, R_quads, dR_quads, B, INC, IEN, E_coeff, nu_coeff, nobU, nobV):
    """
    Compute the stresses and strains at the gauss points using the shape function values and derivative computed during assembly
    :param u: displacement at the control points
    :param R_quads: shape function values at the gauss point for each element
    :type R_quads: np.array(p, q, nquadsquare, nel)
    :param dR_quads: shape function derivatives at the Gauss points
    :param B: control points
    :return:
    """
    [p, q, nquadsquare, nel] = R_quads.shape
    q = q-1
    p = p-1
    nquad = int(math.sqrt(nquadsquare))
    eps = np.zeros((2, 2, nel, nquad, nquad))
    for e in range(0, nel):
        ni = int(INC[int(IEN[e, 0]), 0])
        nj = int(INC[int(IEN[e, 0]), 1])
        for gpi in range(0, nquad):
            for gpj in range(0, nquad):
                x_actu = 0
                y_actu = 0
                for i in range(0, p+1):
                    for j in range(0, q+1):
                        i_CP = ni - i
                        j_CP = nj - j
                        x_actu += R_quads[i, j, gpi * nquad + gpj, e] * B[i_CP, j_CP, 0]
                        y_actu += R_quads[i, j, gpi * nquad + gpj, e] * B[i_CP, j_CP, 1]
                        eps[0, 0, e, gpi, gpj] += float(toPlot["u_x"][j_CP * nobU + i_CP]) * dR_quads[i, j, 0, gpi * nquad + gpj, e]
                        eps[1, 1, e, gpi, gpj] += float(toPlot["u_y"][j_CP * nobU + i_CP]) * dR_quads[i, j, 1, gpi * nquad + gpj, e]
                        eps[0, 1, e, gpi, gpj] += 0.5 * (float(toPlot["u_x"][j_CP * nobU + i_CP]) * dR_quads[i, j, 1, gpi * nquad + gpj, e] + float(
                            toPlot["u_y"][j_CP * nobU + i_CP]) * dR_quads[i, j, 0, gpi * nquad + gpj, e])
                        eps[1, 0, e, gpi, gpj] += 0.5 * (float(toPlot["u_x"][j_CP * nobU + i_CP]) * dR_quads[i, j, 1, gpi * nquad + gpj, e] + float(
                            toPlot["u_y"][j_CP * nobU + i_CP]) * dR_quads[i, j, 0, gpi * nquad + gpj, e])

                sig = np.zeros((2, 2, nel, nquad, nquad))
                coef = E_coeff / (1 - nu_coeff ** 2)
                sig[0, 0, e, gpi, gpj] = coef * (eps[0, 0, e, gpi, gpj] + nu_coeff * eps[1, 1, e, gpi, gpj])
                sig[1, 1, e, gpi, gpj] = coef * (eps[1, 1, e, gpi, gpj] + nu_coeff * eps[0, 0, e, gpi, gpj])
                sig[0, 1, e, gpi, gpj] = coef * (1 - nu_coeff) * eps[0, 1, e, gpi, gpj]
                sig[1, 0, e, gpi, gpj] = coef * (1 - nu_coeff) * eps[1, 0, e, gpi, gpj]
    sigxx = np.reshape(sig[0, 0], (nel * nquad * nquad))
    sigyy = np.reshape(sig[1, 1], (nel * nquad * nquad))
    sigxy = np.reshape(sig[1, 0], (nel * nquad * nquad))
    np.savetxt("stress_GP", np.array([sigxx, sigyy, sigxy]).T)
    return sig, eps
