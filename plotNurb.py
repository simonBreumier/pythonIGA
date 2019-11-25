import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import math
from scipy.optimize import minimize
from simonurbs import *

def plot_field(SU, SV, SOL, label="", plotGrid="", boundlow=-10000, boundsup=-10000, nobU=0, nel=0, IEN=np.zeros((1)),
               ctrlpts=np.zeros((1))):
    """
    Plot a given field on a regular gridd
    :param SU, SV: X, Y coordinates matrix
    :param SOL: Field matrix
    :param label: Title string
    :param plotGrid: 'mesh' if mesh is to be superimposed
    :param boundlow, boundsup: contour color lower and upper bounds
    :param nobU: number of CP in U direction (only if plot mesh)
    :param nel: number of elements (only if plot mesh)
    :param IEN: Connectivity table (only if plot mesh)
    :param ctrlpts: Control points (only if plot mesh)
    """
    if not (boundlow == -10000):
        plt.contourf(SU, SV, SOL, vmin=boundlow, vmax=boundsup, levels=25)
    else:
        plt.contourf(SU, SV, SOL, levels=25)
    plt.colorbar()
    plt.title(label)
    if plotGrid == "mesh":
        plotMesh(nel, IEN, nobU, ctrlpts)


def plotMesh(nel, IEN, nobU, ctrlpts):
    ''' Plot the isogeometric mesh given the controlpoints and connectivity array '''
    nbpts = IEN.shape[1]
    col_list = ["red", "black"]
    k = 0

    for i in range(0, nel):
        coord_elem = np.zeros((nbpts, 2))
        pres = False
        for j in range(0, nbpts):
            i_actu = int(IEN[i, j]) % nobU
            j_actu = int(IEN[i, j] / nobU)
            x_actu = ctrlpts[i_actu, j_actu, 0]
            y_actu = ctrlpts[i_actu, j_actu, 1]
            coord_elem[j, 0] = x_actu
            coord_elem[j, 1] = y_actu
            if j_actu * nobU + i_actu == 58:
                pres = True

        hull = ConvexHull(coord_elem)
        plt.plot(coord_elem[:, 0], coord_elem[:, 1], 'o', color="black")

        for simplex in hull.simplices:
            plt.plot(coord_elem[simplex, 0], coord_elem[simplex, 1], 'k-')


def plotNode(x, y, nobU, nobV):
    plt.scatter(x, y, marker='o')
    for i in range(0, nobU * nobV):
        plt.annotate(str(i), (x[i], y[i]), color="black")


def plot_nurbs(ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, u, GP_coord, nel, IEN, toPlot, plotGrid, E, nu,
               plotFields, numrefine, meshPrev=np.zeros((1))):
    nobU = ctrlpts.shape[0]
    nobV = ctrlpts.shape[1]
    weights = np.zeros((nobU, nobV))
    ctrl_mat = np.zeros((nobU, nobV, 2))
    x = []
    y = []
    for i in range(0, nobU):
        for j in range(0, nobV):
            weights[i, j] = ctrlpts[i, j, -1]
            ctrl_mat[i, j, :] = ctrlpts[i, j, 0:2]
    resol = 100

    if meshPrev.shape[0] > 1:
        lenMesh = meshPrev.shape[1]
        u_interpX = np.zeros(lenMesh)
        u_interpY = np.zeros(lenMesh)
        xtransp = np.zeros(lenMesh)
        ytransp = np.zeros(lenMesh)
        for i in range(0, lenMesh):
            isctrl = False
            p = 0
            k = 0
            while not isctrl and p < nobU:
                while not isctrl and k < nobV:
                    if ctrlpts[p, k, 0] == meshPrev[0, i] and ctrlpts[p, k, 1] == meshPrev[1, i]:
                        isctrl = True
                        xtransp[i] = ctrlpts[p, k, 0]
                        ytransp[i] = ctrlpts[p, k, 1]
                        u_interpX[i] = toPlot["u_x"][k * nobU + p]
                        u_interpY[i] = toPlot["u_y"][k * nobU + p]
                    k += 1
                k = 0
                p += 1
            if not (isctrl):
                xi, eta = findcoord(meshPrev[:, i], ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, weights)
                xtransp[i] = float(
                    compRval(ctrlpts[:, :, 0], xi, eta, ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v,
                             weights))
                ytransp[i] = float(
                    compRval(ctrlpts[:, :, 1], xi, eta, ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v,
                             weights))
                u_interpX[i] = float(
                    compRval(toPlot["u_x"].reshape((nobU, nobV)), xi, eta, ctrlpts, knotvector_u, knotvector_v,
                             degree_u, degree_v, weights))
                u_interpY[i] = float(
                    compRval(toPlot["u_y"].reshape((nobU, nobV)), xi, eta, ctrlpts, knotvector_u, knotvector_v,
                             degree_u, degree_v, weights))
        np.savetxt("SQR_vals/u_CP_" + str(numrefine), np.array([xtransp, ytransp, u_interpX, u_interpY]).T, fmt="%.5e")

    gp_num = GP_coord.shape[1]
    box_ecart = 0.5
    C, dC_dxi, dC_deta = make_plot_matrix(ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, resol, weights)
    field = {}
    SU = np.zeros((resol, resol))
    SV = np.zeros((resol, resol))
    eps = np.zeros((2, 2, resol, resol))
    sig = np.zeros((2, 2, resol, resol))
    plotnum = len(toPlot.keys())
    dx_dxi = np.zeros((2, 2, resol, resol))
    dxi_dx = np.zeros((2, 2, resol, resol))
    dR_dx = np.zeros((2, nobU, nobV, resol, resol))
    for elem in toPlot.keys():
        field[elem] = np.zeros((resol, resol))

    for j in range(0, nobV):
        for i in range(0, nobU):
            SU = SU + ctrl_mat[i, j, 0] * C[i, j, :, :]
            SV = SV + ctrl_mat[i, j, 1] * C[i, j, :, :]
            dx_dxi[0, 0] += np.multiply(ctrl_mat[i, j, 0], dC_dxi[i, j, :, :])
            dx_dxi[0, 1] += np.multiply(ctrl_mat[i, j, 0], dC_deta[i, j, :, :])
            dx_dxi[1, 0] += np.multiply(ctrl_mat[i, j, 1], dC_dxi[i, j, :, :])
            dx_dxi[1, 1] += np.multiply(ctrl_mat[i, j, 1], dC_deta[i, j, :, :])
            for elem in toPlot.keys():
                field[elem] += float(toPlot[elem][j * nobU + i]) * C[i, j, :, :]
            x.append(ctrlpts[i, j, 0])
            y.append(ctrlpts[i, j, 1])
    det = np.zeros((resol, resol))
    for i in range(0, resol):
        for j in range(0, resol):
            dxi_dx[:, :, i, j] = np.linalg.inv(np.matrix(dx_dxi[:, :, i, j]))
            det[i, j] = np.linalg.det(np.matrix(dx_dxi[:, :, i, j]))

    for j in range(0, nobV):
        for i in range(0, nobU):
            dR_dx[0, i, j] = dxi_dx[0, 0] * dC_dxi[i, j] + dxi_dx[1, 0] * dC_deta[i, j]
            dR_dx[1, i, j] = dxi_dx[0, 1] * dC_dxi[i, j] + dxi_dx[1, 1] * dC_deta[i, j]

    for j in range(0, nobV):
        for i in range(0, nobU):
            if "u_x" in toPlot.keys() and "u_y" in toPlot.keys():
                eps[0, 0, :, :] += float(toPlot["u_x"][j * nobU + i]) * dR_dx[0, i, j, :, :]
                eps[1, 1, :, :] += float(toPlot["u_y"][j * nobU + i]) * dR_dx[1, i, j, :, :]
                eps[0, 1, :, :] += 0.5 * (float(toPlot["u_x"][j * nobU + i]) * dR_dx[1, i, j, :, :] + float(
                    toPlot["u_y"][j * nobU + i]) * dR_dx[0, i, j, :, :])
                eps[1, 0, :, :] += 0.5 * (float(toPlot["u_x"][j * nobU + i]) * dR_dx[1, i, j, :, :] + float(
                    toPlot["u_y"][j * nobU + i]) * dR_dx[0, i, j, :, :])

    coef = E / (1 - nu ** 2)
    for j in range(0, resol):
        for i in range(0, resol):
            if "u_x" in toPlot.keys() and "u_y" in toPlot.keys():
                sig[0, 0, i, j] = coef * (eps[0, 0, i, j] + nu * eps[1, 1, i, j])
                sig[1, 1, i, j] = coef * (eps[1, 1, i, j] + nu * eps[0, 0, i, j])
                sig[0, 1, i, j] = coef * (1 - nu) * eps[0, 1, i, j]
                sig[1, 0, i, j] = coef * (1 - nu) * eps[1, 0, i, j]

    if plotGrid == "mesh":
        plotMesh(nel, IEN, nobU, ctrlpts)
    elif plotGrid == "nodes":
        plotNode(x, y, nobU, nobV)

    if plotFields:
        plt.figure(figsize=(12, 4))
        i = 0
        for elem in toPlot.keys():
            i += 1
            plt.subplot(1, 2, i)
            plot_field(SU, SV, field[elem], nobU=nobU, plotGrid=plotGrid, nel=nel, IEN=IEN, ctrlpts=ctrlpts)

        plt.show()
        if "u_x" in toPlot.keys() and "u_y" in toPlot.keys():
            plt.figure(figsize=(12, 12))
            plt.subplot(2, 2, 1)
            plot_field(SU, SV, eps[0, 0], "$\\varepsilon_{xx}$", nobU=nobU, plotGrid=plotGrid, nel=nel, IEN=IEN,
                       ctrlpts=ctrlpts)
            plt.subplot(2, 2, 2)
            plot_field(SU, SV, eps[1, 1], "$\\varepsilon_{yy}$", nobU=nobU, plotGrid=plotGrid, nel=nel, IEN=IEN,
                       ctrlpts=ctrlpts)
            plt.subplot(2, 2, 3)
            plot_field(SU, SV, eps[0, 1], "$\\varepsilon_{xy}$", nobU=nobU, plotGrid=plotGrid, nel=nel, IEN=IEN,
                       ctrlpts=ctrlpts)
            plt.subplot(2, 2, 4)
            plot_field(SU, SV, eps[1, 0], "$\\varepsilon_{yx}$", nobU=nobU, plotGrid=plotGrid, nel=nel, IEN=IEN,
                       ctrlpts=ctrlpts)
            plt.show()
            plt.figure(figsize=(12, 12))
            plt.subplot(2, 2, 1)
            plot_field(SU, SV, sig[0, 0], "$\\sigma_{xx}$", nobU=nobU, plotGrid=plotGrid, nel=nel, IEN=IEN,
                       ctrlpts=ctrlpts)
            plt.subplot(2, 2, 2)
            plot_field(SU, SV, sig[1, 1], "$\\sigma_{yy}$", nobU=nobU, plotGrid=plotGrid, nel=nel, IEN=IEN,
                       ctrlpts=ctrlpts)
            plt.subplot(2, 2, 3)
            plot_field(SU, SV, sig[0, 1], "$\\sigma_{xy}$", nobU=nobU, plotGrid=plotGrid, nel=nel, IEN=IEN,
                       ctrlpts=ctrlpts)
            plt.subplot(2, 2, 4)
            plot_field(SU, SV, sig[1, 0], "$\\sigma_{yx}$", nobU=nobU, plotGrid=plotGrid, nel=nel, IEN=IEN,
                       ctrlpts=ctrlpts)
            plt.show()

    shapeList = (1, resol * resol)
    epsxx = np.reshape(eps[0, 0], shapeList)
    epsyy = np.reshape(eps[1, 1], shapeList)
    epsxy = np.reshape(eps[0, 1], shapeList)
    np.savetxt("SQR_vals/u_CP_" + str(numrefine), np.array(
        [np.reshape(SU, shapeList), np.reshape(SV, shapeList), np.reshape(field["u_x"], shapeList),
         np.reshape(field["u_y"], shapeList)]).T[:, 0], fmt="%.5e")
    np.savetxt("SQR_vals/eps_xx_" + str(numrefine),
               np.array([np.reshape(SU, shapeList), np.reshape(SV, shapeList), epsxx, epsyy, epsxy]).T[:, 0],
               fmt="%.5e")

    sigxx = np.reshape(sig[0, 0], shapeList)
    sigyy = np.reshape(sig[1, 1], shapeList)
    sigxy = np.reshape(sig[0, 1], shapeList)
    np.savetxt("SQR_vals/sigma_" + str(numrefine),
               np.array([np.reshape(SU, shapeList), np.reshape(SV, shapeList), sigxx, sigyy, sigxy]).T[:, 0],
               fmt="%.5e")


def findcoord(targ, ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, weights):
    xtarg = targ[0]
    ytarg = targ[1]
    safe = 1.e-3
    cons = ({'type': 'ineq', 'fun': lambda x: x[0]}, {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: 1. - safe - x[0]}, {'type': 'ineq', 'fun': lambda x: 1. - safe - x[1]})
    res = minimize(compCoordResVal, [0, 0],
                   args=(xtarg, ytarg, ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, weights),
                   constraints=cons)

    return res.x[0], res.x[1]


def compCoordResVal(paramcoord, xtarg, ytarg, ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, weights):
    xi = abs(paramcoord[0])
    eta = abs(paramcoord[1])
    x = compRval(ctrlpts[:, :, 0], xi, eta, ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, weights)
    y = compRval(ctrlpts[:, :, 1], xi, eta, ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, weights)
    return math.sqrt((xtarg - x) ** 2 + (ytarg - y) ** 2)

def make_plot_matrix(ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, resol, weights):
    ''' Generate a C matrix 'such that C[i,j,k,l] = R[i,j] (xi[k], eta[l])'''

    plotVectorU = np.linspace(0, max(knotvector_u), resol);
    plotVectorV = np.linspace(0, max(knotvector_v), resol);

    sU = len(plotVectorU);
    sV = len(plotVectorV);

    # create span indices
    tableSpanU = lookuptablespan(knotvector_u, len(knotvector_u), plotVectorU, sU);
    tableSpanV = lookuptablespan(knotvector_v, len(knotvector_v), plotVectorV, sV);

    # initialize data matrix C
    nobU = ctrlpts.shape[0]
    nobV = ctrlpts.shape[1]
    C = np.zeros((nobU, nobV, sU, sV))
    dC_dxi = np.zeros((nobU, nobV, sU, sV))
    dC_deta = np.zeros((nobU, nobV, sU, sV))
    weightmatrix = np.reshape(weights, (nobU, nobV));
    for i in range(0, sU):
        for j in range(0, sV):
            startX = int(tableSpanU[i] - tableSpanU[0]);
            startY = int(tableSpanV[j] - tableSpanV[0]);

            entryU = BasisFun(int(tableSpanU[i]), plotVectorU[i], degree_u, knotvector_u);
            entryV = BasisFun(int(tableSpanV[j]), plotVectorV[j], degree_v, knotvector_v);
            entryUder = BasisFunDers(int(tableSpanU[i]), plotVectorU[i], degree_u, knotvector_u, 1);
            entryVder = BasisFunDers(int(tableSpanV[j]), plotVectorV[j], degree_v, knotvector_v, 1);
            localweights = weightmatrix[startX:startX + degree_u + 1, startY:startY + degree_v + 1];
            NM = (entryU.transpose() * entryV)
            dNM = (entryUder[1, :].transpose() * entryVder[0, :])
            NdM = (entryUder[0, :].transpose() * entryVder[1, :])

            wNM = np.multiply(NM, localweights);
            wdNM = np.multiply(dNM, localweights);
            wNdM = np.multiply(NdM, localweights);

            W = np.sum(wNM);
            WprimedNM = np.sum(wdNM);
            WprimeNdM = np.sum(wNdM);

            numeratordNM = W * wdNM - WprimedNM * wNM
            numeratorNdM = W * wNdM - WprimeNdM * wNM
            for iU in range(0, degree_u + 1):
                for iV in range(0, degree_v + 1):
                    C[startX + iU, startY + iV, i, j] = wNM[iU, iV] / W;
                    dC_dxi[startX + iU, startY + iV, i, j] = numeratordNM[iU, iV] / W ** 2;
                    dC_deta[startX + iU, startY + iV, i, j] = numeratorNdM[iU, iV] / W ** 2;

    return C, dC_dxi, dC_deta
