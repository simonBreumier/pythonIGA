'''
Functions to compute NURBS shape function values
'''
import numpy as np


def compRval(param, xi, eta, ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, weights):
    """
    Interpolate a field value at a given point using the NURBS shape function
    :param param: field values at the control points
    :param xi, eta: parametric coordinate where to compute the field
    :param ctrlpts: control points
    :param knotvector_u, knotvector_v: knot vectors in u and v direction
    :param degree_u, degree_v: degrees in u and v directions
    :param weights: weight vector for eqch control point
    :return: float field value
    """
    tableSpanU = lookuptablespan(knotvector_u, len(knotvector_u), [xi], 1)
    tableSpanV = lookuptablespan(knotvector_v, len(knotvector_v), [eta], 1)

    # initialize data matrix C
    nobU = ctrlpts.shape[0]
    nobV = ctrlpts.shape[1]
    C = np.zeros((nobU, nobV, 1, 1))
    dC_dxi = np.zeros((nobU, nobV, 1, 1))
    dC_deta = np.zeros((nobU, nobV, 1, 1))
    weightmatrix = np.reshape(weights, (nobU, nobV))

    startX = int(tableSpanU[0] - tableSpanU[0])
    startY = int(tableSpanV[0] - tableSpanV[0])

    entryU = BasisFun(int(tableSpanU[0]), xi, degree_u, knotvector_u)
    entryV = BasisFun(int(tableSpanV[0]), eta, degree_v, knotvector_v)
    entryUder = BasisFunDers(int(tableSpanU[0]), xi, degree_u, knotvector_u, 1)
    entryVder = BasisFunDers(int(tableSpanV[0]), eta, degree_v, knotvector_v, 1)
    localweights = weightmatrix[startX:startX + degree_u + 1, startY:startY + degree_v + 1]
    NM = (entryU.transpose() * entryV)
    dNM = (entryUder[1, :].transpose() * entryVder[0, :])
    NdM = (entryUder[0, :].transpose() * entryVder[1, :])

    wNM = np.multiply(NM, localweights)
    wdNM = np.multiply(dNM, localweights)
    wNdM = np.multiply(NdM, localweights)

    W = np.sum(wNM)
    WprimedNM = np.sum(wdNM)
    WprimeNdM = np.sum(wNdM)

    numeratordNM = W * wdNM - WprimedNM * wNM
    numeratorNdM = W * wNdM - WprimeNdM * wNM
    for iU in range(0, degree_u + 1):
        for iV in range(0, degree_v + 1):
            C[startX + iU, startY + iV, 0, 0] = wNM[iU, iV] / W
            dC_dxi[startX + iU, startY + iV, 0, 0] = numeratordNM[iU, iV] / W ** 2
            dC_deta[startX + iU, startY + iV, 0, 0] = numeratorNdM[iU, iV] / W ** 2
    field = 0.
    for i in range(0, nobU):
        for j in range(nobV):
            field += C[i, j, 0, 0] * param[i, j]
    return field


def lookuptablespan(knotVector, k, plotVector, s):
    """ make a matrix for which A[i] is the knot span in which plotVector[i] lays in
    :param knotVector: knotVector to span
    :param k: knotVector length
    :param plotVector: parametric coordinate vector to span in the knotVector
    :param s: plotVector length
    :return a matrix contaning the knotvector span for each element of plotVector
    source: the NURBS book
    """
    tablespan = np.zeros((s))
    left = 0
    indexActu = 0
    for i in range(1, k):
        right = knotVector[i]
        if (knotVector[left] == right):
            continue
        left = i - 1
        while (plotVector[indexActu] < right):
            tablespan[indexActu] = int(left)
            indexActu += 1
    tablespan[indexActu] = left - 1
    return tablespan


def BasisFun(i, u, p, U):
    """
    Returns the value of the basis function i, of degree p with knots U, at point u
    source: the NURBS book
    """
    N = np.matrix(np.zeros((p + 1)));
    N[0, 0] = 1;

    left = np.zeros((p + 1));
    right = np.zeros((p + 1));
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j];
        right[j] = U[i + j] - u;
        saved = 0;
        for r in range(0, j):
            temp = N[0, r] / (right[r + 1] + left[j - r]);
            N[0, r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;

        N[0, j] = saved;

    return N


def BasisFunDers(i, u, p, U, n):
    """ Compute the basis function and the n-th first derivatives
    :param i: knot span for u
    :param u: evaluation point
    :param p: basis order
    :param U: knot vector
    :param n: maximum derivative degree
    :return matrix ders[p+1,p+1] where ders[i,j] is the value of the i-th derivative of Nj function
    source: the NURBS book
    """

    N = np.matrix(np.zeros((p + 1, p + 1)))
    N[0, 0] = 1
    left = np.zeros((p + 1))
    right = np.zeros((p + 1))
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0
        for r in range(0, j):
            N[j, r] = right[r + 1] + left[j - r]
            temp = N[r, j - 1] / N[j, r]
            N[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j, j] = saved

    # Load the basis function
    ders = np.matrix(np.zeros((n + 1, p + 1)))
    for j in range(0, p + 1):
        ders[0, j] = N[j, p]

    # Compute the derivatives
    a = np.matrix(np.zeros((2, p + 1)))
    for r in range(0, p + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1
        for k in range(1, n + 1):
            d = 0.
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / N[pk + 1, rk]
                d = a[s2, 0] * N[rk, pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / N[pk + 1, rk + j]
                d += a[s2, j] * N[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / N[pk + 1, r]
                d += a[s2, k] * N[r, pk]
            ders[k, r] = d
            j = s1
            s1 = s2
            s2 = j
    r = p
    for k in range(1, n + 1):
        for j in range(0, p + 1):
            ders[k, j] *= r
        r *= (p - k)

    return ders
