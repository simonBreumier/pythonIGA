"""
Functions to solve the plate with a whole problem.
"""
import numpy as np

from assembly import make_KF, impose_BC
from connec_mat import make_IEN_INC, make_IEN_INC_BC
from impose_pres import compute_Fimp_alt
from make_geom import build_BC_surf
from meshOpener import load_geom
from mesh_rule import Gauss_rule
from plotNurb import plot_nurbs
from solve import solve_KU


def make_holeplate(numrefine, disp_solve, plotGrid=""):
    """ Solve the plate with hole problem with linear elasticity
    :param numrefine: Refined mesh number to use
    :param dispSolve: Boolean, activate the result display
    :param plotGrid: "mesh" to superimpose the mesh to the results
    """
    print("------------------- Building geometry...")
    knotvector_u, knotvector_v, p, q, nobU, nobV, ctrlpts = load_geom("meshes/mesh"+str(numrefine))
    L = 4
    R = 1
    n = len(knotvector_u) - (p+1)
    m = len(knotvector_v) - (q+1)

    print("------------------- Building connectivity matrices")
    INC, IEN = make_IEN_INC(n, p, m, q)
    BC_ctrlpts, BC_knot, BC_corres = build_BC_surf(knotvector_u, ctrlpts, 4)

    print("------------------- Generate Gauss quadrature points")
    nquad = 4
    gp, gw = Gauss_rule(nquad)

    print("------------------- Assemble K and F matrix")
    E_coeff = 200000.
    nu_coeff = 0.3
    nen = ctrlpts.shape[0]*ctrlpts.shape[1]
    Fb = np.zeros(nen)
    nel = (n-p)*(m-q)
    print("Number of elements: "+str(nel))
    nobU = ctrlpts.shape[0]
    nobV = ctrlpts.shape[1]
    K, F, GP_coord = make_KF(knotvector_u, knotvector_v, ctrlpts, p, q, nel, INC, IEN, gp, gw, nquad, E_coeff, nu_coeff, Fb)

    print("------------------- Assemble boundary conditions")
    print("Dirichlet...")
    Ub = []
    for j in range(0, nobV):
        for i in range(0, nobU):
            if ctrlpts[i,j,0] == 0.:
                Ub.append([2*(j*nobU+i), 0, 0.])
            # if ctrlpts[i,j,0] == -L:
            #     Ub.append([2*(j*nobU+i), 0, -0.01])
            # if ctrlpts[i,j,1] == L:
            #     Ub.append([2*(j*nobU+i), 1, 0.01])
            if ctrlpts[i,j,1] == 0:
                Ub.append([2*(j*nobU+i), 1, 0.])
    Ub = sorted(Ub)
    method = "reduce"
    F, Ub_vect, free_dofs, fixed_dofs = impose_BC(Ub, K, F, method)

    print("Pressure...")
    n_BC = BC_ctrlpts.shape[0]
    INC_BC, IEN_BC = make_IEN_INC_BC(n_BC, p)
    nel_BC = n_BC - p
    toImpose = []
    for i in range(0, n_BC):
        if BC_ctrlpts[i,0] == -L:
            toImpose.append(i)

    sig_imp = np.matrix([[-10, 0.], [0., 0.]])
    compute_Fimp_alt(nel_BC, IEN_BC, INC_BC, gp, gw, BC_knot, p, BC_ctrlpts, ctrlpts, BC_corres, nquad, sig_imp, F, toImpose)

    print("------------------- Solving system (get a cup of coffee ;) )")
    u, f_rea = solve_KU(K, F, free_dofs, fixed_dofs, Ub_vect, nen, method="reduce")

    print("------------------- Plotting results")
    to_plot = {"u_x": u[range(0, len(u), 2)], "u_y": u[range(1, len(u), 2)]}
    plot_nurbs(ctrlpts, knotvector_u, knotvector_v, p, q, u, GP_coord, nel, IEN, to_plot, plotGrid, E_coeff, nu_coeff, disp_solve, numrefine)
