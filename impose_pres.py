"""
Several functions to compute the stress contribution to the force vector
"""
import math

import numpy as np

from assembly import build_shapeFun
from simonurbs import BasisFunDers


def compute_Fimp(nel, IEN, INC, gp, gw, knot_u, knot_v, p, q, ctrlpts, dof_BC, nquad, sig_imp, F):
	for e in range(0, nel):
		ni = int(INC[int(IEN[e, 0]), 0])
		nj = int(INC[int(IEN[e, 0]), 1])
		
		if (knot_u[ni+1] == knot_u[ni]) or (knot_v[nj+1] == knot_v[nj]):
			continue
			
		dof_glob = []
		dof_loc = []
		for j in range(0,q+1):
			for i in range(0,p+1):
				aa = j*(p+1)+i
				if 2*int(IEN[e, aa]) in dof_BC:
					dof_glob.append(2*int(IEN[e, aa]))
					dof_loc.append([i,j])
				if 2*int(IEN[e, aa])+1 in dof_BC:
					dof_glob.append(2*int(IEN[e, aa])+1)
					dof_loc.append([i,j])
		lendofs = len(dof_loc)
		F_BC_loc = np.zeros((lendofs))
		if not(len(dof_loc) == 0):			
			for gp_act in range(0, nquad):	
				R, dR_dx, J, J_mat, gp_x = build_shapeFun(e, INC, IEN, gp[gp_act], 1., knot_u, knot_v, p, q, ctrlpts)
				Jmod_BC = math.sqrt(J_mat[0,0]**2+J_mat[0,1]**2)*gw[gp_act]
				build_F_BC(R, Jmod_BC, p, F_BC_loc, sig_imp, dof_loc)		

			for i in range(0, lendofs):
				indGlob = dof_glob[i]
				if indGlob%2 == 1:
					F[indGlob] += F_BC_loc[i]
				else:
					F[indGlob] -= F_BC_loc[i]

def build_F_BC(R_BC, Jmod_BC, p, F_BC_loc, sig_imp, dof_loc):
	lenDat = len(dof_loc)
	for i in range(0, lenDat):
		F_BC_loc[i] += sig_imp*R_BC[dof_loc[i][0], dof_loc[i][1]]*Jmod_BC

def compute_Fimp_alt(nel, IEN, INC, gp, gw, knot_u, p, ctrlpts_BC, corres_BC, nquad, sig_imp, F, toImpose):
	"""Compute the imposed stress contribution on the force vector

	:param nel: number of elements
	:param IEN: element to node connectivity matrix
	:param INC: globale to local shape function number connectivity matrix
	:param gp: Gauss points
	:param gw: Gauss weights
	:param knot_u: knot vector of the boundary
	:param p: boundary degree
	:param ctrlpts_BC: boundary control points
	:param corres_BC: correspondence table between the boundary nodes and the global nodes
	:param nquad: number of quadratic points
	:param sig_imp: imposed stress tensor (2x2 np.matrix)
	:param F: load vector
	:param toImpose: boundary nodes ID to be imposed the stress
	:return: Nothing, but changes the load vector F
	"""
	for e in range(0, nel):
		ni = int(INC[int(IEN[e, 0])])
		
		if (knot_u[ni+1] == knot_u[ni]):
			continue

		dof_glob = []
		for i in range(0,p+1):
			coor_glob = corres_BC[int(IEN[e, i])]
			dof_glob.append(2*coor_glob)
			dof_glob.append(2*coor_glob+1)
				
		lendofs = len(dof_glob)
		F_BC_loc = np.zeros((lendofs))
		if not(lendofs == 0):			
			for gp_act in range(0, nquad):	
				R_BC, J, dx_dtiltexi, dy_dtiltexi, dx_dxi, dy_dxi = build_shape_BC(e, INC, IEN, gp[gp_act], knot_u, p, ctrlpts_BC)
				Jmod_BC = gw[gp_act]
				build_F_BC_alt(R_BC, Jmod_BC, p, F_BC_loc, sig_imp, ctrlpts_BC, ni, dx_dtiltexi, dy_dtiltexi, toImpose, dx_dxi, dy_dxi)

			F[dof_glob] += F_BC_loc
					
def build_F_BC_alt(R_BC, Jmod_BC, p, F_BC_loc, sig_imp, ctrlpts_BC, ni, dx_dtiltexi, dy_dtiltexi, toImpose, dx_dxi, dy_dxi):
	"""Build the local element load vector

	:param R_BC: shape function values at gauss point
	:param Jmod_BC: actual Gauss point name (bad name...)
	:param p: curve degree
	:param F_BC_loc: local load vector to be filled
	:param sig_imp: imposed stress tensor (2x2 np.matrix)
	:param ctrlpts_BC: boundary control points (local numbering)
	:param ni: initial node number for the actual element
	:param dx_dtiltexi,dy_dtiltexi: vector for pull back from the reference element to the geometric space
	:param toImpose: Node tom impose sig_imp to
	:return: Nothing but changes F_BC_loc
	"""
	norm = np.matrix([-dy_dtiltexi, dx_dtiltexi]).T
	norm = norm/np.linalg.norm(norm)
	#print(str(dx_dxi)+", "+str(dy_dxi))
	sigmaNorm = np.array((sig_imp[0,0]*norm).T.tolist()[0])
	for i in range(0, p+1):
		if ni-i in toImpose:
		#if ctrlpts_BC[i,0] == -4.:
			#F_BC_loc[2*i:2*i+2] += R_BC[i]*Jmod_BC*sigmaNorm
			F_BC_loc[2 * i] += R_BC[i] * Jmod_BC * sig_imp[0,0]*(-dy_dtiltexi)
		#if ctrlpts_BC[ni-i,1] == 4.0:
			# F_BC_loc[2*i] += sig_imp*R_BC[i]*Jmod_BC	
		#	F_BC_loc[2*i+1] -= sig_imp*R_BC[i]*Jmod_BC*abs(dx_dtiltexi)
			
def build_shape_BC(e, INC, IEN, xi_tilde, BC_knot, p, B):
	""" Build the boundary shape function and compute its value at the Gauss points

	:param e: actual element number
	:param IEN: element to node connectivity matrix
	:param INC: global to local shape function number connectivity matrix
	:param xi_tilde: gauss point coordinate
	:param BC_knot: boundary knot vector
	:param p: boundary degree
	:param B: boundary control points
	"""
	n = len(BC_knot)
	R = np.zeros((p+1))
	dR_dx = np.zeros((p+1, 2)) 
	J = 0;
	dR_dxi = np.zeros((p+1, 2)) # wrt parametric coordinates
	dx_dxi = 0
	dy_dxi = 0
	loc_num = 0
	sum_xi = 0
	sum_tot = 0
	
	ni = int(INC[int(IEN[e, 0])])
	xi = 0.5*((BC_knot[ni+1]- BC_knot[ni])*xi_tilde + BC_knot[ni+1] + BC_knot[ni])
	
	dN = BasisFunDers(ni,xi,p,BC_knot, 1)

	
	for i in range(0,p+1):
		loc_num += 1
		R[i] = dN[0,p - i]*B[ni-i,2]
		sum_tot += R[i]
		dR_dxi[i,0] = dN[1,p - i]*B[ni-i,2]#############
		sum_xi += dR_dxi[i,0]	

	for i in range(0, p+1):
		R[i] = R[i]/sum_tot
		dR_dxi[i, 0] = (dR_dxi[i,0]*sum_tot - R[i]*sum_xi) / sum_tot**2

	for i in range(0, p+1):
		dx_dxi = dx_dxi + B[ni-i, 0]*dR_dxi[i,0]
		dy_dxi = dy_dxi + B[ni-i, 1]*dR_dxi[i,0]
	dxi_dtildexi = 0.5*(BC_knot[ni+1] - BC_knot[ni])

	dx_dtiltexi = dx_dxi * dxi_dtildexi
	dy_dtiltexi = dy_dxi * dxi_dtildexi

	J = math.sqrt(dx_dtiltexi**2+dy_dtiltexi**2)
	print(str(dy_dtiltexi)+" "+str(dx_dtiltexi))
	return R, J, dx_dtiltexi, dy_dtiltexi, dx_dxi, dy_dxi