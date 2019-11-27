import numpy as np

from simonurbs import BasisFunDers

def build_shapeFun(e, INC, IEN, xi_tilde, eta_tilde, knot_u, knot_v, p, q, B):
	"""Compute the shape function values and derivatives at a gauss point

	:param e: actual element number
	:param IEN: element to node connectivity matrix
	:param INC: global to local shape function number connectivity matrix
	:param xi_tilde,eta_tilde: gauss point coordinates
	:param knot_u,knotV: knot vector
	:param p,v: degree in U and V directions
	:param B: Control points
	:return:
	"""
	R = np.zeros((p+1, q+1))
	dR_dx = np.zeros((p+1, q+1, 2)) # wrt geometric coordinates

	dR_dxi = np.zeros((p+1, q+1, 2)) # wrt parametric coordinates
	dx_dxi = np.zeros((2,2))
	dxi_dtildexi = np.zeros((2,2))
	
	J_mat = np.zeros((2,2))
	loc_num = 0
	sum_xi = 0
	sum_eta = 0
	sum_tot = 0
	
	ni = int(INC[int(IEN[e, 0]), 0])
	nj = int(INC[int(IEN[e, 0]), 1])
	
	xi = 0.5*((knot_u[ni+1]- knot_u[ni])*xi_tilde + knot_u[ni+1] + knot_u[ni])
	eta = 0.5*((knot_v[nj+1]- knot_v[nj])*eta_tilde + knot_v[nj+1] + knot_v[nj])
	
	dN = BasisFunDers(ni,xi,p,knot_u, 1)
	dM = BasisFunDers(nj,eta,q,knot_v, 1)

	for j in range(0,q+1):
		for i in range(0, p+1):
			loc_num += 1
			R[i,j] = dN[0,p-i]*dM[0, q-j]*B[ni-i, nj-j,2]
			sum_tot += R[i,j]
			dR_dxi[i,j,0] = dN[1,p-i]*dM[0, q-j]*B[ni-i, nj-j,2]
			sum_xi += dR_dxi[i,j,0]
			dR_dxi[i,j,1] = dN[0,p-i]*dM[1, q-j]*B[ni-i, nj-j,2]
			sum_eta += dR_dxi[i,j,1]
	
	for j in range(0,q+1):
		for i in range(0, p+1):
			R[i,j] = R[i,j]/sum_tot
			dR_dxi[i,j, 0] = (dR_dxi[i,j,0]*sum_tot - R[i,j]*sum_xi) / sum_tot**2
			dR_dxi[i,j, 1] = (dR_dxi[i,j,1]*sum_tot - R[i,j]*sum_eta) / sum_tot**2
	
	for j in range(0,q+1):
		for i in range(0, p+1):
			for aa in range(0,2):
				for bb in range(0,2):
					dx_dxi[aa, bb] = dx_dxi[aa, bb] + B[ni-i, nj-j, aa]*dR_dxi[i,j,bb]

	dxi_dx = np.linalg.inv(np.asmatrix(dx_dxi))
	
	for j in range(0,q+1):
		for i in range(0, p+1):
			for aa in range(0,2):
				for bb in range(0,2):
					dR_dx[i,j, aa] = dR_dx[i,j, aa] + dR_dxi[i,j, bb]*dxi_dx[bb, aa]

	dxi_dtildexi[0,0] = 0.5*(knot_u[ni+1] - knot_u[ni])
	dxi_dtildexi[1, 1] = 0.5*(knot_v[nj+1] - knot_v[nj])
	
	for aa in range(0,2):
		for bb in range(0,2):
			for cc in range(0,2):
				J_mat[aa, bb] = J_mat[aa, bb] + dx_dxi[aa, cc] * dxi_dtildexi[cc, bb]
				
	J = np.linalg.det(np.asmatrix(J_mat))
	gp_xitilde = np.matrix([xi_tilde, eta_tilde])
	gp_x = J_mat*gp_xitilde.T
	return R, dR_dx, J, J_mat, gp_x


def make_KF(knot_u, knot_v, B, p, q, nel, INC, IEN, gp, gw, nquad, E_coeff, nu_coeff, Fb):
	"""Compute the global K and F matrix

	:param knot_u,knotv: knot vectors
	:param B: control points
	:param p,q: degrees in U and V directions
	:param nel: number of elements
	:param INC: globale to local shape function number connectivity matrix
	:param IEN: element to node connectivity matrix
	:param gp: Gauss points
	:param gw: Gauss weights
	:param nquad: number of quadratic points
	:param E_coeff: Youngs modulus
	:param nu_coeff: poissons ratio
	:param Fb: volume force
	"""
	nen = (p + 1) * (q + 1)
	npts = B.shape[0] * B.shape[1]
	K_global = np.zeros((2 * npts, 2 * npts))
	F_global = np.zeros((2 * npts))
	GP_coord = np.zeros((nel, nquad * nquad, 2))

	R_quads = np.zeros((p+1, q+1, nquad**2, nel))
	dR_quads = np.zeros((p + 1, q + 1, 2, nquad**2, nel))

	for e in range(0, nel):
		ni = int(INC[int(IEN[e, 0]), 0])
		nj = int(INC[int(IEN[e, 0]), 1])
				
		if (knot_u[ni+1] == knot_u[ni]) or (knot_v[nj+1] == knot_v[nj]):
			continue
			
		K_local = np.zeros((2*nen, 2*nen))
		F_local = np.zeros((2*nen))
		
		#*************** BUILD K AND F LOCAL *********************
		for i in range(0, nquad):
			for j in range(0, nquad):
				R, dR_dx, J, J_mat, gp_x = build_shapeFun(e, INC, IEN, gp[i], gp[j], knot_u, knot_v, p, q, B)
				R_quads[:, :, i * nquad + j, e] = R
				dR_quads[:, :, :, i * nquad + j, e] = dR_dx
				Jmod = J*gw[i]*gw[j]
				build_Klocal(dR_dx, Jmod, p, q, K_local, E_coeff, nu_coeff)
				build_Flocal(R, Jmod, Fb, p, q, F_local)
				GP_coord[e, j * nquad + i, :] = np.array(gp_x.tolist()[0])
		
		#*************** Global ASSEMBLY *********************
		for j in range(0, q+1):
			for i in range(0, p+1):
				aa = j * (p + 1) + i
				F_global[2 * int(IEN[e, aa]):2 * int(IEN[e, aa]) + 2] += F_local[2 * aa:2 * aa + 2]
				for l in range(0, q + 1):
					for k in range(0, p + 1):
						bb = l * (p + 1) + k
						K_global[2 * int(IEN[e, aa]):2 * int(IEN[e, aa]) + 2,
						2 * int(IEN[e, bb]):2 * int(IEN[e, bb]) + 2] += K_local[2 * aa:2 * aa + 2, 2 * bb:2 * bb + 2]
	return K_global, F_global, GP_coord, R_quads, dR_quads


def build_Klocal(dR_dx, Jmod, p, q, K_local, E, nu):
	"""Build the local K matrix

	:param dR_dx: shaper function deriviative at the gauss point
	:param Jmod: Jacobian (from the master element to the geometric space) multiplied by the Gauss weight
	:param p,q: degrees in the U and V directions
	:param K_local: local K tensor
	:param E: Youngs modulus
	:param nu: poisson's ratio
	:return: Nothing but changes K_local
	"""
	coef = E / (1 - nu ** 2)
	for i in range(0, p + 1):
		for j in range(0, q + 1):
			for k in range(0, p + 1):
				for l in range(0, q + 1):
					aa = j * (p + 1) + i
					bb = l * (p + 1) + k
					K_local[2 * aa, 2 * bb] += coef * (
								dR_dx[i, j, 0] * dR_dx[k, l, 0] + (1 - nu) * dR_dx[i, j, 1] * dR_dx[k, l, 1]) * Jmod
					K_local[2 * aa + 1, 2 * bb + 1] += coef * (
								dR_dx[i, j, 1] * dR_dx[k, l, 1] + (1 - nu) * dR_dx[i, j, 0] * dR_dx[k, l, 0]) * Jmod
					K_local[2 * aa, 2 * bb + 1] += coef * (
								nu * dR_dx[i, j, 0] * dR_dx[k, l, 1] + (1 - nu) * dR_dx[i, j, 1] * dR_dx[
							k, l, 0]) * Jmod
					K_local[2 * aa + 1, 2 * bb] += coef * (
								nu * dR_dx[i, j, 1] * dR_dx[k, l, 0] + (1 - nu) * dR_dx[i, j, 0] * dR_dx[
							k, l, 1]) * Jmod
	
def build_Flocal(R, Jmod, Fb, p, q, F_local):
	"""Build local load vector with volume forces

	:param R: shape functions
	:param Jmod: jacobian from the master element to the geometric space
	:param Fb: volume force vector
	:param p,q: degrees in U and V directions
	:param F_local: loacl load vector
	:return: NOthing but changes F_local
	"""
	for i in range(0, p+1):
		for j in range(0, q+1):
			aa = j*(p+1)+i
			F_local[2*aa] += Fb[0]*R[i,j]*Jmod
			F_local[2*aa+1] += Fb[1]*R[i,j]*Jmod

		
def impose_BC(Ub, K, F, method):
	"""Impose Dirichlet boundary conditions (don't ask me why it's here)

	:param Ub: (nb,3) vectors containing the node to be imposed, the dof number (1 or 2) and the displacement value
	:param K: global K matrix
	:param F: global load vector
	:param method: "reduce" or "penalty
	:return: F, Ub_vect, free_coord, fixed_coord
	"""
	fixed_coord = []
	Ub_vect = np.zeros(len(Ub))
	spanrange = range(0,K.shape[0])
	i=0

	for elem in Ub:
		fixed_coord.append(elem[0]+elem[1])
		Ub_vect[i] = elem[2]
		i+=1
	Ub_vect = np.matrix(Ub_vect)
	
	free_coord = [x for x in spanrange if x not in fixed_coord]
	if method=="reduce":
		tmp = K[free_coord,:]
		tmp = tmp[:,fixed_coord]
		fub = tmp*Ub_vect.transpose()
		F[free_coord] = F[free_coord]-fub[:,0].transpose()
	elif method=="penalty":
		F[fixed_coord] = 1e7*Ub_vect
	
	return F, Ub_vect, free_coord, fixed_coord