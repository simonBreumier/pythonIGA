from simonurbs import make_plot_matrix
import numpy as np
from scipy.interpolate import griddata

def comp_stress(ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, toPlot, E, nu):
	nobU = ctrlpts.shape[0]
	nobV = ctrlpts.shape[1]
	weights = np.zeros((nobU, nobV))
	ctrl_mat = np.zeros((nobU, nobV, 2))
	
	for i in range(0,nobU):
		for j in range(0,nobV):
			weights[i,j] = ctrlpts[i,j,-1]
			ctrl_mat[i,j,:] = ctrlpts[i,j,0:2]
	resol = 100
	x = []
	y = []
	box_ecart = 0.5
	C, dC_dxi, dC_deta = make_plot_matrix(ctrlpts, knotvector_u, knotvector_v, degree_u, degree_v, resol, weights)
	field = {}
	SU = np.zeros((resol,resol))
	SV = np.zeros((resol,resol))
	eps = np.zeros((2,2,resol,resol))
	sig = np.zeros((2,2,resol,resol))
	plotnum = len(toPlot.keys())
	dx_dxi = np.zeros((2,2,resol,resol))
	dxi_dx = np.zeros((2,2,resol,resol))
	dR_dx = np.zeros((2,nobU, nobV, resol,resol))
	for elem in toPlot.keys():
		field[elem] = np.zeros((resol,resol))

	for j in range(0,nobV):
		for i in range(0,nobU):
			SU = SU + ctrl_mat[i,j,0] * C[i,j,:,:]
			SV = SV + ctrl_mat[i,j,1] * C[i,j,:,:]
			dx_dxi[0,0] += np.multiply(ctrl_mat[i,j,0],dC_dxi[i,j,:,:])
			dx_dxi[0,1] += np.multiply(ctrl_mat[i,j,0] , dC_deta[i,j,:,:])
			dx_dxi[1,0] += np.multiply(ctrl_mat[i,j,1], dC_dxi[i,j,:,:])
			dx_dxi[1,1] += np.multiply(ctrl_mat[i,j,1], dC_deta[i,j,:,:])
			for elem in toPlot.keys():
				field[elem] += float(toPlot[elem][j*nobU + i])*C[i,j,:,:]
			x.append(ctrlpts[i,j,0])
			y.append(ctrlpts[i,j,1])	
	det = np.zeros((resol,resol))	
	for i in range(0,resol):
		for j in range(0,resol):
			dxi_dx[:,:,i,j] = np.linalg.inv(np.matrix(dx_dxi[:,:,i,j]))
			det[i,j] = np.linalg.det(np.matrix(dx_dxi[:,:,i,j]))

	for j in range(0,nobV):
		for i in range(0,nobU):
			dR_dx[0,i,j] = dxi_dx[0,0]*dC_dxi[i,j] + dxi_dx[1,0]*dC_deta[i,j]
			dR_dx[1,i,j] = dxi_dx[0,1] *dC_dxi[i,j] + dxi_dx[1,1]*dC_deta[i,j]
	
	for j in range(0,nobV):
		for i in range(0,nobU):
			if "u_x" in toPlot.keys() and "u_y" in toPlot.keys(): 
				eps[0,0,:,:] += float(toPlot["u_x"][j*nobU + i]) * dR_dx[0,i,j,:,:]
				eps[1,1,:,:] += float(toPlot["u_y"][j*nobU + i]) * dR_dx[1,i,j,:,:]
				eps[0,1,:,:] += 0.5*(float(toPlot["u_x"][j*nobU + i]) * dR_dx[1,i,j,:,:] + float(toPlot["u_y"][j*nobU + i]) * dR_dx[0,i,j,:,:])
				eps[1,0,:,:] += 0.5*(float(toPlot["u_x"][j*nobU + i]) * dR_dx[1,i,j,:,:] + float(toPlot["u_y"][j*nobU + i]) * dR_dx[0,i,j,:,:])
	
	coef = E/(1-nu**2)
	for j in range(0,resol):
		for i in range(0,resol):
			if "u_x" in toPlot.keys() and "u_y" in toPlot.keys(): 
				sig[0,0,i,j] = coef*(eps[0,0,i,j] + nu*eps[1,1,i,j])
				sig[1,1,i,j] = coef*(eps[1,1,i,j] + nu*eps[0,0,i,j])
				sig[0,1,i,j] = coef*(1-nu)*eps[0,1,i,j] 
				sig[1,0,i,j] = coef*(1-nu)*eps[1,0,i,j]
	
	shapeList = (resol*resol)
	sigxx = np.reshape(sig[0, 0],shapeList)
	sigyy = np.reshape(sig[1, 1],shapeList)
	x = np.reshape(SU,shapeList)
	y = np.reshape(SV,shapeList)
	lenDat = len(sigxx)
	nbpts = 10
	xy = []
	for i in range(0, lenDat):
		xy.append([x[i], y[i]])
	x_g = [-4.0]#np.linspace(np.min(x), np.max(x), nbpts)
	y_g = np.linspace(np.min(y), np.max(y), nbpts)
	X,Y = np.meshgrid(x_g,y_g)
	Zxx = griddata(xy,sigxx,(X,Y), method="nearest")
	x_g = np.linspace(np.min(x), np.max(x), nbpts)
	y_g = [4.0]#np.linspace(np.min(y), np.max(y), nbpts)
	X,Y = np.meshgrid(x_g,y_g)
	Zyy = griddata(xy,sigyy,(X,Y), method="nearest")
	res = 0.5*np.sum(abs(Zxx+100*np.ones(Zxx.shape))**2 + abs(Zyy-100*np.ones(Zyy.shape))**2)/nbpts
	return res
			