import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.interpolate import griddata

def plot_irreg_field(x,y,z, scalez_min, scalez_max, title):
	cntr2 = plt.tricontourf(x, y, z, levels=25, vmin=scalez_min, vmax=scalez_max)
	plt.colorbar()
	#plt.plot(x, y, '.', ms=3)
	plt.xlim((np.min(x), np.max(x)))
	plt.ylim((np.min(y), np.max(y)))
	plt.title(title)

def interp_grid(x,y,z, nbpts):
	lenDat = len(x)
	xy = []
	for i in range(0, lenDat):
		xy.append([x[i], y[i]])
	x_g =[-4.0]#np.linspace(np.min(x), np.max(x), nbpts)
	y_g = np.linspace(np.min(y), np.max(y), nbpts)
	X,Y = np.meshgrid(x_g,y_g)
	Z = griddata(xy,z,(X,Y), method="nearest")
	
	return X,Y,Z

#x_iso,y_iso,ux_iso, uy_iso = np.loadtxt("../../SQR_vals/u_CP", unpack=True)

errs = []
# mesh_num = [1, 16, 100, 484]
# for i in range(0,4):
x_iso,y_iso,sig_iso = np.loadtxt("../../SQR_vals/sigma_xx_0", unpack=True)
	# sol_ana = 0.1*np.ones(len(x_iso))
	# err_actu = np.sum(sig_iso-sol_ana)/len(x_iso)
	# errs.append(err_actu)

X_iso, Y_iso, Z_iso = interp_grid(x_iso, y_iso, sig_iso, 300)
# plt.plot(X_iso.tolist()[0],Z_iso.tolist()[0],label= "isogeo")
plt.plot(Y_iso,Z_iso,label= "isogeo")
plt.ylim((0, 150))
# lognum = np.log10(np.array(mesh_num)/4.)
# logerr = np.log10(errs)
# lin_approx = np.polyfit(lognum, logerr, deg=1)
# plt.plot(lognum, logerr)
# plt.plot(mesh_num, errs)
# plt.annotate("pente: "+str(lin_approx[0]), (lognum[1],logerr[1]))
# plt.legend()
plt.show()
