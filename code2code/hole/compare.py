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
	x_g = np.linspace(np.min(x), np.max(x), nbpts)
	y_g = np.linspace(np.min(y), np.max(y), nbpts)
	X,Y = np.meshgrid(x_g,y_g)
	Z = griddata(xy,z,(X,Y), method="linear")
	
	return X,Y,Z

x_iso,y_iso,ux_iso,uy_iso = np.loadtxt("u_CP", unpack=True)
x_fe,y_fe,ux_fe,uy_fe = np.loadtxt("u_FE", unpack=True)
scalez_min = np.min(ux_fe)
scalez_max = np.max(ux_fe)

X_fe, Y_fe, Z_fe = interp_grid(x_fe, y_fe, ux_fe, 300)
X_iso, Y_iso, Z_iso = interp_grid(x_iso, y_iso, ux_iso, 300)

plt.figure(figsize=(10,8))
plt.subplot(221)
plt.contourf(X_fe, Y_fe, Z_fe, levels=25)
plt.colorbar()
plt.title("Ux: Finite element")
plt.subplot(222)
plt.contourf(X_iso, Y_iso, Z_iso, levels=25)
plt.colorbar()
plt.title("Ux: Isogeometric")

diff = Z_fe - Z_iso
plt.subplot(223)
plt.contourf(X_fe, Y_fe, diff)
plt.colorbar()
plt.title("FE - isogeometric")
plt.show()