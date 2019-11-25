import numpy as np
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from latexify import latexify



def ana_hole(t, R, r, theta):
	sig_rr = 0.5*(1-R**2/r**2) + 0.5*(1-4*R**2/r**2 + 3*R**4/r**4)*math.cos(2*theta)
	sig_thth = 0.5*(1+R**2/r**2) - 0.5*(1 + 3*R**4/r**4)*math.cos(2*theta)
	sig_rth = -0.5*(1+2*R**2/r**2 - 3*R**4/r**4)*math.sin(2*theta)
	return t*sig_rr

def beam_disp(P,x,E,h,L):
	I = h**4/12
	y = (P*x**2)*(3*L-x)/(6*E*I)
	return y

def FE_comp(x,y,ux, resol):
	xFE, yFE, uxFE, uyFE = np.loadtxt("SQR_vals/FE/u_FE_stress", unpack=True)
	xyFE = []
	lenxFE = len(xFE)
	lenx = len(x)
	xy = []
	for i in range(0, lenxFE):
		xyFE.append([xFE[i]-4., yFE[i]])
		xFE[i] = xFE[i]-4.
	for i in range(0, lenx):
		xy.append([x[i]-4., y[i]])
	X = np.reshape(x, (resol,resol))
	Y = np.reshape(y, (resol,resol))
	interpFE = griddata(xyFE, uxFE, (X, Y), method="cubic")
	UX_ISO = np.reshape(ux, (resol,resol))
	err = np.zeros((resol,resol))
	theta = np.linspace(math.pi/2,math.pi,50)
	xcircle = np.cos(theta)
	ycircle = np.sin(theta)

	for i in range(0, resol):
		for j in range(0, resol):
				err[i,j] = (UX_ISO[i,j] - interpFE[i,j])/np.max(abs(interpFE))

	res = np.sqrt(np.sum(abs(UX_ISO - interpFE)**2)/(resol*resol))

	return res

def comp_diff(file_actu):
	t = 10
	R = 1
	L = 4
	x, y, sig_xx, sig_yy, sig_xy= np.loadtxt(file_actu, unpack=True)

	lenx = x.shape[0]
	xy = []
	res = []
	anasol = []
	sig_rr = np.zeros((lenx))
	for i in range(0, lenx):
		xy.append([x[i],y[i]])
		r = math.sqrt(x[i]**2+y[i]**2)
		if not(x[i] == 0.):
			theta = math.atan(y[i]/x[i])
		else:
			theta = math.pi/2
		sig_rr[i] = sig_xx[i]*np.cos(theta)**2 + 2*sig_xy[i]*np.cos(theta)*np.sin(theta) + sig_yy[i]*np.sin(theta)**2
		anasol.append(ana_hole(t, R, r, theta))
		res.append(abs(sig_rr[i]-ana_hole(t, R, r, theta)))
	
	resFinal = math.sqrt(np.sum(np.array(res)**2)/lenx)
	#resFinal = abs(u_y[-1] - beam_disp(P,L,E,h,L))
	return resFinal, anasol, sig_rr, xy, x, y, L, sig_xx

resvar = []
restot = []
resSig = []
file = "SQR_vals/"
resol = 100
resSigAct, anasol, sig_rr, xy, x, y, L, sig_xxprev = comp_diff(file+"sigma_0")

xprev, yprev, uxprev, uyprev = np.loadtxt(file+"/u_CP_0", unpack=True)
resFinal = FE_comp(xprev,yprev,uxprev, resol)
restot.append(resFinal)

for p in range(1,4):
	resFinal, anasol, sig_rr, xy, x, y, L, sig_xx = comp_diff(file + "sigma_"+str(p))
	resSigAct = math.sqrt(np.sum((sig_xx-sig_xxprev)**2)/len(sig_xx))
	resSig.append(resSigAct)
	sig_xxprev = sig_xx
	x, y, ux, uy = np.loadtxt(file+"/u_CP_"+str(p), unpack=True)
	#plt.contourf(np.reshape(x, (resol, resol)), np.reshape(y, (resol, resol)), np.reshape(sig_xx, (resol, resol)))
	#plt.colorbar()
	#plt.show()
	resFinal = FE_comp(x,y,ux, resol)
	restot.append(resFinal)
	lenX = len(x)
	resComp = 0
	count = 0
	errvals = []
	sigfront = []
	sigy = []
	for i in range(0, lenX):
	# 	if x[i] >= -0.01:
	# 		sigy.append(y[i])
	# 		sigfront.append(sig_xx[i])
		if abs(x[i] - xprev[i]) <1.e-5 and abs(y[i] - yprev[i]) <1.e-5:
	 		resComp += (ux[i]-uxprev[i])**2
	 		count +=1
	 		errvals.append([x[i], y[i], abs(ux[i]-uxprev[i])])
	#plt.plot(sigy,sigfront, label=str(p))
	# plt.contourf(np.reshape(x, (resol, resol)), np.reshape(y, (resol, resol)), np.reshape(sig_xx, (resol, resol)))
	# plt.colorbar()
	# plt.show()
	resComp = math.sqrt(resComp/count)
	uxprev = ux
	resvar.append(resComp)

#plt.legend()
#plt.show()

# numMesh = [1, 16, 100, 484]
# numMesh = [1, 9, 49, 225, 961]
# numMesh = [4, 30, 154, 690, 2914]
# numMesh = [4, 7, 13, 25, 49]
numMesh = [2, 4, 8, 16]
# numMesh = [3, 6, 12, 24]
lognum = np.log10(np.array(numMesh))
logresvar = np.log10(resvar)
r= numMesh[1]/numMesh[0]
lin_approx = np.polyfit(lognum[1:], logresvar, deg=1)

latexify(10, 8)
ptes = np.log10((resSig[2]-resSig[1])/(resSig[1]-resSig[0]))/np.log10(r)
plt.plot(lognum[1:], logresvar, 'o-', linewidth=4, markersize=6, color="#205996")
plt.annotate("pente: "+str(lin_approx[0]), (lognum[1], logresvar[1]))
plt.xlabel("h")
plt.ylabel("$||\sigma_{analytique} - \sigma_{IGA}||_2$")
print(ptes)
gci = (3/(r**ptes-1))*(resSig[2]-resSig[1])
print("gci: "+str(gci))
plt.savefig("Figure/Convergence.jpg")
plt.show()


lognum = np.log10(np.array(numMesh))
logvalres = np.log10(restot)
lin_approx = np.polyfit(lognum, logvalres, deg=1)
plt.plot(lognum, logvalres)
plt.title("Erreur a la solution analytique")
plt.annotate("pente: "+str(lin_approx[0]), (lognum[1], logvalres[1]))
plt.show()
# x_g = np.linspace(np.min(x), np.max(x), 100)
# y_g = np.linspace(np.min(y), np.max(y), 100)
# X,Y = np.meshgrid(x_g, y_g)

# interpSol = griddata(xy, sig_xx,(X,Y), method="cubic")
# interpana = griddata(xy, anasol,(X,Y), method="cubic")
# resol = 200
# X = np.reshape(x, (resol,resol))
# Y = np.reshape(y, (resol,resol))
# SIG = np.reshape(sig_xx, (resol,resol))
# ANA = np.reshape(anasol, (resol,resol))

# plt.figure(figsize=(10,8))
# plt.subplot(221)
# plt.contourf(X,Y,SIG)
# plt.colorbar()
# plt.title("$\sigma_{xx}$: IGA")

# plt.subplot(222)
# plt.contourf(X,Y,ANA)
# plt.colorbar()
# plt.title("$\sigma_{xx}$: Analytique")

# plt.subplot(223)
# plt.contourf(X,Y,abs(SIG-ANA))
# plt.colorbar()
# plt.title("|$\sigma_{IGA}$ - $\sigma_{analytique}$|")

# plt.show()
