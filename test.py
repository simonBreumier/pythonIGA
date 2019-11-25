import numpy as np
from simonurbs import BasisFun, BasisFunDers
import matplotlib.pyplot as plt

xi = np.linspace(0,8,1000)
N = np.zeros((1000,2,3))
knot = [0,0,0,1,2,3,4,5,6,7,8,8,8]
lenKnot = len(knot)
for i in range(0,1000):
	indAct = 0
	while indAct < lenKnot and xi[i] > knot[indAct]:
		indAct+=1

	N[i,:,:] = BasisFunDers(indAct-1,xi[i],2,knot, 1)

plt.subplot(2,1,1)
plt.plot(xi,N[:,0,0], label="N1")
#plt.plot(xi,N[:,0,1], label="N2")
#plt.plot(xi,N[:,0,2], label="N3")
plt.legend()

plt.subplot(2,1,2)
plt.plot(xi,N[:,1,0], label="N1'")
#plt.plot(xi,N[:,1,1], label="N2'")
#plt.plot(xi,N[:,1,2], label="N3'")
plt.legend()

plt.show()
	