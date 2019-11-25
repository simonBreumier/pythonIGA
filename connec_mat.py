import numpy as np

def make_IEN_INC(n, p, m, q):
	'''Build the IEN array such that given an element number e and a local basis function number b, gives the global basis function number B
	Build the INC matrix such that INC(A,i) gives the local shape functiopn number in direction i associated to the global shape function number A'''
	nel = (n-p)*(m-q)
	nnp = n*m
	nen = (p+1)*(q+1)
	INC = np.zeros((nnp,2))
	IEN = np.zeros((nel, nen))
	
	e = 0
	A = 0
	B = 0
	b = 0
	
	for j in range(0, m):
		for i in range(0, n):
			INC[A, 0] = i
			INC[A, 1] = j
			if i >= p and j >= q:
				e +=1
				for jloc in range(0,q+1):
					for iloc in range(0,p+1):
						B = A - jloc * n - iloc
						b = jloc * (p+1) + iloc
						IEN[e-1,b] = B
			A = A+1
	return INC, IEN

def make_IEN_INC_BC(n, p):
	'''Build the IEN array such that given an element number e and a local basis function number b, gives the global basis function number B
	Build the INC matrix such that INC(A,i) gives the local shape functiopn number in direction i associated to the global shape function number A'''
	nel = (n-p)
	nnp = n
	nen = (p+1)
	INC = np.zeros((nnp))
	IEN = np.zeros((nel, nen))
	
	e = 0
	A = 0
	B = 0
	b = 0

	for i in range(0, n):
		INC[A] = i
		if i >= p:
			e +=1
			for iloc in range(0,p+1):
				B = A - iloc
				b = iloc
				IEN[e-1,b] = B
		A = A+1
	return INC, IEN